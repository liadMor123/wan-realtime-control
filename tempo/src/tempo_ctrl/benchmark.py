"""Benchmark data, run configurations and output conventions shared by every script.

Loads TempoControl's one-object / two-object CSVs (and the showcase prompts), names video files the way the official
metric expects, derives the run tag (results/videos directory name) of a run config, configures the global
cross-attention controller for one (run, prompt) pair, and writes videos and per-video JSONL rows. Paths are rooted at
TEMPO_ROOT (default ~/tempo); the upstream checkouts live in ext/ and the weights in cache/.
"""
import csv
import json
import os

import torch

from .masks import parse_benchmark_mask
from .tokens import object_token_indices, temporal_token_indices
from .cross_attention_arms import CTRL

TEMPO = os.environ.get("TEMPO_ROOT", os.path.expanduser("~/tempo"))
WAN_ROOT = os.path.join(TEMPO, "ext", "Wan2.1")
TC_ROOT = os.path.join(TEMPO, "ext", "TempoControl")
CKPT = os.path.join(TEMPO, "cache", "Wan2.1-T2V-1.3B")
ONE_OBJECT_CSV = os.path.join(TC_ROOT, "data", "one_object.csv")
TWO_OBJECTS_CSV = os.path.join(TC_ROOT, "data", "two_objects.csv")
TEXT_CACHE = os.path.join(TEMPO, "cache", "text_enc.pt")
TEXT_CACHE_TWO = os.path.join(TEMPO, "cache", "text_enc_two.pt")     # two-object prompts (+ the same __neg__)
SHOWCASE_CSV = os.path.join(TEMPO, "data", "showcase_prompts.csv")    # non-benchmark prompts (TempoControl's website)

ARMS = ("B0", "U", "L", "S", "P")


def load_one_object():
    """Rows in file order; prompt_id = 0-based row index (header excluded)."""
    with open(ONE_OBJECT_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    for i, r in enumerate(rows):
        r["prompt_id"] = i
        r["mask"] = parse_benchmark_mask(r["control_signal1"])
    return rows


def load_two_objects():
    """Two-object pairs in file order; prompt_id = 0-based row index (header excluded). As in TempoControl's
    generate_benchmark_videos.py and metrics/temporal_accuracy_two_objects.py: the model gets `prompt`; temp_object is timed by
    control_signal1 (the metric's frame_config), static_object by control_signal2; the video is named by
    `original_prompt`."""
    with open(TWO_OBJECTS_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    for i, r in enumerate(rows):
        r["prompt_id"] = i
        r["mask"] = parse_benchmark_mask(r["control_signal1"])
        r["static_mask"] = parse_benchmark_mask(r["control_signal2"])
    return rows


def load_benchmark(name):
    if name == "one_object":
        return load_one_object()
    if name == "two_objects":
        return load_two_objects()
    raise ValueError(f"unknown benchmark {name!r}")


def _parse_id_range(s):
    """'60-79' -> [60, ..., 79]; '3' -> [3] (the source_ids column of data/showcase_prompts.csv)."""
    a, _, b = s.partition("-")
    return list(range(int(a), int(b) + 1)) if b else [int(a)]


def load_showcase(name):
    """Showcase rows of benchmark `name` (data/showcase_prompts.csv: prompts verbatim from TempoControl's project
    page, not in the benchmark), as {showcase id: row} with the same keys as load_benchmark rows, so configure,
    row_video_name and the step-4 / step-5a protocols apply unchanged. control_signal1/2 are copied from the source
    benchmark rows (source_ids); this raises unless every source row has the same masks and the file's masks equal
    them, and unless the video name differs from every benchmark video name. One-object rows carry no
    original_prompt (the official one-object metric names videos by prompt)."""
    with open(SHOWCASE_CSV, newline="") as f:
        raw = list(csv.DictReader(f))
    src = load_benchmark(name)
    taken = {row_video_name(r) for b in ("one_object", "two_objects") for r in load_benchmark(b)}
    cols = ("control_signal1",) if name == "one_object" else ("control_signal1", "control_signal2")
    out = {}
    for r in raw:
        if r["bench"] != name:
            continue
        i = int(r["id"])
        sids = _parse_id_range(r["source_ids"])
        for c in cols:
            vals = {src[j][c] for j in sids}
            if len(vals) != 1 or r[c] != vals.pop():
                raise ValueError(f"showcase {i}: {c} differs from (or among) benchmark rows {r['source_ids']}")
        row = {"prompt_id": i, "prompt": r["prompt"], "temp_object": r["temp_object"],
               "control_signal1": r["control_signal1"], "mask": parse_benchmark_mask(r["control_signal1"]),
               "source_ids": sids, "prompt_set": "showcase"}
        if name == "two_objects":
            row.update(original_prompt=r["original_prompt"], static_object=r["static_object"],
                       control_signal2=r["control_signal2"], static_mask=parse_benchmark_mask(r["control_signal2"]))
        elif r["original_prompt"] or r["static_object"] or r["control_signal2"]:
            raise ValueError(f"showcase {i}: one-object row with two-object fields")
        if row_video_name(row) in taken or i in out:
            raise ValueError(f"showcase {i}: video name {row_video_name(row)!r} or id collides")
        taken.add(row_video_name(row))
        out[i] = row
    if not out:
        raise ValueError(f"no showcase rows for {name!r}")
    return out


def video_name(prompt):
    """temporal_accuracy_one_object.py looks for f"{prompt}-0.mp4" in the videos dir."""
    return f"{prompt}-0.mp4"


def row_video_name(row):
    """Video file of a benchmark row: two-object pairs are looked up by original_prompt, one-object rows by prompt."""
    return video_name(row.get("original_prompt", row["prompt"]))


def run_tag(run):
    """Directory name for a run config: one directory per (arm, params, seed, schedule, branches)."""
    arm = run["arm"]
    parts = [arm]
    if arm == "U":
        parts.append(f"b{run['beta']:g}")
    elif arm == "L":
        parts.append(f"b{run['beta']:g}g{run['gamma']:g}")
    elif arm == "S":
        parts.append(f"hi{run['s_hi']:g}lo{run['s_lo']:g}")
    if run.get("k_steps") is not None:
        parts.append(f"k{run['k_steps']}")
    if run.get("branches", "cond") != "cond":
        parts.append(run["branches"])
    if run.get("path", "explicit") != "explicit":
        parts.append(run["path"])
    if run.get("bench", "one_object") == "two_objects":
        parts.append("2obj")
    parts.append(f"s{run['seed']}")
    if run.get("set"):                                      # a non-benchmark prompt set (e.g. showcase): own dirs
        parts.insert(0, run["set"])
    return "_".join(parts)


def configure_controller(run, row, tok):
    """Reset CTRL for one (run config, benchmark row). Returns index info for the JSONL row."""
    if run["arm"] not in ARMS:
        raise ValueError(f"unknown arm {run['arm']!r}")
    if ("static_object" in row) != (run.get("bench", "one_object") == "two_objects"):
        raise ValueError(f"run bench {run.get('bench', 'one_object')!r} does not match row {row['prompt_id']}")
    obj_idx, n_real, obj_toks = object_token_indices(tok, row["prompt"], row["temp_object"])
    tmp_idx, _, tmp_toks = temporal_token_indices(tok, row["prompt"])
    extra, info2 = [], {}
    if "static_object" in row:                                # two-object pair: K = 2, static object second
        st_idx, _, st_toks = object_token_indices(tok, row["prompt"], row["static_object"])
        if set(st_idx) & set(obj_idx):
            raise ValueError(f"object token sets overlap in pair {row['prompt_id']}: {obj_idx} vs {st_idx}")
        extra = [(st_idx, torch.tensor(row["static_mask"], dtype=torch.float32))]
        info2 = {"static_idx": st_idx, "static_tokens": st_toks}
    CTRL.path = run.get("path", "explicit")
    CTRL.arm = run["arm"]
    CTRL.beta = float(run.get("beta", 0.0))
    CTRL.gamma = float(run.get("gamma", 0.0))
    CTRL.s_hi = float(run.get("s_hi", 0.0))
    CTRL.s_lo = float(run.get("s_lo", 0.0))
    CTRL.mask = torch.tensor(row["mask"], dtype=torch.float32)
    CTRL.obj_idx = obj_idx
    CTRL.temporal_idx = tmp_idx
    CTRL.extra_objs = extra
    CTRL.branches = run.get("branches", "cond")
    CTRL.k_steps = run.get("k_steps")
    CTRL.record_shares = bool(run.get("record_shares", False))
    CTRL.share_groups = {"obj": obj_idx, "tmp": tmp_idx,
                         "real": list(range(n_real)), "pad": list(range(n_real, 512))}
    CTRL.shares = []
    CTRL._cache = {}
    if CTRL.branches != "cond":
        raise NotImplementedError("branches='both' is undefined (see cross_attention_arms.Controller.edit_active)")
    if CTRL.path == "flash" and CTRL.arm != "B0":
        raise ValueError("the flash path carries no edit")
    if CTRL.path in ("fused", "fa2kv") and CTRL.arm not in ("B0", "L"):
        raise ValueError(f"only B0 and L are defined on the {CTRL.path} path")
    if CTRL.path not in ("flash", "explicit", "fused", "fa2kv"):
        raise ValueError(f"unknown path {CTRL.path!r}")
    if extra and CTRL.arm not in ("B0", "L"):
        raise ValueError(f"two-object pairs are defined for B0 and L only, not {CTRL.arm}")
    if extra and CTRL.path != "explicit":
        raise ValueError(f"two-object pairs run on the explicit path only, not {CTRL.path}")
    CTRL.objects()                                            # overlap check on the configured controller
    return {"obj_idx": obj_idx, "obj_tokens": obj_toks, "tmp_idx": tmp_idx,
            "tmp_tokens": tmp_toks, "n_real_tokens": n_real, **info2}


def save_video(video, path):
    from wan.utils.utils import cache_video
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = cache_video(tensor=video[None], save_file=path, fps=16, nrow=1,
                      normalize=True, value_range=(-1, 1))
    if out is None or not os.path.isfile(path):
        raise RuntimeError(f"cache_video failed for {path}")


def append_jsonl(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")
