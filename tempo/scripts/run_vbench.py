#!/usr/bin/env python3
"""VBench (pinned 0.1.5, custom-input mode) imaging_quality + subject_consistency on one-object benchmark videos.

  run_vbench.py --tags B0_s42,L_b2g2_s42 --ids 0-79 --out results/vbench/step6_all80.json [--device cuda]

Runs in ~/tempo/venv_vbench (setup/build_vbench_env.sbatch). For each tag, the videos for --ids are
~/tempo/videos/<tag>/<benchmark.video_name(prompt)>; they are symlinked into a scratch dir and scored with
VBench(...).evaluate(mode='custom_input', local=True), i.e. VBench's own code path, unmodified.
Scale: imaging_quality per video = VBench's per-video MUSIQ mean / 100 (the scale of VBench's reported aggregate);
subject_consistency per video = VBench's per-video value (already 0-1).
Fails loudly on a missing video, an unexpected frame shape, or a non-finite score. Resumable per tag: finished tags are
cached in <out>.parts/<tag>.json and reused when ids, dimensions and versions match.
Weights are read from $VBENCH_CACHE_DIR (default ~/tempo/cache/vbench) and $TORCH_HOME (default <that>/torch); the
GPU job sets HF_HUB_OFFLINE=1 etc., and VBench's local mode only downloads when a file is missing.
"""
import argparse
import datetime
import json
import math
import os
import shutil
import sys
import tempfile

TEMPO = os.environ.get("TEMPO_ROOT", os.path.expanduser("~/tempo"))
VB_CACHE = os.environ.setdefault("VBENCH_CACHE_DIR", os.path.join(TEMPO, "cache", "vbench"))
os.environ.setdefault("TORCH_HOME", os.path.join(VB_CACHE, "torch"))
os.environ.setdefault("HF_HOME", os.path.join(VB_CACHE, "hf"))
sys.path.insert(0, os.path.join(TEMPO, "src"))

import importlib.metadata as md  # noqa: E402

import torch  # noqa: E402

from tempo_ctrl import benchmark  # noqa: E402

DIMS = ["imaging_quality", "subject_consistency"]
SCHEMA = 1  # bump when the scoring/scaling code changes (invalidates .parts caches)
IQ_MODE = "longer"  # VBench CLI default for imaging_quality_preprocessing_mode
REQUIRED_WEIGHTS = [
    os.path.join(VB_CACHE, "pyiqa_model", "musiq_spaq_ckpt-358bb6af.pth"),
    os.path.join(VB_CACHE, "dino_model", "dino_vitbase16_pretrain.pth"),
    os.path.join(VB_CACHE, "dino_model", "facebookresearch_dino_main", "hubconf.py"),
    os.path.join(os.environ["TORCH_HOME"], "hub", "checkpoints", "dino_vitbase16_pretrain.pth"),
]


def parse_ids(s):
    """Same syntax as scripts/generate_benchmark_videos.py parse_ids ("0-7,20,21"); also accepts @file."""
    if s.startswith("@"):
        s = open(s[1:]).read().strip()
    out = []
    for part in s.split(","):
        a, _, b = part.strip().partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def versions():
    return {k: md.version(k) for k in ["vbench", "torch", "torchvision", "pyiqa", "timm", "decord", "numpy"]}


def video_shape(path):
    from decord import VideoReader, cpu
    vr = VideoReader(path, ctx=cpu(0), num_threads=1)
    h, w, _ = vr[0].shape
    return len(vr), h, w


def score_tag(tag, ids, names, videos_root, device, expect, scratch):
    from vbench import VBench
    vdir = os.path.join(videos_root, tag)
    paths = {i: os.path.join(vdir, names[i]) for i in ids}
    missing = [p for p in paths.values() if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(f"{tag}: {len(missing)} missing videos, e.g. {missing[:3]}")
    shapes = {}
    for i, p in paths.items():
        shapes[i] = video_shape(p)
        if expect and shapes[i] != expect:
            raise ValueError(f"{tag} id {i}: frames/H/W {shapes[i]} != expected {expect} ({p})")
        if shapes[i][0] < 2:
            raise ValueError(f"{tag} id {i}: {shapes[i][0]} frames, subject_consistency needs >= 2")
    link_dir = os.path.join(scratch, "videos", tag)
    os.makedirs(link_dir)
    link_to_id = {}
    for i, p in paths.items():
        link = os.path.join(link_dir, names[i])
        os.symlink(os.path.abspath(p), link)
        link_to_id[link] = i
    out_dir = os.path.join(scratch, "vbench_out", tag)
    full_info = os.path.join(os.path.dirname(__import__("vbench").__file__), "VBench_full_info.json")
    vb = VBench(device, full_info, out_dir)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    vb.evaluate(videos_path=link_dir, name=tag, dimension_list=DIMS, local=True, read_frame=False,
                mode="custom_input", imaging_quality_preprocessing_mode=IQ_MODE)
    res = json.load(open(os.path.join(out_dir, f"{tag}_eval_results.json")))
    per_video = {i: {} for i in ids}
    vbench_agg = {}
    for dim in DIMS:
        agg, vids = res[dim]
        vbench_agg[dim] = float(agg)
        got = {}
        for v in vids:
            i = link_to_id[v["video_path"]]
            s = float(v["video_results"])
            got[i] = s / 100.0 if dim == "imaging_quality" else s
        if sorted(got) != sorted(ids):
            raise RuntimeError(f"{tag}/{dim}: VBench scored ids {sorted(got)} != requested {sorted(ids)}")
        for i, s in got.items():
            if not math.isfinite(s):
                raise ValueError(f"{tag}/{dim} id {i}: non-finite score {s}")
            per_video[i][dim] = s
    return {"per_video": {str(i): per_video[i] for i in ids},
            "shape_frames_h_w": {str(i): list(shapes[i]) for i in ids},
            "vbench_aggregate": vbench_agg}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True, help="comma list of video dirs under --videos_root")
    ap.add_argument("--ids", required=True, help='one-object prompt ids, e.g. "0-79" or "@data/heldout_ids.txt"')
    ap.add_argument("--out", required=True, help="results/vbench/<name>.json (relative to ~/tempo)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--videos_root", default=os.path.join(TEMPO, "videos"))
    ap.add_argument("--expect", default="81x480x832", help='frames x H x W every video must have; "" disables')
    ap.add_argument("--no_resume", action="store_true")
    a = ap.parse_args()

    out = a.out if os.path.isabs(a.out) else os.path.join(TEMPO, a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    parts_dir = out + ".parts"
    os.makedirs(parts_dir, exist_ok=True)
    tags = [t for t in a.tags.split(",") if t]
    ids = parse_ids(a.ids)
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate ids")
    expect = tuple(int(x) for x in a.expect.split("x")) if a.expect else None
    prompts = benchmark.load_one_object()
    names = {i: benchmark.video_name(prompts[i]["prompt"]) for i in ids}
    if len(set(names.values())) != len(names):
        raise ValueError("two ids map to the same video file name")
    missing_w = [p for p in REQUIRED_WEIGHTS if not os.path.isfile(p)]
    if missing_w:
        raise FileNotFoundError(f"VBench weights missing (pre-download on the login node): {missing_w}")
    if a.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device cuda but no CUDA device")
    ver = versions()
    meta = {"dims": DIMS, "imaging_quality_preprocessing_mode": IQ_MODE, "mode": "custom_input", "versions": ver,
            "device": a.device, "gpu": torch.cuda.get_device_name(0) if a.device.startswith("cuda") else None,
            "vbench_cache_dir": VB_CACHE, "torch_home": os.environ["TORCH_HOME"], "videos_root": a.videos_root,
            "ids": ids, "tags": tags, "time": datetime.datetime.now().isoformat(timespec="seconds")}
    print("vbench_run", json.dumps(meta), flush=True)

    scratch_root = os.environ.get("TMPDIR") or tempfile.gettempdir()
    result = {"meta": meta, "tags": {}}
    for tag in tags:
        part = os.path.join(parts_dir, f"{tag}.json")
        vdir = os.path.join(a.videos_root, tag)
        stamps = {}
        for i in ids:
            p = os.path.join(vdir, names[i])
            st = os.stat(p) if os.path.isfile(p) else None
            stamps[str(i)] = [st.st_size, st.st_mtime_ns] if st else None
        key = {"schema": SCHEMA, "ids": ids, "dims": DIMS, "versions": ver, "iq_mode": IQ_MODE,
               "videos_root": a.videos_root, "device": a.device, "expect": a.expect, "files": stamps}
        if not a.no_resume and os.path.isfile(part):
            cached = json.load(open(part))
            if cached.get("key") == key:
                print(f"[{tag}] reusing {part}", flush=True)
                result["tags"][tag] = cached["result"]
                continue
        t0 = datetime.datetime.now()
        scratch = tempfile.mkdtemp(prefix=f"vbench_{tag}_", dir=scratch_root)
        try:
            r = score_tag(tag, ids, names, a.videos_root, a.device, expect, scratch)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        r["mean"] = {d: sum(v[d] for v in r["per_video"].values()) / len(ids) for d in DIMS}
        # VBench's own aggregate must agree with our per-video mean (all videos have the same frame count here)
        for d in DIMS:
            if expect and abs(r["mean"][d] - r["vbench_aggregate"][d]) > 1e-6:
                raise RuntimeError(f"{tag}/{d}: mean {r['mean'][d]} != VBench aggregate {r['vbench_aggregate'][d]}")
        r["seconds"] = (datetime.datetime.now() - t0).total_seconds()
        json.dump({"key": key, "result": r}, open(part + ".tmp", "w"), indent=1)
        os.replace(part + ".tmp", part)
        result["tags"][tag] = r
        print(f"[{tag}] n={len(ids)} {json.dumps(r['mean'])} in {r['seconds']:.0f}s", flush=True)

    json.dump(result, open(out, "w"), indent=1)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
