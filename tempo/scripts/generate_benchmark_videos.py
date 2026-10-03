#!/usr/bin/env python3
"""Generate benchmark videos for a list of run configs x prompt ids (Phases 1-2, part 2 steps 4 and 5a).

  generate_benchmark_videos.py --runs runs/phase1.json --ids 0-7 [--shard 0/4] [--phase 1]

A run config is a dict: {"arm": "L", "beta": 2, "gamma": 2, "seed": 42,
"k_steps": null, "branches": "cond", "bench": "one_object"}. With "bench": "two_objects" (all runs of a file must
agree) the ids are two-object pair ids, L edits both objects (K = 2), videos are named by original_prompt, and the
prompts' T5 encodings come from cache/text_enc_two.pt, written here if needed. Work items are (prompt, run) pairs in
prompt-major order; prompts are dealt round-robin to shards. With --prompt-set showcase the ids are showcase ids of
data/showcase_prompts.csv (benchmark.load_showcase; every run must carry "set": "showcase", so tags get the showcase_
prefix and no benchmark dir is touched) and all prompts, one- or two-object, are T5-encoded by load_or_build_text_cache into
cache/text_enc_showcase_<array task>.pt. Resumable: an item whose video
already exists in ~/tempo/videos/<run_tag>/ is skipped (a100-public requeues on
preemption). One JSONL row per video in results/rows/.
"""
import argparse
import json
import os
import shutil
import sys
import time

TEMPO = os.environ.get("TEMPO_ROOT", os.path.expanduser("~/tempo"))
sys.path.insert(0, os.path.join(TEMPO, "ext", "Wan2.1"))
sys.path.insert(0, os.path.join(TEMPO, "src"))

import torch  # noqa: E402

from tempo_ctrl import benchmark  # noqa: E402
from tempo_ctrl.sampler import Sampler  # noqa: E402
from tempo_ctrl.tokens import load_tokenizer  # noqa: E402


def parse_ids(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def load_or_build_text_cache(prompts, dev, path=None):
    """Encodings of the two-object prompts, cached in cache/text_enc_two.pt with the step-4 negative-prompt tensor
    (copied from cache/text_enc.pt). Missing prompts are encoded with the T5 call that wrote cache/text_enc.pt
    (validate_pipeline_equivalence.py section 1: WanT2V's T5EncoderModel, built on CPU, moved to the GPU, text_encoder([p], dev)[0].cpu()),
    after checking that the call reproduces text_enc.pt on one-object prompt 0 and on the negative prompt. Every shard
    encodes the missing prompts of its id list into its own file (cache/text_enc_two_<array task>.pt), so concurrent shards
    never write the same file; on a requeue the tensors already in the file are kept (B0 and L of a pair share them).
    `path` (prompt sets other than the benchmarks) replaces the per-shard file name; the call and checks are the same."""
    one = torch.load(benchmark.TEXT_CACHE)
    if path is None:
        # TEMPO_TEXT_CACHE_TAG (set by two_objects_remaining_62_generate.sbatch to 5b_<task>) keeps later steps out of step 5a's per-shard files
        shard_tag = os.environ.get("TEMPO_TEXT_CACHE_TAG") or os.environ.get("SLURM_ARRAY_TASK_ID", "local")
        path = benchmark.TEXT_CACHE_TWO.replace(".pt", f"_{shard_tag}.pt")
    cache = torch.load(path) if os.path.isfile(path) else {"__neg__": one["__neg__"]}
    if not torch.equal(cache["__neg__"], one["__neg__"]):
        raise RuntimeError(f"{path}: __neg__ differs from {benchmark.TEXT_CACHE}")
    missing = [p for p in dict.fromkeys(prompts) if p not in cache]
    if not missing:
        return cache
    from wan.configs import WAN_CONFIGS
    from wan.modules.t5 import T5EncoderModel
    cfg = WAN_CONFIGS["t2v-1.3B"]
    enc = T5EncoderModel(text_len=cfg.text_len, dtype=cfg.t5_dtype, device=torch.device("cpu"),
                         checkpoint_path=os.path.join(benchmark.CKPT, cfg.t5_checkpoint),
                         tokenizer_path=os.path.join(benchmark.CKPT, cfg.t5_tokenizer), shard_fn=None)
    enc.model.to(dev)
    p0 = benchmark.load_one_object()[0]["prompt"]
    with torch.no_grad():
        chk = {p0: enc([p0], dev)[0].cpu(), "__neg__": enc([cfg.sample_neg_prompt], dev)[0].cpu()}
        new = {p: enc([p], dev)[0].cpu() for p in missing}
    enc.model.cpu()
    del enc
    torch.cuda.empty_cache()
    diffs = {}
    for k, v in chk.items():
        if v.shape != one[k].shape or v.dtype != one[k].dtype:
            raise RuntimeError(f"T5 re-encode of {k[:40]!r}: {v.shape} {v.dtype} vs cached {one[k].shape} {one[k].dtype}")
        diffs[k] = (v.float() - one[k].float()).abs().max().item() / one[k].float().abs().max().item()
    exact = all(torch.equal(v, one[k]) for k, v in chk.items())
    print(f"[gen] T5 encoded {len(missing)} two-object prompts; re-encode reproduces text_enc.pt: "
          f"{'bitwise' if exact else 'NOT bitwise, max rel diff %.2e' % max(diffs.values())}", flush=True)
    if max(diffs.values()) > 1e-2:
        raise RuntimeError(f"T5 call does not reproduce cache/text_enc.pt (max rel diff {max(diffs.values()):.3e})")
    for p, v in new.items():
        if v.dim() != 2 or v.shape[1] != 4096 or not torch.isfinite(v.float()).all():
            raise RuntimeError(f"bad encoding {tuple(v.shape)} for {p!r}")
    cache = {**new, **(torch.load(path) if os.path.isfile(path) else cache)}
    cache["__meta__"] = {"t5_reencode_bitwise": exact, "t5_reencode_max_rel_diff": max(diffs.values()),
                         "job": os.environ.get("SLURM_JOB_ID", "local"), "host": os.uname().nodename}
    tmp = f"{path}.{os.uname().nodename}.{os.getpid()}.tmp"
    torch.save(cache, tmp)
    os.replace(tmp, path)
    return cache


def check_prompt_set(runs, prompt_set):
    """Runs of a prompt set must say so ("set": <prompt set>, which prefixes the run tag), and benchmark runs must not:
    set prompts never land in benchmark dirs and benchmark prompts never in set dirs."""
    bad = [r for r in runs if r.get("set") != prompt_set]
    if bad:
        raise ValueError(f"--prompt-set {prompt_set} needs \"set\": {prompt_set!r} in every run, got {bad}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--phase", type=int, required=True)
    ap.add_argument("--prompt-set", choices=("showcase",), default=None,
                    help="ids index data/showcase_prompts.csv instead of the benchmark")
    a = ap.parse_args()
    runs = json.load(open(a.runs))
    si, sn = map(int, a.shard.split("/"))
    benches = {r.get("bench", "one_object") for r in runs}
    if len(benches) != 1:
        raise ValueError(f"one benchmark per run file, got {benches}")
    bname = benches.pop()
    check_prompt_set(runs, a.prompt_set)
    rows = benchmark.load_showcase(bname) if a.prompt_set else benchmark.load_benchmark(bname)
    # shard by prompt: every shard runs all arms on its prompts, so per-arm timing is paired on one node
    items = [(rows[i], r) for i in parse_ids(a.ids)[si::sn] for r in runs]
    job = f"{os.environ.get('SLURM_JOB_ID', 'local')}_{si}of{sn}"
    scr = os.path.join(os.environ.get("TMPDIR", "/tmp"), "tempo", job)  # Slurm node-local
    rows_path = os.path.join(TEMPO, "results", "rows", f"phase{a.phase}_{job}.jsonl")
    todo = [(row, r) for row, r in items
            if not os.path.isfile(os.path.join(TEMPO, "videos", benchmark.run_tag(r), benchmark.row_video_name(row)))]
    print(f"[gen] {len(items)} items in shard {a.shard}, {len(todo)} to do", flush=True)
    if not todo:
        return

    if a.prompt_set:
        cache = load_or_build_text_cache([rows[i]["prompt"] for i in parse_ids(a.ids)], torch.device("cuda:0"),
                               path=os.path.join(TEMPO, "cache", f"text_enc_{a.prompt_set}_"
                                                 f"{os.environ.get('SLURM_ARRAY_TASK_ID', 'local')}.pt"))
    elif bname == "two_objects":
        cache = load_or_build_text_cache([rows[i]["prompt"] for i in parse_ids(a.ids)], torch.device("cuda:0"))
    else:
        cache = torch.load(benchmark.TEXT_CACHE)
    missing = [row["prompt_id"] for row, _ in todo if row["prompt"] not in cache]
    if missing:
        raise KeyError(f"no cached text encoding for prompt ids {sorted(set(missing))}")
    tok = load_tokenizer(benchmark.WAN_ROOT, benchmark.CKPT)
    S = Sampler(benchmark.CKPT)
    dev = S.device
    neg = [cache["__neg__"].to(dev)]

    # untimed warmup on the explicit path
    benchmark.configure_controller({"arm": "B0", "seed": 0, "bench": bname}, todo[0][0], tok)
    S.generate([cache[todo[0][0]["prompt"]].to(dev)], neg, seed=0, steps=2, decode=False)
    # run files with other attention paths (step2acc: flash, fa2kv) also warm each of those paths, the last one with
    # a VAE decode, so no kernel's first call and not the first decode land in a timed video. Explicit-only run
    # files (all earlier steps, 5b) skip this and are unchanged.
    other = list(dict.fromkeys(r.get("path", "explicit") for r in runs if r.get("path", "explicit") != "explicit"))
    for k, pth in enumerate(other):
        wr = next(r for r in runs if r.get("path", "explicit") == pth)
        benchmark.configure_controller({**wr, "seed": 0}, todo[0][0], tok)
        S.generate([cache[todo[0][0]["prompt"]].to(dev)], neg, seed=0, steps=2, decode=k == len(other) - 1)
        print(f"[gen] warmup on the {pth} path done", flush=True)

    for row, r in todo:
        tag = benchmark.run_tag(r)
        info = benchmark.configure_controller(r, row, tok)
        torch.cuda.reset_peak_memory_stats()
        tm = {}
        video, _ = S.generate([cache[row["prompt"]].to(dev)], neg, seed=int(r["seed"]), timing=tm)
        peak = torch.cuda.max_memory_allocated() / 2**30
        if not torch.isfinite(video).all():
            raise RuntimeError(f"non-finite video for {tag} prompt {row['prompt_id']}")
        name = benchmark.row_video_name(row)
        tmp = os.path.join(scr, tag, name)
        benchmark.save_video(video.cpu(), tmp)
        dst_dir = os.path.join(TEMPO, "videos", tag)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy2(tmp, os.path.join(dst_dir, name + ".part"))
        os.replace(os.path.join(dst_dir, name + ".part"), os.path.join(dst_dir, name))
        benchmark.append_jsonl(rows_path, {
            "phase": a.phase, "run_tag": tag, "arm": r["arm"],
            "params": {k: v for k, v in r.items() if k not in ("arm", "seed")},
            "prompt_id": row["prompt_id"], "temp_object": row["temp_object"], "seed": int(r["seed"]),
            "wall_s": tm["total_s"], "denoise_s": tm["denoise_s"], "peak_gb": peak,
            "video_path": os.path.join(dst_dir, name), "job": job, "host": os.uname().nodename,
            "obj_idx": info["obj_idx"], "tmp_idx": info["tmp_idx"], "n_real_tokens": info["n_real_tokens"],
            **({"static_object": row["static_object"], "static_idx": info["static_idx"]} if "static_idx" in info else {}),
            **({"prompt_set": a.prompt_set, "prompt": row["prompt"], "source_ids": row["source_ids"],
                "video_name": name} if a.prompt_set else {}),
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "temporal_accuracy": None})
        print(f"[gen] {tag} p{row['prompt_id']:02d} {row['temp_object']}: {tm['total_s']:.1f}s peak {peak:.1f}GB",
              flush=True)


if __name__ == "__main__":
    main()
