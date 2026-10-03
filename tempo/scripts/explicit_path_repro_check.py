#!/usr/bin/env python3
"""Step-4 preflight (reviewer finding): does the current code reproduce part-1 videos bitwise?

The explicit-path code (cross_attention_arms.py, benchmark.py) was edited on 2026-09-26 (fused/fa2kv paths added) after part 1. Before
step 4 adds 52 videos to part 1's B0_s42 / L_b2g2_s42 directories, regenerate prompt 2 for both tags with the
current code (same protocol as generate_benchmark_videos.py) into videos/step4_repro/<tag>/ and compare with the part-1 file. Gate:
decoded frames bitwise equal (exit 1 otherwise); mp4 byte equality is recorded, not gated. Part 1 saved no tensors,
so this is the strictest comparison available.
"""
import filecmp
import json
import os
import sys

TEMPO = os.environ.get("TEMPO_ROOT", os.path.expanduser("~/tempo"))
sys.path.insert(0, os.path.join(TEMPO, "ext", "Wan2.1"))
sys.path.insert(0, os.path.join(TEMPO, "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from tempo_ctrl import benchmark  # noqa: E402
from tempo_ctrl.sampler import Sampler  # noqa: E402
from tempo_ctrl.tokens import load_tokenizer  # noqa: E402


def read_frames(p):
    cap = cv2.VideoCapture(p)
    v = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        v.append(f)
    return np.stack(v)


rows = benchmark.load_one_object()
tok = load_tokenizer(benchmark.WAN_ROOT, benchmark.CKPT)
cache = torch.load(benchmark.TEXT_CACHE)
S = Sampler(benchmark.CKPT)
dev = S.device
neg = [cache["__neg__"].to(dev)]
row = rows[2]
rep, ok = {"job": os.environ.get("SLURM_JOB_ID", "local"), "prompt_id": 2}, True
benchmark.configure_controller({"arm": "B0", "seed": 0}, row, tok)                   # same untimed warmup as generate_benchmark_videos.py
S.generate([cache[row["prompt"]].to(dev)], neg, seed=0, steps=2, decode=False)
for run in ({"arm": "B0", "seed": 42}, {"arm": "L", "beta": 2.0, "gamma": 2.0, "seed": 42}):
    tag = benchmark.run_tag(run)
    benchmark.configure_controller(run, row, tok)
    video, _ = S.generate([cache[row["prompt"]].to(dev)], neg, seed=42)
    name = benchmark.video_name(row["prompt"])
    new = os.path.join(TEMPO, "videos", "step4_repro", tag, name)
    benchmark.save_video(video.cpu(), new)
    old = os.path.join(TEMPO, "videos", tag, name)
    fo, fn = read_frames(old), read_frames(new)
    r = {"mp4_bytes_equal": filecmp.cmp(old, new, shallow=False),
         "frames_equal": bool(fo.shape == fn.shape and np.array_equal(fo, fn)),
         "frames_maxabs": int(np.abs(fo.astype(int) - fn.astype(int)).max()) if fo.shape == fn.shape else None}
    rep[tag] = r
    ok &= r["frames_equal"]
    print(f"[repro] {tag} p2: {r}", flush=True)
json.dump(rep, open(os.path.join(TEMPO, "results", "step4_repro.json"), "w"), indent=1)
sys.exit(0 if ok else 1)
