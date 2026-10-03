#!/usr/bin/env python3
"""
Per-position chunk-latency harness for eager Self-Forcing (J2).

Implements the frozen per-position timing protocol: t_gen is a synchronized
host timestamp before the first generator pass of a video, t_chunk[i] is a
synchronized host timestamp after the 5th (clean-context) pass of chunk i,
chunk time T[i] is the difference of consecutive boundaries, and per-pass
times come from CUDA events read only after generation ends. Exactly one
device synchronization per chunk boundary. The VAE decode is timed as its own
stage and never charged to a chunk.

Modes:
  --mode sweep     6 videos in one process (video 0 = warmup, 1-5 measured);
                   writes <out>/j2_sweep.json with per-position median, min,
                   max, IQR and spread, and appends one row to --jsonl.
  --mode envelope  1 video, fixed prompt and seed, dumps the uint8 video to
                   <out>/envelope_<tag>.npy so two runs can be compared by PSNR.
"""
import argparse
import contextlib
import io
import json
import os
import socket
import statistics as st
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch

GB = 1024 ** 3
CEILING_GB = 36.0          # pre-registered memory ceiling
SPREAD_PCT_MAX = 3.0       # pre-registered chunk-6 spread criterion


def fail(msg, code=2):
    print(f"\n!!! J2 FAILURE: {msg}\n", flush=True)
    sys.exit(code)


def spread_pct(xs):
    """Frozen spread estimator: 100 * (max - min) / median."""
    m = st.median(xs)
    return 100.0 * (max(xs) - min(xs)) / m if m > 0 else float("nan")


def summarize(xs):
    xs = list(xs)
    q = np.percentile(xs, [25, 75]) if len(xs) > 1 else (xs[0], xs[0])
    return {"n": len(xs), "median": st.median(xs), "min": min(xs), "max": max(xs),
            "iqr": float(q[1] - q[0]), "spread_pct": spread_pct(xs),
            "values": [round(v, 3) for v in xs]}


class Timer:
    """Per-video timing state implementing the frozen per-position protocol."""

    def __init__(self, first_ts):
        self.first_ts = first_ts
        self.t_gen = None
        self.t_chunk = []          # synchronized host timestamps, one per chunk
        self.pass_events = []      # (chunk, idx, start_event, end_event, timestep)
        self.chunk = -1
        self.in_chunk = 0

    def before_pass(self, tsv):
        if tsv == self.first_ts:
            self.chunk += 1
            self.in_chunk = 0
        if self.t_gen is None:                      # first pass of the video
            torch.cuda.synchronize()
            self.t_gen = time.perf_counter()
        s, e = (torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True))
        self.pass_events.append([self.chunk, self.in_chunk, s, e, tsv])
        self.in_chunk += 1
        return s, e

    def after_pass(self, tsv):
        # the 5th pass of a chunk is the clean-context pass at the context timestep
        if self.in_chunk == 5:
            torch.cuda.synchronize()               # exactly one sync per chunk
            self.t_chunk.append(time.perf_counter())

    def reduce(self):
        torch.cuda.synchronize()
        passes = {}
        for c, i, s, e, tsv in self.pass_events:
            passes.setdefault(c, []).append(
                {"idx": i, "timestep": tsv, "ms": s.elapsed_time(e)})
        chunk_ms, resid = [], []
        prev = self.t_gen
        for i, t in enumerate(self.t_chunk):
            T = (t - prev) * 1e3
            chunk_ms.append(T)
            psum = sum(p["ms"] for p in passes.get(i, []))
            resid.append(100.0 * (T - psum) / T if T > 0 else float("nan"))
            prev = t
        return passes, chunk_ms, resid


def build_pipeline(args, device):
    from omegaconf import OmegaConf
    from pipeline import CausalInferencePipeline
    from demo_utils.memory import gpu, DynamicSwapInstaller

    cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"),
                          OmegaConf.load(args.config_path))
    pipe = CausalInferencePipeline(cfg, device=device)
    sd = torch.load(args.checkpoint_path, map_location="cpu")
    pipe.generator.load_state_dict(sd["generator_ema"])
    del sd
    pipe = pipe.to(dtype=torch.bfloat16)
    # text-encoder offload ON is the protocol default from J2 onward
    DynamicSwapInstaller.install_model(pipe.text_encoder, device=gpu)
    pipe.generator.to(device=gpu)
    pipe.vae.to(device=gpu)
    return pipe, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_path", default="configs/self_forcing_dmd.yaml")
    ap.add_argument("--checkpoint_path", default="checkpoints/self_forcing_dmd.pt")
    ap.add_argument("--data_path", default="prompts/MovieGenVideoBench_extended.txt")
    ap.add_argument("--mode", choices=["sweep", "envelope"], required=True)
    ap.add_argument("--n_measured", type=int, default=5)
    ap.add_argument("--envelope_tag", default="a")
    ap.add_argument("--out", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--num_output_frames", type=int, default=21)
    args = ap.parse_args()

    from utils.misc import set_seed
    device = torch.device("cuda")
    torch.set_grad_enabled(False)
    os.makedirs(args.out, exist_ok=True)

    with open(args.data_path, encoding="utf-8") as f:
        prompts = [f.readline().rstrip() for _ in range(args.n_measured)]
    if any(not p for p in prompts):
        fail("fewer prompts available than requested")

    pipe, cfg = build_pipeline(args, device)
    first_ts = float(pipe.denoising_step_list.tolist()[0])
    import triton, flash_attn
    print(f"[J2] torch={torch.__version__} triton={triton.__version__} "
          f"flash_attn={flash_attn.__version__}", flush=True)

    # ---- instrumentation ------------------------------------------------
    timer = {"t": None}
    gen = pipe.generator
    original_generator_forward = gen.forward

    def timed_generator_forward(*a, **kw):
        ts = kw.get("timestep")
        tsv = float(ts.flatten()[0].item()) if ts is not None else float("nan")
        T = timer["t"]
        s, e = T.before_pass(tsv)
        s.record()
        out = original_generator_forward(*a, **kw)
        e.record()
        T.after_pass(tsv)
        return out

    gen.forward = timed_generator_forward

    vae = pipe.vae
    original_decode_to_pixel = vae.decode_to_pixel
    vae_stats = {}

    def timed_decode_to_pixel(*a, **kw):
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter()
        s.record()
        out = original_decode_to_pixel(*a, **kw)
        e.record()
        torch.cuda.synchronize()
        vae_stats["wall_ms"] = (time.perf_counter() - t0) * 1e3
        vae_stats["event_ms"] = s.elapsed_time(e)
        return out

    vae.decode_to_pixel = timed_decode_to_pixel

    def one_video(prompt, seed):
        set_seed(seed)
        timer["t"] = Timer(first_ts)
        torch.cuda.reset_peak_memory_stats()
        noise = torch.randn([1, args.num_output_frames, 16, 60, 104],
                            device=device, dtype=torch.bfloat16)
        buf = io.StringIO()
        t0 = time.perf_counter()
        # suppress the repo's per-pass print without editing the repo
        with contextlib.redirect_stdout(buf):
            video, latents = pipe.inference(noise=noise, text_prompts=[prompt],
                                            return_latents=True, initial_latent=None,
                                            low_memory=True)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        passes, chunk_ms, resid = timer["t"].reduce()
        return {
            "video": video, "wall_s": wall, "passes": passes,
            "chunk_ms": chunk_ms, "residual_pct": resid,
            "vae_wall_ms": vae_stats.get("wall_ms"),
            "vae_event_ms": vae_stats.get("event_ms"),
            "peak_alloc_gb": torch.cuda.max_memory_allocated() / GB,
            "peak_reserved_gb": torch.cuda.max_memory_reserved() / GB,
        }

    results = {"mode": args.mode, "videos": []}

    if args.mode == "envelope":
        # envelope protocol: prompt 0, seed 1000, one video per process
        r = one_video(prompts[0], 1000)
        arr = (255.0 * r["video"].float().clamp(0, 1)).round().clamp(0, 255)
        arr = arr[0].permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()
        np.save(os.path.join(args.out, f"envelope_{args.envelope_tag}.npy"), arr)
        print(f"[J2] envelope run {args.envelope_tag}: saved {arr.shape} "
              f"peak_reserved={r['peak_reserved_gb']:.3f} GB", flush=True)
        return

    # ---- sweep: 1 warmup + n_measured videos, one process ----------------
    print("[J2] video 0 = WARMUP (excluded, reported separately)", flush=True)
    w = one_video(prompts[0], 999)
    results["warmup"] = {"chunk_ms": [round(x, 3) for x in w["chunk_ms"]],
                         "wall_s": round(w["wall_s"], 3),
                         "vae_wall_ms": round(w["vae_wall_ms"], 2),
                         "peak_reserved_gb": round(w["peak_reserved_gb"], 3)}
    print(f"[J2]   warmup chunk_ms = {[round(x,1) for x in w['chunk_ms']]}", flush=True)
    del w

    for k in range(args.n_measured):
        seed = 1000 + k
        r = one_video(prompts[k], seed)
        results["videos"].append({
            "index": k, "seed": seed, "prompt_head": prompts[k][:70],
            "chunk_ms": [round(x, 3) for x in r["chunk_ms"]],
            "residual_pct": [round(x, 3) for x in r["residual_pct"]],
            "pass_ms": {str(c): [round(p["ms"], 3) for p in sorted(v, key=lambda d: d["idx"])]
                        for c, v in r["passes"].items()},
            "wall_s": round(r["wall_s"], 3),
            "vae_wall_ms": round(r["vae_wall_ms"], 2),
            "vae_event_ms": round(r["vae_event_ms"], 2),
            "peak_alloc_gb": round(r["peak_alloc_gb"], 3),
            "peak_reserved_gb": round(r["peak_reserved_gb"], 3),
        })
        print(f"[J2] video {k} seed {seed}: "
              f"chunks={[round(x,1) for x in r['chunk_ms']]} "
              f"vae={r['vae_wall_ms']:.1f}ms peak={r['peak_reserved_gb']:.2f}GB", flush=True)
        del r

    # ---- per-position statistics across the measured videos ---------------
    n_pos = len(results["videos"][0]["chunk_ms"])
    per_pos = {}
    for i in range(n_pos):
        per_pos[i] = summarize([v["chunk_ms"][i] for v in results["videos"]])
    results["per_position"] = per_pos

    resid_pos = {i: summarize([v["residual_pct"][i] for v in results["videos"]])
                 for i in range(n_pos)}
    results["residual_per_position"] = resid_pos

    vae_all = [v["vae_wall_ms"] for v in results["videos"]]
    results["vae"] = summarize(vae_all)
    peak_all = [v["peak_reserved_gb"] for v in results["videos"]]
    results["peak_reserved_gb"] = summarize(peak_all)

    print("\n[J2] per-position chunk time (ms), 5 measured videos")
    print(f"  {'pos':>3} {'median':>9} {'min':>9} {'max':>9} {'spread%':>8} {'resid%':>8}")
    for i in range(n_pos):
        p, rr = per_pos[i], resid_pos[i]
        print(f"  {i:>3} {p['median']:>9.2f} {p['min']:>9.2f} {p['max']:>9.2f} "
              f"{p['spread_pct']:>8.2f} {rr['median']:>8.2f}")

    c6 = per_pos[n_pos - 1]
    results["criterion"] = {
        "statistic": "spread_pct of chunk-6 time across 5 measured videos",
        "value": c6["spread_pct"], "threshold": SPREAD_PCT_MAX,
        "pass": c6["spread_pct"] < SPREAD_PCT_MAX,
    }
    maxpeak = max(peak_all)
    results["memory_gate"] = {"peak_reserved_gb": maxpeak, "ceiling_gb": CEILING_GB,
                              "pass": maxpeak <= CEILING_GB}

    row = {
        "run_id": f"j2-sweep-{os.environ.get('SLURM_JOB_ID')}",
        "job_id": os.environ.get("SLURM_JOB_ID"), "node": socket.gethostname(),
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__, "triton_version": triton.__version__,
        "flash_attn_version": flash_attn.__version__,
        "repo_revision": os.environ.get("J1_REPO_REV"),
        "checkpoint_sha256": os.environ.get("J1_CKPT_SHA"),
        "resolution": f"{int(cfg.height)}x{int(cfg.width)}",
        "num_frame_per_block": pipe.num_frame_per_block,
        "history_window_tokens": 32760, "passes_per_chunk": 5,
        "precision": "bfloat16", "attention_backend": "flash_attn_varlen (FA2)",
        "graph_mode": "eager", "fusion_mode": "none",
        "warmup_excluded": True, "profiler_attached": False,
        "n_videos_measured": args.n_measured,
        "t_per_chunk_ms_median_pos3": per_pos[3]["median"],
        "t_per_chunk_ms_median_pos6": per_pos[n_pos - 1]["median"],
        "chunk6_spread_pct": c6["spread_pct"],
        "vae_decode_ms_median": results["vae"]["median"],
        "max_memory_reserved_bytes": int(maxpeak * GB),
        "memory_points": {"P5_vae_peak_gb": maxpeak},
        "notes": "J2 frozen per-position timing definitions",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(args.jsonl, "a") as f:
        f.write(json.dumps(row) + "\n")
    with open(os.path.join(args.out, "j2_sweep.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[J2] GATE spread@chunk6 = {c6['spread_pct']:.3f} % "
          f"(threshold < {SPREAD_PCT_MAX} %)")
    print(f"[J2] GATE peak reserved  = {maxpeak:.3f} GB (ceiling {CEILING_GB} GB)")
    if not results["memory_gate"]["pass"]:
        fail(f"peak reserved {maxpeak:.3f} GB exceeds {CEILING_GB} GB ceiling")
    if not results["criterion"]["pass"]:
        fail(f"chunk-6 spread {c6['spread_pct']:.3f} % exceeds {SPREAD_PCT_MAX} %; "
            f"threshold NOT loosened -- raise repetitions instead")
    print("[J2] PASS", flush=True)


if __name__ == "__main__":
    main()
