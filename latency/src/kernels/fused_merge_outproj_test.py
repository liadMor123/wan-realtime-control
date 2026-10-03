#!/usr/bin/env python3
"""
Correctness test and kernel-level microbenchmark for fused_merge_outproj (J13 Step 1 / Step 2).

Correctness, at the real shapes (Q = 4680, H = 12, d = 128, N = 1536) for every
requested (num_splits, chunk position): the fused kernel's output is compared
with an fp32 merge of the same FA2 partials followed by an fp32 GEMM. It
passes if its error is within 2x the band of the PyTorch bf16 path
(FA2 combine -> bf16 O -> F.linear) on both max_abs and mean_abs, and is
finite with the expected shape.

With --bench, times (median/min/max of 50 iterations after warmup) the unfused
total, the fused total, and each component alone at positions 0, 3, 6, and
reports the per-layer and per-pass (x30 layers) net recovery.

Requires the patched flash_attn build that exposes return_partials
(PYTHONPATH pointing at the fa2_patched_site install).

Writes --result (JSON with info, correctness rows, autotune choice, bench rows)
and optionally appends one row to --runs-jsonl. Exit code 0 on pass, 3 on fail.
"""
import argparse
import datetime
import json
import os
import platform
import statistics as st
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fused_merge_outproj import (fused_merge_outproj, merge_partials_fp32,  # noqa: E402
                                 _fused_merge_outproj_kernel)
import flash_attn  # noqa: E402
from flash_attn import flash_attn_with_kvcache  # noqa: E402

B, Q, H, D, N, KMAX = 1, 4680, 12, 128, 1536, 32760
dev = "cuda"


def timed(fn, iters=50, warm=10):
    """Median, min and max CUDA-event time (ms) of fn over iters after warm calls."""
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record(); fn(); e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    return st.median(ts), min(ts), max(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True)
    ap.add_argument("--runs-jsonl", default=None)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--splits", default="2,4")
    ap.add_argument("--positions", default="0,1,2,3,4,5,6")
    args = ap.parse_args()
    assert "fa2_patched_site" in flash_attn.__file__, flash_attn.__file__
    torch.manual_seed(0)
    q = torch.randn(B, Q, H, D, device=dev, dtype=torch.bfloat16)
    k = torch.randn(B, KMAX, H, D, device=dev, dtype=torch.bfloat16)
    v = torch.randn(B, KMAX, H, D, device=dev, dtype=torch.bfloat16)
    W = (torch.randn(N, H * D, device=dev) * 0.02).to(torch.bfloat16)
    bias = (torch.randn(N, device=dev) * 0.02).to(torch.bfloat16)
    results = {"info": {"torch": torch.__version__, "triton": __import__("triton").__version__,
                        "flash_attn": flash_attn.__version__, "flash_attn_file": flash_attn.__file__,
                        "gpu": torch.cuda.get_device_name(0), "node": platform.node(),
                        "job_id": os.environ.get("SLURM_JOB_ID"), "spec": "§19 v12"},
               "correctness": [], "bench": []}
    splits = [int(x) for x in args.splits.split(",")]
    positions = [int(x) for x in args.positions.split(",")]

    all_ok = True
    for s in splits:
        for pos in positions:
            klen = (pos + 1) * 4680
            cache_seqlens = torch.full((B,), klen, dtype=torch.int32, device=dev)
            # PyTorch bf16 path: FA2 combine -> bf16 O -> F.linear
            o_bf16 = flash_attn_with_kvcache(q, k, v, cache_seqlens=cache_seqlens, num_splits=s)
            y_ref_bf16 = F.linear(o_bf16.flatten(2), W, bias)
            # fused path: FA2 partials -> fused merge + out-projection
            _, _, lse_acc, o_acc = flash_attn_with_kvcache(q, k, v, cache_seqlens=cache_seqlens,
                                                           num_splits=s, return_partials=True)
            y_fused = fused_merge_outproj(o_acc, lse_acc, W, bias)
            torch.cuda.synchronize()
            # fp32 reference: fp32 merge of the same partials, fp32 GEMM
            merged32 = merge_partials_fp32(o_acc, lse_acc)
            y_ref32 = merged32 @ W.float().t() + bias.float()
            band = (y_ref_bf16.float() - y_ref32).abs()
            cand = (y_fused.float() - y_ref32).abs()
            direct = (y_fused.float() - y_ref_bf16.float()).abs()
            row = {"splits": s, "position": pos, "K": klen,
                   "shape_ok": tuple(y_fused.shape) == (1, Q, N),
                   "band_max_abs": float(band.max()), "band_mean_abs": float(band.mean()),
                   "fused_max_abs": float(cand.max()), "fused_mean_abs": float(cand.mean()),
                   "ratio_max": float(cand.max() / band.max()),
                   "ratio_mean": float(cand.mean() / band.mean()),
                   "fused_vs_pytorch_bf16_max_abs": float(direct.max()),
                   "fused_vs_pytorch_bf16_mean_abs": float(direct.mean()),
                   "merged_O_max_abs_vs_fa2_combine": float(
                       (merged32.to(torch.bfloat16).float() - o_bf16.flatten(2).float()).abs().max()),
                   "finite": bool(torch.isfinite(y_fused).all())}
            row["pass_2x_band"] = (row["finite"] and row["shape_ok"]
                                   and row["ratio_max"] <= 2.0 and row["ratio_mean"] <= 2.0)
            all_ok &= row["pass_2x_band"]
            results["correctness"].append(row)
            print(json.dumps(row), flush=True)
    results["step1_pass_all"] = all_ok
    try:
        results["autotune_best"] = {str(kk): str(vv) for kk, vv in _fused_merge_outproj_kernel.cache.items()}
    except Exception as e:  # the autotune cache attribute is a Triton implementation detail
        results["autotune_best"] = f"n/a ({e})"
    print("STEP1", "PASS" if all_ok else "FAIL", "| autotune:", results["autotune_best"], flush=True)

    if args.bench:
        for s in splits:
            for pos in (0, 3, 6):
                klen = (pos + 1) * 4680
                cache_seqlens = torch.full((B,), klen, dtype=torch.int32, device=dev)

                def unfused_total():
                    return F.linear(flash_attn_with_kvcache(q, k, v, cache_seqlens=cache_seqlens,
                                                            num_splits=s).flatten(2), W, bias)

                def fused_total():
                    _, _, l_, o_ = flash_attn_with_kvcache(q, k, v, cache_seqlens=cache_seqlens,
                                                           num_splits=s, return_partials=True)
                    return fused_merge_outproj(o_, l_, W, bias)

                def fa2_with_combine():
                    return flash_attn_with_kvcache(q, k, v, cache_seqlens=cache_seqlens, num_splits=s)

                def fa2_partials_only():
                    return flash_attn_with_kvcache(q, k, v, cache_seqlens=cache_seqlens,
                                                   num_splits=s, return_partials=True)

                def fa2_s1_varlen_equiv():
                    return flash_attn_with_kvcache(q, k, v, cache_seqlens=cache_seqlens, num_splits=1)

                o_bf16 = fa2_with_combine()
                _, _, lse_acc, o_acc = fa2_partials_only()

                def linear_alone():
                    return F.linear(o_bf16.flatten(2), W, bias)

                def fused_kernel_alone():
                    return fused_merge_outproj(o_acc, lse_acc, W, bias)

                b = {"splits": s, "position": pos, "K": klen}
                for name, fn in [("unfused_total", unfused_total), ("fused_total", fused_total),
                                 ("fa2_with_combine", fa2_with_combine),
                                 ("fa2_partials_only", fa2_partials_only),
                                 ("linear_alone", linear_alone),
                                 ("fused_kernel_alone", fused_kernel_alone),
                                 ("fa2_s1_varlen_equiv", fa2_s1_varlen_equiv)]:
                    med, mn, mx = timed(fn)
                    b[name + "_ms"] = med; b[name + "_min_ms"] = mn; b[name + "_max_ms"] = mx
                b["combine_kernel_ms"] = b["fa2_with_combine_ms"] - b["fa2_partials_only_ms"]
                b["net_recovery_ms_per_layer"] = b["unfused_total_ms"] - b["fused_total_ms"]
                b["net_recovery_ms_per_pass_x30"] = 30 * b["net_recovery_ms_per_layer"]
                results["bench"].append(b)
                print(json.dumps({kk: (round(vv, 4) if isinstance(vv, float) else vv)
                                  for kk, vv in b.items()}), flush=True)

    json.dump(results, open(args.result, "w"), indent=1)
    if args.runs_jsonl:
        info = results["info"]
        row = {"run_id": f"j13-step1-{info['job_id']}", "job_id": info["job_id"], "node": info["node"],
               "gpu_name": info["gpu"], "gpu_uuid": None, "driver": None,
               "cuda_runtime": torch.version.cuda, "torch_version": torch.__version__,
               "triton_version": info["triton"],
               "repo_revision": "fused_merge_outproj.py + flash-attention v2.8.3 return_partials patch",
               "checkpoint_sha256": None, "resolution": None, "num_frame_per_block": None,
               "denoise_steps": None, "context_timestep": None, "history_window_tokens": KMAX,
               "passes_per_chunk": None, "precision": "bf16 (fp32 partials)",
               "attention_backend": "flash_attn_with_kvcache return_partials + fused_merge_outproj",
               "graph_mode": None, "fusion_mode": "triton_fused", "warmup_excluded": True,
               "profiler_attached": False, "n_chunks": None, "t_per_pass_ms_median": None,
               "t_per_pass_ms_p95": None, "t_per_chunk_ms_median": None, "t_per_chunk_ms_p95": None,
               "t_first_chunk_ms": None,
               "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
               "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
               "memory_points": None, "output_paths": [args.result],
               "notes": f"J13 Step 1 correctness ({'PASS' if all_ok else 'FAIL'}) + kernel-level "
                        f"microbenchmark (spec §19); synthetic inputs at real shapes",
               "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        open(args.runs_jsonl, "a").write(json.dumps(row) + "\n")
    sys.exit(0 if all_ok else 3)


if __name__ == "__main__":
    main()
