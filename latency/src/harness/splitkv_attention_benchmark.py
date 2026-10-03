#!/usr/bin/env python3
"""
Split-KV attention benchmark: flash_attn_with_kvcache vs. the varlen FA2 call (J7 a+b).

At the real shapes (Q = 4680, H = 12, d = 128, K in {4680, 18720, 32760}) it
measures, for num_splits in {1, 2, 4, 8, heuristic}: kernel time, speed-up
over the varlen call, the combine-kernel share, partial-buffer memory and
agreement with the varlen output (bitwise for num_splits = 1, 2x the bf16
band vs an fp32 reference otherwise). It then repeats the head-count
staircase at the best split and probes the query-length axis at fixed K.
MEASUREMENT ONLY -- no kernel is written here.

Writes the file named by $J7_OUT (default j7_attn.json) with keys
splits, correctness, staircase, q_probe.
"""
import json, math, os, statistics as st, sys
import torch
sys.path.insert(0, os.getcwd())
from wan.modules.attention import flash_attention
from flash_attn import flash_attn_with_kvcache

dev = torch.device("cuda"); torch.set_grad_enabled(False)
Q, H, HD, KMAX = 4680, 12, 128, 32760
POS_K = {0: 4680, 3: 18720, 6: 32760}
results = {"splits": [], "staircase": [], "q_probe": [], "correctness": []}


def timeit(fn, n=25, warm=5):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(n):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return st.median(ts)


def kernel_breakdown(fn):
    from torch.profiler import profile, ProfilerActivity
    from torch.autograd import DeviceType
    for _ in range(3): fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        fn(); torch.cuda.synchronize()
    ks = [(e.key, e.count, e.device_time_total / 1e3)
          for e in p.key_averages() if e.device_type == DeviceType.CUDA]
    return sorted(ks, key=lambda x: -x[2])


HMAX = 24   # cache must carry every head count the staircase sweeps
kc_full = torch.randn([1, KMAX, HMAX, HD], device=dev, dtype=torch.bfloat16) * 0.05
vc_full = torch.randn([1, KMAX, HMAX, HD], device=dev, dtype=torch.bfloat16) * 0.05
kc = kc_full[:, :, :H].contiguous(); vc = vc_full[:, :, :H].contiguous()
q = torch.randn([1, Q, H, HD], device=dev, dtype=torch.bfloat16) * 0.05

print("=== (a) flash_attn_with_kvcache, num_splits sweep ===", flush=True)
print(f"{'pos':>4}{'K':>7}{'splits':>9}{'ms':>9}{'vs varlen':>11}{'combine ms':>12}{'partial MB':>12}")
for pos, K in POS_K.items():
    cs = torch.full((1,), K, dtype=torch.int32, device=dev)
    base = timeit(lambda: flash_attention(q, kc[:, :K], vc[:, :K]))
    ref = flash_attention(q, kc[:, :K], vc[:, :K])
    for sp in (1, 2, 4, 8, 0):
        spl = None if sp == 0 else sp
        f = lambda: flash_attn_with_kvcache(q, kc, vc, cache_seqlens=cs,
                                            causal=False, num_splits=(spl or 0))
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        m0 = torch.cuda.memory_allocated()
        y = f(); torch.cuda.synchronize()
        partial_mb = (torch.cuda.max_memory_allocated() - m0) / 2**20
        ms = timeit(f)
        ks = kernel_breakdown(f)
        # isolate the COMBINE kernel only: flash_fwd_splitkv_kernel is the main
        # split kernel, not the reduction, and must not be counted here
        comb = sum(t for n, c, t in ks
                   if ("combine" in n.lower() or "splitkv_combine" in n.lower())
                   and "splitkv_kernel" not in n.lower())
        main = sum(t for n, c, t in ks if "flash" in n.lower()) - comb
        d = (y.float() - ref.float()).abs()
        exact = bool(torch.equal(y, ref))
        row = {"position": pos, "K": K, "num_splits": ("heuristic" if sp == 0 else sp),
               "ms": round(ms, 4), "varlen_ms": round(base, 4),
               "speedup_pct": round(100 * (base - ms) / base, 2),
               "combine_ms": round(comb, 4), "main_kernel_ms": round(main, 4), "partial_MB": round(partial_mb, 2),
               "bitwise_equal_to_varlen": exact,
               "max_abs_vs_varlen": d.max().item(), "mean_abs_vs_varlen": d.mean().item(),
               "kernels": [(n[:50], c, round(t, 4)) for n, c, t in ks[:4]]}
        results["splits"].append(row)
        print(f"{pos:>4}{K:>7}{str(row['num_splits']):>9}{ms:>9.4f}"
              f"{row['speedup_pct']:>10.2f}%{comb:>12.4f}{partial_mb:>12.2f}"
              f"{'  BITWISE=' + str(exact) if sp == 1 else ''}", flush=True)

# --- fp32 band for the non-exact splits -------------------------------------
print("\n=== (a) fp32-band agreement ===", flush=True)
for pos, K in ((3, 18720), (6, 32760)):
    cs = torch.full((1,), K, dtype=torch.int32, device=dev)
    q32 = q.float().transpose(1, 2); k32 = kc[:, :K].float().transpose(1, 2)
    v32 = vc[:, :K].float().transpose(1, 2)
    ref32 = torch.nn.functional.scaled_dot_product_attention(q32, k32, v32).transpose(1, 2)
    band = (torch.nn.functional.scaled_dot_product_attention(
        q32.bfloat16(), k32.bfloat16(), v32.bfloat16()).transpose(1, 2).float() - ref32).abs()
    for sp in (2, 4, 8, 0):
        y = flash_attn_with_kvcache(q, kc, vc, cache_seqlens=cs, causal=False,
                                    num_splits=sp).float()
        d = (y - ref32).abs()
        ok = (d.max().item() <= 2 * band.max().item() and
              d.mean().item() <= 2 * band.mean().item())
        results["correctness"].append({"position": pos, "num_splits": "heuristic" if sp == 0 else sp,
                                   "max_abs": d.max().item(), "mean_abs": d.mean().item(),
                                   "band_max": band.max().item(), "band_mean": band.mean().item(),
                                   "within_2x_band": ok})
        print(f"  pos {pos} splits={'heur' if sp==0 else sp}: max {d.max().item():.5g} "
              f"(band {band.max().item():.5g}) mean {d.mean().item():.5g} "
              f"(band {band.mean().item():.5g}) -> {'PASS' if ok else 'FAIL'}", flush=True)

# --- head staircase at the chosen split -------------------------------------
best = min([r for r in results["splits"] if r["position"] == 6], key=lambda r: r["ms"])
best_splits = best["num_splits"]
best_splits_arg = 0 if best_splits == "heuristic" else best_splits   # 0 = FA2 heuristic
print(f"\n=== (a) head staircase at num_splits={best_splits}, position 6 ===", flush=True)
print(f"{'H':>4}{'CTAs(Q)':>9}{'ms':>10}{'us/CTA':>10}")
for h in range(9, 25):
    qq = torch.randn([1, Q, h, HD], device=dev, dtype=torch.bfloat16) * 0.05
    kk = kc_full[:, :, :h].contiguous(); vv = vc_full[:, :, :h].contiguous()
    cs = torch.full((1,), KMAX, dtype=torch.int32, device=dev)
    ms = timeit(lambda: flash_attn_with_kvcache(qq, kk, vv, cache_seqlens=cs,
                                                causal=False, num_splits=best_splits_arg), n=15, warm=3)
    cta = math.ceil(Q / 128) * h
    results["staircase"].append({"heads": h, "ctas_q": cta, "ms": round(ms, 4),
                             "us_per_cta": round(1e3 * ms / cta, 4)})
    print(f"{h:>4}{cta:>9}{ms:>10.4f}{1e3*ms/cta:>10.4f}", flush=True)
    del qq, kk, vv; torch.cuda.empty_cache()

# --- (b) Q-length probe: CTAs along Q at fixed K -----------------------------
print("\n=== (b) Q-length probe (CTAs along Q, fixed K=32760) ===", flush=True)
print("  note: FA2 does not expose BLOCK_M through its API; the installed build")
print("  fixes the query tile at 128 (grid 37 for Q=4680). Shipping the Triton")
print("  tutorial attention kernel would be kernel writing, which J7 excludes,")
print("  so the Q axis is probed by varying Q instead.")
print(f"{'Q':>7}{'tiles':>7}{'CTAs':>7}{'waves':>8}{'ms':>10}{'ms/1k-q-tok':>14}")
for qq_len in (1560, 3120, 4680, 6240):
    qq = torch.randn([1, qq_len, H, HD], device=dev, dtype=torch.bfloat16) * 0.05
    cs = torch.full((1,), KMAX, dtype=torch.int32, device=dev)
    ms = timeit(lambda: flash_attn_with_kvcache(qq, kc, vc, cache_seqlens=cs,
                                                causal=False, num_splits=best_splits_arg), n=15, warm=3)
    tiles = math.ceil(qq_len / 128); cta = tiles * H
    results["q_probe"].append({"Q": qq_len, "tiles": tiles, "ctas": cta,
                           "waves": round(cta / 108, 3), "ms": round(ms, 4)})
    print(f"{qq_len:>7}{tiles:>7}{cta:>7}{cta/108:>8.2f}{ms:>10.4f}{1000*ms/qq_len:>14.4f}",
          flush=True)
    del qq; torch.cuda.empty_cache()

json.dump(results, open(os.environ.get("J7_OUT", "j7_attn.json"), "w"), indent=2)
print("\n[j7] written", flush=True)
