#!/usr/bin/env python3
"""
Kernel census of one generator pass (J5) and KV-cache contents checks (J4).

Runs the patched model under compile_nocg. For each requested chunk position
it advances the KV cache exactly as inference does, profiles one generator
pass with torch.profiler and assigns every CUDA kernel to a pipeline stage
(STAGE_RULES), giving device time and launch count per stage. The census is
reconciled against the J3b compile_nocg fit (pass time must agree within 3 %).
Before the census it verifies, at every position, that the KV cache's
local_end_index, the attention K length and the zeroed region beyond the
window are what the protocol expects, and hashes the written region.

Every profiler-attached number here is ATTRIBUTION DATA, never headline latency.
Writes <out>/j54.json.
"""
import argparse, contextlib, hashlib, io, json, os, re, time
from collections import defaultdict
import torch

TOKENS_PER_CHUNK = 4680
K_OF_POS = [(i + 1) * TOKENS_PER_CHUNK for i in range(7)]
# J3b compile_nocg fit of chunk time vs history length K: t = a + b*K
J3B_COMPILE_NOCG_FIT_A_MS = 500.86
J3B_COMPILE_NOCG_FIT_B_MS_PER_TOKEN = 25.1892 / 1000.0

# stage -> ordered list of regex matched against the aten/kernel name chain
STAGE_RULES = [
    ("self-attention",        r"flash_fwd|flash::|flash_attn"),
    ("FFN up",                r"__FFN_UP__"),
    ("FFN down",              r"__FFN_DOWN__"),
    ("QKV GEMM",              r"__QKV__"),
    ("out-proj",              r"__OPROJ__"),
    ("cross-attn projections", r"__XPROJ__"),
    ("GELU",                  r"gelu|__GELU__"),
    ("QK-norm",               r"__QKNORM__|rms_norm|layer_norm"),
    ("RoPE",                  r"__ROPE__|rope"),
    ("adaLN/modulation",      r"__ADALN__|modulat"),
    ("norm2",                 r"__NORM2__"),
    ("residual",              r"__RESID__|add_|aten::add"),
    ("timestep embedding",    r"__TSEMB__|sinusoid"),
    ("cache write",           r"__CACHEW__|copy_|slice_scatter|index_put"),
    ("cross-attn",            r"__XATTN__"),
]


def classify_kernel(name):
    for stage, rx in STAGE_RULES:
        if re.search(rx, name, re.I):
            return stage
    if re.search(r"gemm|cutlass|s16816|triton_.*mm|addmm|matmul", name, re.I):
        return "GEMM (unattributed shape)"
    if re.search(r"elementwise|vectorized_elementwise|mul|div|sub|pow|exp|silu|cat|clone|contiguous|triton_poi|triton_red", name, re.I):
        return "elementwise/norm (generic)"
    return "other"


def measure_hbm_bandwidth(dev):
    """Measured HBM bandwidth (stream-triad style), not the datasheet number."""
    n = 1 << 28
    a = torch.empty(n, device=dev, dtype=torch.float16)
    b = torch.empty(n, device=dev, dtype=torch.float16)
    a.fill_(1.0); b.fill_(2.0)
    for _ in range(3): a.copy_(b)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    it = 20
    for _ in range(it): a.copy_(b)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    bw = it * 2 * n * 2 / dt          # read + write, 2 bytes each
    del a, b; torch.cuda.empty_cache()
    return bw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--positions", default="3,6")
    ap.add_argument("--try_compile_cg", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    positions = [int(x) for x in args.positions.split(",")]

    from omegaconf import OmegaConf
    from pipeline import CausalInferencePipeline
    from pipeline.causal_inference import prime_crossattn_cache
    from demo_utils.memory import gpu, DynamicSwapInstaller
    from utils.misc import set_seed
    from wan import patch_flags
    from wan.modules.model import prebuild_sinusoid_cache
    import triton, flash_attn

    dev = torch.device("cuda"); torch.set_grad_enabled(False)
    patch_flags.set_enabled(True)
    cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"),
                          OmegaConf.load("configs/self_forcing_dmd.yaml"))
    pipe = CausalInferencePipeline(cfg, device=dev)
    sd = torch.load("checkpoints/self_forcing_dmd.pt", map_location="cpu")
    pipe.generator.load_state_dict(sd["generator_ema"]); del sd
    pipe = pipe.to(dtype=torch.bfloat16)
    DynamicSwapInstaller.install_model(pipe.text_encoder, device=gpu)
    pipe.generator.to(device=gpu); pipe.vae.to(device=gpu)
    prebuild_sinusoid_cache(pipe.generator.model.freq_dim if hasattr(pipe.generator.model, "freq_dim") else 256, dev)
    with open("prompts/MovieGenVideoBench_extended.txt", encoding="utf-8") as f:
        prompt = f.readline().rstrip()

    report = {"torch": torch.__version__, "triton": triton.__version__,
           "flash_attn": flash_attn.__version__, "positions": positions}
    report["hbm_bw_measured_GBs"] = round(measure_hbm_bandwidth(dev) / 1e9, 1)
    print(f"[J5] measured HBM bandwidth: {report['hbm_bw_measured_GBs']} GB/s", flush=True)

    pipe._initialize_kv_cache(1, torch.bfloat16, dev)
    pipe._initialize_crossattn_cache(1, torch.bfloat16, dev)
    set_seed(1000)
    noise = torch.randn([1, 21, 16, 60, 104], device=dev, dtype=torch.bfloat16)
    cond = pipe.text_encoder(text_prompts=[prompt])
    with contextlib.redirect_stdout(io.StringIO()):
        prime_crossattn_cache(pipe, cond, noise)

    import torch._dynamo as dynamo
    dynamo.config.cache_size_limit = 64
    base = pipe.generator.model
    pipe.generator.model = torch.compile(base, mode="max-autotune-no-cudagraphs",
                                         dynamic=False)

    # ---------------- J4: cache-contents instrumentation ----------------
    cache_checks = []
    attn_shapes = []
    import wan.modules.causal_model as wc
    _orig_attention = wc.attention

    def attention_probe(q, k, v, *a, **kw):
        attn_shapes.append({"q_len": int(q.shape[1]), "k_len": int(k.shape[1]),
                            "heads": int(q.shape[2]), "dim": int(q.shape[3])})
        return _orig_attention(q, k, v, *a, **kw)
    wc.attention = attention_probe

    def advance_to(pos):
        """Run the generator forward through chunk `pos` exactly as inference
        does, so the cache state at `pos` is the real one."""
        for b in pipe.kv_cache1:
            b["k"].zero_(); b["v"].zero_()
            b["global_end_index"] = 0; b["local_end_index"] = 0
        dsl = [float(x) for x in pipe.denoising_step_list.tolist()]
        cur = 0
        for p in range(pos + 1):
            x = noise[:, p * 3:(p + 1) * 3]
            for j, tstep in enumerate(dsl + [0.0]):
                ts = torch.ones([1, 3], device=dev, dtype=torch.int64) * int(tstep)
                out = pipe.generator(noisy_image_or_video=x, conditional_dict=cond,
                                     timestep=ts, kv_cache=pipe.kv_cache1,
                                     crossattn_cache=pipe.crossattn_cache,
                                     current_start=cur * 1560)
                if isinstance(out, (tuple, list)): x = out[1]
            cur += 3
        torch.cuda.synchronize()

    for pos in range(7):
        attn_shapes.clear()
        advance_to(pos)
        want = (pos + 1) * TOKENS_PER_CHUNK
        got = pipe.kv_cache1[0]["local_end_index"]
        got = got if isinstance(got, int) else int(got.item())
        blk0 = pipe.kv_cache1[0]
        written = blk0["k"][:, :want].float()
        tail = blk0["k"][:, want:].float()
        h = hashlib.sha256(written.cpu().numpy().tobytes()).hexdigest()[:16]
        sh = attn_shapes[-1] if attn_shapes else {}
        cache_checks.append({
            "position": pos, "local_end_index": got, "expected": want,
            "index_ok": got == want,
            "beyond_window_all_zero": bool(torch.count_nonzero(tail).item() == 0),
            "written_region_sha16": h,
            "attn_q_len": sh.get("q_len"), "attn_k_len": sh.get("k_len"),
            "attn_k_len_ok": sh.get("k_len") == want,
        })
        print(f"[J4] pos {pos}: local_end_index={got} (want {want}) "
              f"attn K={sh.get('k_len')} zeros-beyond={cache_checks[-1]['beyond_window_all_zero']} "
              f"sha={h}", flush=True)
    report["j4_cache_checks"] = cache_checks

    # prefix stability across positions is measured in triton_fence_and_kv_fidelity.py;
    # the key is kept so the JSON schema is unchanged
    report["j4_prefix_stable"] = None
    wc.attention = _orig_attention

    # ---------------- J5: census at the requested positions ----------------
    from torch.profiler import profile, ProfilerActivity
    from torch.autograd import DeviceType
    census = {}
    for pos in positions:
        advance_to(pos - 1 if pos > 0 else 0)
        x = noise[:, pos * 3:(pos + 1) * 3]
        ts = torch.ones([1, 3], device=dev, dtype=torch.int64) * 1000
        kw = dict(noisy_image_or_video=x, conditional_dict=cond, timestep=ts,
                  kv_cache=pipe.kv_cache1, crossattn_cache=pipe.crossattn_cache,
                  current_start=pos * 3 * 1560)
        for _ in range(3): pipe.generator(**kw)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=True) as prof:
            pipe.generator(**kw)
            torch.cuda.synchronize()
        rows = []
        for e in prof.key_averages(group_by_input_shape=True):
            if e.device_type != DeviceType.CUDA:
                continue
            rows.append({"name": e.key, "count": int(e.count),
                         "ms": e.device_time_total / 1e3,
                         "shapes": str(e.input_shapes)[:200]})
        tot = sum(r["ms"] for r in rows)
        by_stage = defaultdict(lambda: {"count": 0, "ms": 0.0, "kernels": set()})
        for r in rows:
            stage = classify_kernel(r["name"])
            by_stage[stage]["count"] += r["count"]; by_stage[stage]["ms"] += r["ms"]
            by_stage[stage]["kernels"].add(r["name"][:70])
        census[pos] = {
            "total_device_ms": tot,
            "n_kernel_launches": sum(r["count"] for r in rows),
            "by_stage": {k: {"count": v["count"], "ms": round(v["ms"], 4),
                             "pct": round(100 * v["ms"] / tot, 2),
                             "kernels": sorted(v["kernels"])[:6]}
                         for k, v in sorted(by_stage.items(), key=lambda kv: -kv[1]["ms"])},
            "top_kernels": sorted(rows, key=lambda r: -r["ms"])[:25],
        }
        print(f"\n[J5] position {pos}: {census[pos]['n_kernel_launches']} CUDA kernel "
              f"launches, {tot:.2f} ms device time", flush=True)
        for k, v in census[pos]["by_stage"].items():
            print(f"    {k:28s} n={v['count']:5d} {v['ms']:9.3f} ms {v['pct']:6.2f}%", flush=True)
    report["census"] = census

    # ---------------- reconciliation vs the J3b compile_nocg fit ----------
    rec = {}
    for pos in positions:
        pred_chunk = J3B_COMPILE_NOCG_FIT_A_MS + J3B_COMPILE_NOCG_FIT_B_MS_PER_TOKEN * K_OF_POS[pos]
        pred_pass = pred_chunk / 5.0
        meas = census[pos]["total_device_ms"]
        rec[pos] = {"fit_pass_ms": round(pred_pass, 3), "census_pass_ms": round(meas, 3),
                    "delta_pct": round(100 * (meas - pred_pass) / pred_pass, 2),
                    "within_3pct": abs(100 * (meas - pred_pass) / pred_pass) <= 3.0}
        print(f"[J5] reconcile pos {pos}: fit {pred_pass:.2f} ms vs census {meas:.2f} ms "
              f"-> {rec[pos]['delta_pct']:+.2f}% "
              f"{'OK' if rec[pos]['within_3pct'] else 'OUT OF BAND'}", flush=True)
    report["reconciliation"] = rec

    # ---------------- compile_cg re-attempt (reported, not blocking) ------
    if args.try_compile_cg:
        try:
            pipe.generator.model = base
            for blk in pipe.kv_cache1:
                for t in ("k", "v"): torch._dynamo.mark_static_address(blk[t])
            for c in pipe.crossattn_cache:
                for t in ("k", "v"):
                    if torch.is_tensor(c.get(t)): torch._dynamo.mark_static_address(c[t])
            pipe.generator.model = torch.compile(base, mode="max-autotune", dynamic=False)
            advance_to(0)
            report["compile_cg_retry"] = {"status": "SUCCESS"}
            print("[J5] compile_cg retry: SUCCESS", flush=True)
        except Exception as ex:
            report["compile_cg_retry"] = {"status": "FAILED", "error": str(ex)[:800]}
            print(f"[J5] compile_cg retry: FAILED {type(ex).__name__}: {str(ex)[:300]}", flush=True)

    with open(os.path.join(args.out, "j54.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("\n[J5/J4] written", flush=True)


if __name__ == "__main__":
    main()
