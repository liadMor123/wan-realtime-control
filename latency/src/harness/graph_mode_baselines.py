#!/usr/bin/env python3
"""
Graph-mode baselines on the unpatched Self-Forcing model (J3).

Runs the per-position timing protocol under one of four execution modes:
  eager_ref     eager PyTorch (reference video for PSNR comparisons)
  manual_graph  one hand-captured torch.cuda.CUDAGraph per (chunk, pass),
                with two capture-safe shims (cached sinusoid table, host-side
                KV-cache index) that are numerically identical to the original
  compile_nocg  torch.compile(mode="max-autotune-no-cudagraphs")
  compile_cg    torch.compile(mode="max-autotune")

For each mode it records a capture proof (kernel launches on one diagnostic
pass), per-position chunk-time medians over 5 measured videos, the linear fit
t = a + b*K over the history length K, and peak memory.

Writes <out>/j3_<mode>.json, optionally the first measured video as uint8
.npy (--save_video), and appends one row to --jsonl.
"""
import argparse, contextlib, io, json, os, socket, statistics as st, sys, time, traceback
from datetime import datetime, timezone

import numpy as np
import torch

GB = 1024 ** 3
CEILING_GB = 36.0
PSNR_FLOOR_DB = 40.0          # pre-registered PSNR floor (checked by frame_psnr.py)
K_OF_POS = [(i + 1) * 4680 for i in range(7)]


def fail(msg, code=2):
    print(f"\n!!! J3 FAILURE: {msg}\n", flush=True); sys.exit(code)


def spread_pct(xs):
    m = st.median(xs)
    return 100.0 * (max(xs) - min(xs)) / m if m else float("nan")


def summarize(xs):
    xs = list(xs)
    q = np.percentile(xs, [25, 75]) if len(xs) > 1 else (xs[0], xs[0])
    return {"n": len(xs), "median": st.median(xs), "min": min(xs), "max": max(xs),
            "iqr": float(q[1] - q[0]), "spread_pct": spread_pct(xs),
            "values": [round(v, 3) for v in xs]}


def fit_intercept_slope(medians):
    """OLS t = a + b*K. Returns a (ms), b (ms per 1000 K-tokens), R^2."""
    K = np.array(K_OF_POS, dtype=float)
    t = np.array(medians, dtype=float)
    A = np.vstack([np.ones_like(K), K]).T
    (a, b), *_ = np.linalg.lstsq(A, t, rcond=None)
    pred = a + b * K
    ss_res = float(((t - pred) ** 2).sum())
    ss_tot = float(((t - t.mean()) ** 2).sum())
    return {"a_ms": float(a), "b_ms_per_1k_K": float(b * 1000.0),
            "r2": 1.0 - ss_res / ss_tot if ss_tot else float("nan")}


# ---------------------------------------------------------------- shims
class HostIndexShim:
    """Capture-safe stand-in for the kv-cache index tensors.

    causal_model.py uses these only through .item() and .fill_(). The tensor
    form forces a device->host sync inside the captured region. Holding the
    value on the host is numerically identical and removes the sync.
    """
    __slots__ = ("v",)

    def __init__(self, v=0): self.v = int(v)
    def item(self): return self.v
    def fill_(self, x):
        self.v = int(x.item()) if torch.is_tensor(x) else int(x)
        return self


def install_capture_shims(pipe, dev):
    """Two capture blockers, both numerically identical replacements."""
    notes = []
    import wan.modules.model as wmodel
    import wan.modules.causal_model as wcausal

    # (a) sinusoidal_embedding_1d builds torch.arange on CPU then copies H2D
    #     on every call -> illegal during capture. Cache the frequency vector.
    _cache = {}

    def safe_sinusoidal_embedding_1d(dim, position):
        half = dim // 2
        key = (dim, position.device, position.dtype)
        if key not in _cache:
            _cache[key] = torch.pow(
                10000, -torch.arange(half, device=position.device,
                                     dtype=position.dtype).div(half))
        position = position.type(_cache[key].dtype)
        sinusoid = torch.outer(position, _cache[key])
        return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)

    for mod in (wmodel, wcausal):
        if hasattr(mod, "sinusoidal_embedding_1d"):
            mod.sinusoidal_embedding_1d = safe_sinusoidal_embedding_1d
    notes.append("sinusoidal_embedding_1d: cached device freq vector (was CPU arange + H2D per call)")

    # (b) kv-cache index tensors -> host-side shims (removes .item() sync)
    for blk in pipe.kv_cache1:
        blk["global_end_index"] = HostIndexShim(0)
        blk["local_end_index"] = HostIndexShim(0)
    notes.append("kv_cache global/local_end_index: host-side shim (was .item() D2H sync)")
    return notes


def snapshot_kv_indices(pipe):
    return [(b["global_end_index"].item(), b["local_end_index"].item()) for b in pipe.kv_cache1]


def restore_kv_indices(pipe, snap):
    for b, (g, l) in zip(pipe.kv_cache1, snap):
        b["global_end_index"].fill_(g); b["local_end_index"].fill_(l)


# ---------------------------------------------------------------- pipeline
def build_pipeline(args, dev):
    from omegaconf import OmegaConf
    from pipeline import CausalInferencePipeline
    from demo_utils.memory import gpu, DynamicSwapInstaller
    cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"),
                          OmegaConf.load(args.config_path))
    pipe = CausalInferencePipeline(cfg, device=dev)
    sd = torch.load(args.checkpoint_path, map_location="cpu")
    pipe.generator.load_state_dict(sd["generator_ema"]); del sd
    pipe = pipe.to(dtype=torch.bfloat16)
    DynamicSwapInstaller.install_model(pipe.text_encoder, device=gpu)
    pipe.generator.to(device=gpu); pipe.vae.to(device=gpu)
    return pipe, cfg


def count_cuda_kernels(fn):
    """Kernel launches actually issued to the device (CUDA-side events only)."""
    from torch.profiler import profile, ProfilerActivity
    from torch.autograd import DeviceType
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn(); torch.cuda.synchronize()
    n = 0; names = {}
    for e in prof.key_averages():
        if e.device_type == DeviceType.CUDA:
            n += e.count; names[e.key[:70]] = names.get(e.key[:70], 0) + e.count
    return n, sorted(names.items(), key=lambda kv: -kv[1])[:10]


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_path", default="configs/self_forcing_dmd.yaml")
    ap.add_argument("--checkpoint_path", default="checkpoints/self_forcing_dmd.pt")
    ap.add_argument("--data_path", default="prompts/MovieGenVideoBench_extended.txt")
    ap.add_argument("--mode", required=True,
                    choices=["eager_ref", "manual_graph", "compile_nocg", "compile_cg"])
    ap.add_argument("--n_measured", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--num_output_frames", type=int, default=21)
    ap.add_argument("--save_video", default="")
    args = ap.parse_args()

    from utils.misc import set_seed
    dev = torch.device("cuda"); torch.set_grad_enabled(False)
    os.makedirs(args.out, exist_ok=True)
    with open(args.data_path, encoding="utf-8") as f:
        prompts = [f.readline().rstrip() for _ in range(args.n_measured)]

    pipe, cfg = build_pipeline(args, dev)
    first_ts = float(pipe.denoising_step_list.tolist()[0])
    import triton, flash_attn
    report = {"mode": args.mode, "torch": torch.__version__,
              "triton": triton.__version__, "flash_attn": flash_attn.__version__,
              "capture": {}, "notes": []}

    # caches must exist before shims are installed
    pipe._initialize_kv_cache(1, torch.bfloat16, dev)
    pipe._initialize_crossattn_cache(1, torch.bfloat16, dev)

    graph_state = {"enabled": False, "graphs": {}, "pool": None,
                   "static": {}, "replays": 0, "capturing": False}

    if args.mode == "manual_graph":
        report["notes"] += install_capture_shims(pipe, dev)

    if args.mode in ("compile_nocg", "compile_cg"):
        import torch._dynamo as dynamo
        dynamo.config.cache_size_limit = 64          # >= 7 chunk-position shapes
        dynamo.config.accumulated_cache_size_limit = 256
        mode = ("max-autotune-no-cudagraphs" if args.mode == "compile_nocg"
                else "max-autotune")
        pipe.generator.model = torch.compile(pipe.generator.model,
                                             mode=mode, dynamic=False)
        report["notes"].append(f"torch.compile(generator.model, mode={mode}, dynamic=False)")

    # ---- timing wrapper (frozen per-position protocol) ----
    state = {"chunk": -1, "in_chunk": 0, "t_gen": None, "t_chunk": [], "ev": []}
    gen = pipe.generator
    original_forward = gen.forward

    def timed_forward(*a, **kw):
        ts = kw.get("timestep")
        tsv = float(ts.flatten()[0].item()) if ts is not None else float("nan")
        if tsv == first_ts:
            state["chunk"] += 1; state["in_chunk"] = 0
        if state["t_gen"] is None:
            torch.cuda.synchronize(); state["t_gen"] = time.perf_counter()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        state["ev"].append([state["chunk"], state["in_chunk"], s, e])
        pos, idx = state["chunk"], state["in_chunk"]
        state["in_chunk"] += 1
        s.record()
        if graph_state["enabled"]:
            out = run_graphed(pos, idx, kw)
        else:
            out = original_forward(*a, **kw)
        e.record()
        if state["in_chunk"] == 5:
            torch.cuda.synchronize(); state["t_chunk"].append(time.perf_counter())
        return out

    gen.forward = timed_forward

    # ---- manual graph capture / replay ----
    def capture(pos, idx, kw):
        st_ = graph_state["static"]
        key = (pos, idx)
        if "noisy" not in st_:
            st_["noisy"] = kw["noisy_image_or_video"].clone()
            st_["ts"] = kw["timestep"].clone()
            st_["cond"] = {k: (v.clone() if torch.is_tensor(v) else v)
                           for k, v in kw["conditional_dict"].items()}
        st_["noisy"].copy_(kw["noisy_image_or_video"])
        st_["ts"].copy_(kw["timestep"])
        for k, v in kw["conditional_dict"].items():
            if torch.is_tensor(v): st_["cond"][k].copy_(v)
        call_kw = dict(kw)
        call_kw["noisy_image_or_video"] = st_["noisy"]
        call_kw["timestep"] = st_["ts"]
        call_kw["conditional_dict"] = st_["cond"]

        snap = snapshot_kv_indices(pipe)
        sstream = torch.cuda.Stream(); sstream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(sstream):
            for _ in range(3):
                restore_kv_indices(pipe, snap); original_forward(**call_kw)
        torch.cuda.current_stream().wait_stream(sstream); torch.cuda.synchronize()

        restore_kv_indices(pipe, snap)
        g = torch.cuda.CUDAGraph()
        ctx = (torch.cuda.graph(g) if graph_state["pool"] is None
               else torch.cuda.graph(g, pool=graph_state["pool"]))
        with ctx:
            out = original_forward(**call_kw)
        if graph_state["pool"] is None:
            graph_state["pool"] = g.pool()
        graph_state["graphs"][key] = (g, out, snap)
        # capture records without executing -> replay once to realise the state
        restore_kv_indices(pipe, snap)
        g.replay(); torch.cuda.synchronize()
        return out

    def run_graphed(pos, idx, kw):
        key = (pos, idx)
        if graph_state["capturing"]:
            return capture(pos, idx, kw)
        g, out, snap = graph_state["graphs"][key]
        st_ = graph_state["static"]
        st_["noisy"].copy_(kw["noisy_image_or_video"])
        st_["ts"].copy_(kw["timestep"])
        for k, v in kw["conditional_dict"].items():
            if torch.is_tensor(v): st_["cond"][k].copy_(v)
        restore_kv_indices(pipe, snap)
        g.replay()
        graph_state["replays"] += 1
        return out

    # ---- one video ----
    def one_video(prompt, seed, capture_pass=False):
        set_seed(seed)
        state.update({"chunk": -1, "in_chunk": 0, "t_gen": None,
                      "t_chunk": [], "ev": []})
        torch.cuda.reset_peak_memory_stats()
        graph_state["capturing"] = capture_pass
        noise = torch.randn([1, args.num_output_frames, 16, 60, 104],
                            device=dev, dtype=torch.bfloat16)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            video, latents = pipe.inference(noise=noise, text_prompts=[prompt],
                                            return_latents=True,
                                            initial_latent=None, low_memory=True)
        torch.cuda.synchronize()
        graph_state["capturing"] = False
        passes = {}
        for c, i, s, e in state["ev"]:
            passes.setdefault(c, []).append(s.elapsed_time(e))
        chunk_ms, resid, prev = [], [], state["t_gen"]
        for i, t in enumerate(state["t_chunk"]):
            T = (t - prev) * 1e3; chunk_ms.append(T)
            psum = sum(passes.get(i, []))
            resid.append(100.0 * (T - psum) / T if T else float("nan"))
            prev = t
        return {"video": video, "chunk_ms": chunk_ms, "residual_pct": resid,
                "pass_ms": passes,
                "peak_reserved_gb": torch.cuda.max_memory_reserved() / GB}

    # ---- warmup (capture happens here for manual_graph) ----
    if args.mode == "manual_graph":
        graph_state["enabled"] = True
    t0 = time.perf_counter()
    try:
        w = one_video(prompts[0], 999, capture_pass=(args.mode == "manual_graph"))
    except Exception as ex:
        report["capture"] = {"status": "FAILED", "error_type": type(ex).__name__,
                             "error": str(ex)[:1200],
                             "traceback_tail": traceback.format_exc()[-2000:]}
        print(f"[J3:{args.mode}] WARMUP/CAPTURE FAILED: {type(ex).__name__}: {str(ex)[:400]}",
              flush=True)
        print(traceback.format_exc()[-2000:], flush=True)
        with open(os.path.join(args.out, f"j3_{args.mode}.json"), "w") as f:
            json.dump(report, f, indent=2)
        fail(f"mode {args.mode} failed during warmup/capture", 5)
    warm_s = time.perf_counter() - t0
    report["warmup_s"] = round(warm_s, 2)
    report["warmup_chunk_ms"] = [round(x, 2) for x in w["chunk_ms"]]
    print(f"[J3:{args.mode}] warmup {warm_s:.1f}s chunks="
          f"{[round(x,1) for x in w['chunk_ms']]}", flush=True)
    if args.mode == "manual_graph":
        report["capture"] = {"status": "SUCCESS",
                             "graphs_captured": len(graph_state["graphs"])}
        print(f"[J3] captured {len(graph_state['graphs'])} graphs", flush=True)
    del w

    # ---- capture proof: kernel launches on one diagnostic pass ----
    try:
        n_k, top = count_cuda_kernels(lambda: pipe.generator(
            noisy_image_or_video=torch.randn([1, 3, 16, 60, 104], device=dev, dtype=torch.bfloat16),
            conditional_dict=pipe.text_encoder(text_prompts=[prompts[0]]),
            timestep=torch.ones([1, 3], device=dev, dtype=torch.int64) * int(first_ts),
            kv_cache=pipe.kv_cache1, crossattn_cache=pipe.crossattn_cache,
            current_start=0) if args.mode != "manual_graph" else
            graph_state["graphs"][(0, 0)][0].replay())
        report["capture"]["kernel_launches_diagnostic_pass"] = n_k
        report["capture"]["top_kernels"] = top
        print(f"[J3:{args.mode}] diagnostic pass CUDA kernel launches: {n_k}", flush=True)
    except Exception as ex:
        report["capture"]["kernel_launches_diagnostic_pass"] = None
        report["capture"]["launch_count_error"] = str(ex)[:400]
        print(f"[J3:{args.mode}] launch count unavailable: {str(ex)[:200]}", flush=True)

    # reset dynamo recompile tracking after warmup
    if args.mode in ("compile_nocg", "compile_cg"):
        try:
            from torch._dynamo.utils import counters
            counters.clear()
        except Exception:
            pass

    # ---- measured videos ----
    vids = []
    for k in range(args.n_measured):
        r = one_video(prompts[k], 1000 + k)
        vids.append({"index": k, "seed": 1000 + k,
                     "chunk_ms": [round(x, 3) for x in r["chunk_ms"]],
                     "residual_pct": [round(x, 3) for x in r["residual_pct"]],
                     "peak_reserved_gb": round(r["peak_reserved_gb"], 3)})
        print(f"[J3:{args.mode}] video {k}: {[round(x,1) for x in r['chunk_ms']]} "
              f"peak={r['peak_reserved_gb']:.2f}GB", flush=True)
        if k == 0 and args.save_video:
            arr = (255.0 * r["video"].float().clamp(0, 1)).round().clamp(0, 255)
            np.save(args.save_video, arr[0].permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy())
        del r

    if args.mode in ("compile_nocg", "compile_cg"):
        try:
            from torch._dynamo.utils import counters
            rec = dict(counters.get("frames", {}))
            report["recompiles_after_warmup"] = rec
            print(f"[J3:{args.mode}] dynamo counters after warmup: {rec}", flush=True)
        except Exception:
            pass

    n_pos = len(vids[0]["chunk_ms"])
    per_pos = {i: summarize([v["chunk_ms"][i] for v in vids]) for i in range(n_pos)}
    report["videos"] = vids
    report["per_position"] = per_pos
    report["fit"] = fit_intercept_slope([per_pos[i]["median"] for i in range(n_pos)])
    report["peak_reserved_gb"] = summarize([v["peak_reserved_gb"] for v in vids])
    if args.mode == "manual_graph":
        report["capture"]["total_replays"] = graph_state["replays"]
        expected = 35 * args.n_measured
        report["capture"]["expected_replays"] = expected
        report["capture"]["one_replay_per_pass"] = (graph_state["replays"] == expected)

    print(f"\n[J3:{args.mode}] per-position medians (ms):")
    for i in range(n_pos):
        p = per_pos[i]
        print(f"   pos {i}  median {p['median']:9.2f}  spread {p['spread_pct']:5.2f}%")
    f = report["fit"]
    print(f"[J3:{args.mode}] fit t = {f['a_ms']:.2f} ms + {f['b_ms_per_1k_K']:.4f} ms/1k-K "
          f"(R^2={f['r2']:.5f})")

    mx = report["peak_reserved_gb"]["max"]
    if mx > CEILING_GB:
        with open(os.path.join(args.out, f"j3_{args.mode}.json"), "w") as fh:
            json.dump(report, fh, indent=2)
        fail(f"{args.mode} peak reserved {mx:.3f} GB exceeds {CEILING_GB} GB ceiling")

    with open(os.path.join(args.out, f"j3_{args.mode}.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    with open(args.jsonl, "a") as fh:
        fh.write(json.dumps({
            "run_id": f"j3-{args.mode}-{os.environ.get('SLURM_JOB_ID')}",
            "job_id": os.environ.get("SLURM_JOB_ID"), "node": socket.gethostname(),
            "graph_mode": args.mode, "fusion_mode": "none",
            "torch_version": torch.__version__, "triton_version": triton.__version__,
            "attention_backend": "flash_attn_varlen (FA2)",
            "warmup_excluded": True, "profiler_attached": False,
            "t_per_chunk_ms_median_pos3": per_pos[3]["median"],
            "t_per_chunk_ms_median_pos6": per_pos[n_pos - 1]["median"],
            "fit_a_ms": f["a_ms"], "fit_b_ms_per_1k_K": f["b_ms_per_1k_K"],
            "max_memory_reserved_bytes": int(mx * GB),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }) + "\n")
    print(f"[J3:{args.mode}] done", flush=True)


if __name__ == "__main__":
    main()
