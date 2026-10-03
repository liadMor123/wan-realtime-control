#!/usr/bin/env python3
"""
Per-position latency benchmark across execution modes of the patched Self-Forcing model.

This is the main timing harness of Part A (jobs J3b, J6, J7, J8, J11). Every
mode runs the frozen per-position protocol (1 warmup video + --n_videos measured
videos, fixed prompts and seeds); when several modes are given they are
interleaved by video so slow drift cancels. Modes:

  eager-original, eager-patched      unpatched / capture-enabling-patched eager
  <mode>-perf                        same mode with the perf patches enabled
  manual_graph                       hand-captured CUDA graph per (chunk, pass)
  manual_graph_compiled              manual graphs over Inductor's compile_nocg kernels
  compile_nocg, compile_cg           torch.compile without / with Inductor cudagraphs
  final, final_no_*, final_rule      the integrated configuration and its
                                     leave-one-out ablations (see FINAL_MODES)

--attn_splits > 0 routes self-attention through flash_attn_with_kvcache
(split-KV). Writes <out>/j3b.json (per-mode per-position statistics, the
t = a + b*K fit, intercept deltas vs eager-original, graph replay counts,
failures) and <out>/vid_<mode>.npy (first measured video, uint8) for the
Tier-1 bitwise checks.
"""
import argparse, contextlib, io, json, os, statistics as st, time, traceback
import numpy as np, torch

GB = 1024 ** 3; CEILING_GB = 36.0
K_OF_POS = [(i + 1) * 4680 for i in range(7)]
MODES = ["eager-original", "eager-patched", "manual_graph", "compile_nocg", "compile_cg"]
GRAPH_MODES = {"manual_graph", "manual_graph_compiled",
               "final", "final_no_perf", "final_no_split", "final_no_inductor",
               "final_rule"}
# J8 composite modes: each is `final` minus exactly one component
FINAL_MODES = {
    "final":            dict(perf=True,  splits=4, compile=True,  graphs=True,  rule=None),
    "final_no_perf":    dict(perf=False, splits=4, compile=True,  graphs=True,  rule=None),
    "final_no_split":   dict(perf=True,  splits=0, compile=True,  graphs=True,  rule=None),
    "final_no_inductor":dict(perf=True,  splits=4, compile=False, graphs=True,  rule=None),
    "final_no_graphs":  dict(perf=True,  splits=4, compile=True,  graphs=False, rule=None),
    "final_rule":       dict(perf=True,  splits=4, compile=True,  graphs=True,  rule="J10"),
}
J10_RULE = {0: 2, 1: 3, 2: 4, 3: 4, 4: 5, 5: 5, 6: 6}
# "<mode>-perf" runs the same mode with the perf patches enabled


def spread_pct(xs):
    m = st.median(xs); return 100.0 * (max(xs) - min(xs)) / m if m else float("nan")


def summarize(xs):
    xs = list(xs); q = np.percentile(xs, [25, 75]) if len(xs) > 1 else (xs[0], xs[0])
    return {"n": len(xs), "median": st.median(xs), "min": min(xs), "max": max(xs),
            "iqr": float(q[1] - q[0]), "spread_pct": spread_pct(xs),
            "values": [round(v, 3) for v in xs]}


def fit_intercept_slope(med):
    K = np.array(K_OF_POS, float); t = np.array(med, float)
    A = np.vstack([np.ones_like(K), K]).T
    (a, b), *_ = np.linalg.lstsq(A, t, rcond=None)
    ss = float(((t - (a + b * K)) ** 2).sum()); tot = float(((t - t.mean()) ** 2).sum())
    return {"a_ms": float(a), "b_ms_per_1k_K": float(b * 1000), "r2": 1 - ss / tot if tot else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--jsonl", required=True,
                    help="accepted for interface parity with the other harnesses; "
                         "this script writes only <out>/j3b.json")
    ap.add_argument("--n_videos", type=int, default=5)
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--attn_splits", type=int, default=0,
                    help="J7: >0 routes self-attention through flash_attn_with_kvcache")
    ap.add_argument("--perf", choices=["on","off","per-mode"], default="per-mode",
                    help="process-level perf-patch flag; compile modes need it fixed")
    args = ap.parse_args()
    modes = [m for m in args.modes.split(",") if m]
    print(f"[J3b] modes this process: {modes}", flush=True)
    os.makedirs(args.out, exist_ok=True)

    from omegaconf import OmegaConf
    from pipeline import CausalInferencePipeline
    from pipeline.causal_inference import prime_crossattn_cache
    from demo_utils.memory import gpu, DynamicSwapInstaller
    from utils.misc import set_seed
    from wan import patch_flags
    from wan.modules.model import prebuild_sinusoid_cache
    import triton, flash_attn

    dev = torch.device("cuda"); torch.set_grad_enabled(False)
    cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"),
                          OmegaConf.load("configs/self_forcing_dmd.yaml"))
    pipe = CausalInferencePipeline(cfg, device=dev)
    sd = torch.load("checkpoints/self_forcing_dmd.pt", map_location="cpu")
    pipe.generator.load_state_dict(sd["generator_ema"]); del sd
    pipe = pipe.to(dtype=torch.bfloat16)
    DynamicSwapInstaller.install_model(pipe.text_encoder, device=gpu)
    pipe.generator.to(device=gpu); pipe.vae.to(device=gpu)
    # patch 7/7: build the sinusoidal constant OUTSIDE any compiled region, or
    # Inductor's cudagraph tree adopts it and compile_cg fails on the next run
    prebuild_sinusoid_cache(256, dev)
    with open("prompts/MovieGenVideoBench_extended.txt", encoding="utf-8") as f:
        prompts = [f.readline().rstrip() for _ in range(args.n_videos)]
    first_ts = float(pipe.denoising_step_list.tolist()[0])

    base_model = pipe.generator.model
    compiled = {}
    import torch._dynamo as dynamo
    dynamo.config.cache_size_limit = 64
    dynamo.config.accumulated_cache_size_limit = 256

    timing = {"chunk": -1, "in_chunk": 0, "t_gen": None, "t_chunk": [], "ev": [], "graph": None}
    mark_step = {"on": False}
    original_forward = pipe.generator.forward

    def timed_forward(*a, **kw):
        if mark_step["on"]:
            # J11: the Inductor cudagraph error names this as the remedy --
            # tell the cudagraph tree a new step began so it stops treating the
            # previous run's outputs (the KV-cache writes) as live.
            torch.compiler.cudagraph_mark_step_begin()
        ts = kw.get("timestep")
        tsv = float(ts.flatten()[0].item()) if ts is not None else float("nan")
        if tsv == first_ts:
            timing["chunk"] += 1; timing["in_chunk"] = 0
        if timing["t_gen"] is None:
            torch.cuda.synchronize(); timing["t_gen"] = time.perf_counter()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        timing["ev"].append([timing["chunk"], timing["in_chunk"], s, e])
        pos, idx = timing["chunk"], timing["in_chunk"]; timing["in_chunk"] += 1
        s.record()
        g = timing["graph"]
        out = g(pos, idx, kw, original_forward) if g is not None else original_forward(*a, **kw)
        e.record()
        if timing["in_chunk"] == 5:
            torch.cuda.synchronize(); timing["t_chunk"].append(time.perf_counter())
        return out
    pipe.generator.forward = timed_forward

    # ---------------- manual CUDA graph machinery ----------------
    class ManualGraphs:
        """One captured CUDA graph per (chunk position, pass index), sharing a pool."""
        total_replays = 0

        def __init__(self): self.g = {}; self.pool = None; self.static = {}
        def snap(self):
            return [(b["global_end_index"], b["local_end_index"]) for b in pipe.kv_cache1]
        def restore(self, s):
            for b, (gi, li) in zip(pipe.kv_cache1, s):
                b["global_end_index"], b["local_end_index"] = gi, li
        def bind(self, kw):
            st_ = self.static
            if "noisy" not in st_:
                st_["noisy"] = kw["noisy_image_or_video"].clone()
                st_["ts"] = kw["timestep"].clone()
                st_["cond"] = {k: (v.clone() if torch.is_tensor(v) else v)
                               for k, v in kw["conditional_dict"].items()}
            st_["noisy"].copy_(kw["noisy_image_or_video"]); st_["ts"].copy_(kw["timestep"])
            for k, v in kw["conditional_dict"].items():
                if torch.is_tensor(v): st_["cond"][k].copy_(v)
            nk = dict(kw); nk["noisy_image_or_video"] = st_["noisy"]
            nk["timestep"] = st_["ts"]; nk["conditional_dict"] = st_["cond"]
            return nk
        def capture(self, pos, idx, kw, fwd0):
            nk = self.bind(kw); s0 = self.snap()
            sstream = torch.cuda.Stream(); sstream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(sstream):
                for _ in range(3):
                    self.restore(s0); fwd0(**nk)
            torch.cuda.current_stream().wait_stream(sstream); torch.cuda.synchronize()
            self.restore(s0)
            g = torch.cuda.CUDAGraph()
            ctx = torch.cuda.graph(g) if self.pool is None else torch.cuda.graph(g, pool=self.pool)
            with ctx:
                out = fwd0(**nk)
            if self.pool is None: self.pool = g.pool()
            self.g[(pos, idx)] = (g, out, s0)
            self.restore(s0); g.replay(); torch.cuda.synchronize()
            return out
        def replay(self, pos, idx, kw, fwd0):
            g, out, s0 = self.g[(pos, idx)]
            self.bind(kw); self.restore(s0); g.replay()
            ManualGraphs.total_replays += 1
            return out

    graphs = ManualGraphs()
    capture_mode = {"on": False}

    def graph_hook(pos, idx, kw, fwd0):
        return graphs.capture(pos, idx, kw, fwd0) if capture_mode["on"] else graphs.replay(pos, idx, kw, fwd0)

    # ---------------- mode setup ----------------
    def configure_mode(mode):
        if mode in FINAL_MODES:
            cf = FINAL_MODES[mode]
            patch_flags.set_enabled(True)
            patch_flags.set_perf(cf["perf"])
            patch_flags.set_attn_splits(cf["splits"])
            patch_flags.set_attn_split_rule(J10_RULE if cf["rule"] else None)
            pipe.generator.model = base_model
            timing["graph"] = None
            if cf["compile"]:
                if "nocg" not in compiled:
                    compiled["nocg"] = torch.compile(base_model,
                        mode="max-autotune-no-cudagraphs", dynamic=False)
                pipe.generator.model = compiled["nocg"]
            if cf["graphs"]:
                timing["graph"] = graph_hook
            return
        patch_flags.set_attn_split_rule(None)
        patch_flags.set_enabled(mode != "eager-original")
        if args.perf == "per-mode":
            # perf patches 8/9+9/9 are A/B-ed by mode name; safe for eager only
            patch_flags.set_perf(mode.endswith("-perf"))
        else:
            patch_flags.set_perf(args.perf == "on")
        patch_flags.set_attn_splits(args.attn_splits)
        mode = mode[:-5] if mode.endswith("-perf") else mode
        pipe.generator.model = base_model
        timing["graph"] = None
        if mode == "compile_nocg":
            if "nocg" not in compiled:
                compiled["nocg"] = torch.compile(base_model, mode="max-autotune-no-cudagraphs",
                                                 dynamic=False)
            pipe.generator.model = compiled["nocg"]
        elif mode == "compile_cg":
            mark_step["on"] = True
            if "cg" not in compiled:
                # caches must exist before their addresses can be marked static
                if pipe.kv_cache1 is None:
                    pipe._initialize_kv_cache(1, torch.bfloat16, dev)
                    pipe._initialize_crossattn_cache(1, torch.bfloat16, dev)
                for b in pipe.kv_cache1:
                    for t in ("k", "v"):
                        torch._dynamo.mark_static_address(b[t])
                for c in pipe.crossattn_cache:
                    for t in ("k", "v"):
                        if torch.is_tensor(c.get(t)):
                            torch._dynamo.mark_static_address(c[t])
                compiled["cg"] = torch.compile(base_model, mode="max-autotune", dynamic=False)
            pipe.generator.model = compiled["cg"]
        elif mode == "manual_graph":
            timing["graph"] = graph_hook
        elif mode == "manual_graph_compiled":
            # fallback arm: manual per-position graphs over the kernels
            # Inductor generated for compile_nocg ("fusion + graphs" arm).
            if "nocg" not in compiled:
                compiled["nocg"] = torch.compile(base_model, mode="max-autotune-no-cudagraphs",
                                                 dynamic=False)
            pipe.generator.model = compiled["nocg"]
            timing["graph"] = graph_hook

    def one_video(prompt, seed, mode, capture=False):
        set_seed(seed)
        timing.update({"chunk": -1, "in_chunk": 0, "t_gen": None, "t_chunk": [], "ev": []})
        torch.cuda.reset_peak_memory_stats()
        noise = torch.randn([1, 21, 16, 60, 104], device=dev, dtype=torch.bfloat16)
        if mode in GRAPH_MODES or mode == "compile_cg":
            cond = pipe.text_encoder(text_prompts=[prompt])
            if pipe.kv_cache1 is None:
                pipe._initialize_kv_cache(1, torch.bfloat16, dev)
                pipe._initialize_crossattn_cache(1, torch.bfloat16, dev)
            sg = timing["graph"]; timing["graph"] = None
            with contextlib.redirect_stdout(io.StringIO()):
                prime_crossattn_cache(pipe, cond, noise)
            timing["graph"] = sg
            # the priming pass goes through the timing wrapper; discard its
            # timestamps so chunk 0 is not charged for it (protocol: t_gen is
            # taken immediately before the FIRST generator pass of the video)
            timing.update({"chunk": -1, "in_chunk": 0, "t_gen": None, "t_chunk": [], "ev": []})
        capture_mode["on"] = capture
        buf = io.StringIO(); t0 = time.perf_counter()
        with contextlib.redirect_stdout(buf):
            video, _ = pipe.inference(noise=noise, text_prompts=[prompt], return_latents=True,
                                      initial_latent=None, low_memory=True)
        torch.cuda.synchronize(); capture_mode["on"] = False
        wall = time.perf_counter() - t0
        passes = {}
        for c, i, s, e in timing["ev"]:
            passes.setdefault(c, []).append(s.elapsed_time(e))
        ch, rs, prev = [], [], timing["t_gen"]
        for i, t in enumerate(timing["t_chunk"]):
            T = (t - prev) * 1e3; ch.append(T)
            ps = sum(passes.get(i, [])); rs.append(100 * (T - ps) / T if T else float("nan")); prev = t
        return {"video": video, "chunk_ms": ch, "residual_pct": rs, "wall_s": wall,
                "peak_reserved_gb": torch.cuda.max_memory_reserved() / GB}

    report = {"modes": {}, "torch": torch.__version__, "triton": triton.__version__,
           "flash_attn": flash_attn.__version__, "failures": {}}
    ok_modes = []

    # ---- warmup per mode (capture happens here for manual_graph) ----
    for m in modes:
        try:
            configure_mode(m)
            t0 = time.perf_counter()
            w = one_video(prompts[0], 999, m, capture=(m in GRAPH_MODES))
            report["modes"][m] = {"warmup_s": round(time.perf_counter() - t0, 2),
                               "warmup_chunk_ms": [round(x, 2) for x in w["chunk_ms"]],
                               "videos": []}
            if m in GRAPH_MODES:
                report["modes"][m]["graphs_captured"] = len(graphs.g)
            print(f"[J3b] {m}: warmup ok ({report['modes'][m]['warmup_s']}s)"
                  f"{' graphs=' + str(len(graphs.g)) if m in GRAPH_MODES else ''}", flush=True)
            ok_modes.append(m); del w
        except Exception as ex:
            report["failures"][m] = {"error_type": type(ex).__name__, "error": str(ex)[:1200],
                                  "traceback_tail": traceback.format_exc()[-2500:]}
            print(f"[J3b] {m}: WARMUP/CAPTURE FAILED: {type(ex).__name__}: {str(ex)[:300]}", flush=True)

    # ---- measured videos, INTERLEAVED by video ----
    for v in range(args.n_videos):
        for m in ok_modes:
            configure_mode(m)
            _t0 = time.time()
            r = one_video(prompts[v], 1000 + v, m)
            report["modes"][m]["videos"].append({
                "index": v, "seed": 1000 + v,
                "t_start_unix": _t0, "t_end_unix": time.time(),
                "chunk_ms": [round(x, 3) for x in r["chunk_ms"]],
                "residual_pct": [round(x, 3) for x in r["residual_pct"]],
                "peak_reserved_gb": round(r["peak_reserved_gb"], 3)})
            if v == 0:
                a = (255.0 * r["video"].float().clamp(0, 1)).round().clamp(0, 255)
                np.save(os.path.join(args.out, f"vid_{m}.npy"),
                        a[0].permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy())
            print(f"[J3b] v{v} {m:15s} {[round(x,1) for x in r['chunk_ms']]} "
                  f"peak={r['peak_reserved_gb']:.2f}", flush=True)
            del r

    for m in ok_modes:
        vids = report["modes"][m]["videos"]
        n = len(vids[0]["chunk_ms"])
        pp = {i: summarize([x["chunk_ms"][i] for x in vids]) for i in range(n)}
        report["modes"][m]["per_position"] = pp
        report["modes"][m]["fit"] = fit_intercept_slope([pp[i]["median"] for i in range(n)])
        report["modes"][m]["peak_reserved_gb"] = summarize([x["peak_reserved_gb"] for x in vids])
    for gm in GRAPH_MODES & set(ok_modes):
        report["modes"][gm]["total_replays"] = ManualGraphs.total_replays
        report["modes"][gm]["expected_replays"] = 35 * args.n_videos
        report["modes"][gm]["one_replay_per_pass"] = (ManualGraphs.total_replays == 35 * args.n_videos)
    if "eager-original" in ok_modes:
        a0 = report["modes"]["eager-original"]["fit"]["a_ms"]
        for m in ok_modes:
            report["modes"][m]["intercept_delta_vs_eager_original_ms"] = \
                report["modes"][m]["fit"]["a_ms"] - a0

    with open(os.path.join(args.out, "j3b.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("\n[J3b] summary:")
    for m in ok_modes:
        f_ = report["modes"][m]["fit"]
        print(f"  {m:15s} a={f_['a_ms']:8.2f} b={f_['b_ms_per_1k_K']:.4f} "
              f"R2={f_['r2']:.5f} Δa={report['modes'][m].get('intercept_delta_vs_eager_original_ms', 0):8.2f}")
    for m, e in report["failures"].items():
        print(f"  {m:15s} FAILED: {e['error_type']}: {e['error'][:120]}")


if __name__ == "__main__":
    main()
