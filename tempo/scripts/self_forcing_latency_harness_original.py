#!/usr/bin/env python3
"""J3b: five modes, interleaved by video. experiment_spec.md v4 section 10."""
import argparse, contextlib, io, json, os, socket, statistics as st, sys, time, traceback
from datetime import datetime, timezone
import numpy as np, torch

GB = 1024 ** 3; CEILING_GB = 36.0
K_OF_POS = [(i + 1) * 4680 for i in range(7)]
MODES = ["eager-original", "eager-patched", "manual_graph", "compile_nocg", "compile_cg"]
GRAPH_MODES = {"manual_graph", "manual_graph_compiled",
               "final", "final_no_perf", "final_no_split", "final_no_inductor",
               "final_rule", "final_fused"}
# J8 composite modes: each is `final` minus exactly one component
FINAL_MODES = {
    "final":            dict(perf=True,  splits=4, compile=True,  graphs=True,  rule=None),
    "final_no_perf":    dict(perf=False, splits=4, compile=True,  graphs=True,  rule=None),
    "final_no_split":   dict(perf=True,  splits=0, compile=True,  graphs=True,  rule=None),
    "final_no_inductor":dict(perf=True,  splits=4, compile=False, graphs=True,  rule=None),
    "final_no_graphs":  dict(perf=True,  splits=4, compile=True,  graphs=False, rule=None),
    "final_rule":       dict(perf=True,  splits=4, compile=True,  graphs=True,  rule="J10"),
    # J13: final + fused split-KV merge in the out-proj GEMM (spec §19); needs the patched flash_attn
    "final_fused":      dict(perf=True,  splits=4, compile=True,  graphs=True,  rule=None, fused=True),
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


def fit_aK(med):
    K = np.array(K_OF_POS, float); t = np.array(med, float)
    A = np.vstack([np.ones_like(K), K]).T
    (a, b), *_ = np.linalg.lstsq(A, t, rcond=None)
    ss = float(((t - (a + b * K)) ** 2).sum()); tot = float(((t - t.mean()) ** 2).sum())
    return {"a_ms": float(a), "b_ms_per_1k_K": float(b * 1000), "r2": 1 - ss / tot if tot else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--jsonl", required=True)
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

    S = {"chunk": -1, "in_chunk": 0, "t_gen": None, "t_chunk": [], "ev": [], "graph": None}
    MARK_STEP = {"on": False}
    _fwd = pipe.generator.forward

    def fwd(*a, **kw):
        if MARK_STEP["on"]:
            # J11: the Inductor cudagraph error names this as the remedy --
            # tell the cudagraph tree a new step began so it stops treating the
            # previous run's outputs (the KV-cache writes) as live.
            torch.compiler.cudagraph_mark_step_begin()
        ts = kw.get("timestep")
        tsv = float(ts.flatten()[0].item()) if ts is not None else float("nan")
        if tsv == first_ts:
            S["chunk"] += 1; S["in_chunk"] = 0
        if S["t_gen"] is None:
            torch.cuda.synchronize(); S["t_gen"] = time.perf_counter()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        S["ev"].append([S["chunk"], S["in_chunk"], s, e])
        pos, idx = S["chunk"], S["in_chunk"]; S["in_chunk"] += 1
        s.record()
        g = S["graph"]
        out = g(pos, idx, kw, _fwd) if g is not None else _fwd(*a, **kw)
        e.record()
        if S["in_chunk"] == 5:
            torch.cuda.synchronize(); S["t_chunk"].append(time.perf_counter())
        return out
    pipe.generator.forward = fwd

    # ---------------- manual CUDA graph machinery ----------------
    class Graphs:
        def __init__(self): self.g = {}; self.pool = None; self.static = {}
        self_replays = 0
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
            Graphs.self_replays += 1
            return out

    G = Graphs()
    cap_mode = {"on": False}

    def graph_hook(pos, idx, kw, fwd0):
        return G.capture(pos, idx, kw, fwd0) if cap_mode["on"] else G.replay(pos, idx, kw, fwd0)

    # ---------------- mode setup ----------------
    def setup(mode):
        if mode in FINAL_MODES:
            cf = FINAL_MODES[mode]
            patch_flags.set_enabled(True)
            patch_flags.set_perf(cf["perf"])
            patch_flags.set_attn_splits(cf["splits"])
            patch_flags.set_attn_split_rule(J10_RULE if cf["rule"] else None)
            patch_flags.set_fused_merge(bool(cf.get("fused", False)))
            if cf.get("fused", False):
                # Triton autotuning must not run inside a CUDA-graph capture: warm the kernel once here
                from wan.modules.fused_merge_outproj import prewarm
                prewarm(cf["splits"]); print("[J3b] fused_merge_outproj prewarmed", flush=True)
            pipe.generator.model = base_model
            S["graph"] = None
            if cf["compile"]:
                if "nocg" not in compiled:
                    compiled["nocg"] = torch.compile(base_model,
                        mode="max-autotune-no-cudagraphs", dynamic=False)
                pipe.generator.model = compiled["nocg"]
            if cf["graphs"]:
                S["graph"] = graph_hook
            return
        patch_flags.set_attn_split_rule(None)
        patch_flags.set_fused_merge(False)
        patch_flags.set_enabled(mode != "eager-original")
        if args.perf == "per-mode":
            # perf patches 8/9+9/9 are A/B-ed by mode name; safe for eager only
            patch_flags.set_perf(mode.endswith("-perf"))
        else:
            patch_flags.set_perf(args.perf == "on")
        patch_flags.set_attn_splits(args.attn_splits)
        mode = mode[:-5] if mode.endswith("-perf") else mode
        pipe.generator.model = base_model
        S["graph"] = None
        if mode == "compile_nocg":
            if "nocg" not in compiled:
                compiled["nocg"] = torch.compile(base_model, mode="max-autotune-no-cudagraphs",
                                                 dynamic=False)
            pipe.generator.model = compiled["nocg"]
        elif mode == "compile_cg":
            MARK_STEP["on"] = True
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
            S["graph"] = graph_hook
        elif mode == "manual_graph_compiled":
            # spec 10.2 fallback: manual per-position graphs over the kernels
            # Inductor generated for compile_nocg ("fusion + graphs" arm).
            if "nocg" not in compiled:
                compiled["nocg"] = torch.compile(base_model, mode="max-autotune-no-cudagraphs",
                                                 dynamic=False)
            pipe.generator.model = compiled["nocg"]
            S["graph"] = graph_hook

    def one_video(prompt, seed, mode, capture=False):
        set_seed(seed)
        S.update({"chunk": -1, "in_chunk": 0, "t_gen": None, "t_chunk": [], "ev": []})
        torch.cuda.reset_peak_memory_stats()
        noise = torch.randn([1, 21, 16, 60, 104], device=dev, dtype=torch.bfloat16)
        if mode in GRAPH_MODES or mode == "compile_cg":
            cond = pipe.text_encoder(text_prompts=[prompt])
            if pipe.kv_cache1 is None:
                pipe._initialize_kv_cache(1, torch.bfloat16, dev)
                pipe._initialize_crossattn_cache(1, torch.bfloat16, dev)
            sg = S["graph"]; S["graph"] = None
            with contextlib.redirect_stdout(io.StringIO()):
                prime_crossattn_cache(pipe, cond, noise)
            S["graph"] = sg
            # the priming pass goes through the timing wrapper; discard its
            # timestamps so chunk 0 is not charged for it (spec 8.2: t_gen is
            # taken immediately before the FIRST generator pass of the video)
            S.update({"chunk": -1, "in_chunk": 0, "t_gen": None, "t_chunk": [], "ev": []})
        cap_mode["on"] = capture
        buf = io.StringIO(); t0 = time.perf_counter()
        with contextlib.redirect_stdout(buf):
            video, _ = pipe.inference(noise=noise, text_prompts=[prompt], return_latents=True,
                                      initial_latent=None, low_memory=True)
        torch.cuda.synchronize(); cap_mode["on"] = False
        wall = time.perf_counter() - t0
        passes = {}
        for c, i, s, e in S["ev"]:
            passes.setdefault(c, []).append(s.elapsed_time(e))
        ch, rs, prev = [], [], S["t_gen"]
        for i, t in enumerate(S["t_chunk"]):
            T = (t - prev) * 1e3; ch.append(T)
            ps = sum(passes.get(i, [])); rs.append(100 * (T - ps) / T if T else float("nan")); prev = t
        return {"video": video, "chunk_ms": ch, "residual_pct": rs, "wall_s": wall,
                "peak_reserved_gb": torch.cuda.max_memory_reserved() / GB}

    rep = {"modes": {}, "torch": torch.__version__, "triton": triton.__version__,
           "flash_attn": flash_attn.__version__, "failures": {}}
    ok_modes = []

    # ---- warmup per mode (capture happens here for manual_graph) ----
    for m in modes:
        try:
            setup(m)
            t0 = time.perf_counter()
            w = one_video(prompts[0], 999, m, capture=(m in GRAPH_MODES))
            rep["modes"][m] = {"warmup_s": round(time.perf_counter() - t0, 2),
                               "warmup_chunk_ms": [round(x, 2) for x in w["chunk_ms"]],
                               "videos": []}
            if m in GRAPH_MODES:
                rep["modes"][m]["graphs_captured"] = len(G.g)
            print(f"[J3b] {m}: warmup ok ({rep['modes'][m]['warmup_s']}s)"
                  f"{' graphs=' + str(len(G.g)) if m in GRAPH_MODES else ''}", flush=True)
            ok_modes.append(m); del w
        except Exception as ex:
            rep["failures"][m] = {"error_type": type(ex).__name__, "error": str(ex)[:1200],
                                  "traceback_tail": traceback.format_exc()[-2500:]}
            print(f"[J3b] {m}: WARMUP/CAPTURE FAILED: {type(ex).__name__}: {str(ex)[:300]}", flush=True)

    # ---- measured videos, INTERLEAVED by video (spec 10.2) ----
    for v in range(args.n_videos):
        for m in ok_modes:
            setup(m)
            _t0 = time.time()
            r = one_video(prompts[v], 1000 + v, m)
            rep["modes"][m]["videos"].append({
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
        vids = rep["modes"][m]["videos"]
        n = len(vids[0]["chunk_ms"])
        pp = {i: summarize([x["chunk_ms"][i] for x in vids]) for i in range(n)}
        rep["modes"][m]["per_position"] = pp
        rep["modes"][m]["fit"] = fit_aK([pp[i]["median"] for i in range(n)])
        rep["modes"][m]["peak_reserved_gb"] = summarize([x["peak_reserved_gb"] for x in vids])
    for gm in GRAPH_MODES & set(ok_modes):
        rep["modes"][gm]["total_replays"] = Graphs.self_replays
        rep["modes"][gm]["expected_replays"] = 35 * args.n_videos
        rep["modes"][gm]["one_replay_per_pass"] = (Graphs.self_replays == 35 * args.n_videos)
    if "eager-original" in ok_modes:
        a0 = rep["modes"]["eager-original"]["fit"]["a_ms"]
        for m in ok_modes:
            rep["modes"][m]["intercept_delta_vs_eager_original_ms"] = \
                rep["modes"][m]["fit"]["a_ms"] - a0

    with open(os.path.join(args.out, "j3b.json"), "w") as f:
        json.dump(rep, f, indent=2)
    print("\n[J3b] summary:")
    for m in ok_modes:
        f_ = rep["modes"][m]["fit"]
        print(f"  {m:15s} a={f_['a_ms']:8.2f} b={f_['b_ms_per_1k_K']:.4f} "
              f"R2={f_['r2']:.5f} Δa={rep['modes'][m].get('intercept_delta_vs_eager_original_ms', 0):8.2f}")
    for m, e in rep["failures"].items():
        print(f"  {m:15s} FAILED: {e['error_type']}: {e['error'][:120]}")


if __name__ == "__main__":
    main()
