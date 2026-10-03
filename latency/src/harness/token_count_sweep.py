#!/usr/bin/env python3
"""
Token-count sweep: chunk time vs. history length K for num_frame_per_block in {1, 2, 3}.

Rebuilds the pipeline for each num_frame_per_block (tokens per chunk 1560,
3120, 4680), runs the per-position timing protocol under compile_nocg with the
perf patches on, fits t = a + b*K per setting and prints the comparison at
matched K (9360, 18720, 28080 tokens), where the three settings can be
compared at equal attention work.

Writes <out>/j6_sweep.json: per setting, K list, per-position medians and fit.
"""
import argparse, contextlib, io, json, os, statistics as st, sys, time
import numpy as np, torch
sys.path.insert(0, os.getcwd())

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--n_videos", type=int, default=3)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

from omegaconf import OmegaConf
from pipeline import CausalInferencePipeline
from demo_utils.memory import gpu, DynamicSwapInstaller
from utils.misc import set_seed
from wan import patch_flags
from wan.modules.model import prebuild_sinusoid_cache
import torch._dynamo as dynamo

dev = torch.device("cuda"); torch.set_grad_enabled(False)
patch_flags.set_enabled(True); patch_flags.set_perf(True)
dynamo.config.cache_size_limit = 256
dynamo.config.accumulated_cache_size_limit = 1024
with open("prompts/MovieGenVideoBench_extended.txt", encoding="utf-8") as f:
    prompts = [f.readline().rstrip() for _ in range(args.n_videos)]

FRAMES = {1: 21, 2: 20, 3: 21}     # 20 for nfpb=2: the pipeline asserts frames % nfpb == 0
results = {}
for nfpb in (1, 2, 3):
    cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"),
                          OmegaConf.load("configs/self_forcing_dmd.yaml"))
    cfg.num_frame_per_block = nfpb
    pipe = CausalInferencePipeline(cfg, device=dev)
    sd = torch.load("checkpoints/self_forcing_dmd.pt", map_location="cpu")
    pipe.generator.load_state_dict(sd["generator_ema"]); del sd
    pipe = pipe.to(dtype=torch.bfloat16)
    DynamicSwapInstaller.install_model(pipe.text_encoder, device=gpu)
    pipe.generator.to(device=gpu); pipe.vae.to(device=gpu)
    prebuild_sinusoid_cache(256, dev)
    pipe.generator.model = torch.compile(pipe.generator.model,
                                         mode="max-autotune-no-cudagraphs", dynamic=False)
    first_ts = float(pipe.denoising_step_list.tolist()[0])
    timing = {"chunk": -1, "in": 0, "t0": None, "tc": []}
    original_forward = pipe.generator.forward

    def timed_forward(*x, **kw):
        ts = kw.get("timestep")
        tv = float(ts.flatten()[0].item()) if ts is not None else float("nan")
        if tv == first_ts: timing["chunk"] += 1; timing["in"] = 0
        if timing["t0"] is None: torch.cuda.synchronize(); timing["t0"] = time.perf_counter()
        timing["in"] += 1
        o = original_forward(*x, **kw)
        if timing["in"] == 5:
            torch.cuda.synchronize(); timing["tc"].append(time.perf_counter())
        return o
    pipe.generator.forward = timed_forward

    def one_video(prompt, seed):
        set_seed(seed); timing.update({"chunk": -1, "in": 0, "t0": None, "tc": []})
        noise = torch.randn([1, FRAMES[nfpb], 16, 60, 104], device=dev, dtype=torch.bfloat16)
        with contextlib.redirect_stdout(io.StringIO()):
            pipe.inference(noise=noise, text_prompts=[prompt], return_latents=True,
                           initial_latent=None, low_memory=True)
        torch.cuda.synchronize()
        out, prev = [], timing["t0"]
        for t in timing["tc"]:
            out.append((t - prev) * 1e3); prev = t
        return out

    one_video(prompts[0], 999)                              # warmup video
    vids = [one_video(prompts[i], 1000 + i) for i in range(args.n_videos)]
    n = min(len(v) for v in vids)
    med = [st.median([v[i] for v in vids]) for i in range(n)]
    K = [(i + 1) * nfpb * 1560 for i in range(n)]
    A = np.vstack([np.ones(n), np.array(K, float)]).T
    (aa, bb), *_ = np.linalg.lstsq(A, np.array(med, float), rcond=None)
    pred = aa + bb * np.array(K, float); tt = np.array(med, float)
    r2 = 1 - ((tt - pred) ** 2).sum() / ((tt - tt.mean()) ** 2).sum()
    results[nfpb] = {"tokens_per_chunk": nfpb * 1560, "frames": FRAMES[nfpb],
                     "chunks": n, "K": K, "median_ms": [round(x, 2) for x in med],
                     "a_ms": float(aa), "b_ms_per_1k_K": float(bb * 1000), "r2": float(r2)}
    print(f"[sweep] nfpb={nfpb} tokens/chunk={nfpb*1560} chunks={n} "
          f"a={aa:.2f} ms b={bb*1000:.4f} ms/1k-K R2={r2:.5f}", flush=True)
    del pipe; torch.cuda.empty_cache()

print("\n=== matched-K comparison (K equal across token counts) ===")
print(f"{'K':>8}" + "".join(f"{'nfpb='+str(n):>14}" for n in (1, 2, 3)))
for Km in (9360, 18720, 28080):
    row = f"{Km:>8}"
    for n in (1, 2, 3):
        r = results[n]
        row += f"{(r['median_ms'][r['K'].index(Km)] if Km in r['K'] else float('nan')):>14.1f}"
    print(row)
print("\nper-1000-tokens-of-chunk at matched K (chunk time / tokens per chunk):")
for Km in (9360, 18720, 28080):
    row = f"{Km:>8}"
    for n in (1, 2, 3):
        r = results[n]
        v = r['median_ms'][r['K'].index(Km)] / (n * 1.56) if Km in r['K'] else float('nan')
        row += f"{v:>14.2f}"
    print(row)
json.dump(results, open(os.path.join(args.out, "j6_sweep.json"), "w"), indent=2)
