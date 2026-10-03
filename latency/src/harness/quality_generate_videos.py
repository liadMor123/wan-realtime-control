#!/usr/bin/env python3
"""
Generate the J9 quality-check videos for one arm.

Arms: eager_a = eager-original with seeds 1000+i, final = the integrated
configuration (patches + perf + split-KV 4 + compile_nocg) with the same seeds,
eager_b = eager-original with seeds 2000+i (the seed-change control). Prompts
are the first --n_prompts lines of the benchmark file, in file order, no
curation. Descriptive only -- no pass/fail.

Writes <out>/<arm>_p<i>.npy (uint8 video) and <out>/j9_prompts.json.
"""
import argparse, contextlib, io, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.getcwd())

ap = argparse.ArgumentParser()
ap.add_argument("--arm", choices=["eager_a", "final", "eager_b"], required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n_prompts", type=int, default=10)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

from omegaconf import OmegaConf
from pipeline import CausalInferencePipeline
from pipeline.causal_inference import prime_crossattn_cache
from demo_utils.memory import gpu, DynamicSwapInstaller
from utils.misc import set_seed
from wan import patch_flags
from wan.modules.model import prebuild_sinusoid_cache

dev = torch.device("cuda"); torch.set_grad_enabled(False)
FINAL = args.arm == "final"
patch_flags.set_enabled(FINAL)           # eager-original = unpatched semantics
patch_flags.set_perf(FINAL)
patch_flags.set_attn_splits(4 if FINAL else 0)
SEED0 = 2000 if args.arm == "eager_b" else 1000

cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"),
                      OmegaConf.load("configs/self_forcing_dmd.yaml"))
pipe = CausalInferencePipeline(cfg, device=dev)
sd = torch.load("checkpoints/self_forcing_dmd.pt", map_location="cpu")
pipe.generator.load_state_dict(sd["generator_ema"]); del sd
pipe = pipe.to(dtype=torch.bfloat16)
DynamicSwapInstaller.install_model(pipe.text_encoder, device=gpu)
pipe.generator.to(device=gpu); pipe.vae.to(device=gpu)
prebuild_sinusoid_cache(256, dev)
if FINAL:
    import torch._dynamo as dynamo
    dynamo.config.cache_size_limit = 256
    pipe.generator.model = torch.compile(pipe.generator.model,
                                         mode="max-autotune-no-cudagraphs", dynamic=False)

with open("prompts/MovieGenVideoBench_extended.txt", encoding="utf-8") as f:
    prompts = [f.readline().rstrip() for _ in range(args.n_prompts)]
json.dump({"prompts": prompts, "source": "prompts/MovieGenVideoBench_extended.txt",
           "rule": "first ten lines, file order, no curation"},
          open(os.path.join(args.out, "j9_prompts.json"), "w"), indent=2)

for i, p in enumerate(prompts):
    set_seed(SEED0 + i)
    noise = torch.randn([1, 21, 16, 60, 104], device=dev, dtype=torch.bfloat16)
    with contextlib.redirect_stdout(io.StringIO()):
        v, _ = pipe.inference(noise=noise, text_prompts=[p], return_latents=True,
                              initial_latent=None, low_memory=True)
    arr = (255.0 * v.float().clamp(0, 1)).round().clamp(0, 255)
    arr = arr[0].permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()
    np.save(os.path.join(args.out, f"{args.arm}_p{i}.npy"), arr)
    print(f"[j9:{args.arm}] prompt {i} seed {SEED0+i} -> {arr.shape}", flush=True)
    del v, arr
    torch.cuda.empty_cache()
print(f"[j9:{args.arm}] done", flush=True)
