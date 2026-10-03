#!/usr/bin/env python3
"""
Visual check of compile numerics: one video eager and one under compile_nocg.

Same prompt and seed; saves a 2x2 frame grid of each (frames 0, 26, 53, 80)
to <out>/j3_frames_eager.png and <out>/j3_frames_compile_nocg.png and prints
the per-frame PSNR summary between the two videos.
"""
import argparse, json, os, sys
import numpy as np, torch
from omegaconf import OmegaConf
sys.path.insert(0, os.getcwd())
from pipeline import CausalInferencePipeline
from demo_utils.memory import gpu, DynamicSwapInstaller
from utils.misc import set_seed

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
args = ap.parse_args()
dev = torch.device("cuda"); torch.set_grad_enabled(False)
cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"),
                      OmegaConf.load("configs/self_forcing_dmd.yaml"))
pipe = CausalInferencePipeline(cfg, device=dev)
sd = torch.load("checkpoints/self_forcing_dmd.pt", map_location="cpu")
pipe.generator.load_state_dict(sd["generator_ema"]); del sd
pipe = pipe.to(dtype=torch.bfloat16)
DynamicSwapInstaller.install_model(pipe.text_encoder, device=gpu)
pipe.generator.to(device=gpu); pipe.vae.to(device=gpu)
with open("prompts/MovieGenVideoBench_extended.txt", encoding="utf-8") as f:
    prompt = f.readline().rstrip()

def generate_video():
    set_seed(1000)
    noise = torch.randn([1, 21, 16, 60, 104], device=dev, dtype=torch.bfloat16)
    v, _ = pipe.inference(noise=noise, text_prompts=[prompt], return_latents=True,
                          initial_latent=None, low_memory=True)
    a = (255.0 * v.float().clamp(0, 1)).round().clamp(0, 255)
    return a[0].permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()

def save_frame_grid(arr, path, idxs=(0, 26, 53, 80)):
    import imageio.v2 as iio
    fs = [arr[i] for i in idxs]
    top = np.concatenate(fs[:2], axis=1); bot = np.concatenate(fs[2:], axis=1)
    iio.imwrite(path, np.concatenate([top, bot], axis=0))
    print(f"  wrote {path}")

e = generate_video(); save_frame_grid(e, os.path.join(args.out, "j3_frames_eager.png"))
import torch._dynamo as dynamo
dynamo.config.cache_size_limit = 64
pipe.generator.model = torch.compile(pipe.generator.model,
                                     mode="max-autotune-no-cudagraphs", dynamic=False)
c = generate_video(); save_frame_grid(c, os.path.join(args.out, "j3_frames_compile_nocg.png"))
d = (e.astype(np.float64) - c.astype(np.float64))
mse = (d ** 2).mean(axis=(1, 2, 3))
psnr = [float("inf") if m == 0 else 20 * np.log10(255 / np.sqrt(m)) for m in mse]
fin = [p for p in psnr if np.isfinite(p)]
print(json.dumps({"psnr_min": min(fin) if fin else None,
                  "psnr_median": float(np.median(fin)) if fin else None,
                  "identical": int(sum(1 for p in psnr if not np.isfinite(p)))}, indent=2))
