#!/usr/bin/env python3
"""
Tier-1 acceptance test for the cache-write (perf) intervention: eager-patched-perf vs eager-patched.

Generates each of the first five prompts twice with the same seed, with the
perf patches off and on, and compares the uint8 videos by sha256. One
differing byte rejects the intervention (exit code 7).

Writes the JSON named by $J6C_OUT (default j6c_accept.json).
"""
import contextlib, hashlib, io, json, os, sys
import numpy as np, torch
from omegaconf import OmegaConf
sys.path.insert(0, os.getcwd())
from pipeline import CausalInferencePipeline
from demo_utils.memory import gpu, DynamicSwapInstaller
from utils.misc import set_seed
from wan import patch_flags
from wan.modules.model import prebuild_sinusoid_cache

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
prebuild_sinusoid_cache(256, dev)
with open("prompts/MovieGenVideoBench_extended.txt", encoding="utf-8") as f:
    prompts = [f.readline().rstrip() for _ in range(5)]

def generate_video(prompt, seed):
    set_seed(seed)
    noise = torch.randn([1, 21, 16, 60, 104], device=dev, dtype=torch.bfloat16)
    with contextlib.redirect_stdout(io.StringIO()):
        v, _ = pipe.inference(noise=noise, text_prompts=[prompt], return_latents=True,
                              initial_latent=None, low_memory=True)
    a = (255.0 * v.float().clamp(0, 1)).round().clamp(0, 255)
    return a[0].permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()

rows, ok = [], True
for i, p in enumerate(prompts):
    patch_flags.set_perf(False); a = generate_video(p, 1000 + i)
    patch_flags.set_perf(True);  b = generate_video(p, 1000 + i)
    ha = hashlib.sha256(a.tobytes()).hexdigest(); hb = hashlib.sha256(b.tobytes()).hexdigest()
    same = ha == hb; ok &= same
    nd = int((a != b).sum()); mx = int(np.abs(a.astype(int)-b.astype(int)).max()) if not same else 0
    rows.append({"prompt": i, "baseline": ha[:16], "perf": hb[:16],
                 "byte_identical": same, "differing_bytes": nd, "max_abs_diff": mx})
    print(f"  prompt {i}: baseline {ha[:16]}  perf {hb[:16]}  "
          f"{'IDENTICAL' if same else f'DIFFERS ({nd} bytes, max {mx})'}", flush=True)
json.dump({"all_byte_identical": ok, "rows": rows},
          open(os.environ.get("J6C_OUT", "j6c_accept.json"), "w"), indent=2)
print(f"\nCACHE-WRITE INTERVENTION ACCEPTANCE: {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 7)
