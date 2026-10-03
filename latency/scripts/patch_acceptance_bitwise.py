#!/usr/bin/env python3
"""
Tier-1 acceptance test for the capture-enabling patch: eager-patched vs eager-original.

Generates each of the first five prompts twice with the same seed, once with
the patch flags off and once on, and compares the uint8 videos by sha256.
J2 proved the eager path is bitwise deterministic, so this is exact, not a
tolerance: one differing byte rejects the patch (exit code 7).

Writes the JSON named by $J3B_ACCEPT_OUT (default j3b_accept.json).
"""
import contextlib, hashlib, io, json, os, sys
import numpy as np, torch
from omegaconf import OmegaConf
sys.path.insert(0, os.getcwd())
from pipeline import CausalInferencePipeline
from demo_utils.memory import gpu, DynamicSwapInstaller
from utils.misc import set_seed
from wan import patch_flags

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
    prompts = [f.readline().rstrip() for _ in range(5)]


def generate_video(prompt, seed):
    set_seed(seed)
    noise = torch.randn([1, 21, 16, 60, 104], device=dev, dtype=torch.bfloat16)
    with contextlib.redirect_stdout(io.StringIO()):
        v, _ = pipe.inference(noise=noise, text_prompts=[prompt],
                              return_latents=True, initial_latent=None,
                              low_memory=True)
    a = (255.0 * v.float().clamp(0, 1)).round().clamp(0, 255)
    return a[0].permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()


rows, all_ok = [], True
for i, p in enumerate(prompts):
    patch_flags.set_enabled(False); a = generate_video(p, 1000 + i)
    patch_flags.set_enabled(True);  b = generate_video(p, 1000 + i)
    ha = hashlib.sha256(a.tobytes()).hexdigest()
    hb = hashlib.sha256(b.tobytes()).hexdigest()
    same = ha == hb
    nd = int((a != b).sum())
    maxd = int(np.abs(a.astype(int) - b.astype(int)).max()) if not same else 0
    rows.append({"prompt": i, "seed": 1000 + i, "sha_original": ha[:16],
                 "sha_patched": hb[:16], "byte_identical": same,
                 "differing_bytes": nd, "max_abs_diff": maxd})
    all_ok &= same
    print(f"  prompt {i}: original {ha[:16]}  patched {hb[:16]}  "
          f"{'IDENTICAL' if same else f'DIFFERS ({nd} bytes, max {maxd})'}", flush=True)

out = {"all_byte_identical": all_ok, "rows": rows}
with open(os.environ.get("J3B_ACCEPT_OUT", "j3b_accept.json"), "w") as f:
    json.dump(out, f, indent=2)
print(f"\nPATCH ACCEPTANCE: {'PASS' if all_ok else 'FAIL'}")
sys.exit(0 if all_ok else 7)
