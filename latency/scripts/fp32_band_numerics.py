#!/usr/bin/env python3
"""
Tier-2 numerics: single-pass agreement vs an fp32 eager reference, on identical inputs and cache state.

Band = |eager-bf16 - fp32|. A kernel-changing mode passes if its own error vs
fp32 is within 2x that band on BOTH max_abs and mean_abs.

Limitation, recorded: FlashAttention-2 accepts only fp16/bf16, so the fp32
reference is fp32 everywhere EXCEPT inside the attention kernel itself, which
casts to bf16 in both arms. The band therefore characterises GEMM, epilogue and
elementwise reduction-order differences -- which is exactly what compilation
changes -- and not attention-internal precision.

Writes the JSON named by $J3B_FP32_OUT (default j3b_fp32.json): the band and,
per candidate mode, its error, the ratio to the band and the pass flag.
"""
import json, os, sys
import torch
from omegaconf import OmegaConf
sys.path.insert(0, os.getcwd())
from pipeline import CausalInferencePipeline
from demo_utils.memory import gpu, DynamicSwapInstaller
from utils.misc import set_seed
from wan import patch_flags

dev = torch.device("cuda"); torch.set_grad_enabled(False)
patch_flags.set_enabled(True)
cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"),
                      OmegaConf.load("configs/self_forcing_dmd.yaml"))
pipe = CausalInferencePipeline(cfg, device=dev)
sd = torch.load("checkpoints/self_forcing_dmd.pt", map_location="cpu")
pipe.generator.load_state_dict(sd["generator_ema"]); del sd
DynamicSwapInstaller.install_model(pipe.text_encoder, device=gpu)
pipe.generator.to(device=gpu); pipe.vae.to(device=gpu)
with open("prompts/MovieGenVideoBench_extended.txt", encoding="utf-8") as f:
    prompt = f.readline().rstrip()

pipe._initialize_kv_cache(1, torch.float32, dev)
pipe._initialize_crossattn_cache(1, torch.float32, dev)
cond = pipe.text_encoder(text_prompts=[prompt])
set_seed(1000)
noisy32 = torch.randn([1, 3, 16, 60, 104], device=dev, dtype=torch.float32)
ts = torch.ones([1, 3], device=dev, dtype=torch.int64) * 1000


def reset_caches(dtype):
    for b in pipe.kv_cache1:
        b["k"] = torch.zeros_like(b["k"], dtype=dtype); b["v"] = torch.zeros_like(b["v"], dtype=dtype)
        b["global_end_index"] = 0; b["local_end_index"] = 0
    for c in pipe.crossattn_cache:
        c["is_init"] = False


def one_pass(noisy):
    o = pipe.generator(noisy_image_or_video=noisy, conditional_dict=cond, timestep=ts,
                       kv_cache=pipe.kv_cache1, crossattn_cache=pipe.crossattn_cache,
                       current_start=0)
    return (o[1] if isinstance(o, (tuple, list)) else o).float().clone()


def abs_error(x, ref):
    d = (x - ref).abs()
    return {"max_abs": d.max().item(), "mean_abs": d.mean().item()}


# fp32 reference
pipe.generator.to(dtype=torch.float32)
cond32 = {k: (v.float() if torch.is_tensor(v) else v) for k, v in cond.items()}
cond = cond32
reset_caches(torch.float32); ref = one_pass(noisy32)
torch.cuda.synchronize()

# eager bf16 -> the inherent band
pipe.generator.to(dtype=torch.bfloat16)
cond = {k: (v.bfloat16() if torch.is_tensor(v) else v) for k, v in cond32.items()}
noisyb = noisy32.bfloat16()
reset_caches(torch.bfloat16); e_bf16 = one_pass(noisyb)
band = abs_error(e_bf16, ref)
torch.cuda.synchronize()

out = {"band_eager_bf16_vs_fp32": band, "modes": {}}
print(f"[tier2] inherent bf16 band vs fp32: max_abs={band['max_abs']:.6g} "
      f"mean_abs={band['mean_abs']:.6g}")

import torch._dynamo as dynamo
dynamo.config.cache_size_limit = 64
base = pipe.generator.model
for name, mode in (("compile_nocg", "max-autotune-no-cudagraphs"),):
    try:
        pipe.generator.model = torch.compile(base, mode=mode, dynamic=False)
        reset_caches(torch.bfloat16); one_pass(noisyb)                  # compile/warm
        reset_caches(torch.bfloat16); y = one_pass(noisyb)
        e = abs_error(y, ref)
        ok = (e["max_abs"] <= 2 * band["max_abs"]) and (e["mean_abs"] <= 2 * band["mean_abs"])
        out["modes"][name] = {**e, "within_2x_band": ok,
                              "ratio_max": e["max_abs"] / band["max_abs"],
                              "ratio_mean": e["mean_abs"] / band["mean_abs"]}
        print(f"[tier2] {name}: max_abs={e['max_abs']:.6g} ({e['max_abs']/band['max_abs']:.2f}x) "
              f"mean_abs={e['mean_abs']:.6g} ({e['mean_abs']/band['mean_abs']:.2f}x) "
              f"-> {'PASS' if ok else 'FAIL'}")
    except Exception as ex:
        out["modes"][name] = {"error": str(ex)[:400]}
        print(f"[tier2] {name}: ERROR {str(ex)[:200]}")
    finally:
        pipe.generator.model = base

with open(os.environ.get("J3B_FP32_OUT", "j3b_fp32.json"), "w") as f:
    json.dump(out, f, indent=2)
