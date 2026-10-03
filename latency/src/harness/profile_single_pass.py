#!/usr/bin/env python3
"""
Profiler target: generator chunk(s) at a chosen position, bracketed by cudaProfilerStart/Stop.

Advances the KV cache to --position exactly as inference does (unprofiled),
warms the compiled kernels for that position without advancing the cache, then
runs --chunks full chunks (5 passes each) inside an explicit CUDA profiler
range so ncu / nsys see only that region. Optional NVTX ranges per pass.
Writes nothing itself; the Slurm scripts collect the ncu CSV / nsys trace.
"""
import argparse, contextlib, io, os, sys
import torch
from omegaconf import OmegaConf
sys.path.insert(0, os.getcwd())
from pipeline import CausalInferencePipeline
from pipeline.causal_inference import prime_crossattn_cache
from demo_utils.memory import gpu, DynamicSwapInstaller
from utils.misc import set_seed
from wan import patch_flags
from wan.modules.model import prebuild_sinusoid_cache

ap = argparse.ArgumentParser()
ap.add_argument("--position", type=int, default=3)
ap.add_argument("--mode", choices=["compile_nocg", "eager"], default="compile_nocg")
ap.add_argument("--chunks", type=int, default=1, help="number of full chunks to run in the profiled region")
ap.add_argument("--nvtx", action="store_true")
args = ap.parse_args()

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
    prompt = f.readline().rstrip()
pipe._initialize_kv_cache(1, torch.bfloat16, dev)
pipe._initialize_crossattn_cache(1, torch.bfloat16, dev)
set_seed(1000)
noise = torch.randn([1, 21, 16, 60, 104], device=dev, dtype=torch.bfloat16)
cond = pipe.text_encoder(text_prompts=[prompt])
with contextlib.redirect_stdout(io.StringIO()):
    prime_crossattn_cache(pipe, cond, noise)

if args.mode == "compile_nocg":
    import torch._dynamo as dynamo
    dynamo.config.cache_size_limit = 64
    pipe.generator.model = torch.compile(pipe.generator.model,
                                         mode="max-autotune-no-cudagraphs", dynamic=False)

dsl = [float(x) for x in pipe.denoising_step_list.tolist()]


def run_chunk(p, x):
    for j, tstep in enumerate(dsl + [0.0]):
        ts = torch.ones([1, 3], device=dev, dtype=torch.int64) * int(tstep)
        if args.nvtx:
            torch.cuda.nvtx.range_push(f"pass{j}")
        out = pipe.generator(noisy_image_or_video=x, conditional_dict=cond, timestep=ts,
                             kv_cache=pipe.kv_cache1, crossattn_cache=pipe.crossattn_cache,
                             current_start=p * 3 * 1560)
        if args.nvtx:
            torch.cuda.nvtx.range_pop()
        if isinstance(out, (tuple, list)):
            x = out[1]
    return x


# advance the cache to the requested position (NOT profiled -- warmup region)
for b in pipe.kv_cache1:
    b["k"].zero_(); b["v"].zero_(); b["global_end_index"] = 0; b["local_end_index"] = 0
for p in range(args.position):
    run_chunk(p, noise[:, p * 3:(p + 1) * 3])
torch.cuda.synchronize()

# warm the compiled kernels for THIS position without advancing the cache
snap = [(b["global_end_index"], b["local_end_index"]) for b in pipe.kv_cache1]
for _ in range(2):
    for b, (g, l) in zip(pipe.kv_cache1, snap):
        b["global_end_index"], b["local_end_index"] = g, l
    run_chunk(args.position, noise[:, args.position * 3:(args.position + 1) * 3])
for b, (g, l) in zip(pipe.kv_cache1, snap):
    b["global_end_index"], b["local_end_index"] = g, l
torch.cuda.synchronize()

torch.cuda.cudart().cudaProfilerStart()
for c in range(args.chunks):
    run_chunk(args.position, noise[:, args.position * 3:(args.position + 1) * 3])
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
print(f"[profile_single_pass] profiled {args.chunks} chunk(s) at position {args.position}, mode {args.mode}")
