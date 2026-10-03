#!/usr/bin/env python3
"""
CUDA-graph capture probe: can one generator pass of the unpatched model be captured at all?

Warms three eager passes, records the eager kernel-launch count per pass (the
baseline for the J3 capture proof), attempts a torch.cuda.CUDAGraph capture of
one pass and lists the .item() host-sync sites in wan/modules/causal_model.py
that block capture.

Writes the JSON named by $J3_PROBE_OUT (default j3_probe.json).
"""
import json
import os
import subprocess
import sys
import traceback

import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.getcwd())
from pipeline import CausalInferencePipeline
from demo_utils.memory import gpu, DynamicSwapInstaller
from utils.misc import set_seed

dev = torch.device("cuda")
torch.set_grad_enabled(False)
set_seed(0)

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
cond = pipe.text_encoder(text_prompts=[prompt])

pipe._initialize_kv_cache(1, torch.bfloat16, dev)
pipe._initialize_crossattn_cache(1, torch.bfloat16, dev)

noisy = torch.randn([1, 3, 16, 60, 104], device=dev, dtype=torch.bfloat16)
ts = (torch.ones([1, 3], device=dev, dtype=torch.int64) * 1000)

def one_pass():
    return pipe.generator(noisy_image_or_video=noisy, conditional_dict=cond,
                          timestep=ts, kv_cache=pipe.kv_cache1,
                          crossattn_cache=pipe.crossattn_cache, current_start=0)

report = {}

# ---- eager warm + launch count -------------------------------------------
for _ in range(3):
    one_pass()
torch.cuda.synchronize()

from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    one_pass()
    torch.cuda.synchronize()
kern = [e for e in prof.key_averages() if e.device_time_total > 0]
n_launch = sum(e.count for e in kern)
report["eager_kernel_launches_per_pass"] = int(n_launch)
report["eager_distinct_kernels"] = len(kern)
print(f"[probe] eager: {n_launch} kernel launches per pass, "
      f"{len(kern)} distinct kernels")
print("[probe] top kernels by total device time:")
for e in sorted(kern, key=lambda x: -x.device_time_total)[:12]:
    print(f"    {e.count:5d}  {e.device_time_total/1e3:9.3f} ms  {e.key[:78]}")

# ---- capture attempt ------------------------------------------------------
print("\n[probe] attempting torch.cuda.CUDAGraph capture of one pass ...")
try:
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            one_pass()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        one_pass()
    torch.cuda.synchronize()
    report["capture"] = "SUCCESS"
    print("[probe] CAPTURE SUCCEEDED")
except Exception as e:
    report["capture"] = "FAILED"
    report["capture_error_type"] = type(e).__name__
    report["capture_error"] = str(e)[:1500]
    tb = traceback.format_exc()
    report["capture_traceback_tail"] = tb[-2500:]
    print(f"[probe] CAPTURE FAILED: {type(e).__name__}: {str(e)[:400]}")
    print("\n[probe] traceback tail:\n" + "\n".join(tb.splitlines()[-25:]))

# ---- count the host-sync sites that block capture -------------------------
gr = subprocess.run(["grep", "-rn", r"\.item()", "wan/modules/causal_model.py"],
                    capture_output=True, text=True).stdout.strip()
report["item_call_sites_causal_model"] = gr.splitlines()
print("\n[probe] .item() sites in wan/modules/causal_model.py:")
print(gr if gr else "  (none)")

with open(os.environ.get("J3_PROBE_OUT", "j3_probe.json"), "w") as f:
    json.dump(report, f, indent=2)
