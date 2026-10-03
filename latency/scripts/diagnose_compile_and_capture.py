#!/usr/bin/env python3
"""
J3 diagnostics on the unpatched model: compile numerics and CUDA-graph capture blockers.

--part numerics  one generator pass, eager vs torch.compile(compile_nocg), on
                 identical inputs and cache state; reports max/mean abs
                 difference of the denoised prediction and of the block-0 KV
                 cache, and whether the divergence appears on the first pass.
--part capture   installs the two capture-safe shims (cached sinusoid table,
                 host-side KV index), attempts a torch.cuda.CUDAGraph capture
                 of one pass and greps the inference path for the remaining
                 capture-hostile constructs (.item(), torch.arange, ...).

Writes the JSON named by --out.
"""
import argparse, json, os, subprocess, sys, traceback
import torch
from omegaconf import OmegaConf
sys.path.insert(0, os.getcwd())
from pipeline import CausalInferencePipeline
from demo_utils.memory import gpu, DynamicSwapInstaller
from utils.misc import set_seed

ap = argparse.ArgumentParser()
ap.add_argument("--part", choices=["numerics", "capture"], required=True)
ap.add_argument("--out", required=True)
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
report = {}

pipe._initialize_kv_cache(1, torch.bfloat16, dev)
pipe._initialize_crossattn_cache(1, torch.bfloat16, dev)
cond = pipe.text_encoder(text_prompts=[prompt])
set_seed(1000)
noisy = torch.randn([1, 3, 16, 60, 104], device=dev, dtype=torch.bfloat16)
ts = torch.ones([1, 3], device=dev, dtype=torch.int64) * 1000


def fresh_caches():
    for b in pipe.kv_cache1:
        b["k"].zero_(); b["v"].zero_()
        b["global_end_index"].fill_(0); b["local_end_index"].fill_(0)
    for c in pipe.crossattn_cache:
        c["is_init"] = False


def one_pass():
    return pipe.generator(noisy_image_or_video=noisy, conditional_dict=cond,
                          timestep=ts, kv_cache=pipe.kv_cache1,
                          crossattn_cache=pipe.crossattn_cache, current_start=0)


if args.part == "numerics":
    fresh_caches(); out_e = one_pass()
    den_e = (out_e[1] if isinstance(out_e, (tuple, list)) else out_e).float().clone()
    kv_e = pipe.kv_cache1[0]["k"][:, :4680].float().clone()
    torch.cuda.synchronize()

    import torch._dynamo as dynamo
    dynamo.config.cache_size_limit = 64
    pipe.generator.model = torch.compile(pipe.generator.model,
                                         mode="max-autotune-no-cudagraphs", dynamic=False)
    fresh_caches(); one_pass()                      # compile/warm
    fresh_caches(); out_c = one_pass()
    den_c = (out_c[1] if isinstance(out_c, (tuple, list)) else out_c).float().clone()
    kv_c = pipe.kv_cache1[0]["k"][:, :4680].float().clone()
    torch.cuda.synchronize()

    def compare(x, y, name):
        d = (x - y).abs()
        rel = d.max().item() / max(y.abs().max().item(), 1e-9)
        r = {"max_abs": d.max().item(), "mean_abs": d.mean().item(),
             "rel_max": rel, "allclose_1e-2": bool(torch.allclose(x, y, atol=1e-2, rtol=1e-2))}
        print(f"  {name:24s} max_abs={r['max_abs']:.6g} mean_abs={r['mean_abs']:.6g} rel={rel:.4g}")
        return r

    print("[diag] single-pass eager vs compiled, identical inputs and cache state:")
    report["denoised_pred"] = compare(den_c, den_e, "denoised_pred")
    report["kv_cache_block0_k"] = compare(kv_c, kv_e, "kv_cache block0 k")
    report["verdict"] = ("DIVERGES ON FIRST PASS" if report["denoised_pred"]["rel_max"] > 0.05
                      else "first pass agrees; divergence accumulates later")
    print(f"[diag] VERDICT: {report['verdict']}")

else:  # capture
    import wan.modules.model as wmodel, wan.modules.causal_model as wcausal
    _c = {}

    def cached_sinusoidal_embedding_1d(dim, position):
        half = dim // 2; key = (dim, position.device, position.dtype)
        if key not in _c:
            _c[key] = torch.pow(10000, -torch.arange(half, device=position.device,
                                                     dtype=position.dtype).div(half))
        return torch.cat([torch.cos(torch.outer(position.type(_c[key].dtype), _c[key])),
                          torch.sin(torch.outer(position.type(_c[key].dtype), _c[key]))], dim=1)
    for m in (wmodel, wcausal):
        if hasattr(m, "sinusoidal_embedding_1d"): m.sinusoidal_embedding_1d = cached_sinusoidal_embedding_1d

    class HostIndexShim:
        __slots__ = ("v",)
        def __init__(s, v=0): s.v = int(v)
        def item(s): return s.v
        def fill_(s, x): s.v = int(x.item()) if torch.is_tensor(x) else int(x); return s
    for b in pipe.kv_cache1:
        b["global_end_index"] = HostIndexShim(0); b["local_end_index"] = HostIndexShim(0)

    for _ in range(3):
        one_pass()
    torch.cuda.synchronize()
    print("[diag] shims applied; attempting capture with CUDA_LAUNCH_BLOCKING="
          f"{os.environ.get('CUDA_LAUNCH_BLOCKING')}")
    try:
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3): one_pass()
        torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            one_pass()
        torch.cuda.synchronize()
        report["capture"] = "SUCCESS"; print("[diag] CAPTURE SUCCEEDED")
    except Exception as e:
        tb = traceback.format_exc()
        report["capture"] = "FAILED"; report["error"] = str(e)[:800]
        report["traceback_tail"] = tb[-3000:]
        print(f"[diag] CAPTURE FAILED: {type(e).__name__}: {str(e)[:300]}")
        print("\n".join(tb.splitlines()[-30:]))

    gr = subprocess.run(["grep", "-rnE", r"\.item\(\)|torch\.arange\(|torch\.tensor\(|\.cpu\(\)",
                         "wan/modules/causal_model.py", "wan/modules/model.py",
                         "wan/modules/attention.py"], capture_output=True, text=True).stdout
    report["capture_hostile_sites"] = gr.strip().splitlines()
    print("\n[diag] capture-hostile constructs on the inference path:")
    print(gr)

with open(args.out, "w") as f:
    json.dump(report, f, indent=2)
