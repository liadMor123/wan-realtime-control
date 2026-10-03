#!/usr/bin/env python3
"""
Triton release/acquire fence test and KV-cache fidelity checks (J6 part f, J4 leftovers).

Three independent measurements, all written to one JSON:
  1. gate item 6: a Triton producer/consumer kernel checks that
     tl.atomic_add(..., sem="release") paired with a volatile load is honoured
     across programs on this GPU/Triton build (20 trials, one program per SM);
  2. KV-cache prefix stability: after each chunk, the sha256 of the already
     written cache region must equal the hash taken one chunk earlier;
  3. dense fp32 attention reference: FlashAttention-2 output on the real cache
     contents vs. fp32 scaled_dot_product_attention on a 2-head slice, judged
     against 2x the inherent bf16 band.

Writes the file named by $J6_EXTRAS_OUT (default j6_extras.json).
"""
import contextlib, hashlib, io, json, os, sys
import torch
sys.path.insert(0, os.getcwd())

results = {}

# ---------------------------------------------------------------- gate item 6
import triton
import triton.language as tl


@triton.jit
def producer_consumer(flag_ptr, data_ptr, out_ptr, N: tl.constexpr):
    """Program 0 writes data then releases a flag; programs 1..N-1 spin on an
    acquire load of that flag, then read the data. If release/acquire is not
    honoured a consumer can observe flag==1 with stale data and write 0.
    """
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(data_ptr + tl.arange(0, 128), tl.full((128,), 7, tl.int32))
        tl.atomic_add(flag_ptr, 1, sem="release")
    else:
        done = 0
        while done == 0:
            done = tl.load(flag_ptr, volatile=True)
        v = tl.load(data_ptr + tl.arange(0, 128))
        ok = tl.sum(tl.where(v == 7, 1, 0))
        tl.store(out_ptr + pid, ok)


def triton_release_acquire_test():
    r = {"triton_version": triton.__version__}
    import inspect
    r["atomic_add_has_sem"] = "sem" in inspect.signature(tl.atomic_add).parameters
    dev = torch.device("cuda")
    N = 108                                   # one program per SM
    try:
        for trial in range(20):
            flag = torch.zeros(1, dtype=torch.int32, device=dev)
            data = torch.zeros(128, dtype=torch.int32, device=dev)
            res = torch.full((N,), -1, dtype=torch.int32, device=dev)
            producer_consumer[(N,)](flag, data, res, N, num_warps=4)
            torch.cuda.synchronize()
            bad = int((res[1:] != 128).sum().item())
            if bad:
                r["status"] = "FAILED"; r["trial"] = trial; r["bad_programs"] = bad
                break
        else:
            r["status"] = "PASS"
            r["trials"] = 20
            r["programs"] = N
    except Exception as ex:
        r["status"] = "ERROR"; r["error"] = str(ex)[:600]
    return r


results["gate_item6_triton_release_acquire"] = triton_release_acquire_test()
print("[f] triton release/acquire:", json.dumps(results["gate_item6_triton_release_acquire"]), flush=True)

# ---------------------------------------------------------------- J4 leftovers
from omegaconf import OmegaConf
from pipeline import CausalInferencePipeline
from pipeline.causal_inference import prime_crossattn_cache
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
    prompt = f.readline().rstrip()
pipe._initialize_kv_cache(1, torch.bfloat16, dev)
pipe._initialize_crossattn_cache(1, torch.bfloat16, dev)
set_seed(1000)
noise = torch.randn([1, 21, 16, 60, 104], device=dev, dtype=torch.bfloat16)
cond = pipe.text_encoder(text_prompts=[prompt])
with contextlib.redirect_stdout(io.StringIO()):
    prime_crossattn_cache(pipe, cond, noise)
dsl = [float(x) for x in pipe.denoising_step_list.tolist()]


def run_chunk(p, x):
    for tstep in dsl + [0.0]:
        ts = torch.ones([1, 3], device=dev, dtype=torch.int64) * int(tstep)
        o = pipe.generator(noisy_image_or_video=x, conditional_dict=cond, timestep=ts,
                           kv_cache=pipe.kv_cache1, crossattn_cache=pipe.crossattn_cache,
                           current_start=p * 3 * 1560)
        if isinstance(o, (tuple, list)): x = o[1]
    return x


for b in pipe.kv_cache1:
    b["k"].zero_(); b["v"].zero_(); b["global_end_index"] = 0; b["local_end_index"] = 0

# --- prefix stability: hash [0:prev_end] at p and compare to the same range at p-1
prefix = []
prev_hash, prev_end = None, 0
for p in range(7):
    run_chunk(p, noise[:, p * 3:(p + 1) * 3])
    torch.cuda.synchronize()
    end = (p + 1) * 4680
    if prev_end:
        h = hashlib.sha256(pipe.kv_cache1[0]["k"][:, :prev_end].float().cpu().numpy().tobytes()).hexdigest()[:16]
        prefix.append({"position": p, "prefix_tokens": prev_end, "hash": h,
                       "matches_previous": h == prev_hash})
        print(f"[J4] prefix[0:{prev_end}] at pos {p}: {h} "
              f"{'STABLE' if h == prev_hash else 'CHANGED'}", flush=True)
    prev_hash = hashlib.sha256(pipe.kv_cache1[0]["k"][:, :end].float().cpu().numpy().tobytes()).hexdigest()[:16]
    prev_end = end
results["j4_prefix_stability"] = prefix
results["j4_prefix_all_stable"] = all(x["matches_previous"] for x in prefix)

# --- dense fp32 reference on a reduced-head slice
from wan.modules.attention import flash_attention
dense_reference = []
for p in (0, 3, 6):
    end = (p + 1) * 4680
    k = pipe.kv_cache1[0]["k"][:, :end]
    v = pipe.kv_cache1[0]["v"][:, :end]
    q = torch.randn([1, 4680, 12, 128], device=dev, dtype=torch.bfloat16) * 0.05
    nh = 2                                            # reduced-head slice
    qs, ks, vs = q[:, :, :nh], k[:, :, :nh], v[:, :, :nh]
    fa = flash_attention(qs, ks, vs).float()
    q32 = qs.float().transpose(1, 2); k32 = ks.float().transpose(1, 2); v32 = vs.float().transpose(1, 2)
    ref = torch.nn.functional.scaled_dot_product_attention(q32, k32, v32).transpose(1, 2)
    d = (fa - ref).abs()
    # bf16 band: same dense reference evaluated in bf16 then upcast
    bf = torch.nn.functional.scaled_dot_product_attention(
        q32.bfloat16(), k32.bfloat16(), v32.bfloat16()).transpose(1, 2).float()
    band = (bf - ref).abs()
    r = {"position": p, "k_len": end, "heads_compared": nh,
         "fa2_vs_fp32_max_abs": d.max().item(), "fa2_vs_fp32_mean_abs": d.mean().item(),
         "bf16_band_max_abs": band.max().item(), "bf16_band_mean_abs": band.mean().item()}
    r["within_2x_band"] = (r["fa2_vs_fp32_max_abs"] <= 2 * r["bf16_band_max_abs"] and
                           r["fa2_vs_fp32_mean_abs"] <= 2 * r["bf16_band_mean_abs"])
    dense_reference.append(r)
    print(f"[J4] pos {p} K={end}: FA2 vs fp32 max {r['fa2_vs_fp32_max_abs']:.5g} "
          f"(band {r['bf16_band_max_abs']:.5g}) mean {r['fa2_vs_fp32_mean_abs']:.5g} "
          f"(band {r['bf16_band_mean_abs']:.5g}) -> {'PASS' if r['within_2x_band'] else 'FAIL'}",
          flush=True)
results["j4_dense_reference"] = dense_reference

with open(os.environ.get("J6_EXTRAS_OUT", "j6_extras.json"), "w") as f:
    json.dump(results, f, indent=2)
print("[f] written", flush=True)
