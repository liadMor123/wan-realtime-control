#!/usr/bin/env python3
"""Part 2, step 2: arm L through a fused SDPA kernel with an additive attention mask (Wan2.1-T2V-1.3B).

  1. GPU preflight on dummy input: fused vs fp64 (full sequence and 3-frame blocks at absolute offsets),
     CUDA-graph capture/replay of the masked call, and the SDPA backend from torch.profiler kernel names.
  2. Probe video (untimed): flash B0, prompt 0, seed 42. At layers 0/14/29 x steps 0/25/49 (cond branch):
       flash (no bias) vs fp64 (no bias)  |  fused zero table vs fp64 (no bias)  |  fused L table vs fp64 (L bias)
     Criterion (pre-registered, part 2): both fused errors <= flash's, on mean-abs and fraction != bf16(fp64).
  3. Timing, prompts 0 and 1, seed 42, one job/node, interleaved per prompt:
       flash B0  |  fused-mask L(2,2)  |  explicit L(2,2)         (full videos incl. VAE decode)

Exit status 1 if the criterion fails or SDPA did not run a fused backend (blocks the step-3 jobs).
"""
import json
import os
import shutil
import sys
import time

TEMPO = os.environ.get("TEMPO_ROOT", os.path.expanduser("~/tempo"))
sys.path.insert(0, os.path.join(TEMPO, "ext", "Wan2.1"))
sys.path.insert(0, os.path.join(TEMPO, "src"))

import torch  # noqa: E402
from torch.autograd import DeviceType  # noqa: E402

from wan.modules.attention import flash_attention  # noqa: E402

from tempo_ctrl import benchmark  # noqa: E402
from tempo_ctrl.fused_attention import (HW, LK, N_LAT, frame_table_L, sdpa_backend_from_kernels, masked_attention,  # noqa: E402
                                        reference_fp64, token_table, zero_table)
from tempo_ctrl.sampler import Sampler  # noqa: E402
from tempo_ctrl.tokens import load_tokenizer  # noqa: E402
from tempo_ctrl.cross_attention_arms import CTRL, Controller, explicit_attention  # noqa: E402

JOB = os.environ.get("SLURM_JOB_ID", "local")
SCR = os.path.join(os.environ.get("TMPDIR", "/tmp"), "tempo", f"step2_{JOB}")
RES = os.path.join(TEMPO, "results", "step2")
ROWS = os.path.join(TEMPO, "results", "rows", f"phase22_{JOB}.jsonl")
DEV = torch.device("cuda:0")
L22 = {"arm": "L", "beta": 2.0, "gamma": 2.0}
PROBE_LAYERS, PROBE_STEPS = (0, 14, 29), (0, 25, 49)
os.makedirs(SCR, exist_ok=True)
os.makedirs(RES, exist_ok=True)
rep = {"job": JOB, "host": os.uname().nodename, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}


def log(*a):
    print(f"[step2 {time.strftime('%H:%M:%S')}]", *a, flush=True)


def write_report():
    json.dump(rep, open(os.path.join(RES, f"step2_{JOB}.json"), "w"), indent=1, default=str)


def error_stats(o, ref):
    e = (o.double() - ref).abs()
    return {"maxabs": e.max().item(), "meanabs": e.mean().item(),
            "frac_ne_bf16_round_of_fp64": (o.double() != ref.to(torch.bfloat16).double()).double().mean().item()}


def cuda_kernel_names(prof):
    return sorted({e.key for e in prof.key_averages() if e.device_type == DeviceType.CUDA})


if "A100-SXM4-40GB" not in rep["gpu"]:
    raise SystemExit(f"wrong GPU {rep['gpu']!r}: A100-SXM4-40GB only")
rows = benchmark.load_one_object()
tok = load_tokenizer(benchmark.WAN_ROOT, benchmark.CKPT)
P0, P1 = rows[0], rows[1]
benchmark.configure_controller(dict(L22, seed=42), P0, tok)
FT_L = frame_table_L(CTRL.mask, CTRL.obj_idx, 2.0, 2.0)          # prompt 0's L(2,2) frame table (probes)
FT_0 = torch.zeros(N_LAT, LK)
TAB_L, TAB_0 = token_table(FT_L, DEV), zero_table(DEV)

# ------------------------------------------------------------------ 1. preflight on dummy input
g = torch.Generator(device=DEV).manual_seed(0)
q = torch.randn(1, N_LAT * HW, 12, 128, device=DEV, generator=g, dtype=torch.bfloat16)
k = torch.randn(1, LK, 12, 128, device=DEV, generator=g, dtype=torch.bfloat16)
v = torch.randn(1, LK, 12, 128, device=DEV, generator=g, dtype=torch.bfloat16)
pre = {"full_L_vs_fp64": error_stats(masked_attention(q, k, v, TAB_L, 0), reference_fp64(q, k, v, FT_L, 0)),
       "full_flash_vs_fp64": error_stats(flash_attention(q, k, v), reference_fp64(q, k, v, FT_0, 0)),
       "full_zero_vs_fp64": error_stats(masked_attention(q, k, v, TAB_0, 0), reference_fp64(q, k, v, FT_0, 0))}
ft_r = torch.randint(-4, 5, (N_LAT, LK), generator=torch.Generator().manual_seed(1)).float()
tab_r = token_table(ft_r, DEV)
ref_r = reference_fp64(q, k, v, ft_r, 0)
blk = []
for f0 in range(0, N_LAT, 3):                                    # Self-Forcing's 3-frame blocks
    sl = slice(f0 * HW, (f0 + 3) * HW)
    blk.append(error_stats(masked_attention(q[:, sl], k, v, tab_r, f0 * HW), ref_r[:, sl])["maxabs"])
pre["block3_absolute_offset_maxabs"] = blk
assert max(blk) < 5e-2 and pre["full_L_vs_fp64"]["maxabs"] < 5e-2, pre
# CUDA-graph capture: static table, no host sync inside the captured call
qs = q[:, :3 * HW].clone()
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        masked_attention(qs, k, v, tab_r, 6 * HW)
torch.cuda.current_stream().wait_stream(s)
gr = torch.cuda.CUDAGraph()
with torch.cuda.graph(gr):
    og = masked_attention(qs, k, v, tab_r, 6 * HW)
qs.copy_(q[:, 6 * HW:9 * HW])
gr.replay()
torch.cuda.synchronize()
pre["cuda_graph_replay_equals_eager"] = bool(torch.equal(og, masked_attention(qs, k, v, tab_r, 6 * HW)))
assert pre["cuda_graph_replay_equals_eager"], "graph replay != eager"
pre["sdpa_kernels"], pre["sdpa_backend"] = {}, {}
for lq in (N_LAT * HW, 3 * HW):                                  # each call classified on its own
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            masked_attention(q[:, :lq], k, v, TAB_L, 0)
        torch.cuda.synchronize()
    pre["sdpa_kernels"][lq] = cuda_kernel_names(prof)
    pre["sdpa_backend"][lq] = sdpa_backend_from_kernels(pre["sdpa_kernels"][lq])
rep["preflight"] = pre
write_report()
log("preflight", json.dumps(pre))
if any(b_ not in ("mem_efficient", "cudnn") for b_ in pre["sdpa_backend"].values()):
    raise SystemExit(f"SDPA did not select a fused masked backend: {pre['sdpa_kernels']} (-> use FlexAttention)")
del q, k, v, qs, og, gr, ref_r

# ------------------------------------------------------------------ 2. probe video (untimed)
cache = torch.load(benchmark.TEXT_CACHE)
S = Sampler(benchmark.CKPT)
neg = [cache["__neg__"].to(DEV)]
probes = []


def probe_hook(mod, args, out):
    if CTRL.branch != "cond" or CTRL.step not in PROBE_STEPS or mod._tempo_layer not in PROBE_LAYERS:
        return
    x, context, context_lens = args
    b, n, d = x.size(0), mod.num_heads, mod.head_dim
    q = mod.norm_q(mod.q(x)).view(b, -1, n, d)
    k = mod.norm_k(mod.k(context)).view(b, -1, n, d)
    v = mod.v(context).view(b, -1, n, d)
    o_fl = flash_attention(q, k, v, k_lens=None)
    o_f0 = masked_attention(q, k, v, TAB_0, 0)
    o_fL = masked_attention(q, k, v, TAB_L, 0)
    o_xL = explicit_attention(q, k, v, Controller(path="explicit", arm="L", beta=2.0, gamma=2.0,
                                                  mask=torch.tensor(P0["mask"], dtype=torch.float32),
                                                  obj_idx=list(CTRL_OBJ)))
    with torch.autocast("cuda", enabled=False):
        ref0 = reference_fp64(q, k, v, FT_0, 0)
        rec = {"step": CTRL.step, "layer": mod._tempo_layer, "flash": error_stats(o_fl, ref0), "fused_zero": error_stats(o_f0, ref0)}
        del ref0
        refL = reference_fp64(q, k, v, FT_L, 0)
        rec["fused_L"] = error_stats(o_fL, refL)
        rec["explicit_L"] = error_stats(o_xL, refL)
        rec["refL_absmax"] = refL.abs().max().item()
        del refL
    probes.append(rec)


CTRL_OBJ = list(CTRL.obj_idx)
hooks = [b_.cross_attn.register_forward_hook(probe_hook) for b_ in S.model.blocks]
benchmark.configure_controller({"arm": "B0", "seed": 42, "path": "flash"}, P0, tok)
S.generate([cache[P0["prompt"]].to(DEV)], neg, seed=42, decode=False)
for h in hooks:
    h.remove()
assert len(probes) == len(PROBE_LAYERS) * len(PROBE_STEPS), len(probes)


def within_flash_error(p, key):
    return (p[key]["meanabs"] <= p["flash"]["meanabs"] and
            p[key]["frac_ne_bf16_round_of_fp64"] <= p["flash"]["frac_ne_bf16_round_of_fp64"])


rep["probes"] = probes
rep["criterion"] = {"fused_L_le_flash_all_probes": all(within_flash_error(p, "fused_L") for p in probes),
                    "fused_zero_le_flash_all_probes": all(within_flash_error(p, "fused_zero") for p in probes)}
rep["criterion"]["pass"] = all(rep["criterion"].values())
write_report()
log("probe criterion", rep["criterion"])
for p in probes:
    log(f"  L{p['layer']:02d} s{p['step']:02d} mean-abs flash {p['flash']['meanabs']:.3e} fused0 "
        f"{p['fused_zero']['meanabs']:.3e} fusedL {p['fused_L']['meanabs']:.3e} explL {p['explicit_L']['meanabs']:.3e} | "
        f"frac!=bf16 flash {p['flash']['frac_ne_bf16_round_of_fp64']:.3f} fused0 "
        f"{p['fused_zero']['frac_ne_bf16_round_of_fp64']:.3f} fusedL {p['fused_L']['frac_ne_bf16_round_of_fp64']:.3f}")

# ------------------------------------------------------------------ 3. timing
RUNS = [{"arm": "B0", "seed": 42, "path": "flash"}, dict(L22, seed=42, path="fused"), dict(L22, seed=42)]
for j, r in enumerate(RUNS):        # untimed 2-step warmups of every path; the first also decodes, so the first
    benchmark.configure_controller(r, P0, tok)     # VAE decode (lazy kernel loading) is not charged to the first timed video
    S.generate([cache[P0["prompt"]].to(DEV)], neg, seed=0, steps=2, decode=(j == 0))
torch.cuda.synchronize()
timing = []
for row, order in ((P0, RUNS), (P1, RUNS[::-1])):             # path order reversed on the second prompt
    for r in order:
        tag = benchmark.run_tag(r)
        info = benchmark.configure_controller(r, row, tok)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        tm = {}
        video, _ = S.generate([cache[row["prompt"]].to(DEV)], neg, seed=42, timing=tm)
        peak = torch.cuda.max_memory_allocated() / 2**30
        video = video.cpu()
        if not torch.isfinite(video).all():
            raise RuntimeError(f"non-finite video {tag} p{row['prompt_id']}")
        name = benchmark.video_name(row["prompt"])
        tmp = os.path.join(SCR, tag, name)
        benchmark.save_video(video, tmp)
        dst = os.path.join(TEMPO, "videos", "step2", tag)      # never overwrite part-1 videos (L_b2g2_s42)
        os.makedirs(dst, exist_ok=True)
        shutil.copy2(tmp, os.path.join(dst, name + ".part"))
        os.replace(os.path.join(dst, name + ".part"), os.path.join(dst, name))
        row_out = {"phase": 22, "run_tag": tag, "arm": r["arm"], "path": r.get("path", "explicit"),
                   "params": {k_: v_ for k_, v_ in r.items() if k_ not in ("arm", "seed")},
                   "prompt_id": row["prompt_id"], "temp_object": row["temp_object"], "seed": 42,
                   "wall_s": tm["total_s"], "denoise_s": tm["denoise_s"], "peak_gb": peak,
                   "video_path": os.path.join(dst, name), "job": JOB, "host": os.uname().nodename,
                   "obj_idx": info["obj_idx"], "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "temporal_accuracy": None}
        benchmark.append_jsonl(ROWS, row_out)
        timing.append(row_out)
        log(f"{tag:22s} p{row['prompt_id']} total {tm['total_s']:.1f}s denoise {tm['denoise_s']:.1f}s peak {peak:.2f}GB")
        del video
rep["timing"] = timing
by = {(t["run_tag"], t["prompt_id"]): t for t in timing}
summ = {}
for pid in (0, 1):
    fl = by[("B0_flash_s42", pid)]
    summ[pid] = {tag: {"total_vs_flash": by[(tag, pid)]["wall_s"] / fl["wall_s"] - 1,
                       "denoise_vs_flash": by[(tag, pid)]["denoise_s"] / fl["denoise_s"] - 1,
                       "peak_gb_minus_flash": by[(tag, pid)]["peak_gb"] - fl["peak_gb"]}
                 for tag in ("L_b2g2_fused_s42", "L_b2g2_s42")}
rep["timing_summary"] = summ
ov = [summ[p]["L_b2g2_fused_s42"]["total_vs_flash"] for p in (0, 1)]
mem = [summ[p]["L_b2g2_fused_s42"]["peak_gb_minus_flash"] for p in (0, 1)]
rep["Q1"] = {"fused_L_overhead_mean": sum(ov) / 2, "per_prompt": ov, "peak_gb_delta": mem,
             "met": bool(sum(ov) / 2 <= 0.02 and all(abs(m) <= 0.5 for m in mem))}
write_report()
log("timing summary", json.dumps(summ), "| Q1", rep["Q1"])
if not rep["criterion"]["pass"]:
    log("### CORRECTNESS CRITERION FAILED -- exiting 1")
    sys.exit(1)
log("DONE")
