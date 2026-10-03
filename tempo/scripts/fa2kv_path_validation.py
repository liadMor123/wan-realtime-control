#!/usr/bin/env python3
"""Part 2, step 2 (re-run with the step-2b choice): arm L through Wan's own FA2 kernel by key augmentation ("fa2kv").

First run: scripts/sdpa_mask_path_validation.py (SDPA mem-efficient mask path, job 162864; correctness criterion missed).
Bake-off: scripts/fused_kernel_bakeoff.py (job 162889) chose fa2kv by the pre-registered rule (decision 14).

  1. GPU preflight on dummy input: fa2kv vs fp64 (full sequence and 3-frame blocks at absolute offsets), CUDA-graph
     capture/replay, kernel names (must be FA2's), and whether zero-table fa2kv is bitwise equal to Wan's flash call.
  2. Probe video (untimed): flash B0, prompt 0, seed 42; 9 probes (layers 0/14/29 x steps 0/25/49, cond branch):
       pre-registered criterion: fa2kv (zero and L table) <= flash on mean-abs and fraction != bf16(fp64)
       noise floor: flash with permuted key slots; exact-rounding floor: fp32 mem-efficient kernel with the same table
  3. Timing, prompts 0 and 1, seed 42, one job/node (order reversed on prompt 1):
       flash B0  |  fa2kv L(2,2)  |  explicit L(2,2)         (full videos incl. VAE decode)

Exit status 1 (blocks step 3) only on a defect (decision 15): a preflight failure; zero-table mean-abs error > 1.5x
flash's; or L-table mean-abs error > 2 x max(1, flash/floor) x the exact-rounding floor. The pre-registered criterion is
reported either way.
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
import torch.nn.functional as F  # noqa: E402
from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: E402
from tempo_ctrl.fused_attention import (HW, LK, N_LAT, fa2_kernel_head_dim, fa2kv_attention, fa2kv_tables,  # noqa: E402
                                        fa2kv_zero_tables, frame_table_L, reference_fp64, token_table, zero_table)
from tempo_ctrl.sampler import Sampler  # noqa: E402
from tempo_ctrl.tokens import load_tokenizer  # noqa: E402
from tempo_ctrl.cross_attention_arms import CTRL, Controller, explicit_attention  # noqa: E402

JOB = os.environ.get("SLURM_JOB_ID", "local")
SCR = os.path.join(os.environ.get("TMPDIR", "/tmp"), "tempo", f"step2_{JOB}")
RES = os.path.join(TEMPO, "results", "step2")
ROWS = os.path.join(TEMPO, "results", "rows", f"phase22_{JOB}.jsonl")
PATH = "fa2kv"
DEV = torch.device("cuda:0")
L22 = {"arm": "L", "beta": 2.0, "gamma": 2.0}
PROBE_LAYERS, PROBE_STEPS = (0, 14, 29), (0, 25, 49)
os.makedirs(SCR, exist_ok=True)
os.makedirs(RES, exist_ok=True)
rep = {"job": JOB, "host": os.uname().nodename, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}


def log(*a):
    print(f"[step2 {time.strftime('%H:%M:%S')}]", *a, flush=True)


def write_report():
    json.dump(rep, open(os.path.join(RES, f"step2_{PATH}_{JOB}.json"), "w"), indent=1, default=str)


def error_stats(o, ref):
    e = (o.double() - ref).abs()
    return {"maxabs": e.max().item(), "meanabs": e.mean().item(),
            "frac_ne_bf16_round_of_fp64": (o.double() != ref.to(torch.bfloat16).double()).double().mean().item()}


def err_norm(o, ref):
    """max |o - ref| / max |ref|: scale-free error (bf16 rounding ~2^-8..2^-7; a frame-offset bug gives ~0.1-1)."""
    return ((o.double() - ref).abs().max() / ref.abs().max()).item()


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
TAB_L, TAB_0 = token_table(FT_L, DEV), zero_table(DEV)          # dense tables (fp32 floor kernel only)
KA_L = fa2kv_tables(CTRL.mask, CTRL.obj_idx, 2.0, 2.0, DEV)
KA_0 = fa2kv_zero_tables(DEV)
_PERM = torch.randperm(LK, generator=torch.Generator().manual_seed(7)).to(DEV)


def fp32_floor(q, k, v, tab):
    """Exact-rounding floor: mem-efficient kernel on fp32 inputs with the dense table, output rounded to bf16."""
    lq = q.shape[1]
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
        o = F.scaled_dot_product_attention(*(t.to(torch.bfloat16).float().transpose(1, 2) for t in (q, k, v)),
                                           attn_mask=tab[:lq].float().view(1, 1, lq, -1))
    return o.transpose(1, 2).to(torch.bfloat16)

# ------------------------------------------------------------------ 1. preflight on dummy input
g = torch.Generator(device=DEV).manual_seed(0)
q = torch.randn(1, N_LAT * HW, 12, 128, device=DEV, generator=g, dtype=torch.bfloat16)
k = torch.randn(1, LK, 12, 128, device=DEV, generator=g, dtype=torch.bfloat16)
v = torch.randn(1, LK, 12, 128, device=DEV, generator=g, dtype=torch.bfloat16)
pre = {"full_L_vs_fp64": error_stats(fa2kv_attention(q, k, v, KA_L, 0), reference_fp64(q, k, v, FT_L, 0)),
       "full_flash_vs_fp64": error_stats(flash_attention(q, k, v), reference_fp64(q, k, v, FT_0, 0)),
       "full_zero_vs_fp64": error_stats(fa2kv_attention(q, k, v, KA_0, 0), reference_fp64(q, k, v, FT_0, 0))}
pre["zero_table_bitwise_equals_flash"] = bool(torch.equal(fa2kv_attention(q, k, v, KA_0, 0).to(torch.bfloat16),
                                                          flash_attention(q, k, v).to(torch.bfloat16)))
# 3-frame blocks at absolute offsets, with a frame-distinct rank-1 bias (per-frame b in -4..4 on 3 object slots)
b_r = torch.randint(-4, 5, (N_LAT,), generator=torch.Generator().manual_seed(1)).float()
obj_r = [3, 40, 41]
ft_r = torch.zeros(N_LAT, LK)
ft_r[:, obj_r] = b_r[:, None]
from tempo_ctrl.fused_attention import KA_C  # noqa: E402
ke_r = torch.zeros(LK, 8, dtype=torch.float64)
ke_r[obj_r, :3] = torch.tensor(KA_C, dtype=torch.float64)
ka_r = (b_r.to(torch.bfloat16).repeat_interleave(HW).to(DEV), ke_r.to(torch.bfloat16).to(DEV))
ref_r = reference_fp64(q, k, v, ft_r, 0)
blk = []
for f0 in range(0, N_LAT, 3):                                    # Self-Forcing's 3-frame blocks
    sl = slice(f0 * HW, (f0 + 3) * HW)
    blk.append(err_norm(fa2kv_attention(q[:, sl], k, v, ka_r, f0 * HW), ref_r[:, sl]))
pre["block3_absolute_offset_maxabs"] = blk
pre["block3_absolute_offset_err_norm"] = pre.pop("block3_absolute_offset_maxabs")
# scale-free errors (max |o - ref| / max |ref|): every offset block within max(2^-6, 2x flash's own on the unbiased
# full sequence); a frame-offset bug gives ~0.1-1
pre["flash_full_err_norm"] = err_norm(flash_attention(q, k, v), reference_fp64(q, k, v, FT_0, 0))
pre["full_L_err_norm"] = err_norm(fa2kv_attention(q, k, v, KA_L, 0), reference_fp64(q, k, v, FT_L, 0))
lim = max(2.0 ** -6, 2 * pre["flash_full_err_norm"])
assert max(pre["block3_absolute_offset_err_norm"]) <= lim and pre["full_L_err_norm"] <= lim, (lim, pre)
# CUDA-graph capture: static table, no host sync inside the captured call
qs = q[:, :3 * HW].clone()
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        fa2kv_attention(qs, k, v, ka_r, 6 * HW)
torch.cuda.current_stream().wait_stream(s)
gr = torch.cuda.CUDAGraph()
with torch.cuda.graph(gr):
    og = fa2kv_attention(qs, k, v, ka_r, 6 * HW)
qs.copy_(q[:, 6 * HW:9 * HW])
gr.replay()
torch.cuda.synchronize()
pre["cuda_graph_replay_equals_eager"] = bool(torch.equal(og, fa2kv_attention(qs, k, v, ka_r, 6 * HW)))
assert pre["cuda_graph_replay_equals_eager"], "graph replay != eager"
pre["kernels"] = {}
for lq in (N_LAT * HW, 3 * HW):                                  # each call on its own
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            fa2kv_attention(q[:, :lq], k, v, KA_L, 0)
        torch.cuda.synchronize()
    pre["kernels"][lq] = cuda_kernel_names(prof)
rep["preflight"] = pre
write_report()
log("preflight", json.dumps(pre))
pre["fa2_kernel_head_dim"] = {lq: fa2_kernel_head_dim(ks) for lq, ks in pre["kernels"].items()}
if not all((hd or 0) >= 136 for hd in pre["fa2_kernel_head_dim"].values()):         # FA2 fwd kernel at d >= 136
    raise SystemExit(f"fa2kv did not run FA2's forward kernel: {pre['kernels']}")
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
    o_f0 = fa2kv_attention(q, k, v, KA_0, 0)
    o_fL = fa2kv_attention(q, k, v, KA_L, 0)
    with torch.autocast("cuda", enabled=False):
        o_perm = flash_attention(q, k[:, _PERM], v[:, _PERM], k_lens=None)
        o_x0, o_xLf = fp32_floor(q, k, v, TAB_0), fp32_floor(q, k, v, TAB_L)
    o_xL = explicit_attention(q, k, v, Controller(path="explicit", arm="L", beta=2.0, gamma=2.0,
                                                  mask=torch.tensor(P0["mask"], dtype=torch.float32),
                                                  obj_idx=list(CTRL_OBJ)))
    with torch.autocast("cuda", enabled=False):
        ref0 = reference_fp64(q, k, v, FT_0, 0)
        rec = {"step": CTRL.step, "layer": mod._tempo_layer, "flash": error_stats(o_fl, ref0), "fused_zero": error_stats(o_f0, ref0),
               "flash_perm": error_stats(o_perm, ref0), "fp32floor_zero": error_stats(o_x0, ref0),
               "zero_bitwise_equals_flash": bool(torch.equal(o_f0.to(torch.bfloat16), o_fl.to(torch.bfloat16)))}
        del ref0
        refL = reference_fp64(q, k, v, FT_L, 0)
        rec["fused_L"] = error_stats(o_fL, refL)
        rec["explicit_L"] = error_stats(o_xL, refL)
        rec["fp32floor_L"] = error_stats(o_xLf, refL)
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
rep["criterion"]["path"] = PATH
rep["noise_floor_flash_perm_passes"] = all(within_flash_error(p, "flash_perm") for p in probes)
rep["zero_bitwise_equals_flash_all_probes"] = all(p["zero_bitwise_equals_flash"] for p in probes)
rep["ratio_to_fp32_floor"] = {tb: max(p[f"fused_{tb}"]["meanabs"] / p[f"fp32floor_{tb}"]["meanabs"] for p in probes)
                              for tb in ("zero", "L")}
rep["ratio_to_fp32_floor"]["flash_zero"] = max(p["flash"]["meanabs"] / p["fp32floor_zero"]["meanabs"] for p in probes)
# defect gate (decision 15, calibrated after review): zero table within 1.5x of flash (noise floor 1.41x in 162889);
# L table within 2 x max(1, flash's own ratio to the fp32 floor) of the exact-rounding floor
rep["ratio_zero_to_flash"] = max(p["fused_zero"]["meanabs"] / p["flash"]["meanabs"] for p in probes)
rep["defect_gate"] = {"zero_to_flash_max": rep["ratio_zero_to_flash"], "zero_limit": 1.5,
                      "L_to_floor_max": rep["ratio_to_fp32_floor"]["L"],
                      "L_limit": 2.0 * max(1.0, rep["ratio_to_fp32_floor"]["flash_zero"])}
rep["defect_gate_pass"] = bool(rep["ratio_zero_to_flash"] <= 1.5 and
                               rep["ratio_to_fp32_floor"]["L"] <= rep["defect_gate"]["L_limit"])
write_report()
log("probe criterion", rep["criterion"], "| noise floor passes:", rep["noise_floor_flash_perm_passes"],
    "| zero bitwise == flash at all probes:", rep["zero_bitwise_equals_flash_all_probes"],
    "| worst mean-abs ratio to fp32 floor:", rep["ratio_to_fp32_floor"], "| defect gate:", rep["defect_gate_pass"])
for p in probes:
    log(f"  L{p['layer']:02d} s{p['step']:02d} mean-abs flash {p['flash']['meanabs']:.3e} fused0 "
        f"{p['fused_zero']['meanabs']:.3e} fusedL {p['fused_L']['meanabs']:.3e} explL {p['explicit_L']['meanabs']:.3e} | "
        f"frac!=bf16 flash {p['flash']['frac_ne_bf16_round_of_fp64']:.3f} fused0 "
        f"{p['fused_zero']['frac_ne_bf16_round_of_fp64']:.3f} fusedL {p['fused_L']['frac_ne_bf16_round_of_fp64']:.3f}")

# ------------------------------------------------------------------ 3. timing
RUNS = [{"arm": "B0", "seed": 42, "path": "flash"}, dict(L22, seed=42, path=PATH), dict(L22, seed=42)]
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
                 for tag in (f"L_b2g2_{PATH}_s42", "L_b2g2_s42")}
rep["timing_summary"] = summ
ov = [summ[p][f"L_b2g2_{PATH}_s42"]["total_vs_flash"] for p in (0, 1)]
mem = [summ[p][f"L_b2g2_{PATH}_s42"]["peak_gb_minus_flash"] for p in (0, 1)]
rep["Q1"] = {"path": PATH, "fused_L_overhead_mean": sum(ov) / 2, "per_prompt": ov, "peak_gb_delta": mem,
             "met": bool(sum(ov) / 2 <= 0.02 and all(abs(m) <= 0.5 for m in mem))}
write_report()
log("timing summary", json.dumps(summ), "| Q1", rep["Q1"])
if not rep["criterion"]["pass"]:
    log("### pre-registered correctness criterion MISSED (reported; decision 15: not a gate for step 3)")
if not rep["defect_gate_pass"]:
    log("### DEFECT GATE FAILED (error > 2x the exact-rounding floor) -- exiting 1")
    sys.exit(1)
log("DONE")
