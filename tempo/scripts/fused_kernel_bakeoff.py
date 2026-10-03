#!/usr/bin/env python3
"""Part 2, step 2b: kernel bake-off for the fused L path (pre-registered protocol, part 2, decision 14).

The SDPA memory-efficient backend missed the step-2 correctness criterion. Here every candidate computes the same
cross-attention on real q/k/v at 9 probes (layers 0/14/29 x steps 0/25/49, cond branch) of pilot prompt 1's flash-B0
trajectory (seed 42), with the zero table and with the L(2,2) table, and is scored like step 2 (<= flash on mean-abs and
on fraction != bf16(fp64), all probes). Then each candidate's kernel time is measured at Wan (32,760 q) and
Self-Forcing (4,680 q) shapes. No video is saved.
"""
import json
import os
import statistics as st
import sys
import time

TEMPO = os.environ.get("TEMPO_ROOT", os.path.expanduser("~/tempo"))
sys.path.insert(0, os.path.join(TEMPO, "ext", "Wan2.1"))
sys.path.insert(0, os.path.join(TEMPO, "src"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: E402

from wan.modules.attention import flash_attention  # noqa: E402

from tempo_ctrl import benchmark  # noqa: E402
from tempo_ctrl.fused_attention import HW, LK, N_LAT, frame_table_L, masked_attention, reference_fp64, token_table  # noqa: E402
from tempo_ctrl.sampler import Sampler  # noqa: E402
from tempo_ctrl.tokens import load_tokenizer  # noqa: E402
from tempo_ctrl.cross_attention_arms import CTRL  # noqa: E402

JOB = os.environ.get("SLURM_JOB_ID", "local")
RES = os.path.join(TEMPO, "results", "step2")
DEV = torch.device("cuda:0")
BF = torch.bfloat16
SCALE = 128 ** -0.5
os.makedirs(RES, exist_ok=True)
torch._dynamo.config.cache_size_limit = 64                  # flex: one closure per call; never fall back to eager
torch._dynamo.config.accumulated_cache_size_limit = 256
rep = {"job": JOB, "host": os.uname().nodename, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}


def log(*a):
    print(f"[step2b {time.strftime('%H:%M:%S')}]", *a, flush=True)


def write_report():
    json.dump(rep, open(os.path.join(RES, f"step2b_{JOB}.json"), "w"), indent=1, default=str)


if "A100-SXM4-40GB" not in rep["gpu"]:
    raise SystemExit(f"wrong GPU {rep['gpu']!r}")

# --------------------------------------------------------------- candidates: (q, k, v [B, L, n, d] bf16, bias) -> out
# bias = (tok_table [21*1560, 512] bf16, tok_scalar [21*1560] fp32 = b(f) per query token, obj_idx) ; q_start = 0 here


def c_memeff(q, k, v, bias, q_start=0):
    return masked_attention(q, k, v, bias[0], q_start)


def c_cudnn(q, k, v, bias, q_start=0):
    lq = q.shape[1]
    with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
        o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                           attn_mask=bias[0][q_start:q_start + lq].view(1, 1, lq, -1))
    return o.transpose(1, 2)


def c_memeff_fp32(q, k, v, bias, q_start=0):
    lq = q.shape[1]
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
        o = F.scaled_dot_product_attention(q.float().transpose(1, 2), k.float().transpose(1, 2),
                                           v.float().transpose(1, 2),
                                           attn_mask=bias[0][q_start:q_start + lq].float().view(1, 1, lq, -1))
    return o.transpose(1, 2).to(BF)


_flex = None


def c_flex(q, k, v, bias, q_start=0):
    global _flex
    from torch.nn.attention.flex_attention import flex_attention
    if _flex is None:
        _flex = torch.compile(flex_attention, dynamic=False)
    rows = bias[0][q_start:q_start + q.shape[1]]

    def score_mod(score, b, h, qi, ki):
        return score + rows[qi, ki]

    o = _flex(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), score_mod=score_mod)
    return o.transpose(1, 2)


# keyaug: bias b(f) on the object keys carried as 8 extra head dims through Wan's FA2 kernel
_T = 1.0 / float(torch.tensor(SCALE, dtype=torch.float32))      # 1 / (fp32 scale FA2 uses), float64
KEYAUG_C = []
_r = _T
for _ in range(3):
    c = float(torch.tensor(_r, dtype=torch.float64).to(BF))
    KEYAUG_C.append(c)
    _r -= c
KEYAUG_REL_ERR = abs(sum(KEYAUG_C) - _T) / _T


def c_keyaug(q, k, v, bias, q_start=0):
    from flash_attn import flash_attn_func
    b_, lq, n, d = q.shape
    lk = k.shape[1]
    qe = torch.zeros(b_, lq, n, 8, dtype=BF, device=q.device)
    qe[..., :3] = bias[1][q_start:q_start + lq].to(BF).view(1, lq, 1, 1)
    ke = torch.zeros(b_, lk, n, 8, dtype=BF, device=q.device)
    ke[:, bias[2], :, :3] = torch.tensor(KEYAUG_C, dtype=BF, device=q.device)
    ve = torch.zeros(b_, lk, n, 8, dtype=BF, device=q.device)
    o = flash_attn_func(torch.cat([q, qe], -1), torch.cat([k, ke], -1), torch.cat([v, ve], -1), softmax_scale=SCALE)
    return o[..., :d]


_PERM = None


def c_flash_perm(q, k, v, bias, q_start=0):
    """Noise floor: Wan's FA2 on the same inputs with the 512 key slots permuted (same math, other summation order).
    Only defined for the zero bias; used to show how much a strict <= flash comparison is rounding-order noise."""
    global _PERM
    if _PERM is None:
        _PERM = torch.randperm(k.shape[1], generator=torch.Generator().manual_seed(7)).to(k.device)
    return flash_attention(q, k[:, _PERM], v[:, _PERM], k_lens=None)


CANDS = {"memeff": c_memeff, "cudnn": c_cudnn, "flex": c_flex, "keyaug": c_keyaug, "memeff_fp32": c_memeff_fp32}


def error_stats(o, ref):
    e = (o.double() - ref).abs()
    return {"maxabs": e.max().item(), "meanabs": e.mean().item(),
            "frac_ne_bf16_round_of_fp64": (o.double() != ref.to(BF).double()).double().mean().item()}


# --------------------------------------------------------------- setup: prompt 1 tables
rows = benchmark.load_one_object()
tok = load_tokenizer(benchmark.WAN_ROOT, benchmark.CKPT)
P1 = rows[1]
benchmark.configure_controller({"arm": "L", "beta": 2.0, "gamma": 2.0, "seed": 42}, P1, tok)
OBJ = list(CTRL.obj_idx)
FT_L = frame_table_L(CTRL.mask, OBJ, 2.0, 2.0)
FT_0 = torch.zeros(N_LAT, LK)
m = torch.tensor(P1["mask"], dtype=torch.float32)
assert (FT_L[:, OBJ] == FT_L[:, OBJ[:1]]).all()             # rank-1 per frame: one value on every object slot
bL = FT_L[:, OBJ[0]].repeat_interleave(HW).to(DEV)          # keyaug's b(f), from the same table as the reference
BIAS = {"zero": (token_table(FT_0, DEV), torch.zeros(N_LAT * HW, device=DEV), OBJ),
        "L": (token_table(FT_L, DEV), bL, OBJ)}
rep["keyaug"] = {"c": KEYAUG_C, "rel_err_sum_vs_1_over_scale": KEYAUG_REL_ERR}
log("keyaug constants", KEYAUG_C, "rel err", KEYAUG_REL_ERR)

# --------------------------------------------------------------- dummy sanity: each candidate runs, shapes/dtypes
g = torch.Generator(device=DEV).manual_seed(0)
qd = torch.randn(1, 3 * HW, 12, 128, device=DEV, generator=g, dtype=BF)
kd = torch.randn(1, LK, 12, 128, device=DEV, generator=g, dtype=BF)
vd = torch.randn(1, LK, 12, 128, device=DEV, generator=g, dtype=BF)
refd = reference_fp64(qd, kd, vd, FT_L, 6 * HW)
qf = torch.randn(1, N_LAT * HW, 12, 128, device=DEV, generator=g, dtype=BF)       # Wan's full length too
ok = {}
for name, fn in CANDS.items():
    try:
        o = fn(qd, kd, vd, BIAS["L"], 6 * HW)
        of = fn(qf, kd, vd, BIAS["L"], 0)
        torch.cuda.synchronize()
        assert o.shape == qd.shape and of.shape == qf.shape, (o.shape, of.shape)
        ok[name] = error_stats(o, refd)
        assert ok[name]["maxabs"] < 5e-2, f"{name}: offset-block error {ok[name]['maxabs']}"
    except Exception as ex:                                  # a candidate that cannot run is recorded, not fatal
        ok[name] = {"error": f"{type(ex).__name__}: {str(ex)[:400]}"}
    log("dummy", name, ok[name])
del qf
rep["dummy_block6"] = ok
CANDS = {k_: v_ for k_, v_ in CANDS.items() if "error" not in ok[k_]}
write_report()

# --------------------------------------------------------------- probes on prompt 1
cache = torch.load(benchmark.TEXT_CACHE)
S = Sampler(benchmark.CKPT)
neg = [cache["__neg__"].to(DEV)]
probes = []


def probe_hook(mod, args, out):
    if CTRL.branch != "cond" or CTRL.step not in (0, 25, 49) or mod._tempo_layer not in (0, 14, 29):
        return
    x, context, _ = args
    b, n, d = x.size(0), mod.num_heads, mod.head_dim
    q = mod.norm_q(mod.q(x)).view(b, -1, n, d)
    k = mod.norm_k(mod.k(context)).view(b, -1, n, d)
    v = mod.v(context).view(b, -1, n, d)
    qb, kb, vb = q.to(BF), k.to(BF), v.to(BF)                # the cast flash_attention applies
    rec = {"step": CTRL.step, "layer": mod._tempo_layer}
    with torch.autocast("cuda", enabled=False):             # candidates run on exactly the bf16 inputs, no autocast
        outs = {"flash": flash_attention(q, k, v, k_lens=None),
                "flash_perm": c_flash_perm(q, k, v, None)}
        for name, fn in CANDS.items():
            for tb in ("zero", "L"):
                try:
                    outs[f"{name}_{tb}"] = fn(qb, kb, vb, BIAS[tb])
                except Exception as ex:                      # record and keep the probe video alive
                    rec[f"{name}_{tb}_error"] = f"{type(ex).__name__}: {str(ex)[:300]}"
        for tb, ft in (("zero", FT_0), ("L", FT_L)):
            ref = reference_fp64(q, k, v, ft, 0)
            if tb == "zero":
                rec["flash"] = error_stats(outs["flash"], ref)
                rec["flash_perm"] = error_stats(outs["flash_perm"], ref)
            for name in CANDS:
                if f"{name}_{tb}" in outs:
                    rec[f"{name}_{tb}"] = error_stats(outs[f"{name}_{tb}"], ref)
            del ref
    probes.append(rec)


hooks = [b_.cross_attn.register_forward_hook(probe_hook) for b_ in S.model.blocks]
benchmark.configure_controller({"arm": "B0", "seed": 42, "path": "flash"}, P1, tok)
S.generate([cache[P1["prompt"]].to(DEV)], neg, seed=42, decode=False)
for h in hooks:
    h.remove()
rep["probes"] = probes
write_report()
assert len(probes) == 9, len(probes)
failed = {k_.rsplit("_", 2)[0] for p in probes for k_ in p if k_.endswith("_error")}
CANDS = {k_: v_ for k_, v_ in CANDS.items() if k_ not in failed}
rep["failed_in_probes"] = sorted(failed)


def passes(name, tb):
    return all(p[f"{name}_{tb}"]["meanabs"] <= p["flash"]["meanabs"] and
               p[f"{name}_{tb}"]["frac_ne_bf16_round_of_fp64"] <= p["flash"]["frac_ne_bf16_round_of_fp64"]
               for p in probes)


def worst_ratio(name):
    return max(max(p[f"{name}_{tb}"]["meanabs"] / p["flash"]["meanabs"],
                   p[f"{name}_{tb}"]["frac_ne_bf16_round_of_fp64"] / p["flash"]["frac_ne_bf16_round_of_fp64"])
               for p in probes for tb in ("zero", "L"))


summary = {name: {"pass_zero": passes(name, "zero"), "pass_L": passes(name, "L"), "worst_ratio": worst_ratio(name)}
           for name in CANDS}
# noise floor: does FA2 itself pass "<= FA2" once only its summation order changes?
nf = {"pass": all(p["flash_perm"]["meanabs"] <= p["flash"]["meanabs"] and
                  p["flash_perm"]["frac_ne_bf16_round_of_fp64"] <= p["flash"]["frac_ne_bf16_round_of_fp64"] for p in probes),
      "worst_ratio": max(max(p["flash_perm"]["meanabs"] / p["flash"]["meanabs"],
                             p["flash_perm"]["frac_ne_bf16_round_of_fp64"] / p["flash"]["frac_ne_bf16_round_of_fp64"])
                         for p in probes)}
summary["_noise_floor_flash_perm"] = nf
log("noise floor (FA2, permuted keys):", nf)
rep["criterion"] = summary
write_report()
for name, s_ in summary.items():
    if name.startswith("_"):
        continue
    log(f"criterion {name:12s} zero {s_['pass_zero']} L {s_['pass_L']} worst ratio to flash {s_['worst_ratio']:.3f}")

# --------------------------------------------------------------- kernel timing
del S
torch.cuda.empty_cache()


def median_time_ms(fn, reps=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return st.median(ts)


timing = {}
for lq, q_start in ((N_LAT * HW, 0), (3 * HW, 9 * HW)):
    q = torch.randn(1, lq, 12, 128, device=DEV, dtype=BF)
    k = torch.randn(1, LK, 12, 128, device=DEV, dtype=BF)
    v = torch.randn(1, LK, 12, 128, device=DEV, dtype=BF)
    t = {"flash_nobias": median_time_ms(lambda: flash_attention(q, k, v))}
    if "keyaug" in CANDS:                                    # keyaug with k/v extensions built once per prompt
        from flash_attn import flash_attn_func
        ke = torch.zeros(1, LK, 12, 8, dtype=BF, device=DEV)
        ke[:, OBJ, :, :3] = torch.tensor(KEYAUG_C, dtype=BF, device=DEV)
        kx, vx = torch.cat([k, ke], -1), torch.cat([v, torch.zeros_like(ke)], -1)
        qe = torch.zeros(1, lq, 12, 8, dtype=BF, device=DEV)

        def keyaug_pre():
            qe[..., :3] = bL[q_start:q_start + lq].to(BF).view(1, lq, 1, 1)
            return flash_attn_func(torch.cat([q, qe], -1), kx, vx, softmax_scale=SCALE)[..., :128]
        t["keyaug_precomputed_kv"] = median_time_ms(keyaug_pre)
    for name, fn in CANDS.items():
        t[name] = median_time_ms(lambda fn=fn: fn(q, k, v, BIAS["L"], q_start))
    timing[lq] = t
    log("kernel ms at Lq", lq, json.dumps({k_: round(v_, 4) for k_, v_ in t.items()}))
rep["kernel_ms"] = timing
write_report()

passing = [nm for nm in ("cudnn", "flex", "keyaug") if nm in summary and summary[nm]["pass_zero"] and summary[nm]["pass_L"]]
if passing:
    choice = min(passing, key=lambda nm: timing[3 * HW][nm])
    rep["choice"] = {"path": choice, "rule": "fastest passing candidate at the Self-Forcing shape", "passing": passing}
else:
    pool = [nm for nm in ("cudnn", "flex", "keyaug") if nm in summary]
    if not pool:
        raise SystemExit("no candidate ran")
    choice = min(pool, key=lambda nm: summary[nm]["worst_ratio"])
    rep["choice"] = {"path": choice, "rule": "no candidate passes: smallest worst-case ratio; criterion MISSED",
                     "passing": []}
write_report()
log("CHOICE", rep["choice"])
