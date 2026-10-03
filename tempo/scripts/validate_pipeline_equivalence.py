#!/usr/bin/env python3
"""Phase 0: text cache, reference equivalence, explicit-path correctness, timing, arm smoke tests.

Videos (6, all seed 42, benchmark protocol):
  v1 prompt 0  wan.WanT2V.generate, untouched model (reference)
  v2 prompt 0  our Sampler, flash path            -> must be byte-identical to v1
  v3 prompt 0  our Sampler, explicit path, B0     -> compared to v2 (not expected identical)
  v4 prompt 0  explicit, L beta=gamma=0, + share recording + per-layer attention probes
                                                  -> must be byte-identical to v3
  v5 prompt 1  flash B0     (timing pair 2)
  v6 prompt 1  explicit B0  (timing pair 2)
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

import wan  # noqa: E402
from wan.configs import WAN_CONFIGS  # noqa: E402
from wan.modules.attention import flash_attention  # noqa: E402

from tempo_ctrl import benchmark  # noqa: E402
from tempo_ctrl.sampler import FRAME_NUM, GUIDE, SHIFT, SIZE, STEPS, Sampler  # noqa: E402
from tempo_ctrl.tokens import load_tokenizer  # noqa: E402
from tempo_ctrl.cross_attention_arms import CTRL, N_LAT, HW, Controller, explicit_attention  # noqa: E402

JOB = os.environ.get("SLURM_JOB_ID", "local")
SCR = os.path.join(os.environ.get("TMPDIR", "/tmp"), "tempo", f"phase0_{JOB}")  # Slurm node-local
RES = os.path.join(TEMPO, "results", "phase0")
VID_HOME = os.path.join(TEMPO, "videos", "phase0")
ROWS = os.path.join(TEMPO, "results", "rows", f"phase0_{JOB}.jsonl")
os.makedirs(SCR, exist_ok=True)
os.makedirs(RES, exist_ok=True)
DEV = torch.device("cuda:0")
torch.backends.cudnn.benchmark = False
report = {"job": JOB, "host": os.uname().nodename, "gpu": torch.cuda.get_device_name(0),
          "protocol": {"size": SIZE, "frames": FRAME_NUM, "steps": STEPS, "shift": SHIFT,
                       "guide": GUIDE, "solver": "unipc", "seed": 42}}


def log(*a):
    print(f"[phase0 {time.strftime('%H:%M:%S')}]", *a, flush=True)


def write_report():
    with open(os.path.join(RES, f"phase0_{JOB}.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)


def psnr(a, b):
    mse = ((a.float() - b.float()) ** 2).mean().item()
    return float("inf") if mse == 0 else 10 * torch.log10(torch.tensor(4.0 / mse)).item()  # range [-1,1]


rows = benchmark.load_one_object()
tok = load_tokenizer(benchmark.WAN_ROOT, benchmark.CKPT)
P0, P1 = rows[0], rows[1]
# token lookup for all 80 prompts before any GPU work (fails loudly on any mismatch)
report["token_lookup"] = {r["prompt_id"]: benchmark.configure_controller({"arm": "B0", "seed": 42}, r, tok) for r in rows}
log("token lookup ok for", len(rows), "prompts; prompt 0:", report["token_lookup"][0])


def save_video_and_row(name, prompt_row, tm):
    """Write the video and its JSONL row immediately, so a later crash loses nothing."""
    pth = os.path.join(SCR, name, benchmark.video_name(prompt_row["prompt"]))
    benchmark.save_video(videos[name].cpu(), pth)
    dst = os.path.join(VID_HOME, name)
    os.makedirs(dst, exist_ok=True)
    shutil.copy2(pth, dst)
    benchmark.append_jsonl(ROWS, {
        "phase": 0, "video": name, "arm": tm.get("arm", "B0"), "path": tm.get("path", "wan_original"),
        "params": {"beta": 0.0, "gamma": 0.0} if name == "v4" else {}, "prompt_id": prompt_row["prompt_id"],
        "seed": 42, "wall_s": tm["total_s"], "denoise_s": tm.get("denoise_s"), "peak_gb": tm["peak_gb"],
        "video_path": os.path.join(dst, benchmark.video_name(prompt_row["prompt"])), "job": JOB,
        "temporal_accuracy": None})

# ---------------------------------------------------------------- 1. WanT2V + text cache
cfg = WAN_CONFIGS["t2v-1.3B"]
t0 = time.perf_counter()
w = wan.WanT2V(config=cfg, checkpoint_dir=benchmark.CKPT, device_id=0, rank=0)
report["load_wan_t2v_s"] = time.perf_counter() - t0
w.text_encoder.model.to(DEV)
cache = {}
with torch.no_grad():
    for r in rows:
        cache[r["prompt"]] = w.text_encoder([r["prompt"]], DEV)[0].cpu()
    cache["__neg__"] = w.text_encoder([cfg.sample_neg_prompt], DEV)[0].cpu()
    again = w.text_encoder([P0["prompt"]], DEV)[0].cpu()
    again_neg = w.text_encoder([cfg.sample_neg_prompt], DEV)[0].cpu()
report["t5_reencode_identical"] = bool(torch.equal(again, cache[P0["prompt"]]) and
                                       torch.equal(again_neg, cache["__neg__"]))
report["text_cache"] = {"n": len(cache), "dtype": str(cache["__neg__"].dtype),
                        "real_len_prompt0": cache[P0["prompt"]].shape[0],
                        "real_len_neg": cache["__neg__"].shape[0]}
torch.save(cache, benchmark.TEXT_CACHE)
log("text cache written", report["text_cache"], "reencode identical:", report["t5_reencode_identical"])
write_report()


def text_context(prompt):
    return [cache[prompt].to(DEV)], [cache["__neg__"].to(DEV)]


videos, lat, timing = {}, {}, {}

# ---------------------------------------------------------------- 2. v1 reference (untouched Wan)
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
t0 = time.perf_counter()
videos["v1"] = w.generate(P0["prompt"], size=SIZE, frame_num=FRAME_NUM, shift=SHIFT,
                          sample_solver="unipc", sampling_steps=STEPS, guide_scale=GUIDE,
                          seed=42, offload_model=False)
torch.cuda.synchronize()
timing["v1"] = {"total_s": time.perf_counter() - t0, "peak_gb": torch.cuda.max_memory_allocated() / 2**30,
                "note": "WanT2V.generate incl. T5 encode; T5 resident on GPU; first video = includes warmup"}
log("v1 reference", timing["v1"])
save_video_and_row("v1", P0, timing["v1"])
w.text_encoder.model.cpu()
del w.text_encoder
torch.cuda.empty_cache()

S = Sampler.from_wan_t2v(w)


def generate_and_record(name, row, path, arm="B0", **kw):
    assert (CTRL.path, CTRL.arm) == (path, arm), (CTRL.path, CTRL.arm, path, arm)
    torch.cuda.reset_peak_memory_stats()
    tm = {}
    c, cn = text_context(row["prompt"])
    videos[name], lat[name] = S.generate(c, cn, seed=42, timing=tm)
    tm["peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
    tm["path"], tm["arm"] = path, arm
    tm.update(kw)
    timing[name] = tm
    log(name, tm)
    report["timing"] = timing
    save_video_and_row(name, row, tm)
    write_report()


# ---------------------------------------------------------------- 3. v2 our loop, flash
benchmark.configure_controller({"arm": "B0", "seed": 42, "path": "flash"}, P0, tok)
generate_and_record("v2", P0, "flash")
report["v2_equals_v1"] = bool(torch.equal(videos["v1"], videos["v2"]))
report["v2_vs_v1"] = {"video_maxabs": (videos["v1"] - videos["v2"]).abs().max().item(),
                      "video_psnr_db": psnr(videos["v1"], videos["v2"])}
log("v2 == v1 (byte-identical):", report["v2_equals_v1"])

# ---------------------------------------------------------------- 4. v3 explicit B0 (after warmup)
benchmark.configure_controller({"arm": "B0", "seed": 42}, P0, tok)
c, cn = text_context(P0["prompt"])
S.generate(c, cn, seed=0, steps=2, decode=False)   # explicit-path warmup, untimed
generate_and_record("v3", P0, "explicit")
report["v3_vs_v2"] = {"video_psnr_db": psnr(videos["v3"], videos["v2"]),
                      "video_maxabs": (videos["v3"] - videos["v2"]).abs().max().item(),
                      "latent_maxabs": (lat["v3"] - lat["v2"]).abs().max().item(),
                      "latent_rel_l2": ((lat["v3"] - lat["v2"]).norm() / lat["v2"].norm()).item()}
log("v3 vs v2", report["v3_vs_v2"])

# ---------------------------------------------------------------- 5. v4 L(0,0) + shares + probes
PROBE_LAYERS, PROBE_STEPS = (0, 14, 29), (0, 25, 49)
probes = []


def probe_hook(mod, args, out):
    if CTRL.branch != "cond" or CTRL.step not in PROBE_STEPS or mod._tempo_layer not in PROBE_LAYERS:
        return
    x, context, context_lens = args
    b, n, d = x.size(0), mod.num_heads, mod.head_dim
    with torch.no_grad():
        q = mod.norm_q(mod.q(x)).view(b, -1, n, d)   # x is already norm3(x): hook sees cross_attn's inputs
        k = mod.norm_k(mod.k(context)).view(b, -1, n, d)
        v = mod.v(context).view(b, -1, n, d)
        o_fl = flash_attention(q, k, v, k_lens=None).float()
        c0 = Controller(path="explicit", arm="B0")
        o_ex = explicit_attention(q, k, v, c0).float()
        qd = q[0].to(torch.bfloat16).double().transpose(0, 1)
        kd = k[0].to(torch.bfloat16).double().transpose(0, 1)
        vd = v[0].to(torch.bfloat16).double().transpose(0, 1)
        ref = torch.softmax(qd @ kd.transpose(1, 2) / d ** 0.5, -1) @ vd
        ref = ref.transpose(0, 1).unsqueeze(0)
        ref_bf = ref.to(torch.bfloat16).double()
        rec = {"step": CTRL.step, "layer": mod._tempo_layer}
        for nm, o in (("flash", o_fl.double()), ("explicit", o_ex.double())):
            e = (o - ref).abs()
            rec[nm] = {"maxabs": e.max().item(), "meanabs": e.mean().item(),
                       "frac_ne_bf16_round_of_fp64": (o != ref_bf).double().mean().item()}
        rec["flash_vs_explicit_maxabs"] = (o_fl - o_ex).abs().max().item()
        rec["ref_absmax"] = ref.abs().max().item()
        probes.append(rec)
        del qd, kd, vd, ref, ref_bf


hooks = [blk.cross_attn.register_forward_hook(probe_hook) for blk in S.model.blocks]
info = benchmark.configure_controller({"arm": "L", "beta": 0.0, "gamma": 0.0, "seed": 42, "record_shares": True}, P0, tok)
report["prompt0_tokens"] = info
generate_and_record("v4", P0, "explicit", arm="L", note="beta=gamma=0, record_shares, probes: timing not representative")
for h in hooks:
    h.remove()
report["v4_equals_v3"] = bool(torch.equal(videos["v4"], videos["v3"]))
report["v4_latent_equals_v3"] = bool(torch.equal(lat["v4"], lat["v3"]))
report["attn_probes"] = probes
assert len(probes) == len(PROBE_LAYERS) * len(PROBE_STEPS), len(probes)
# pass: explicit is at least as close to fp64 as flash, on mean error and on bf16-rounding agreement
ok = all(p["explicit"]["meanabs"] <= p["flash"]["meanabs"] and
         p["explicit"]["frac_ne_bf16_round_of_fp64"] <= p["flash"]["frac_ne_bf16_round_of_fp64"] for p in probes)
report["explicit_within_fp32_band"] = ok
log("v4 == v3:", report["v4_equals_v3"], "| explicit err <= flash err on all probes:", ok)
torch.save(CTRL.shares, os.path.join(RES, f"shares_prompt0_B0_{JOB}.pt"))
# compact share summary: mean over layers at selected steps, per latent frame
summ = {}
for st in (0, 10, 25, 49):
    recs = [r for (s_, l_, r) in CTRL.shares if s_ == st]
    summ[st] = {g: torch.stack([r[g] for r in recs]).mean(0).tolist() for g in recs[0]}
report["share_summary_prompt0"] = summ
CTRL.record_shares = False
write_report()

# ---------------------------------------------------------------- 6. v5/v6 timing pair on prompt 1
benchmark.configure_controller({"arm": "B0", "seed": 42, "path": "flash"}, P1, tok)
generate_and_record("v5", P1, "flash")
benchmark.configure_controller({"arm": "B0", "seed": 42}, P1, tok)
generate_and_record("v6", P1, "explicit")
ov = [(timing[e]["total_s"] / timing[f_]["total_s"] - 1) for f_, e in (("v2", "v3"), ("v5", "v6"))]
report["explicit_overhead_frac"] = ov
log("explicit-path overhead vs flash:", ov)

# ---------------------------------------------------------------- 7. S exactness + arm smoke tests
m = torch.tensor(P0["mask"], dtype=torch.float32)
obj = [7, 8]
report["S_exactness"] = {}
for scale in (0.3, 3.0):      # 3.0: logit std ~9, object shares down to fp32 underflow
    g = torch.Generator(device=DEV).manual_seed(0)
    q = torch.randn(1, N_LAT * HW, 12, 128, device=DEV, generator=g) * scale
    k = torch.randn(1, 512, 12, 128, device=DEV, generator=g) * scale
    v = torch.randn(1, 512, 12, 128, device=DEV, generator=g)
    cS = Controller(path="explicit", arm="S", s_hi=0.3, s_lo=0.02, mask=m, obj_idx=obj)
    o_S = explicit_attention(q, k, v, cS).double()
    qd, kd, vd = (t_[0].to(torch.bfloat16).double().transpose(0, 1) for t_ in (q, k, v))
    lg = qd @ kd.transpose(1, 2) / 128 ** 0.5
    tt = (0.02 + m.double().to(DEV) * (0.3 - 0.02)).repeat_interleave(HW).view(1, -1, 1)
    notO = [i for i in range(512) if i not in obj]
    logit_s = torch.logsumexp(lg[..., obj], -1, keepdim=True) - torch.logsumexp(lg[..., notO], -1, keepdim=True)
    log_s = -torch.nn.functional.softplus(-logit_s)
    delta = torch.log(tt) - torch.log1p(-tt) - logit_s
    lg[..., obj] += delta
    p2 = torch.softmax(lg, -1)
    ref_S = (p2 @ vd).transpose(0, 1).unsqueeze(0)
    ind = torch.zeros_like(v)
    ind[0, obj, :, 0] = 1.0                       # output channel 0 = object share after edit
    ind[0, :, :, 1] = 1.0                         # output channel 1 = row sum after edit
    o_ind = explicit_attention(q, k, ind, cS)[0].float()
    report["S_exactness"][f"scale{scale}"] = {
        "min_log_share_before": log_s.min().item(),
        "out_maxabs_vs_fp64_logit_shift": (o_S - ref_S).abs().max().item(),
        "fp64_share_maxerr_vs_target": (p2[..., obj].sum(-1) - tt.view(1, -1)).abs().max().item(),
        "measured_share_maxerr_vs_target(bf16 out)": (o_ind[..., 0] - tt.view(-1, 1).float()).abs().max().item(),
        "row_sum_maxerr(bf16 out)": (o_ind[..., 1] - 1).abs().max().item()}
    assert report["S_exactness"][f"scale{scale}"]["measured_share_maxerr_vs_target(bf16 out)"] < 4e-3
    assert report["S_exactness"][f"scale{scale}"]["row_sum_maxerr(bf16 out)"] < 8e-3
    del qd, kd, vd, lg, p2, ref_S
    if scale == 0.3:
        cP = Controller(path="explicit", arm="P", mask=m, obj_idx=obj)
        shP = explicit_attention(q, k, ind, cP)[0, :, :, 0].float().view(N_LAT, HW, 12)
        report["P_offframe_obj_share_max"] = shP[m < 0.5].abs().max().item()
log("S exactness", report["S_exactness"], "| P off-frame share max", report["P_offframe_obj_share_max"])
write_report()

smoke = {}
for rc in ({"arm": "U", "beta": 2.0}, {"arm": "L", "beta": 2.0, "gamma": 2.0},
           {"arm": "S", "s_hi": 0.3, "s_lo": 0.02}, {"arm": "P"},
           {"arm": "L", "beta": 2.0, "gamma": 2.0, "k_steps": 1}):
    rc = dict(rc, seed=42)
    benchmark.configure_controller(rc, P0, tok)
    c, cn = text_context(P0["prompt"])
    _, x0 = S.generate(c, cn, seed=42, steps=3, decode=False)
    smoke[benchmark.run_tag(rc)] = {"finite": bool(torch.isfinite(x0).all()), "absmean": x0.abs().mean().item()}
report["arm_smoke_3steps"] = smoke
log("smoke", smoke)
write_report()

report["done"] = True
write_report()
log("DONE")
