#!/usr/bin/env python3
"""Part 2, step 3: temporal-accuracy videos from Self-Forcing (streaming, 3-frame blocks, 4 steps, no CFG).

Two arms, both through the fa2kv cross-attention path (L via Wan's FA2 kernel by key augmentation; the step-2b choice,
pre-registered decisions 7-8 and 14):
  SF_B0_fa2kv_s42      : all-zero tables
  SF_L_b2g2_fa2kv_s42  : L(2,2) tables, applied in every pass of every block (4 denoising + 1 KV-cache pass)
Eager mode, capture patches on, perf patches on, attn_splits 0. set_seed(--seed, default 42) before each video's noise
draw, as in Self-Forcing's inference.py (step 3b runs seeds 42 and 43; tags carry the seed).

Run from the staged tempo-L Self-Forcing repo:
  self_forcing_generate_videos.py --ids 2-6,22-26,42-46,62-66
Writes ~/tempo/videos/<tag>/<prompt>-0.mp4 and ~/tempo/results/rows/phase23_<job>.jsonl. Resumable.

Preflight (fails loudly, before any video): A100 check; Self-Forcing's tokenizer ids == tempo's for every prompt;
FA2 kernel at Self-Forcing's shape; compiled == eager; call-count check during the first video (every layer sees
q_start = 0, 4680, ..., 28080, five passes each); the first prompt's L video must differ from its B0 video. Recorded:
whether zero-table fa2kv is bitwise equal to the original flash call, per call and for the whole first B0 video.
"""
import argparse
import contextlib
import io
import json
import os
import shutil
import sys
import time
from collections import Counter

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.environ.get("TEMPO_ROOT", os.path.expanduser("~/tempo")), "scripts"))

import torch  # noqa: E402
from torch.autograd import DeviceType  # noqa: E402

import self_forcing_common as C  # noqa: E402
from tempo_ctrl import benchmark  # noqa: E402
from tempo_ctrl.fused_attention import HW, LK, N_LAT  # noqa: E402
from tempo_ctrl.tokens import load_tokenizer  # noqa: E402

TEMPO = C.TEMPO
SEED = 42                                              # default; --seed overrides (step 3b runs 42 and 43)


def arm_tags(seed):
    return [(f"SF_B0_fa2kv_s{seed}", None), (f"SF_L_b2g2_fa2kv_s{seed}", (2.0, 2.0))]


def parse_ids(s):
    """Same as generate_benchmark_videos.parse_ids; generate_benchmark_videos.py is not imported because it puts ext/Wan2.1 (another `wan`) on sys.path."""
    out = []
    for part in s.split(","):
        x, _, y = part.partition("-")
        out += list(range(int(x), int(y) + 1)) if y else [int(x)]
    return out


def log(*a):
    print(f"[sfgen {time.strftime('%H:%M:%S')}]", *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    seed = a.seed
    ARMS = arm_tags(seed)
    job = os.environ.get("SLURM_JOB_ID", "local")
    dev = torch.device("cuda")
    torch.set_grad_enabled(False)
    gpu = torch.cuda.get_device_name(0)
    if "A100-SXM4-40GB" not in gpu:
        raise SystemExit(f"wrong GPU {gpu!r}: A100-SXM4-40GB only")
    from utils.misc import set_seed
    from wan import patch_flags
    patch_flags.set_enabled(True)
    patch_flags.set_perf(True)
    patch_flags.set_attn_splits(0)
    patch_flags.set_attn_split_rule(None)
    patch_flags.set_fused_merge(False)

    rows = benchmark.load_one_object()
    ids = parse_ids(a.ids)
    tok = load_tokenizer(benchmark.WAN_ROOT, benchmark.CKPT)
    scr = os.path.join(os.environ.get("TMPDIR", "/tmp"), "tempo", f"sfgen_{job}")
    rows_path = os.path.join(TEMPO, "results", "rows", f"phase23_{job}.jsonl")
    res_dir = os.path.join(TEMPO, "results", "step3")
    rep_name = f"sfgen_{job}_s{seed}.json"
    os.makedirs(res_dir, exist_ok=True)
    rep = {"seed": seed, "ids": ids, "job": job, "host": os.uname().nodename, "gpu": gpu, "torch": torch.__version__,
           "patch_flags": {"enabled": True, "perf": True, "attn_splits": 0}, "weights": "generator_ema"}

    pipe = C.build_pipeline(dev)
    model = pipe.generator.model
    TAB = C.fa2kv_zero_tables(dev)                           # the one static (qb, ke) pair; per-run values are copied in
    C.install_tempo_bias(model, TAB)

    # --- preflight 1: tokenizer agreement (object slots come from tempo's tokenizer)
    for i in ids:
        p = rows[i]["prompt"]
        ids_sf, _ = pipe.text_encoder.tokenizer([p], return_mask=True, add_special_tokens=True)
        ids_t = tok([p], return_mask=False)
        if not torch.equal(ids_sf.cpu(), ids_t.cpu()):
            raise SystemExit(f"tokenizer mismatch on prompt {i}")
    # --- preflight 2: kernel at Self-Forcing's shape (4,680 queries, offset rows), with L-like tables
    from wan.modules.attention import flash_attention
    g = torch.Generator(device=dev).manual_seed(0)
    q = torch.randn(1, 3 * HW, 12, 128, device=dev, generator=g, dtype=torch.bfloat16)
    k = torch.randn(1, LK, 12, 128, device=dev, generator=g, dtype=torch.bfloat16)
    TL = C.fa2kv_tables(rows[ids[0]]["mask"], [5, 6], 2.0, 2.0, dev)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            C.fa2kv_attention(q, k, k, TL, 9 * HW)
        torch.cuda.synchronize()
    kn = sorted({e.key for e in prof.key_averages() if e.device_type == DeviceType.CUDA})
    rep["kernels"] = kn
    log("fa2kv kernels", kn)
    rep["fa2_kernel_head_dim"] = C.fa2_kernel_head_dim(kn)
    if (rep["fa2_kernel_head_dim"] or 0) < 136:                         # FA2 forward kernel at d >= 136
        raise SystemExit("fa2kv did not run FA2's forward kernel")
    rep["zero_call_bitwise_equals_flash"] = bool(torch.equal(C.fa2kv_attention(q, k, k, TAB, 9 * HW),
                                                             flash_attention(q, k, k)))
    log("zero-table fa2kv call bitwise == flash_attention:", rep["zero_call_bitwise_equals_flash"])
    # --- preflight 3: the fa2kv call under torch.compile (as in the latency harness's `final` mode) must give the same
    # output as eager and still run FA2's kernel
    comp = torch.compile(C.fa2kv_attention, mode="max-autotune-no-cudagraphs", dynamic=False)
    o_e = C.fa2kv_attention(q, k, k, TL, 9 * HW)
    o_c = comp(q, k, k, TL, 9 * HW)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        comp(q, k, k, TL, 9 * HW)
        torch.cuda.synchronize()
    kc = sorted({e.key for e in prof.key_averages() if e.device_type == DeviceType.CUDA})
    rep["compiled_kernels"], rep["compiled_equals_eager"] = kc, bool(torch.equal(o_e, o_c))
    log("compiled fa2kv kernels", kc, "equal to eager:", rep["compiled_equals_eager"])
    if not rep["compiled_equals_eager"] or (C.fa2_kernel_head_dim(kc) or 0) < 136:
        raise SystemExit(f"compiled fa2kv call differs from eager: {kc} vs {kn}")
    del TL
    del q, k, o_e, o_c

    calls = Counter()

    def counting_attn(q, k, v, table, q_start):
        calls[(q.shape[1], q_start)] += 1
        return C.fa2kv_attention(q, k, v, table, q_start)

    first = {}
    for n_item, i in enumerate(ids):
        row = rows[i]
        for tag, bg in ARMS:
            dst_dir = os.path.join(TEMPO, "videos", tag)
            name = benchmark.video_name(row["prompt"])
            if os.path.isfile(os.path.join(dst_dir, name)) and n_item > 0:
                continue
            if bg is None:
                TAB[0].zero_()
                TAB[1].zero_()
                obj_idx = C.l_frame_table(row, tok)[1]
            else:
                tl, obj_idx = C.l_fa2kv_tables(row, tok, dev, *bg)
                TAB[0].copy_(tl[0])
                TAB[1].copy_(tl[1])
            check = n_item == 0
            if check:
                calls.clear()
                C.install_tempo_bias(model, TAB, counting_attn)
            set_seed(seed)
            noise = torch.randn([1, N_LAT, 16, 60, 104], device=dev, dtype=torch.bfloat16)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                video, lat = pipe.inference(noise=noise, text_prompts=[row["prompt"]], return_latents=True,
                                            initial_latent=None, low_memory=True)
            torch.cuda.synchronize()
            wall = time.perf_counter() - t0
            peak = torch.cuda.max_memory_allocated() / 2**30
            pipe.vae.model.clear_cache()
            if check:
                C.install_tempo_bias(model, TAB)
                want = {(3 * HW, f0 * HW): 5 * C.N_BLOCKS for f0 in range(0, N_LAT, 3)}
                if dict(calls) != want:
                    raise SystemExit(f"cross-attention call pattern {dict(calls)} != {want}")
                first[tag] = (video.clone(), lat.clone())
            video = video[0].float().cpu()                         # [T, C, H, W] in [0, 1]
            if video.shape != (81, 3, 480, 832) or not torch.isfinite(video).all():
                raise RuntimeError(f"bad video {tuple(video.shape)} for {tag} prompt {i}")
            tmp = os.path.join(scr, tag, name)
            benchmark.save_video((video * 2 - 1).permute(1, 0, 2, 3), tmp)   # [C, T, H, W] in [-1, 1]
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy2(tmp, os.path.join(dst_dir, name + ".part"))
            os.replace(os.path.join(dst_dir, name + ".part"), os.path.join(dst_dir, name))
            benchmark.append_jsonl(rows_path, {
                "phase": 23, "run_tag": tag, "arm": "B0" if bg is None else "L", "path": "fa2kv",
                "model": "Self-Forcing dmd (generator_ema)", "params": {} if bg is None else {"beta": bg[0], "gamma": bg[1]},
                "prompt_id": i, "temp_object": row["temp_object"], "seed": seed, "wall_s": wall, "peak_gb": peak,
                "timing_note": "descriptive only (eager, first prompt includes CUDA init); latency is the step3_lat job",
                "video_path": os.path.join(dst_dir, name), "job": job, "host": os.uname().nodename,
                "obj_idx": obj_idx, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "temporal_accuracy": None})
            log(f"{tag} p{i:02d} {row['temp_object']}: {wall:.1f}s peak {peak:.2f}GB")
        if n_item == 0:
            # unmodified Self-Forcing (original flash call) on the same prompt and seed: is B0 bitwise equal to it?
            C.install_tempo_bias(model, None)
            set_seed(seed)
            noise = torch.randn([1, N_LAT, 16, 60, 104], device=dev, dtype=torch.bfloat16)
            with contextlib.redirect_stdout(io.StringIO()):
                v_orig, l_orig = pipe.inference(noise=noise, text_prompts=[row["prompt"]], return_latents=True,
                                                initial_latent=None, low_memory=True)
            pipe.vae.model.clear_cache()
            C.install_tempo_bias(model, TAB)
            vb0, lb0 = first[ARMS[0][0]]
            rep["first_prompt_B0_vs_unmodified"] = {           # reported only: B0 is fa2kv with zero tables
                "latent_bitwise": bool(torch.equal(l_orig, lb0)), "video_bitwise": bool(torch.equal(v_orig, vb0)),
                "latent_rel_l2": ((l_orig.float() - lb0.float()).norm() / l_orig.float().norm()).item(),
                "video_maxabs": (v_orig - vb0).abs().max().item()}
            log("first prompt: B0 (fa2kv, zero tables) vs unmodified Self-Forcing:", rep["first_prompt_B0_vs_unmodified"])
            del v_orig, l_orig
            d = (first[ARMS[1][0]][0] - vb0).abs().max().item()
            first.clear()
            rep["first_prompt_L_vs_B0_maxabs"] = d
            rep["call_pattern_ok"] = True
            log("first prompt: call pattern ok; L vs B0 video max-abs diff", d)
            if d == 0:
                raise SystemExit("L video identical to B0: the bias is not reaching the model")
            json.dump(rep, open(os.path.join(res_dir, rep_name), "w"), indent=1)
    rep["done"] = True
    json.dump(rep, open(os.path.join(res_dir, rep_name), "w"), indent=1)
    log("DONE")


if __name__ == "__main__":
    main()
