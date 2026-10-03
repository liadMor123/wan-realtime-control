#!/usr/bin/env python3
"""Part 2, step 3b-2: prompt-switching baseline in Self-Forcing (pre-registered protocol, part 2, step 3b).

Unmodified Self-Forcing cross-attention (tempo_bias None). Chunks before the onset chunk use the pre-onset prompt
"An empty scene."; from the first pass of the onset chunk (= first_on // 3) the full benchmark prompt's T5 encoding is
used and ONLY the cross-attention K/V cache is re-computed (crossattn_cache[*]["is_init"] = False; a switch at a chunk
boundary as in Rolling Forcing, 2509.25161 App. C). The self-attention KV cache of past chunks is kept unchanged (it was
computed under the pre-onset prompt). LongLive's KV-recache, which also rebuilds the self-attention cache, is NOT this.

Run from the staged tempo-L Self-Forcing repo:  self_forcing_prompt_switch_heldout.py --ids 2-6,22-26,42-46,62-66 [--seed 42]
Writes ~/tempo/videos/SF_PS_s<seed>/<prompt>-0.mp4 and results/rows/phase23_<job>_ps.jsonl, with per-chunk timings
(synchronised host time after each chunk's 5th pass, as in the ~/proj protocol; passes counted on the host) and the T5 time of the full prompt.
"""
import argparse
import contextlib
import io
import json
import os
import re
import shutil
import sys
import time

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.environ.get("TEMPO_ROOT", os.path.expanduser("~/tempo")), "scripts"))

import torch  # noqa: E402

import self_forcing_common as C  # noqa: E402
from tempo_ctrl import benchmark  # noqa: E402
from tempo_ctrl.fused_attention import N_LAT  # noqa: E402

TEMPO = C.TEMPO
PRE = "An empty scene."
TEMPLATE = re.compile(r"^An empty scene\. Suddenly, during the (second|third|fourth|last) second of the video, "
                      r"an? .+? appears out of nowhere, drawing all attention\.$")
NFPB = 3


def parse_ids(s):
    out = []
    for part in s.split(","):
        x, _, y = part.partition("-")
        out += list(range(int(x), int(y) + 1)) if y else [int(x)]
    return out


def log(*a):
    print(f"[sfps {time.strftime('%H:%M:%S')}]", *a, flush=True)


def onset_chunk(mask):
    m = [int(x) for x in mask]
    first = m.index(1)
    assert all(m[first:]), "mask is not 'on from the onset to the end'"
    return first // NFPB, first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
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
    for i in ids:
        if not TEMPLATE.match(rows[i]["prompt"]):
            raise SystemExit(f"prompt {i} does not follow the benchmark template")
    tag = f"SF_PS_s{a.seed}"
    scr = os.path.join(os.environ.get("TMPDIR", "/tmp"), "tempo", f"sfps_{job}")
    rows_path = os.path.join(TEMPO, "results", "rows", f"phase23_{job}_ps.jsonl")
    res_path = os.path.join(TEMPO, "results", "step3", f"sfps_{job}.json")
    rep = {"job": job, "host": os.uname().nodename, "gpu": gpu, "pre_prompt": PRE, "seed": a.seed, "videos": []}

    pipe = C.build_pipeline(dev)
    C.install_tempo_bias(pipe.generator.model, None)          # unmodified cross-attention (original flash call)
    S = {"n": 0, "chunk": -1, "in_chunk": 0, "t0": None, "tc": [], "switch": None, "cond": None, "switched": False}
    _fwd = pipe.generator.forward

    def fwd(*args, **kw):
        # chunks counted on the host (5 passes each: 4 denoising + 1 KV-cache pass); no per-pass device sync
        S["chunk"], S["in_chunk"] = divmod(S["n"], 5)
        S["n"] += 1
        if S["t0"] is None:
            torch.cuda.synchronize()
            S["t0"] = time.perf_counter()
        if S["switch"] is not None and S["chunk"] >= S["switch"]:
            if not S["switched"]:
                assert S["chunk"] == S["switch"] and S["in_chunk"] == 0
                for c in pipe.crossattn_cache:           # re-cache ONLY the cross-attention K/V
                    c["is_init"] = False
                S["switched"] = True
            kw["conditional_dict"] = S["cond"]
        out = _fwd(*args, **kw)
        if S["in_chunk"] == 4:
            torch.cuda.synchronize()
            S["tc"].append(time.perf_counter())
        return out

    pipe.generator.forward = fwd

    def generate(first_prompt, switch, cond_after, seed):
        S.update({"n": 0, "chunk": -1, "in_chunk": 0, "t0": None, "tc": [], "switch": switch, "cond": cond_after,
                  "switched": False})
        set_seed(seed)
        noise = torch.randn([1, N_LAT, 16, 60, 104], device=dev, dtype=torch.bfloat16)
        with contextlib.redirect_stdout(io.StringIO()):
            video, lat = pipe.inference(noise=noise, text_prompts=[first_prompt], return_latents=True,
                                        initial_latent=None, low_memory=True)
        torch.cuda.synchronize()
        pipe.vae.model.clear_cache()
        if switch is not None and not S["switched"]:
            raise RuntimeError("switch never happened")
        ch, prev = [], S["t0"]
        for t in S["tc"]:
            ch.append((t - prev) * 1e3)
            prev = t
        if len(ch) != 7 or S["n"] != 35:
            raise RuntimeError(f"{len(ch)} chunks timed / {S['n']} passes, expected 7 / 35")
        return video, lat, ch

    def encode(p):
        torch.cuda.synchronize()
        t = time.perf_counter()
        c = pipe.text_encoder(text_prompts=[p])
        torch.cuda.synchronize()
        return c, (time.perf_counter() - t) * 1e3

    # --- machinery check: a switch at chunk 0 == plain generation from the full prompt (bitwise); also a warmup
    r0 = rows[ids[0]]
    cond_full, _ = encode(r0["prompt"])
    v_sw, l_sw, _ = generate(PRE, 0, cond_full, a.seed)
    v_pl, l_pl, _ = generate(r0["prompt"], None, None, a.seed)
    kv_full = [c["k"].clone() for c in pipe.crossattn_cache]           # cross-attention K after a full-prompt run
    rep["check_switch_at_0_equals_plain_full"] = {"latent": bool(torch.equal(l_sw, l_pl)),
                                                  "video": bool(torch.equal(v_sw, v_pl))}
    log("machinery check (switch at chunk 0 == plain full prompt):", rep["check_switch_at_0_equals_plain_full"])
    if not rep["check_switch_at_0_equals_plain_full"]["latent"]:
        raise SystemExit("prompt-switch machinery check failed")
    # mid-stream switch (chunk 1): chunk 0 must equal a PRE-only run bitwise, later chunks must differ, and the
    # cross-attention cache after the run must be the full prompt's (i.e. the re-cache really happened)
    _, l_pre, _ = generate(PRE, None, None, a.seed)
    _, l_s1, _ = generate(PRE, 1, cond_full, a.seed)
    chk = {"chunk0_equals_pre_only": bool(torch.equal(l_s1[:, :3], l_pre[:, :3])),
           "later_chunks_differ_from_pre_only": bool(not torch.equal(l_s1[:, 3:], l_pre[:, 3:])),
           "crossattn_k_equals_full_prompt": all(torch.equal(c["k"], kf)
                                                 for c, kf in zip(pipe.crossattn_cache, kv_full))}
    rep["check_switch_at_1"] = chk
    log("machinery check (switch at chunk 1):", chk)
    if not all(chk.values()):
        raise SystemExit("mid-stream prompt-switch check failed")
    del v_sw, l_sw, v_pl, l_pl, l_pre, l_s1, kv_full

    for i in ids:
        row = rows[i]
        sw, first_on = onset_chunk(row["mask"])
        cond_full, t5_ms = encode(row["prompt"])
        torch.cuda.reset_peak_memory_stats()
        t_wall = time.perf_counter()
        video, _, ch = generate(PRE, sw, cond_full, a.seed)
        wall = time.perf_counter() - t_wall
        peak = torch.cuda.max_memory_allocated() / 2**30
        video = video[0].float().cpu()
        if video.shape != (81, 3, 480, 832) or not torch.isfinite(video).all():
            raise RuntimeError(f"bad video for prompt {i}")
        name = benchmark.video_name(row["prompt"])
        tmp = os.path.join(scr, tag, name)
        benchmark.save_video((video * 2 - 1).permute(1, 0, 2, 3), tmp)
        dst = os.path.join(TEMPO, "videos", tag)
        os.makedirs(dst, exist_ok=True)
        shutil.copy2(tmp, os.path.join(dst, name + ".part"))
        os.replace(os.path.join(dst, name + ".part"), os.path.join(dst, name))
        rec = {"phase": 23, "run_tag": tag, "arm": "PS", "path": "unmodified", "prompt_id": i,
               "temp_object": row["temp_object"], "seed": a.seed, "pre_prompt": PRE, "first_on": first_on,
               "switch_chunk": sw, "chunk_ms": ch, "t5_full_prompt_ms": t5_ms, "wall_s": wall, "peak_gb": peak,
               "video_path": os.path.join(dst, name), "job": job, "host": os.uname().nodename,
               "timing_note": "eager; wall excludes the full-prompt T5 encode (t5_full_prompt_ms)",
               "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "temporal_accuracy": None}
        benchmark.append_jsonl(rows_path, rec)
        rep["videos"].append(rec)
        json.dump(rep, open(res_path, "w"), indent=1)
        log(f"p{i:02d} {row['temp_object']}: switch at chunk {sw} (first on {first_on}); chunks "
            f"{[round(x) for x in ch]} ms; T5 {t5_ms:.0f} ms")

    # switch-chunk latency: at each onset position, switch videos vs videos where that position is a normal chunk
    lat = {}
    import statistics as st
    for p in sorted({v["switch_chunk"] for v in rep["videos"]}):
        sw_ = [v["chunk_ms"][p] for v in rep["videos"] if v["switch_chunk"] == p]
        no_ = [v["chunk_ms"][p] for v in rep["videos"] if v["switch_chunk"] != p]
        lat[p] = {"switch_median_ms": st.median(sw_), "normal_median_ms": st.median(no_), "n_switch": len(sw_),
                  "n_normal": len(no_), "overhead": st.median(sw_) / st.median(no_) - 1}
    rep["switch_latency"] = lat
    rep["t5_full_prompt_ms_median"] = st.median(v["t5_full_prompt_ms"] for v in rep["videos"])
    rep["done"] = True
    json.dump(rep, open(res_path, "w"), indent=1)
    log("switch-chunk latency", json.dumps(lat), "| T5 median ms", rep["t5_full_prompt_ms_median"])
    log("DONE")


if __name__ == "__main__":
    main()
