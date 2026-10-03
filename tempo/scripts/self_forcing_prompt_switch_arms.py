#!/usr/bin/env python3
"""Part 2, step 3c: prompt-switching baselines in Self-Forcing on the fa2kv path (pre-registered protocol, part 2, step 3c).

Modes (one per job):
  rf   PS-RF (Rolling-Forcing-style): at the first pass of the onset chunk the conditioning becomes the full prompt
       and ONLY the cross-attention K/V is re-computed (crossattn_cache[*]["is_init"] = False); the self-attention
       cache keeps the K/V computed under "An empty scene." (3b-2's mechanism). Zero fa2kv table.
  ll   PS-LongLive-style: at the first pass of the onset chunk both caches are rebuilt under the full prompt, as
       LongLive v1 `InteractiveCausalInferencePipeline._recache_after_switch` (commit 9b2102b): zero the self-attention
       k/v, zero the cross-attention cache, re-run the generator on all previous clean chunks at timestep
       context_noise (= 0) with the new prompt, zero the cross-attention cache again. LongLive does the re-run as one
       block-causal pass; Self-Forcing's KV-cache path has no block mask, so it is one context pass per previous chunk
       in chunk order (the same computation). Zero fa2kv table. LongLive's model was trained with this re-cache;
       Self-Forcing's was not.
  rfL  PS-RF + L: mode rf with the full prompt's L(2,2) fa2kv table installed for every pass of every chunk.

Run from the staged tempo-L Self-Forcing repo:  self_forcing_prompt_switch_arms.py --mode rf|ll|rfL --ids 0-79 [--seed 42]
Writes ~/tempo/videos/<tag>/<prompt>-0.mp4, results/rows/phase23_<job>_<mode>.jsonl and results/step3c/sfsw_<job>.json,
with per-chunk timings (host time after a sync at each chunk's 5th pass, as in the ~/proj protocol; passes counted on
the host) and the T5 time of the full prompt.
"""
import argparse
import contextlib
import hashlib
import io
import json
import os
import re
import shutil
import statistics as st
import sys
import time

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.environ.get("TEMPO_ROOT", os.path.expanduser("~/tempo")), "scripts"))

import torch  # noqa: E402

import self_forcing_common as C  # noqa: E402
from tempo_ctrl import benchmark  # noqa: E402
from tempo_ctrl.fused_attention import N_LAT  # noqa: E402
from tempo_ctrl.tokens import load_tokenizer  # noqa: E402

TEMPO = C.TEMPO
PRE = "An empty scene."
TEMPLATE = re.compile(r"^An empty scene\. Suddenly, during the (second|third|fourth|last) second of the video, "
                      r"an? .+? appears out of nowhere, drawing all attention\.$")
NFPB = 3
PASSES = 5                                   # per chunk: 4 denoising + 1 clean-context KV-cache pass
TAGS = {"rf": "SF_PSRF_fa2kv_s{}", "ll": "SF_PSLL_fa2kv_s{}", "rfL": "SF_PSRFL_fa2kv_s{}"}
ARM = {"rf": "PS-RF", "ll": "PS-LongLive-style", "rfL": "PS-RF+L"}


def parse_ids(s):
    out = []
    for part in s.split(","):
        x, _, y = part.partition("-")
        out += list(range(int(x), int(y) + 1)) if y else [int(x)]
    return out


def log(*a):
    print(f"[sfsw {time.strftime('%H:%M:%S')}]", *a, flush=True)


def onset_chunk(mask):
    m = [int(x) for x in mask]
    first = m.index(1)
    assert all(m[first:]), "mask is not 'on from the onset to the end'"
    return first // NFPB, first


class Switcher:
    """Wraps pipe.generator.forward. Counts passes on the host, records each chunk's clean frames (the input of its
    KV-cache pass), performs the switch at the first pass of the switch chunk, and times chunks (one sync per chunk;
    in mode ll also one sync before and after the re-cache, to time it)."""

    def __init__(self, pipe, fwd, sync=None, clock=time.perf_counter):
        self.pipe, self._fwd = pipe, fwd
        self.sync = sync or torch.cuda.synchronize
        self.clock = clock
        self.reset(None, None, None)

    def reset(self, mode, switch, cond):
        self.mode, self.switch, self.cond = mode, switch, cond
        self.n, self.t0, self.tc, self.clean = 0, None, [], []
        self.switched, self.n_recache, self.recache_ms = False, 0, None

    def recache(self):
        """LongLive v1 _recache_after_switch (9b2102b, lines 34-96), global_sink = False branch."""
        p = self.pipe
        for blk in p.kv_cache1:                               # 1. reset the self-attention cache (k, v; not indices)
            blk["k"].zero_()
            blk["v"].zero_()
        for blk in p.crossattn_cache:                         # 2. reset the cross-attention cache
            blk["k"].zero_()
            blk["v"].zero_()
            blk["is_init"] = False
        if len(self.clean) != self.switch:
            raise RuntimeError(f"{len(self.clean)} clean chunks recorded before switch chunk {self.switch}")
        fs = p.frame_seq_length
        for j, (x, ts) in enumerate(self.clean):              # 3. re-run all previous clean chunks, new prompt
            self._fwd(noisy_image_or_video=x, conditional_dict=self.cond, timestep=ts, kv_cache=p.kv_cache1,
                      crossattn_cache=p.crossattn_cache, current_start=j * NFPB * fs)
            self.n_recache += 1
        for blk in p.crossattn_cache:                         # 4. reset the cross-attention cache again
            blk["k"].zero_()
            blk["v"].zero_()
            blk["is_init"] = False

    def __call__(self, *args, **kw):
        if args or "noisy_image_or_video" not in kw or "conditional_dict" not in kw:
            raise RuntimeError("generator called with an unexpected signature")
        chunk, in_chunk = divmod(self.n, PASSES)
        self.n += 1
        if self.t0 is None:
            self.sync()
            self.t0 = self.clock()
        if self.switch is not None and chunk >= self.switch:
            if not self.switched:
                if chunk != self.switch or in_chunk != 0:
                    raise RuntimeError(f"switch at chunk {chunk} pass {in_chunk}, expected chunk {self.switch} pass 0")
                if self.mode == "ll":
                    self.sync()
                    t = self.clock()
                    self.recache()
                    self.sync()
                    self.recache_ms = (self.clock() - t) * 1e3
                else:
                    for c in self.pipe.crossattn_cache:      # re-cache ONLY the cross-attention K/V
                        c["is_init"] = False
                self.switched = True
            kw["conditional_dict"] = self.cond
        if in_chunk == PASSES - 1:                            # the KV-cache pass: its input is the chunk's clean frames
            if kw["current_start"] != chunk * NFPB * self.pipe.frame_seq_length:
                raise RuntimeError("unexpected current_start in the KV-cache pass")
            x, ts = kw["noisy_image_or_video"], kw["timestep"]
            self.clean.append((x.clone(), ts.clone()) if self.mode == "ll" else (x, ts))
        out = self._fwd(*args, **kw)
        if in_chunk == PASSES - 1:
            self.sync()
            self.tc.append(self.clock())
        return out

    def chunk_ms(self):
        ch, prev = [], self.t0
        for t in self.tc:
            ch.append((t - prev) * 1e3)
            prev = t
        return ch


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=sorted(TAGS))
    ap.add_argument("--ids", required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    mode = a.mode
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
    tok = load_tokenizer(benchmark.WAN_ROOT, benchmark.CKPT)
    tag = TAGS[mode].format(a.seed)
    scr = os.path.join(os.environ.get("TMPDIR", "/tmp"), "tempo", f"sfsw_{job}")
    rows_path = os.path.join(TEMPO, "results", "rows", f"phase23_{job}_{mode}.jsonl")
    res_dir = os.path.join(TEMPO, "results", "step3c")
    os.makedirs(res_dir, exist_ok=True)
    res_path = os.path.join(res_dir, f"sfsw_{job}.json")
    rep = {"job": job, "mode": mode, "arm": ARM[mode], "tag": tag, "host": os.uname().nodename, "gpu": gpu,
           "pre_prompt": PRE, "seed": a.seed, "path": "fa2kv", "table": "L(2,2)" if mode == "rfL" else "zero",
           "longlive_ref": "NVlabs/LongLive 9b2102b pipeline/interactive_causal_inference.py:_recache_after_switch"
           if mode == "ll" else None, "videos": []}

    pipe = C.build_pipeline(dev)
    model = pipe.generator.model
    TAB = C.fa2kv_zero_tables(dev)                               # the one static (qb, ke) pair; per-prompt values copied in
    C.install_tempo_bias(model, TAB)

    def set_table(row, L):
        if L:
            tl, obj_idx = C.l_fa2kv_tables(row, tok, dev, 2.0, 2.0)
            TAB[0].copy_(tl[0])
            TAB[1].copy_(tl[1])
        else:
            TAB[0].zero_()
            TAB[1].zero_()
            obj_idx = C.l_frame_table(row, tok)[1]
        return obj_idx

    n_pre = len(tok.tokenizer(PRE, add_special_tokens=True)["input_ids"])
    rep["pre_prompt_tokens"] = n_pre

    SW = Switcher(pipe, pipe.generator.forward)
    pipe.generator.forward = SW

    def generate(first_prompt, switch, cond_after, seed, m=mode):
        SW.reset(m, switch, cond_after)
        set_seed(seed)
        noise = torch.randn([1, N_LAT, 16, 60, 104], device=dev, dtype=torch.bfloat16)
        with contextlib.redirect_stdout(io.StringIO()):
            video, lat = pipe.inference(noise=noise, text_prompts=[first_prompt], return_latents=True,
                                        initial_latent=None, low_memory=True)
        torch.cuda.synchronize()
        pipe.vae.model.clear_cache()
        if switch is not None and not SW.switched:
            raise RuntimeError("switch never happened")
        ch = SW.chunk_ms()
        if len(ch) != 7 or SW.n != 7 * PASSES:
            raise RuntimeError(f"{len(ch)} chunks timed / {SW.n} passes, expected 7 / {7 * PASSES}")
        want_rc = switch if (m == "ll" and switch is not None) else 0
        if SW.n_recache != want_rc:
            raise RuntimeError(f"{SW.n_recache} re-cache passes, expected {want_rc}")
        return video, lat, ch

    def encode(p):
        torch.cuda.synchronize()
        t = time.perf_counter()
        c = pipe.text_encoder(text_prompts=[p])
        torch.cuda.synchronize()
        return c, (time.perf_counter() - t) * 1e3

    def self_attention_k_chunk0(blocks=(0, 1, 15, 29)):
        fs3 = NFPB * pipe.frame_seq_length
        return [pipe.kv_cache1[b]["k"][:, :fs3].clone() for b in blocks]

    # ---------------- machinery checks (pre-registered protocol, part 2, 3c-2) ----------------
    r0 = rows[ids[0]]
    obj0 = set_table(r0, mode == "rfL")
    if min(obj0) < n_pre:
        raise SystemExit(f"object slots {obj0} overlap the pre-onset prompt's {n_pre} tokens")
    cond_pre, _ = encode(PRE)
    embs = [v for v in cond_pre.values() if torch.is_tensor(v) and v.dim() == 3]
    if len(embs) != 1 or embs[0][0, n_pre:].abs().max().item() != 0:
        raise SystemExit("pre-onset T5 context is not zero past its length (padding slots not identical)")
    del cond_pre, embs
    cond_full, _ = encode(r0["prompt"])
    chk = {}
    if float(pipe.args.context_noise) != 0:
        raise SystemExit("context_noise is not 0")
    # M1: switch at chunk 0 == plain full prompt
    _, l_sw0, _ = generate(PRE, 0, cond_full, a.seed)
    v_full, l_full, _ = generate(r0["prompt"], None, None, a.seed)
    ref_tag = f"SF_L_b2g2_fa2kv_s{a.seed}" if mode == "rfL" else f"SF_B0_fa2kv_s{a.seed}"
    name0 = benchmark.video_name(r0["prompt"])
    m0 = os.path.join(scr, "m0", name0)
    benchmark.save_video((v_full[0].float().cpu() * 2 - 1).permute(1, 0, 2, 3), m0)
    chk[f"M0_plain_full_equals_stored_{ref_tag}_mp4"] = md5(m0) == md5(os.path.join(TEMPO, "videos", ref_tag, name0))
    del v_full
    kx_full = [c["k"].clone() for c in pipe.crossattn_cache]            # cross-attention K after a full-prompt run
    chk["M1_switch_at_0_equals_full"] = bool(torch.equal(l_sw0, l_full))
    # M2: identity switch (full -> full) at chunk 3 == plain full prompt
    _, l_id3, _ = generate(r0["prompt"], 3, cond_full, a.seed)
    chk["M2_identity_switch_at_3_equals_full"] = bool(torch.equal(l_id3, l_full))
    # M3: PRE -> full at chunk 1
    _, l_pre, _ = generate(PRE, None, None, a.seed)
    _, l_s1, _ = generate(PRE, 1, cond_full, a.seed)
    k0_s1 = self_attention_k_chunk0()
    chk["M3_chunk0_equals_pre_only"] = bool(torch.equal(l_s1[:, :3], l_pre[:, :3]))
    chk["M3_later_chunks_differ_from_pre_only"] = bool(not torch.equal(l_s1[:, 3:], l_pre[:, 3:]))
    chk["M3_crossattn_k_equals_full"] = all(torch.equal(c["k"], kf) for c, kf in zip(pipe.crossattn_cache, kx_full))
    if mode == "ll":                                    # M4: the self-attention cache was really rebuilt
        _, l_s1_rf, _ = generate(PRE, 1, cond_full, a.seed, m="rf")
        k0_rf = self_attention_k_chunk0()
        # block 0's self-attention K sees no text (cross-attention comes after it): equal = same frames, position and
        # timestep; blocks >= 1 must differ, because they were recomputed under the full prompt
        chk["M4_self_k_chunk0_block0_equals_rf"] = bool(torch.equal(k0_s1[0], k0_rf[0]))
        chk["M4_self_k_chunk0_blocks_1_15_29_differ_from_rf"] = all(not torch.equal(x, y)
                                                                   for x, y in zip(k0_s1[1:], k0_rf[1:]))
        chk["M4_later_chunks_differ_from_rf"] = bool(not torch.equal(l_s1[:, 3:], l_s1_rf[:, 3:]))
        del l_s1_rf, k0_rf
    if mode == "rfL":                                   # M5: the bias reaches the model
        sw0, _ = onset_chunk(r0["mask"])
        _, l_L, _ = generate(PRE, sw0, cond_full, a.seed)
        set_table(r0, False)
        _, l_Z, _ = generate(PRE, sw0, cond_full, a.seed)
        set_table(r0, True)
        chk["M5_L_differs_from_zero_table"] = bool(not torch.equal(l_L, l_Z))
        del l_L, l_Z
    rep["checks"] = chk
    log("machinery checks:", chk)
    json.dump(rep, open(res_path, "w"), indent=1)
    if not all(chk.values()):
        raise SystemExit("3c machinery check failed")
    del l_sw0, l_full, l_id3, l_pre, l_s1, kx_full, k0_s1

    # ---------------- measured videos ----------------
    for i in ids:
        row = rows[i]
        sw, first_on = onset_chunk(row["mask"])
        obj_idx = set_table(row, mode == "rfL")
        if min(obj_idx) < n_pre:
            raise SystemExit(f"prompt {i}: object slots overlap the pre-onset prompt")
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
        rec = {"phase": 23, "step": "3c", "run_tag": tag, "arm": ARM[mode], "mode": mode, "path": "fa2kv",
               "table": rep["table"], "params": {"beta": 2.0, "gamma": 2.0} if mode == "rfL" else {},
               "prompt_id": i, "temp_object": row["temp_object"], "seed": a.seed, "pre_prompt": PRE,
               "first_on": first_on, "switch_chunk": sw, "chunk_ms": ch, "recache_ms": SW.recache_ms,
               "n_recache_passes": SW.n_recache, "t5_full_prompt_ms": t5_ms, "wall_s": wall, "peak_gb": peak,
               "obj_idx": obj_idx, "video_path": os.path.join(dst, name), "job": job, "host": os.uname().nodename,
               "timing_note": "eager; one sync per chunk (mode ll: +1 sync before and after the re-cache, which runs "
                              "inside the switch chunk); wall excludes the full-prompt T5 encode",
               "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "temporal_accuracy": None}
        benchmark.append_jsonl(rows_path, rec)
        rep["videos"].append(rec)
        json.dump(rep, open(res_path, "w"), indent=1)
        log(f"p{i:02d} {row['temp_object']}: switch at chunk {sw} (first on {first_on}); chunks "
            f"{[round(x) for x in ch]} ms; re-cache {SW.recache_ms} ms; T5 {t5_ms:.0f} ms")

    # switch-chunk latency: at each onset position, switch videos vs videos where that position is a normal chunk
    lat = {}
    for p in sorted({v["switch_chunk"] for v in rep["videos"]}):
        sw_ = [v["chunk_ms"][p] for v in rep["videos"] if v["switch_chunk"] == p]
        no_ = [v["chunk_ms"][p] for v in rep["videos"] if v["switch_chunk"] != p]
        if not no_:
            continue
        e = {"switch_median_ms": st.median(sw_), "normal_median_ms": st.median(no_), "n_switch": len(sw_),
             "n_normal": len(no_), "overhead": st.median(sw_) / st.median(no_) - 1}
        if mode == "ll":
            rc = [v["recache_ms"] for v in rep["videos"] if v["switch_chunk"] == p]
            e["recache_median_ms"] = st.median(rc)
            d = e["switch_median_ms"] - e["normal_median_ms"]
            e["recache_share_of_overhead"] = e["recache_median_ms"] / d if d > 0 else None
        lat[p] = e
    rep["switch_latency"] = lat
    rep["t5_full_prompt_ms_median"] = st.median(v["t5_full_prompt_ms"] for v in rep["videos"])
    if mode == "rf":                                    # reported only: fa2kv zero table vs 3b-2's unmodified path
        old = os.path.join(TEMPO, "videos", f"SF_PS_s{a.seed}")
        cmp_ = {}
        for v in rep["videos"]:
            o = os.path.join(old, os.path.basename(v["video_path"]))
            if os.path.isfile(o):
                cmp_[v["prompt_id"]] = md5(o) == md5(v["video_path"])
        rep["vs_3b2_SF_PS_mp4_bytes_equal"] = {"n": len(cmp_), "n_equal": sum(cmp_.values()), "per_prompt": cmp_}
        log("PS-RF (fa2kv zero) vs 3b-2 SF_PS mp4 bytes:", rep["vs_3b2_SF_PS_mp4_bytes_equal"]["n_equal"], "/",
            len(cmp_), "equal")
    rep["done"] = True
    json.dump(rep, open(res_path, "w"), indent=1)
    log("switch-chunk latency", json.dumps(lat), "| T5 median ms", rep["t5_full_prompt_ms_median"])
    log("DONE")


if __name__ == "__main__":
    main()
