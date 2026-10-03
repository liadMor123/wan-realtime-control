#!/usr/bin/env python3
"""GIF set for the project page (brief v3 §8), from cached Wan2.1 videos, CPU only.

  make_comparison_gifs.py [--fps 16] [--width 416] [--max-mb 8]  -> results/gifs/*.gif, results/gifs/index.md

Each GIF: B0 (videos/B0_s42) and L beta=gamma=2 (videos/L_b2g2_s42) side by side, one-object benchmark, captioned with the
prompt; under each clip a timeline bar over the 81 output frames marking where the object should be visible (benchmark
mask: latent frame j <-> output frame 0 for j = 0, frames 4j-3..4j for j >= 1) and a cursor at the current frame.
Choice rule (committed in the brief): the first 2 held-out prompts of each timing in file order, ids 2, 3, 22, 23, 42,
43, 62, 63; plus the failure case by rule: among the 20 held-out ids (data/heldout_ids.txt), the most negative
L - B0 accuracy at seed 42 (results/phase2_summary.json); if none is negative, the smallest delta (ties: first id).
If a GIF exceeds --max-mb it is re-written with 96, then 64 palette colours, then at half the frame rate.
"""
import argparse
import json
import os
import sys

import cv2
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

TEMPO = os.environ.get("TEMPO_ROOT", os.path.expanduser("~/tempo"))
sys.path.insert(0, os.path.join(TEMPO, "src"))
sys.path.insert(0, os.path.join(TEMPO, "scripts"))
from tempo_ctrl import benchmark  # noqa: E402
from tempo_ctrl.masks import N_OUT, latent_to_output_frames  # noqa: E402
from generate_benchmark_videos import parse_ids  # noqa: E402

CHOSEN = [2, 3, 22, 23, 42, 43, 62, 63]
REF, ARM = "B0_s42", "L_b2g2_s42"
SURFACE, INK, INK2, GRID = (252, 252, 251), (11, 11, 11), (82, 81, 78), (228, 227, 223)
ON, FAIL = (42, 120, 214), (227, 73, 72)
FONT_DIR = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")


def font(size, bold=False):
    return ImageFont.truetype(os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"), size)


def output_frame_mask(lat_mask):
    m = np.zeros(N_OUT, dtype=bool)
    for j, v in enumerate(lat_mask):
        if v:
            m[latent_to_output_frames(j)] = True
    return m


def read_video(path, width):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        h = round(f.shape[0] * width / f.shape[1])
        frames.append(cv2.cvtColor(cv2.resize(f, (width, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
    cap.release()
    if len(frames) != N_OUT:
        raise RuntimeError(f"{path}: {len(frames)} frames, expected {N_OUT}")
    return frames


def wrap_text(draw, text, fnt, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=fnt) <= width:
            cur = t
        else:
            lines.append(cur)
            cur = w
    return lines + [cur]


def compose_frames(b_frames, l_frames, mask, prompt, pid, acc_b, acc_l, fail_d, fps_note):
    W = b_frames[0].shape[1]
    H = b_frames[0].shape[0]
    PAD, GAP, BAR = 14, 12, 12
    f_cap, f_lab, f_small = font(15), font(13, True), font(11)
    scratch = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    total_w = 2 * W + GAP + 2 * PAD
    head = f"#{pid}  " + ("FAILURE CASE (L − B0 = %+.2f)  " % fail_d if fail_d is not None else "")
    cap_lines = wrap_text(scratch, prompt, f_cap, total_w - 2 * PAD)
    y_clip = PAD + 18 + 20 * len(cap_lines) + 8 + 20
    y_bar = y_clip + H + 8
    total_h = y_bar + BAR + 6 + 16 + PAD
    base = Image.new("RGB", (total_w, total_h), SURFACE)
    d = ImageDraw.Draw(base)
    d.text((PAD, PAD), head, font=f_lab, fill=FAIL if fail_d is not None else INK2)
    for k, line in enumerate(cap_lines):
        d.text((PAD, PAD + 18 + 20 * k), line, font=f_cap, fill=INK)
    xs = [PAD, PAD + W + GAP]
    for x, lab in zip(xs, (f"B0 (Wan2.1, no control)   acc {acc_b:.2f}", f"L β=γ=2 (ours)   acc {acc_l:.2f}")):
        d.text((x, y_clip - 20), lab, font=f_lab, fill=INK)
    cell = W / N_OUT
    for x in xs:                                      # static timeline: object-due frames filled
        d.rectangle([x, y_bar, x + W - 1, y_bar + BAR - 1], fill=GRID)
        for k in np.flatnonzero(mask):
            d.rectangle([x + k * cell, y_bar, x + (k + 1) * cell - 0.01, y_bar + BAR - 1], fill=ON)
    d.rectangle([PAD, y_bar + BAR + 8, PAD + 10, y_bar + BAR + 18], fill=ON)
    d.text((PAD + 15, y_bar + BAR + 6), "object should be visible (benchmark mask)   ▎ current frame   " + fps_note,
           font=f_small, fill=INK2)
    out = []
    for t, (fb, fl) in enumerate(zip(b_frames, l_frames)):
        im = base.copy()
        im.paste(Image.fromarray(fb), (xs[0], y_clip))
        im.paste(Image.fromarray(fl), (xs[1], y_clip))
        dd = ImageDraw.Draw(im)
        for x in xs:
            cx = x + (t + 0.5) * cell
            dd.rectangle([cx - 1.5, y_bar - 4, cx + 1.5, y_bar + BAR + 3], fill=INK)
        dd.text((xs[1] + W - 60, y_bar + BAR + 6), f"t = {t / 16:.2f} s", font=f_small, fill=INK2)
        out.append(im)
    return out, [(x, y_clip, W, H) for x in xs]


def nearest_lut(P):
    """Exact nearest-palette-entry index for every colour of a 6-bit-per-channel RGB cube (Pillow's own palette
    mapping is approximate)."""
    g = np.arange(64, dtype=np.float32) * 4 + 2
    cube = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    lut = np.empty(len(cube), dtype=np.uint8)
    for k in range(0, len(cube), 16384):
        lut[k:k + 16384] = ((cube[k:k + 16384, None, :] - P[None]) ** 2).sum(-1).argmin(1)
    return lut


def write_gif(frames, rects, path, fps, colors=192):
    """One global palette per GIF, so static regions stay identical across frames and Pillow stores only the changed
    rectangle of each frame. Video pixels (inside `rects`) map only to entries learned from video content; the UI
    (text, timeline) maps to all entries. This keeps the saturated UI blue out of the video."""
    pick = frames[:: max(1, len(frames) // 8)][:8]
    crops = [f.crop((x, y, x + w, y + h)) for f in pick for (x, y, w, h) in rects]
    cw, ch = crops[0].size
    vm = Image.new("RGB", (cw, ch * len(crops)))
    for k, c in enumerate(crops):
        vm.paste(c, (0, k * ch))
    nv = colors - 16
    Pv = np.array(vm.quantize(colors=nv, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
                  .getpalette()[: 3 * nv], dtype=np.float32).reshape(-1, 3)
    ui = frames[0].copy()
    for (x, y, w, h) in rects:
        ui.paste((252, 252, 251), (x, y, x + w, y + h))
    Pu = np.array(ui.quantize(colors=16, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
                  .getpalette()[: 3 * 16], dtype=np.float32).reshape(-1, 3)
    P = np.concatenate([Pv, Pu])
    lut_v, lut_all = nearest_lut(Pv), nearest_lut(P)
    flat = P.astype(np.uint8).ravel().tolist()
    pal = []
    for f in frames:
        q = np.asarray(f, dtype=np.uint8) >> 2
        key = (q[..., 0].astype(np.int32) << 12) | (q[..., 1].astype(np.int32) << 6) | q[..., 2]
        idx = lut_all[key]
        for (x, y, w, h) in rects:
            idx[y:y + h, x:x + w] = lut_v[key[y:y + h, x:x + w]]
        im = Image.fromarray(idx, "P")
        im.putpalette(flat)
        pal.append(im)
    # GIF delays are in 1/100 s: 60, 70, 60, 60 ms averages 62.5 ms = 16 fps
    base = 1000 / fps
    dur = [int(base // 10 * 10) + (10 if (k % 4 == 1 and base % 10) else 0) for k in range(len(pal))]
    pal[0].save(path, save_all=True, append_images=pal[1:], duration=dur, loop=0, optimize=True, disposal=1)
    return os.path.getsize(path) / 2**20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--width", type=int, default=416)
    ap.add_argument("--max-mb", type=float, default=8.0)
    ap.add_argument("--out", default=os.path.join(TEMPO, "results", "gifs"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    prompts = benchmark.load_one_object()
    heldout = parse_ids(open(os.path.join(TEMPO, "data", "heldout_ids.txt")).read().strip())
    S = json.load(open(os.path.join(TEMPO, "results", "phase2_summary.json")))
    acc_b = {int(k): v for k, v in S[REF]["acc_per_prompt"].items()}
    acc_l = {int(k): v for k, v in S[ARM]["acc_per_prompt"].items()}
    assert set(heldout) <= set(acc_b) and set(heldout) <= set(acc_l), "held-out ids missing from phase2_summary"
    assert set(CHOSEN) <= set(heldout)
    delta = {i: round(acc_l[i] - acc_b[i], 6) for i in heldout}
    neg = [i for i in heldout if delta[i] < 0]
    fail = min(neg or heldout, key=lambda i: (delta[i], heldout.index(i)))
    fail_rule = "most negative L − B0" if neg else "no negative Δ; smallest Δ"
    print(f"failure case: id {fail} delta {delta[fail]:+.3f} ({fail_rule}); negatives: {[(i, delta[i]) for i in neg]}")

    items = [(i, None) for i in CHOSEN] + [(fail, delta[fail])]
    md = ["# GIF set (Wan2.1-T2V-1.3B, seed 42, one-object benchmark): B0 vs L β=γ=2", "",
          "Left: B0 (`videos/B0_s42`, no control). Right: L β=γ=2 (`videos/L_b2g2_s42`). Under each clip: the 81 output "
          "frames, filled where the benchmark mask says the object should be visible (latent frame j ↔ output frame 0 "
          "for j = 0, frames 4j−3…4j for j ≥ 1), with a cursor at the current frame. Accuracy = official TempoControl "
          "temporal accuracy per video (`results/phase2_summary.json`).", "",
          "**Choice rule (committed in the brief before making the set):** the first 2 held-out prompts of each timing in "
          "file order: ids 2, 3, 22, 23, 42, 43, 62, 63. **Failure case, also by rule:** among the 20 held-out ids "
          "(`data/heldout_ids.txt`), the one with the most negative L − B0 accuracy at seed 42; if none is negative, the "
          f"smallest Δ. Result: id {fail}, Δ = {delta[fail]:+.2f} ({fail_rule}; "
          f"{len(neg)} of 20 held-out prompts have Δ < 0).", "",
          "| GIF | id | prompt | B0 acc | L acc | Δ | size | fps |", "|---|---|---|---|---|---|---|---|"]
    for pid, fd in items:
        p = prompts[pid]
        name = benchmark.video_name(p["prompt"])
        vb = read_video(os.path.join(TEMPO, "videos", REF, name), a.width)
        vl = read_video(os.path.join(TEMPO, "videos", ARM, name), a.width)
        mask = output_frame_mask(p["mask"])
        fname = f"{'failure_' if fd is not None else ''}{pid:02d}_{p['temp_object'].replace(' ', '_')}.gif"
        path = os.path.join(a.out, fname)
        fr, rects = compose_frames(vb, vl, mask, p["prompt"], pid, acc_b[pid], acc_l[pid], fd, "(16 fps)")
        fps, step = a.fps, 1
        for colors in (192, 96, 64):                  # fewer palette colours before dropping frames
            mb = write_gif(fr, rects, path, fps, colors)
            if mb <= a.max_mb:
                break
        else:                                         # last resort: every 2nd frame at half the frame rate
            fps, step = a.fps // 2, 2
            fr, rects = compose_frames(vb, vl, mask, p["prompt"], pid, acc_b[pid], acc_l[pid], fd, "(every 2nd frame)")
            mb = write_gif(fr[::step], rects, path, fps, colors)
        print(f"{fname}: {mb:.2f} MB, {fps} fps, {len(fr[::step])} frames, {colors} colours")
        md.append(f"| [{fname}]({fname}) | {pid} | {p['prompt']} | {acc_b[pid]:.2f} | {acc_l[pid]:.2f} | "
                  f"{acc_l[pid] - acc_b[pid]:+.2f}{' **(failure case)**' if fd is not None else ''} | {mb:.1f} MB | {fps} |")
    open(os.path.join(a.out, "index.md"), "w").write("\n".join(md) + "\n")
    print("wrote", os.path.join(a.out, "index.md"))


if __name__ == "__main__":
    main()
