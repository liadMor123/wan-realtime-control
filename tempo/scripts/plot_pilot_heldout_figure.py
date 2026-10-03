#!/usr/bin/env python3
"""findings figure: (a) held-out temporal accuracy per run, 95 % bootstrap CI of the paired difference vs
B0 seed 42 drawn around each run's mean, with the seed-to-seed control shown as its own row; (b) wall-time
overhead vs B0 on the same prompt and node, all arms (pilot + held-out). Two single-axis panels.

  plot_pilot_heldout_figure.py --summary results/phase2_summary.json --pilot results/phase1_summary.json -> results/findings_figure.png
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TEMPO = os.environ.get("TEMPO_ROOT", os.path.expanduser("~/tempo"))
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
EDIT, BASE = "#2a78d6", "#8a8984"          # categorical slot 1; neutral for baselines
LABEL = {"B0_s42": "B0 (seed 42)", "B0_s43": "B0 (seed 43)", "L_b2g2_s42": "L β=γ=2",
         "L_b4g4_s42": "L β=γ=4", "L_b2g2_k10_s42": "L β=γ=2, steps<10", "P_s42": "P (mask)",
         "U_b2_s42": "U β=2", "S_hi0.01lo0.0001_s42": "S (0.01, 1e-4)", "S_hi0.05lo0.0001_s42": "S (0.05, 1e-4)",
         "L_b2g2_s43": "L β=γ=2 (seed 43)", "L_b2g2_k10_s43": "L β=γ=2, steps<10 (seed 43)"}


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--out", default=os.path.join(TEMPO, "results", "findings_figure.png"))
    a = ap.parse_args()
    S = json.load(open(a.summary))
    P = json.load(open(a.pilot))
    runs = [t for t in S if not t.startswith("_") and t.endswith("_s42") or t == "B0_s43"]
    runs = sorted(runs, key=lambda t: S[t]["acc_mean"])
    ref = S["B0_s42"]["acc_mean"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.25, 1]})
    fig.patch.set_facecolor(SURFACE)
    style(ax1)
    for y, t in enumerate(runs):
        o = S[t]
        base = t.startswith("B0")
        c = BASE if base else EDIT
        # paired-difference CI, centred on this run's own mean
        lo = o["acc_mean"] + o["d_vs_ref_ci95"][0] - o["d_vs_ref_mean"]
        hi = o["acc_mean"] + o["d_vs_ref_ci95"][1] - o["d_vs_ref_mean"]
        lo, hi = max(lo, 0.0), min(hi, 1.0)          # accuracy is bounded in [0, 1]
        if t != "B0_s42":
            ax1.plot([lo, hi], [y, y], color=c, lw=2, solid_capstyle="round")
        ax1.plot(o["acc_mean"], y, "o", ms=8, color=c, mec=SURFACE, mew=2)
        ax1.text(1.02, y, f"{o['acc_mean']:.2f}", va="center", color=INK, fontsize=9,
                 transform=ax1.get_yaxis_transform())
    ax1.axvline(ref, color=BASE, lw=1, ls=(0, (3, 3)))
    ax1.set_yticks(range(len(runs)), [LABEL.get(t, t) for t in runs], color=INK, fontsize=9)
    ax1.set_xlim(0.3, 1.0)
    ax1.set_xlabel("temporal accuracy (official metric), held-out prompts", color=INK2, fontsize=9)
    n = S[runs[0]]["n"]
    ax1.set_title(f"(a) Temporal accuracy, n = {n} prompts", loc="left", color=INK, fontsize=10, pad=16)
    ax1.text(0, 1.01, "bars: 95% bootstrap CI of the paired difference vs B0 seed 42 (dashed)",
             transform=ax1.transAxes, color=INK2, fontsize=8, va="bottom")

    style(ax2)
    arms = {t: P[t]["overhead_vs_ref_mean"] for t in P if not t.startswith("_") and t != "B0_s42"}
    arms.update({t: S[t]["overhead_vs_ref_mean"] for t in S if not t.startswith("_") and t not in ("B0_s42", "B0_s43")})
    order = sorted(arms, key=lambda t: arms[t])
    for y, t in enumerate(order):
        v = 100 * arms[t]
        ax2.plot([0, v], [y, y], color=EDIT, lw=2, solid_capstyle="round")
        ax2.plot(v, y, "o", ms=8, color=EDIT, mec=SURFACE, mew=2)
        ax2.text(v + 0.12, y, f"+{v:.1f}%", va="center", color=INK, fontsize=9)
    ax2.set_yticks(range(len(order)), [LABEL.get(t, t) for t in order], color=INK, fontsize=9)
    ax2.set_xlim(0, max(4.0, 100 * max(arms.values()) + 1.0))
    ax2.set_xlabel("wall-time overhead, same prompt and node (%)", color=INK2, fontsize=9)
    ax2.set_title("(b) Time overhead per arm", loc="left", color=INK, fontsize=10, pad=16)
    ax2.text(0, 1.01, "vs B0 on the explicit path (≈203 s/video)", transform=ax2.transAxes, color=INK2,
             fontsize=8, va="bottom")
    fig.tight_layout()
    fig.savefig(a.out, dpi=160, facecolor=SURFACE)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
