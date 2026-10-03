#!/usr/bin/env python3
"""Part-2 deliverable figures (brief v3 §8), CPU only, re-runnable. Missing inputs are skipped with a warning.

  plot_deliverable_figures.py [--out results/figures]
    fig_accuracy.{png,pdf}       temporal accuracy per arm; bars = 95 % bootstrap CI of the paired Δ vs B0, drawn around
                                 each arm's mean (as scripts/plot_pilot_heldout_figure.py). Panels: Wan2.1 single object (step 4),
                                 Self-Forcing single object (step 3b seeds 42/43 + step 3c prompt-switching arms),
                                 two objects (step 5a). Filled = all 80 prompts, open = the 72 non-pilot prompts.
    fig_time_memory.{png,pdf}    step 2: time per video and peak-memory delta per attention path (Wan2.1, prompts 0-1)
    fig_sf_block_cost.{png,pdf}  step 3: Self-Forcing per-chunk cost of L vs chunk position, final and eager-patched

Inputs: results/step4_summary.json, results/step3b_summary.json, results/step3c_summary.json (optional),
results/step5a_summary.json (optional), results/rows/phase22_*.jsonl (+ results/step2/step2_fa2kv_*.json cross-check),
results/step3/lat_*/summary.json.
"""
import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

TEMPO = os.environ.get("TEMPO_ROOT", os.path.expanduser("~/tempo"))
R = os.path.join(TEMPO, "results")
# same tokens as scripts/plot_pilot_heldout_figure.py (validated reference palette, light mode)
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 9})


def warn(msg):
    print("WARNING:", msg, file=sys.stderr)


def load(name):
    p = os.path.join(R, name)
    if not os.path.isfile(p):
        warn(f"{p} not found; its panel/arms are skipped (re-run this script once it exists)")
        return None
    return json.load(open(p))


def style(ax, grid_axis="x"):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    ax.grid(axis=grid_axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def save(fig, out, stem):
    for ext in ("png", "pdf"):
        p = os.path.join(out, f"{stem}.{ext}")
        fig.savefig(p, dpi=180, facecolor=SURFACE, bbox_inches="tight")
        print("wrote", p)
    plt.close(fig)


# ---------------------------------------------------------------- (a) accuracy
def arm_ci(o):
    """Paired-Δ CI re-centred on the arm's own mean, clipped to [0, 1]."""
    lo = o["acc_mean"] + o["d_vs_ref_ci95"][0] - o["d_vs_ref_mean"]
    hi = o["acc_mean"] + o["d_vs_ref_ci95"][1] - o["d_vs_ref_mean"]
    return max(lo, 0.0), min(hi, 1.0)


def accuracy_panels():
    """Each panel: (title, subtitle, groups); group = (name, [(label, color, is_ref, {subset: stats})])."""
    panels = []
    s4 = load("step4_summary.json")
    if s4:
        panels.append(("Wan2.1-T2V-1.3B, single object (step 4, seed 42)", "explicit path; 80 prompts", [
            ("", [("B0 (no control)", MUTED, True, {"80": s4["all80"]["B0"], "72": s4["np72"]["B0"]}),
                  ("L β=γ=2", BLUE, False, {"80": s4["all80"]["L"], "72": s4["np72"]["L"]})])]))
    s3b, s3c = load("step3b_summary.json"), load("step3c_summary.json")
    if s3b:
        g42 = [("B0 (unmodified SF)", MUTED, True, {"80": s3b["step3b_s42_all80"]["B0"], "72": s3b["step3b_s42_np72"]["B0"]}),
               ("L β=γ=2", BLUE, False, {"80": s3b["step3b_s42_all80"]["L"], "72": s3b["step3b_s42_np72"]["L"]})]
        if s3c:
            for arm, lab, c in (("PS-RF", "PS-RF", ORANGE),
                                ("PS-LongLive-style", "PS-LongLive-style", ORANGE),
                                ("PS-RF+L", "PS-RF + L", AQUA)):
                g42.append((lab, c, False, {"80": s3c["all80"]["vs_B0"][arm], "72": s3c["np72"]["vs_B0"][arm]}))
                # 3c's B0 must be step 3b's seed-42 B0 (same videos)
            assert abs(s3c["all80"]["vs_B0"]["B0"]["acc_mean"] - s3b["step3b_s42_all80"]["B0"]["acc_mean"]) < 1e-9
        g43 = [("B0 (unmodified SF)", MUTED, True, {"80": s3b["step3b_s43_all80"]["B0"], "72": s3b["step3b_s43_np72"]["B0"]}),
               ("L β=γ=2", BLUE, False, {"80": s3b["step3b_s43_all80"]["L"], "72": s3b["step3b_s43_np72"]["L"]})]
        panels.append(("Self-Forcing (streaming, 3-frame chunks), single object (steps 3b, 3c)",
                       "fa2kv path; 80 prompts" + ("" if s3c else "; step-3c prompt-switching arms not yet available"),
                       [("seed 42", g42), ("seed 43", g43)]))
    s5 = load("step5a_summary.json")
    if s5:
        n = s5["L"]["n"]
        s5["B0"].setdefault("d_vs_ref_mean", 0.0)             # the reference carries no Δ fields in step5a_summary
        s5["B0"].setdefault("d_vs_ref_ci95", [0.0, 0.0])
        groups = [("5a: first 20" if load("step5b_summary.json") else "",
                   [("B0 (no control)", MUTED, True, {"pairs": s5["B0"]}),
                    ("L β=γ=2, K = 2", BLUE, False, {"pairs": s5["L"]})])]
        s5b = load("step5b_summary.json")
        title, sub = "Wan2.1-T2V-1.3B, two objects (step 5a, seed 42)", f"explicit path; {n} scorable pairs"
        if s5b:
            r82 = s5b["results"]["all82"]
            r82["B0"].setdefault("d_vs_ref_mean", 0.0)
            r82["B0"].setdefault("d_vs_ref_ci95", [0.0, 0.0])
            groups.insert(0, ("5b: all 82 pairs",
                              [("B0 (no control)", MUTED, True, {"pairs": r82["B0"]}),
                               ("L β=γ=2, K = 2", BLUE, False, {"pairs": r82["L"]})]))
            title, sub = "Wan2.1-T2V-1.3B, two objects (steps 5a, 5b, seed 42)", "explicit path; 5b = all 82 pairs (TempoControl: 37.5 → 53.2 %)"
        panels.append((title, sub, groups))
    return panels


def fig_accuracy(out):
    panels = accuracy_panels()
    if not panels:
        warn("no accuracy summaries found; fig_accuracy skipped")
        return
    OFF = 0.17                                            # vertical offset: all-80 above, 72 non-pilot below
    layout = []                                           # per panel: list of (y, label, color, is_ref, stats, group)
    for title, sub, groups in panels:
        rows, y = [], 0.0
        for gname, arms in groups:
            for lab, c, ref, st in arms:
                rows.append((y, lab, c, ref, st, gname))
                y += 1
            y += 0.6                                      # gap between groups
        layout.append(rows)
    heights = [max(r[0] for r in rows) + 1.6 for rows in layout]
    xs = [v for rows in layout for r in rows for s in r[4].values() for v in arm_ci(s) + (s["acc_mean"],)]
    xlo = min(0.3, min(xs) - 0.04)
    fig, axes = plt.subplots(len(panels), 1, figsize=(9.2, 0.42 * sum(heights) + 1.2 * len(panels)),
                             gridspec_kw={"height_ratios": heights}, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, (title, sub, groups), rows in zip(axes[:, 0], panels, layout):
        style(ax)
        ticks, labels = [], []
        for gname in dict.fromkeys(r[5] for r in rows):
            grows = [r for r in rows if r[5] == gname]
            refrow = next(r for r in grows if r[3])
            y0, y1 = grows[0][0] - 0.45, grows[-1][0] + 0.45
            for key, ls in (("80", (0, (3, 3))), ("pairs", (0, (3, 3))), ("72", (0, (1, 2)))):
                if key in refrow[4]:
                    ax.plot([refrow[4][key]["acc_mean"]] * 2, [y0, y1], color=MUTED, lw=1, ls=ls, zorder=1)
            if gname:
                ax.text(-0.01, y0 - 0.05, gname, transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                        color=INK2, fontsize=8.5, style="italic")
        for y, lab, c, ref, st, _ in rows:
            ticks.append(y)
            labels.append(lab)
            subsets = [k for k in ("80", "72", "pairs") if k in st]
            txt = []
            for k in subsets:
                o = st[k]
                yy = y - OFF if k == "80" else y + OFF if k == "72" else y
                lo, hi = arm_ci(o)
                if not ref:
                    ax.plot([lo, hi], [yy, yy], color=c, lw=2, solid_capstyle="round", zorder=2)
                open_ = k == "72"
                ax.plot(o["acc_mean"], yy, "o", ms=7, color=SURFACE if open_ else c, mec=c if open_ else SURFACE,
                        mew=1.6 if open_ else 1.5, zorder=3)
                d = "" if ref else f"  {100 * o['d_vs_ref_mean']:+.1f} [{100 * o['d_vs_ref_ci95'][0]:+.1f}, " \
                                   f"{100 * o['d_vs_ref_ci95'][1]:+.1f}]"
                nlab = {"80": "n=80", "72": "n=72", "pairs": f"n={o['n']}"}[k]
                txt.append((yy, f"{o['acc_mean']:.3f}{d}  ({nlab})"))
            for yy, t in txt:
                ax.text(1.01, yy, t, transform=ax.get_yaxis_transform(), va="center", color=INK, fontsize=7.8)
        ax.set_yticks(ticks, labels, color=INK, fontsize=9)
        ax.set_ylim(max(r[0] for r in rows) + 0.7, -0.9)
        ax.set_xlim(xlo, 1.0)
        ax.set_title(title, loc="left", color=INK, fontsize=10, pad=14)
        ax.text(0, 1.005, sub, transform=ax.transAxes, color=INK2, fontsize=8, va="bottom")
    axes[-1, 0].set_xlabel("temporal accuracy (official TempoControl metric)", color=INK2)
    fig.text(0.01, 0.005, "Dots: mean accuracy. Bars: 95 % bootstrap CI (10,000 resamples) of the paired difference vs "
             "B0 of the same group, drawn around the arm's mean. Filled = all 80 prompts, open = 72 non-pilot prompts;\n"
             "grey dashed / dotted line = B0 mean on 80 / 72 prompts. Right: accuracy, Δ vs B0 in points [95 % CI]. "
             "PS = prompt switching at the scheduled time (step 3c baselines).", color=INK2, fontsize=7.5, va="bottom")
    handles = [Line2D([], [], marker="o", ls="", color=INK2, mec=SURFACE, ms=7, label="all 80 prompts (or all pairs)"),
               Line2D([], [], marker="o", ls="", color=SURFACE, mec=INK2, mew=1.6, ms=7, label="72 non-pilot prompts")]
    axes[0, 0].legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=False, fontsize=8,
                      labelcolor=INK2, borderaxespad=0.2)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    save(fig, out, "fig_accuracy")


# ---------------------------------------------------------------- (b) time / memory
def fused_path_timing_rows():
    by_job = {}
    for p in sorted(glob.glob(os.path.join(R, "rows", "phase22_*.jsonl"))):
        for l in open(p):
            if l.strip():
                r = json.loads(l)
                by_job.setdefault(r["job"], []).append(r)
    return by_job


def fig_time_memory(out):
    by_job = fused_path_timing_rows()
    fa_job = next((j for j, rs in by_job.items() if any(r.get("path") == "fa2kv" for r in rs)), None)
    sd_job = next((j for j, rs in by_job.items() if any(r.get("path") == "fused" for r in rs)), None)
    if fa_job is None:
        warn("no phase-22 rows with path fa2kv; fig_time_memory skipped")
        return

    def get(job, path):
        rs = sorted([r for r in by_job[job] if r.get("path", "explicit") == path], key=lambda r: r["prompt_id"])
        return {r["prompt_id"]: r for r in rs}

    ref = {fa_job: get(fa_job, "flash")}
    paths = [("flash (Wan default), B0", "flash", fa_job, MUTED), ("fa2kv (FA2 + additive KV bias), L", "fa2kv", fa_job, BLUE)]
    if sd_job:
        ref[sd_job] = get(sd_job, "flash")
        paths.append(("SDPA with additive mask, L", "fused", sd_job, ORANGE))
    paths.append(("explicit (unfused), L", "explicit", fa_job, AQUA))
    # cross-check with the step-2 JSON of the fa2kv job
    js = os.path.join(R, "step2", f"step2_fa2kv_{fa_job}.json")
    if os.path.isfile(js):
        ts = json.load(open(js))["timing_summary"]
        for pid, v in ts.items():
            r_l, r_f = get(fa_job, "fa2kv")[int(pid)], ref[fa_job][int(pid)]
            assert abs(r_l["wall_s"] / r_f["wall_s"] - 1 - v["L_b2g2_fa2kv_s42"]["total_vs_flash"]) < 1e-9

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.3), gridspec_kw={"width_ratios": [1.35, 1]})
    fig.patch.set_facecolor(SURFACE)
    style(ax1), style(ax2)
    labels = []
    for y, (lab, path, job, c) in enumerate(paths):
        rs = get(job, path)
        labels.append(lab)
        walls = [rs[p]["wall_s"] for p in sorted(rs)]
        ovh = [rs[p]["wall_s"] / ref[job][p]["wall_s"] - 1 for p in sorted(rs)]
        dmem = [1024 * (rs[p]["peak_gb"] - ref[job][p]["peak_gb"]) for p in sorted(rs)]
        for k, w in enumerate(walls):
            ax1.plot(w, y + (k - 0.5) * 0.18, "o", ms=7, color=c, mec=SURFACE, mew=1.5)
        t = "  ".join(f"{w:.1f}" for w in walls)
        if path != "flash":
            t += "  s   (" + " / ".join(f"{100 * o:+.2f} %" for o in ovh) + ")"
        else:
            t += "  s   (reference)"
        ax1.text(max(walls) + 0.5, y, t, va="center", color=INK, fontsize=8)
        for k, m in enumerate(dmem):
            ax2.plot([0, m], [y + (k - 0.5) * 0.18] * 2, color=c, lw=2, solid_capstyle="round")
            ax2.plot(m, y + (k - 0.5) * 0.18, "o", ms=7, color=c, mec=SURFACE, mew=1.5)
        mt = ("peak " + " / ".join(f"{rs[p]['peak_gb']:.2f}" for p in sorted(rs)) + " GB") if path == "flash" else \
            " / ".join(f"{m:+.1f}" for m in dmem) + " MB"
        ax2.text(max(max(dmem), 0) + 3, y, mt, va="center", color=INK, fontsize=8)
        print(f"{lab:38s} job {job}  wall {walls}  overhead {[round(100 * o, 2) for o in ovh]} %  "
              f"peak {[round(rs[p]['peak_gb'], 4) for p in sorted(rs)]}  dmem MB {[round(m, 1) for m in dmem]}")
    for ax in (ax1, ax2):
        ax.set_yticks(range(len(paths)), labels if ax is ax1 else [""] * len(paths), color=INK)
        ax.set_ylim(len(paths) - 0.5, -0.6)
    ax1.set_xlim(185, 222)
    ax1.set_xlabel("wall time per video (s), 81 frames, 50 steps + CFG", color=INK2)
    ax1.set_title("(a) Time per video by attention path", loc="left", color=INK, fontsize=10, pad=14)
    ax1.text(0, 1.01, "prompts 0 and 1 (one dot each); % vs flash B0, same prompt and job", transform=ax1.transAxes,
             color=INK2, fontsize=7.5, va="bottom")
    ax2.axvline(0, color=MUTED, lw=1, ls=(0, (3, 3)))
    ax2.set_xlim(-10, 130)
    ax2.set_xlabel("peak allocated memory minus flash B0, same prompt and job (MB)", color=INK2)
    ax2.set_title("(b) Peak GPU memory vs flash B0", loc="left", color=INK, fontsize=10, pad=14)
    ax2.text(0, 1.01, "same prompt and job; flash B0 absolute peak shown", transform=ax2.transAxes, color=INK2,
             fontsize=7.5, va="bottom")
    fig.text(0.01, 0.005, f"Wan2.1-T2V-1.3B (81 frames, 50 steps, CFG), A100-40GB, one node. Step-2 jobs {fa_job} (flash, fa2kv, "
             f"explicit)" + (f" and {sd_job} (SDPA mask, compared with that job's own flash B0 run)" if sd_job else "")
             + ". L = β=γ=2 temporal bias.", color=INK2, fontsize=7.5, va="bottom")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, out, "fig_time_memory")


# ---------------------------------------------------------------- (c) Self-Forcing per-chunk cost
def fig_sf_block_cost(out):
    cands = [p for p in sorted(glob.glob(os.path.join(R, "step3", "lat_*", "summary.json")))
             if "final" in json.load(open(p))]
    if not cands:
        warn("no results/step3/lat_*/summary.json with a `final` mode; fig_sf_block_cost skipped")
        return
    p = cands[-1]
    S = json.load(open(p))
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    fig.patch.set_facecolor(SURFACE)
    style(ax, "y")
    ax.axhline(2.0, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(6.55, 2.0, "2 % budget (Q2)", va="center", ha="left", color=INK2, fontsize=8)
    for mode, c, lab in (("final", BLUE, "final (Inductor + CUDA graphs)"), ("eager-patched", ORANGE, "eager-patched")):
        if mode not in S:
            continue
        pos = S[mode]["positions"]
        x = [q["pos"] for q in pos]
        y = [100 * q["cost"] for q in pos]
        ax.plot(x, y, color=c, lw=2, zorder=2)
        ax.plot(x, y, "o", ms=7, color=c, mec=SURFACE, mew=1.5, zorder=3, label=f"{lab}: mean {100 * S[mode]['mean_cost']:+.2f} %")
        for xi, yi, q in zip(x, y, pos):
            ax.text(xi, yi + (0.13 if mode == "eager-patched" else -0.13), f"{yi:+.2f}", ha="center",
                    va="bottom" if mode == "eager-patched" else "top", color=INK, fontsize=7.5, zorder=4,
                    bbox=dict(boxstyle="square,pad=0.1", fc=SURFACE, ec="none"))
        ax.axhline(100 * S[mode]["mean_cost"], color=c, lw=0.8, ls=(0, (1, 2)), zorder=1)
        ax.text(6.55, 100 * S[mode]["mean_cost"], f"mean {100 * S[mode]['mean_cost']:+.2f} %", va="center",
                ha="left", color=INK2, fontsize=7.5)
        print(mode, "per position %:", [round(v, 2) for v in y], "mean", round(100 * S[mode]["mean_cost"], 2),
              "delta ms:", [round(q["delta_ms"], 1) for q in pos])
    ax.set_xticks(range(7))
    ax.set_xlim(-0.4, 6.4)
    ax.set_ylim(0, 4.0)
    ax.set_xlabel("chunk position in the stream (3 latent frames per chunk; KV cache grows with position)", color=INK2)
    ax.set_ylabel("cost of L per chunk (%)", color=INK2)
    ax.set_title("Self-Forcing: per-chunk cost of L vs unmodified, by chunk position", loc="left", color=INK,
                 fontsize=10, pad=34)
    ax.text(0, 1.055, "Q2 cost: met on the mean (+1.58 %), missed at positions 0–1 (+2.2 %, +2.1 %)",
            transform=ax.transAxes, color=INK, fontsize=9, fontweight="bold", va="bottom")
    nf, ne = S.get("final", {}).get("n_videos"), S.get("eager-patched", {}).get("n_videos")
    ax.text(0, 1.005, f"median chunk time per position, L / unmodified − 1; final {nf} vs {nf} videos (ABBA), "
            f"eager {ne} vs {ne}; A100-40GB", transform=ax.transAxes, color=INK2, fontsize=7.5, va="bottom")
    ax.legend(loc="upper right", frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout()
    save(fig, out, "fig_sf_block_cost")
    print("latency source:", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(R, "figures"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fig_accuracy(a.out)
    fig_time_memory(a.out)
    fig_sf_block_cost(a.out)


if __name__ == "__main__":
    main()
