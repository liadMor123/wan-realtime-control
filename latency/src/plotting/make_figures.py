#!/usr/bin/env python3
"""
Regenerate every Part-A figure (figures/fig1_*.png ... fig8_*.png) from results/*.json. No GPU needed.

Figures 3, 4, 7 and 8 are drawn from the result files j6_fa2sweep.json,
j7_attn2.json, j6_sweep.json and j9_metrics.json. Figures 1, 2, 5 and 6 plot
headline per-pass / per-chunk numbers derived from the J3b, J5, J6 and J8
result files (j3b_*.json, j54.json, j6_gaps.json, j8_*.json); those values
are written out as literals below so each figure reads as a summary.

Usage: python src/plotting/make_figures.py   (run from latency/ or anywhere)
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
RESULTS_DIR = os.path.join(ROOT, "results")
FIGURES_DIR = os.path.join(ROOT, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False})
COLORS = {"eager": "#4C566A", "final": "#BF616A", "a": "#5E81AC", "b": "#A3BE8C",
          "c": "#D08770", "d": "#B48EAD", "e": "#8FBCBB"}


def load_json(name):
    return json.load(open(os.path.join(RESULTS_DIR, name)))


comp = [("host-device\nsync sites", 9.60), ("launch boundaries\n(eager kernels)", 4.99),
        ("Inductor fusion +\nkernel selection", 26.81), ("launch boundaries\nafter fusion", 3.11),
        ("remaining\n(best mode)", 97.06)]
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.bar([c[0] for c in comp], [c[1] for c in comp],
       color=[COLORS["a"], COLORS["b"], COLORS["c"], COLORS["d"], COLORS["eager"]])
for i, c in enumerate(comp): ax.text(i, c[1] + 1.5, f"{c[1]:.2f}", ha="center", fontsize=8)
ax.set_ylabel("ms per pass")
ax.set_title("Fig 1 — J3b: decomposition of the 136.6 ms/pass K-independent cost")
plt.tight_layout(); plt.savefig(os.path.join(FIGURES_DIR, "fig1_j3b_decomposition.png")); plt.close()

lab = ["GEMM arithmetic\n(at peak)", "GEMM\ninefficiency", "KV-cache\nwrites",
       "elementwise\n/ norm", "cross-\nattention"]
val = [37.58, 20.71, 19.69, 14.78, 3.45]
fig, ax = plt.subplots(figsize=(6.4, 3.4))
ax.barh(lab[::-1], val[::-1], color=[COLORS["e"], COLORS["c"], COLORS["b"], COLORS["a"], COLORS["d"]][::-1])
for i, v in enumerate(val[::-1]):
    ax.text(v + 0.5, i, f"{v:.2f} ms  ({100*v/sum(val):.1f} %)", va="center", fontsize=8)
ax.set_xlim(0, 48); ax.set_xlabel("ms per pass")
ax.set_title("Fig 2 — J5: where the 96.2 ms/pass K-independent work goes")
plt.tight_layout(); plt.savefig(os.path.join(FIGURES_DIR, "fig2_j5_census.png")); plt.close()

st = load_json("j6_fa2sweep.json"); s6 = [r for r in st if r["position"] == 6]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.4))
a1.plot([r["heads"] for r in s6], [r["ms"] for r in s6], "o-", color=COLORS["final"], ms=4)
for h in (12, 15, 18, 21, 24): a1.axvline(h, color=COLORS["eager"], ls=":", lw=0.8)
a1.set_xlabel("attention heads"); a1.set_ylabel("FA2 kernel time (ms)")
a1.set_title("head-count staircase, K = 32,760\nsteps where ceil(37H/108) increments", fontsize=9)
ker = ["FA2\nself-attn", "FFN down", "FFN up", "QKV /\nout-proj", "cuBLAS\nN=1536"]
tl = [19.2, 7.7, 2.9, 10.2, 18.2]
a2.bar(ker, tl, color=[COLORS["final"], COLORS["c"], COLORS["b"], COLORS["a"], COLORS["d"]])
for i, v in enumerate(tl): a2.text(i, v + 0.4, f"{v:.1f}%", ha="center", fontsize=8)
a2.set_ylabel("tail imbalance (%)"); a2.set_title("tail by kernel (ncu, position 6)", fontsize=9)
plt.suptitle("Fig 3 — J6: wave quantization measured two ways", y=1.02, fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(FIGURES_DIR, "fig3_j6_tails.png"), bbox_inches="tight"); plt.close()

j7 = load_json("j7_attn2.json")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.4))
for pos, col in ((0, COLORS["a"]), (3, COLORS["b"]), (6, COLORS["final"])):
    rs = sorted([r for r in j7["splits"] if r["position"] == pos and r["num_splits"] != "heuristic"],
                key=lambda r: r["num_splits"])
    a1.plot([r["num_splits"] for r in rs], [r["speedup_pct"] for r in rs], "o-",
            color=col, ms=4, label=f"position {pos}")
a1.axhline(19.2, color=COLORS["eager"], ls="--", lw=0.9, label="measured tail 19.2 %")
a1.set_xlabel("num_splits"); a1.set_ylabel("speed-up vs varlen (%)")
a1.legend(fontsize=7); a1.set_title("split-KV recovers the tail", fontsize=9)
sc = j7["staircase"]
a2.plot([r["heads"] for r in sc], [r["us_per_cta"] for r in sc], "o-", color=COLORS["b"], ms=4,
        label="num_splits = 4")
a2.plot([r["heads"] for r in s6], [1e3 * r["ms"] / r["ctas"] for r in s6], "s--",
        color=COLORS["eager"], ms=3, alpha=0.7, label="varlen (before)")
a2.set_xlabel("attention heads"); a2.set_ylabel("us per CTA"); a2.legend(fontsize=7)
a2.set_title("staircase flattened: 2.9 % spread", fontsize=9)
plt.suptitle("Fig 4 — J7: the conventional split-KV arm", y=1.02, fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(FIGURES_DIR, "fig4_j7_splitkv.png"), bbox_inches="tight"); plt.close()

bars = [("split-KV", 107.9), ("Inductor fusion", 94.9), ("sync sites*", 48.0),
        ("cache-write fusion", 45.9), ("graphs", 8.0)]
fig, ax = plt.subplots(figsize=(7.6, 3.6))
run = 1498.3
ax.bar(0, run, color=COLORS["eager"], width=0.6); ax.text(0, run + 14, "1498", ha="center", fontsize=8)
for i, (n, v) in enumerate(bars, 1):
    ax.bar(i, v, bottom=run - v, color=COLORS["c"], width=0.6)
    ax.text(i, run + 6, f"-{v:.1f}", ha="center", fontsize=8); run -= v
ax.bar(len(bars) + 1, 1147.0, color=COLORS["final"], width=0.6)
ax.text(len(bars) + 1, 1161, "1147", ha="center", fontsize=8)
ax.axhline(1147.0, color=COLORS["final"], ls=":", lw=0.8)
ax.set_xticks(range(len(bars) + 2))
ax.set_xticklabels(["eager-\noriginal"] + [b[0].replace(" ", "\n") for b in bars] + ["final"], fontsize=7)
ax.set_ylabel("chunk-6 time (ms)"); ax.set_ylim(900, 1580)
ax.set_title("Fig 5 — J8 waterfall (leave-one-out; sum 304.6 vs measured 351.4, -13.3 % interaction)",
             fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(FIGURES_DIR, "fig5_j8_waterfall.png")); plt.close()

eag = [803.7, 912.3, 1029.5, 1146.8, 1264.6, 1381.8, 1498.3]
fin = [559.0, 664.3, 756.0, 855.9, 957.3, 1053.6, 1147.0]
fig, ax = plt.subplots(figsize=(6.4, 3.4))
ax.plot(range(7), eag, "o-", color=COLORS["eager"], label="eager-original  (a=682.8, b=24.86)")
ax.plot(range(7), fin, "o-", color=COLORS["final"], label="final  (a=464.2, b=20.94)")
for i in range(7):
    ax.annotate(f"-{100*(1-fin[i]/eag[i]):.0f}%", (i, fin[i] - 52), ha="center",
                fontsize=7, color=COLORS["final"])
ax.set_xlabel("chunk position  (K = 4,680 ... 32,760)"); ax.set_ylabel("chunk time (ms)")
ax.legend(fontsize=7); ax.set_title("Fig 6 — per-position chunk time, eager-original vs final")
plt.tight_layout(); plt.savefig(os.path.join(FIGURES_DIR, "fig6_per_position.png")); plt.close()

sw = load_json("j6_sweep.json")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.4))
for n, col in (("1", COLORS["a"]), ("2", COLORS["b"]), ("3", COLORS["final"])):
    r = sw[n]
    a1.plot(r["K"], r["median_ms"], "o-", ms=3, color=col,
            label=f"nfpb={n}  a={r['a_ms']:.0f} b={r['b_ms_per_1k_K']:.2f}")
a1.set_xlabel("K (tokens)"); a1.set_ylabel("chunk time (ms)"); a1.legend(fontsize=7)
a1.set_title("token-count sweep", fontsize=9)
a2.bar(["1", "2", "3"], [27.8, 7.4, 17.8], color=[COLORS["a"], COLORS["b"], COLORS["final"]])
for i, (w_, c_) in enumerate(zip([27.8, 7.4, 17.8], [156, 300, 444])):
    a2.text(i, w_ + 0.6, f"{w_:.1f}%\n{c_} CTAs", ha="center", fontsize=7)
a2.set_xlabel("num_frame_per_block"); a2.set_ylabel("wave-quantization waste (%)")
a2.set_ylim(0, 34); a2.set_title("quantization waste from grid arithmetic", fontsize=9)
plt.suptitle("Fig 7 — token count and the attention grid", y=1.02, fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(FIGURES_DIR, "fig7_nfpb_sweep.png"), bbox_inches="tight"); plt.close()

q = load_json("j9_metrics.json"); pv = {a: {x["prompt"]: x for x in v} for a, v in q["per_video"].items()}
mets = [("clip_prompt_sim", "CLIP prompt-sim"), ("clip_temporal_cos", "CLIP temporal"),
        ("lpips_consecutive", "LPIPS consecutive")]
fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
for ax, (m, lb) in zip(axes, mets):
    fa = [abs(pv["final"][i][m] - pv["eager_a"][i][m]) for i in range(10)]
    ba = [abs(pv["eager_b"][i][m] - pv["eager_a"][i][m]) for i in range(10)]
    x = np.arange(10)
    ax.bar(x - 0.2, fa, 0.4, color=COLORS["final"], label="|final - eagerA|")
    ax.bar(x + 0.2, ba, 0.4, color=COLORS["eager"], label="|eagerB - eagerA|")
    ax.set_title(f"{lb}\nseed moves {np.mean(ba)/np.mean(fa):.1f}x further", fontsize=8)
    ax.set_xlabel("prompt"); ax.set_xticks(x); ax.tick_params(labelsize=7)
axes[0].set_ylabel("absolute difference"); axes[0].legend(fontsize=7)
plt.suptitle("Fig 8 — J9: paired per-prompt quality differences (optimisation vs seed)",
             y=1.03, fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(FIGURES_DIR, "fig8_j9_paired.png"), bbox_inches="tight"); plt.close()

print("figures written:")
for f in sorted(os.listdir(FIGURES_DIR)):
    if f.startswith("fig"):
        print(f"   {f}  {os.path.getsize(os.path.join(FIGURES_DIR,f))//1024} KB")
