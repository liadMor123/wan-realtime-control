#!/usr/bin/env python3
"""Step 2's optional accuracy check (brief §2), summary: L(2,2) on the production fa2kv path vs a matched
baseline, B0 on Wan's unmodified flash path. Wan2.1-T2V-1.3B, seed 42, the 20 held-out prompts (data/heldout_ids.txt),
phase 24, tags B0_flash_s42 / L_b2g2_fa2kv_s42, metric and CLIP scored on the GPU (results/metric/phase24/,
results/quality/phase24/).

Primary (all GPU-scored): paired_accuracy_analysis.py --phase 24 (mean, L - B0 paired with a 10,000-resample bootstrap 95 % CI,
better / worse, sign test, absent / present success, CLIP delta, wall overhead on the same node) -> also written as
results/phase24_summary.{md,json}.
Context only (not a comparison of paths: a path change alone changes the sample):
  - part 1's explicit-path result on the same 20 prompts, B0_s42 0.705 -> L_b2g2_s42 0.875 (+17.0 [+9.2, +25.5]),
    read from results/phase2_summary.json (CPU-scored); and the same two tags re-scored on the GPU when available
    (results/metric/phase2_gpu/, the consistency gate);
  - per-prompt agreement of fa2kv-L with explicit-L accuracies (and flash-B0 with explicit-B0): descriptive only,
    next to the seed-to-seed noise floor of part 1.
The GPU-vs-CPU gate (gpu_vs_cpu_metric_check.py, one-object section) is printed next to the numbers.

  summarize_fa2kv_accuracy.py   -> results/step2acc_summary.{md,json}
"""
import json
import os
import subprocess
import sys

import numpy as np

TEMPO = os.environ.get("TEMPO_ROOT", os.path.expanduser("~/tempo"))
sys.path.insert(0, os.path.join(TEMPO, "src"))
sys.path.insert(0, os.path.join(TEMPO, "scripts"))
import gpu_vs_cpu_metric_check  # noqa: E402
from generate_benchmark_videos import parse_ids  # noqa: E402

PHASE = 24
REF, ARM = "B0_flash_s42", "L_b2g2_fa2kv_s42"
EXP_REF, EXP_ARM = "B0_s42", "L_b2g2_s42"                 # part 1, explicit path
STEP2_FA2KV_OVERHEAD = 0.0158                               # step 2 timing (job 162925), fa2kv L vs flash B0
RES = os.path.join(TEMPO, "results")


def int_keys(d):
    """paired_accuracy_analysis.py per-prompt dicts have str keys after JSON."""
    return {int(k): v for k, v in d.items()}


def agreement(a, b, ids):
    x, y = np.array([a[i] for i in ids]), np.array([b[i] for i in ids])
    d = x - y
    r = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else None
    return {"n": len(ids), "n_equal": int((d == 0).sum()), "n_within_0.10": int((np.abs(d) <= 0.10 + 1e-9).sum()),
            "mean_abs_diff": float(np.abs(d).mean()), "mean_diff": float(d.mean()), "pearson_r": r,
            "per_prompt": {i: [float(a[i]), float(b[i])] for i in ids}}


def main():
    ids = parse_ids(open(os.path.join(TEMPO, "data", "heldout_ids.txt")).read().strip())
    assert len(ids) == 20, ids
    subprocess.run([sys.executable, os.path.join(TEMPO, "scripts", "paired_accuracy_analysis.py"), "--phase", str(PHASE),
                    "--ids", ",".join(map(str, ids)), "--ref", REF, "--tags", f"{REF},{ARM}",
                    "--name", f"phase{PHASE}_summary"], check=True, stdout=subprocess.DEVNULL)
    s = json.load(open(os.path.join(RES, f"phase{PHASE}_summary.json")))
    b, l_ = s[REF], s[ARM]
    for k in ("acc_per_prompt", "d_vs_ref_per_prompt"):
        b[k], l_[k] = int_keys(b[k]), int_keys(l_[k])
    l_["n_tied"] = l_["n"] - l_["n_better"] - l_["n_worse"]
    if l_["n_timed"] != len(ids):
        print(f"### only {l_['n_timed']} of {len(ids)} prompts have timing rows for both arms")

    # context: part 1 explicit path (CPU-scored), and its GPU re-score if the gate has run
    p2 = json.load(open(os.path.join(RES, "phase2_summary.json")))
    e_b, e_l = p2[EXP_REF], p2[EXP_ARM]
    exp_cpu = {"B0_acc": e_b["acc_mean"], "L_acc": e_l["acc_mean"], "d": e_l["d_vs_ref_mean"],
               "d_ci95": e_l["d_vs_ref_ci95"], "n_better": e_l["n_better"], "n_worse": e_l["n_worse"],
               "sign_p": e_l["sign_p_vs_ref"], "scorer": "CPU (results/phase2_summary.json)"}
    noise = p2.get("_noise_floor", {}).get("mean_abs_paired_diff")
    gate = gpu_vs_cpu_metric_check.compare("one_object")
    exp_acc_cpu = {EXP_REF: int_keys(e_b["acc_per_prompt"]), EXP_ARM: int_keys(e_l["acc_per_prompt"])}
    exp_acc = dict(exp_acc_cpu)
    exp_gpu = None
    if gate["status"] == "compared":
        g = {t: {p["id"]: p["gpu"] for p in gate["tags"][t]["per_video"]} for t in (EXP_REF, EXP_ARM)}
        exp_acc = g                                         # same scorer as the primary numbers
        from paired_accuracy_analysis import bootstrap_ci, sign_test
        d = np.array([g[EXP_ARM][i] - g[EXP_REF][i] for i in ids])
        exp_gpu = {"B0_acc": float(np.mean([g[EXP_REF][i] for i in ids])),
                   "L_acc": float(np.mean([g[EXP_ARM][i] for i in ids])), "d": float(d.mean()),
                   "d_ci95": bootstrap_ci(d), "n_better": int((d > 0).sum()), "n_worse": int((d < 0).sum()),
                   "sign_p": sign_test(d), "scorer": "GPU (results/metric/phase2_gpu)"}
    agree_L = agreement(l_["acc_per_prompt"], exp_acc[EXP_ARM], ids)
    agree_B = agreement(b["acc_per_prompt"], exp_acc[EXP_REF], ids)

    out = {"ids": ids, "phase": PHASE, "tags": {"B0": REF, "L": ARM}, "B0": b, "L": l_,
           "context_explicit_path": {"cpu": exp_cpu, "gpu_rescore": exp_gpu},
           "agreement_fa2kvL_vs_explicitL": agree_L, "agreement_flashB0_vs_explicitB0": agree_B,
           "agreement_scorer": "GPU (both)" if exp_gpu else "fa2kv/flash GPU vs explicit CPU",
           "seed_noise_mean_abs_paired_diff": noise, "step2_timing_overhead": STEP2_FA2KV_OVERHEAD,
           "gpu_vs_cpu_gate": {k: v for k, v in gate.items() if k != "tags"} |
                              {"tags": {t: {k: v for k, v in x.items() if k != "per_video"}
                                        for t, x in gate["tags"].items()}}}
    ci = l_["d_vs_ref_ci95"]
    ovh = l_.get("overhead_vs_ref_mean")
    md = ["# Step 2 accuracy check: L(2,2) on the fa2kv path vs B0 on Wan's flash path (matched baseline)", "",
          "Wan2.1-T2V-1.3B, seed 42, the 20 held-out prompts (" + ", ".join(map(str, ids)) + "); official one-object "
          "metric and CLIP scored on the GPU (phase 24).", "",
          "| run | accuracy | Δ vs B0 [95 % CI] | better / worse / tied (sign p) | absent ok | present ok | CLIP Δ | "
          "wall s | overhead | peak GB |", "|---|---|---|---|---|---|---|---|---|---|",
          f"| B0, flash path (Wan unmodified) | {b['acc_mean']:.3f} | — | — | {b['absent_rate']:.2f} | "
          f"{b['present_rate']:.2f} | — | {b['wall_s_mean'] or float('nan'):.1f} | — | "
          f"{b['peak_gb_max'] or float('nan'):.2f} |",
          f"| L(2,2), fa2kv path | {l_['acc_mean']:.3f} | **{100 * l_['d_vs_ref_mean']:+.1f} "
          f"[{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]** | {l_['n_better']} / {l_['n_worse']} / {l_['n_tied']} "
          f"({l_['sign_p_vs_ref']:.2g}) | {l_['absent_rate']:.2f} | {l_['present_rate']:.2f} | "
          f"{l_.get('clip_d_vs_ref', float('nan')):+.4f} | {l_['wall_s_mean'] or float('nan'):.1f} | "
          f"{100 * (ovh if ovh is not None else float('nan')):+.2f} % | {l_['peak_gb_max'] or float('nan'):.2f} |", "",
          f"{gpu_vs_cpu_metric_check.gate_one_line(gate)} (B0_s42 and L_b2g2_s42, the same 20 prompts, GPU vs part 1's CPU scores).",
          "",
          f"Wall overhead: {l_['n_timed']} prompts, both arms on the same node; step 2's timing on prompts 0-1 measured "
          f"{100 * STEP2_FA2KV_OVERHEAD:+.2f} %.", "",
          "## Context only: part 1's explicit path on the same 20 prompts", "",
          "A path change alone changes the sample, so these are not paired with the rows above.", "",
          "| scorer | B0_s42 | L_b2g2_s42 | Δ [95 % CI] | better / worse (sign p) |", "|---|---|---|---|---|"]
    for e in (exp_cpu, exp_gpu):
        if e:
            md.append(f"| {e['scorer']} | {e['B0_acc']:.3f} | {e['L_acc']:.3f} | {100 * e['d']:+.1f} "
                      f"[{100 * e['d_ci95'][0]:+.1f}, {100 * e['d_ci95'][1]:+.1f}] | {e['n_better']} / "
                      f"{e['n_worse']} ({e['sign_p']:.2g}) |")
    md += ["", f"Per-prompt agreement across paths (descriptive; scorer: {out['agreement_scorer']}; part 1's seed-to-seed "
               f"mean |Δ| per video: {noise if noise is None else round(noise, 3)}):", "",
           "| pair | equal | within 0.10 | mean \\|Δ\\| | mean Δ | Pearson r |", "|---|---|---|---|---|---|"]
    for name, a in (("fa2kv-L vs explicit-L", agree_L), ("flash-B0 vs explicit-B0", agree_B)):
        r = "n/a" if a["pearson_r"] is None else f"{a['pearson_r']:.2f}"
        md.append(f"| {name} | {a['n_equal']}/{a['n']} | {a['n_within_0.10']}/{a['n']} | {a['mean_abs_diff']:.3f} | "
                  f"{a['mean_diff']:+.3f} | {r} |")
    md += ["", "Per prompt (accuracy):", "",
           "| prompt | B0 flash | L fa2kv | B0 explicit | L explicit |", "|---|---|---|---|---|"]
    for i in ids:
        md.append(f"| {i} | {b['acc_per_prompt'][i]:.2f} | {l_['acc_per_prompt'][i]:.2f} | "
                  f"{exp_acc[EXP_REF][i]:.2f} | {exp_acc[EXP_ARM][i]:.2f} |")
    pa = bool(l_["d_vs_ref_mean"] > 0)
    out["PA"] = {"prediction": "mean paired delta (L fa2kv - B0 flash) > 0", "d": l_["d_vs_ref_mean"], "met": pa}
    md += ["", f"**PA** (pre-registered: mean paired Δ > 0): Δ = {100 * l_['d_vs_ref_mean']:+.1f} pts "
               f"[{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}] → **{'met' if pa else 'MISSED'}**", ""]
    json.dump(out, open(os.path.join(RES, "step2acc_summary.json"), "w"), indent=1, default=float)
    open(os.path.join(RES, "step2acc_summary.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
