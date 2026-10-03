#!/usr/bin/env python3
"""Part 2, step 5b summary (brief §5): Wan2.1 B0 vs L(2,2) with K = 2 objects, seed 42, explicit path, on ALL
82 two-object pairs = step 5a's 20 (data/step5a_ids.txt) + step 5b's 62 (data/step5b_ids.txt); same tags, protocol and
phase-5 rows as 5a.

Primary numbers use one scorer for every video: the official two-object metric and CLIP run on the GPU over all 82
pairs of both tags (results/metric/phase5_gpu/, results/quality/phase5_gpu/; slurm/score_gpu.sbatch). Reported sets:
  all 82      : the paper's set; absolute numbers next to TempoControl's 37.5 -> 53.2 %
  65 scorable : without the 17 pairs whose object names are not COCO classes (select_scorable_two_object_pairs.UNSCORABLE)
  62 new      : step 5b's pairs alone
  20 of 5a    : step 5a's pairs, GPU-scored (5a's own CPU numbers: results/step5a_summary.json)
Each: mean, L - B0 paired with a 10,000-resample bootstrap 95 % CI, better / worse / tied, sign test, off- / on-frame
success, CLIP delta, wall overhead (summarize_two_objects_20_pairs.arm_stats / paired, imported). The GPU-vs-CPU gate on the 20 5a
pairs (gpu_vs_cpu_metric_check.py, two-object section) is stated next to the numbers; if any of those 40 videos disagree, the
CPU-where-available reading (CPU scores for the 20 5a pairs, GPU for the other 62) is also given, for all 82 and the
65 scorable pairs, next to the all-CPU reading of the 20 5a pairs.

  summarize_two_objects_all82.py   -> results/step5b_summary.{md,json}
"""
import json
import os
import sys

import numpy as np

TEMPO = os.environ.get("TEMPO_ROOT", os.path.expanduser("~/tempo"))
sys.path.insert(0, os.path.join(TEMPO, "src"))
sys.path.insert(0, os.path.join(TEMPO, "scripts"))
from tempo_ctrl import benchmark  # noqa: E402
import gpu_vs_cpu_metric_check  # noqa: E402
from paired_accuracy_analysis import bootstrap_ci, load_phase_rows, sign_test  # noqa: E402
from generate_benchmark_videos import parse_ids  # noqa: E402
from select_scorable_two_object_pairs import UNSCORABLE  # noqa: E402
from summarize_two_objects_20_pairs import ARM, PHASE, REF, TC_B0, TC_L, add_paired_stats, arm_stats, load_pair_metric  # noqa: E402

GPU_DIR, CPU_DIR = "phase5_gpu", f"phase{PHASE}"
N_PAIRS = 82


def read_ids(name):
    return parse_ids(open(os.path.join(TEMPO, "data", name)).read().strip())


def id_sets():
    a, b = read_ids("step5a_ids.txt"), read_ids("step5b_ids.txt")
    assert len(a) == 20 and len(b) == 62 and not set(a) & set(b) and sorted(a + b) == list(range(N_PAIRS)), (a, b)
    scorable = [i for i in range(N_PAIRS) if i not in UNSCORABLE]
    assert len(scorable) == 65
    return {"all82": list(range(N_PAIRS)), "scorable65": scorable, "new62": b, "step5a20": a}


def gpu_set(ids, pairs, rows):
    b, acc_b = arm_stats(REF, ids, pairs, rows, GPU_DIR, GPU_DIR)
    l_, acc_l = arm_stats(ARM, ids, pairs, rows, GPU_DIR, GPU_DIR)
    add_paired_stats(b, l_, acc_b, acc_l, ids)
    return {"n": len(ids), "B0": b, "L": l_}


def mixed_set(ids, cpu_ids, pairs):
    """Accuracy statistics with CPU scores for cpu_ids and GPU scores for the rest (the CPU-where-available reading)."""
    o = {"n": len(ids), "n_cpu_scored": len(set(ids) & set(cpu_ids))}
    acc = {}
    for arm, tag in (("B0", REF), ("L", ARM)):
        g, _ = load_pair_metric(tag, ids, pairs, GPU_DIR)
        c_ids = [i for i in ids if i in cpu_ids]
        c, _ = load_pair_metric(tag, c_ids, pairs, CPU_DIR) if c_ids else ({}, None)
        per = {i: (c[i] if i in c else g[i]) for i in ids}
        acc[arm] = np.array([per[i]["video_results"] for i in ids])
        n_off = np.array([int((pairs[i]["mask"] == 0).sum()) for i in ids])
        n_on = np.array([int((pairs[i]["mask"] == 1).sum()) for i in ids])
        o[arm] = {"acc_mean": float(acc[arm].mean()),
                  "off_frame_acc": float((np.array([per[i]["static_object_success_rate"] for i in ids]) * n_off).sum()
                                         / n_off.sum()),
                  "on_frame_acc": float((np.array([per[i]["temp_object_success_rate"] for i in ids]) * n_on).sum()
                                        / n_on.sum())}
    d = acc["L"] - acc["B0"]
    o["L"].update({"d_vs_ref_mean": float(d.mean()), "d_vs_ref_ci95": bootstrap_ci(d), "n_better": int((d > 0).sum()),
                   "n_worse": int((d < 0).sum()), "n_tied": int((d == 0).sum()), "sign_p_vs_ref": sign_test(d)})
    return o


def table_row(label, s):
    b, l_ = s["B0"], s["L"]
    ci = l_["d_vs_ref_ci95"]
    ovh = l_.get("overhead_vs_ref_mean")
    clip = l_.get("clip_d_vs_ref")
    return (f"| {label} | {s['n']} | {100 * b['acc_mean']:.1f} | {100 * l_['acc_mean']:.1f} | "
            f"**{100 * l_['d_vs_ref_mean']:+.1f} [{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]** | "
            f"{l_['n_better']} / {l_['n_worse']} / {l_['n_tied']} ({l_['sign_p_vs_ref']:.2g}) | "
            f"{b['off_frame_acc']:.3f} → {l_['off_frame_acc']:.3f} | {b['on_frame_acc']:.3f} → {l_['on_frame_acc']:.3f} | "
            f"{'—' if clip is None else f'{clip:+.4f}'} | "
            f"{'—' if ovh is None else f'{100 * ovh:+.2f} %'} |")


def main():
    sets = id_sets()
    pairs = benchmark.load_two_objects()
    rows = load_phase_rows(PHASE)
    res = {name: gpu_set(ids, pairs, rows) for name, ids in sets.items()}
    gate = gpu_vs_cpu_metric_check.compare("two_object")
    disagree = gate["status"] != "compared" or not gate["all_equal"]
    alt = None
    if disagree:
        alt = {"cpu_where_available_all82": mixed_set(sets["all82"], sets["step5a20"], pairs),
               "cpu_where_available_scorable65": mixed_set(sets["scorable65"], sets["step5a20"], pairs)}
        try:
            alt["all_cpu_step5a20"] = mixed_set(sets["step5a20"], sets["step5a20"], pairs)
        except (FileNotFoundError, RuntimeError) as e:
            alt["all_cpu_step5a20"] = {"error": str(e)}
    a82 = res["all82"]
    out = {"sets": sets, "scorer": "GPU (results/metric/phase5_gpu, results/quality/phase5_gpu)",
           "results": res, "TempoControl": {"B0_all82": TC_B0, "TempoControl_all82": TC_L,
                                            "gain_pts": 100 * (TC_L - TC_B0),
                                            "gain_pts_scorable65_if_capped_pairs_gained_nothing":
                                                100 * (TC_L - TC_B0) * N_PAIRS / 65},
           "gpu_vs_cpu_gate": {k: v for k, v in gate.items() if k != "tags"} |
                              {"tags": {t: {k: v for k, v in x.items() if k != "per_video"}
                                        for t, x in gate["tags"].items()}},
           "alternative_readings": alt}
    md = ["# Step 5b summary: two objects, all 82 pairs (Wan2.1-T2V-1.3B, explicit path, seed 42, K = 2)", "",
          "Official two-object metric and CLIP, all scored on the GPU (one scorer for every video). Accuracy in %.", "",
          "| set | n | B0 | L(2,2) | Δ [95 % CI] (pts) | better / worse / tied (sign p) | off-frame ok B0 → L | "
          "on-frame ok B0 → L | CLIP Δ | L wall overhead |", "|---|---|---|---|---|---|---|---|---|---|",
          table_row("all 82 (paper's set)", a82), table_row("65 scorable", res["scorable65"]),
          table_row("62 new (5b only)", res["new62"]), table_row("20 of 5a (GPU-scored)", res["step5a20"]), "",
          f"**Absolute comparison with TempoControl (all 82 pairs):** B0 {100 * a82['B0']['acc_mean']:.1f} % → "
          f"L {100 * a82['L']['acc_mean']:.1f} % ({100 * a82['L']['d_vs_ref_mean']:+.1f} pts), against their "
          f"{100 * TC_B0:.1f} % → {100 * TC_L:.1f} % (+{100 * (TC_L - TC_B0):.1f} pts; quoted from their paper, not "
          "measured). Our B0 and theirs are single draws on different hardware/software; the step-4 gate (single "
          "object) found our B0 5.6 pts above theirs.", "",
          f"On the 65 scorable pairs their gain is not published; if the 17 capped pairs gained nothing it would be "
          f"≈ +{100 * (TC_L - TC_B0) * N_PAIRS / 65:.1f} pts. The 17 unscorable pairs (" +
          ", ".join(map(str, UNSCORABLE)) + ") score ≤ about 0.5 whatever the method does; they are in all 82 and in "
          "the 62 new pairs.", "",
          f"{gpu_vs_cpu_metric_check.gate_one_line(gate)} (B0_2obj_s42 and L_b2g2_2obj_s42 on the 20 5a pairs, GPU vs step 5a's CPU "
          "scores).", "",
          "Off-frame: the 10 frames with control_signal1 = 0 (static object detected, temporal object not). On-frame: "
          "the 11 frames with control_signal1 = 1 (both detected). Wall overhead: L vs B0 on the same pair and node, "
          "from the phase-5 rows of both 5a and 5b."]
    if alt:
        md += ["", "## GPU and CPU scores disagree on the 5a pairs: alternative readings", "",
               "| reading | n | B0 | L | Δ [95 % CI] (pts) | better / worse / tied (sign p) |", "|---|---|---|---|---|---|"]
        for k, s in alt.items():
            if "error" in s:
                md.append(f"| {k} | — | — | — | {s['error']} | — |")
                continue
            ci = s["L"]["d_vs_ref_ci95"]
            md.append(f"| {k} | {s['n']} | {100 * s['B0']['acc_mean']:.1f} | {100 * s['L']['acc_mean']:.1f} | "
                      f"{100 * s['L']['d_vs_ref_mean']:+.1f} [{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}] | "
                      f"{s['L']['n_better']} / {s['L']['n_worse']} / {s['L']['n_tied']} ({s['L']['sign_p_vs_ref']:.2g}) |")
    md += ["", "Per pair (all 82, GPU-scored; * = unscorable, 5a = step-5a pair):", "",
           "| pair | static | temporal | B0 | L | Δ |", "|---|---|---|---|---|---|"]
    b, l_ = a82["B0"], a82["L"]
    for i in sets["all82"]:
        flag = ("*" if i in UNSCORABLE else "") + (" 5a" if i in sets["step5a20"] else "")
        md.append(f"| {i}{flag} | {pairs[i]['static_object']} | {pairs[i]['temp_object']} | "
                  f"{b['acc_per_pair'][i]:.2f} | {l_['acc_per_pair'][i]:.2f} | {l_['d_vs_ref_per_pair'][i]:+.2f} |")
    pb = bool(a82["L"]["d_vs_ref_mean"] > 0)
    out["PB"] = {"prediction": "L - B0 > 0 on all 82 pairs", "d": a82["L"]["d_vs_ref_mean"], "met": pb}
    ci82 = a82["L"]["d_vs_ref_ci95"]
    md += ["", f"**PB** (pre-registered: L − B0 > 0 on all 82): Δ = {100 * a82['L']['d_vs_ref_mean']:+.1f} pts "
               f"[{100 * ci82[0]:+.1f}, {100 * ci82[1]:+.1f}] → **{'met' if pb else 'MISSED'}**", ""]
    json.dump(out, open(os.path.join(TEMPO, "results", "step5b_summary.json"), "w"), indent=1, default=float)
    open(os.path.join(TEMPO, "results", "step5b_summary.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
