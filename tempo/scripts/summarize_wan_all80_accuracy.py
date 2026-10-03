#!/usr/bin/env python3
"""Part 2, step 4 summary (CPU; brief §4): Wan2.1 B0 vs L(2,2), seed 42, all 80 single-object prompts.

Runs paired_accuracy_analysis.py (phase 4 metric dir) on all 80 and on the 72 non-pilot prompts, applies the pre-registered calibration
gate (B0 on all 80 within +-5 pts of TempoControl's 63.9 % -> "comparable"), adds per-timing breakdowns and Q3/Q4.
Writes results/step4_summary.{md,json}.
"""
import json
import os
import statistics as st
import subprocess
import sys

TEMPO = os.environ.get("TEMPO_ROOT", os.path.expanduser("~/tempo"))
PY = sys.executable
SETS = {"all80": "0-79", "np72": "2-19,22-39,42-59,62-79"}
TIMING = {"2nd": range(0, 20), "3rd": range(20, 40), "4th": range(40, 60), "last": range(60, 80)}
TC_B0, TC_L = 0.639, 0.836                                     # TempoControl paper, single object, all 80, seed 42


def run_paired_analysis(ids, name):
    cmd = [PY, os.path.join(TEMPO, "scripts", "paired_accuracy_analysis.py"), "--phase", "4", "--ids", ids, "--ref", "B0_s42",
           "--tags", "B0_s42,L_b2g2_s42", "--name", name]
    r = subprocess.run(cmd, cwd=TEMPO, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"analyze failed for {name}:\n{r.stdout[-2000:]}{r.stderr[-2000:]}")
    return json.load(open(os.path.join(TEMPO, "results", f"{name}.json")))


def per_timing(o, key):
    return {k: round(st.mean(o[key][str(i)] for i in rg if str(i) in o[key]), 3)
            for k, rg in TIMING.items() if any(str(i) in o[key] for i in rg)}


def main():
    out, md = {}, ["# Step 4 summary (Wan2.1-T2V-1.3B, explicit path, seed 42; official metric)", ""]
    for sname, ids in SETS.items():
        r = run_paired_analysis(ids, f"step4_{sname}")
        b, l_ = r["B0_s42"], r["L_b2g2_s42"]
        if sname == "all80":                                  # our mean must equal the official script's own mean
            for t, o in (("B0_s42", b), ("L_b2g2_s42", l_)):
                off = json.load(open(os.path.join(TEMPO, "results", "metric", "phase4", t,
                                                  "temporal_accuracy_one_object.json")))["temporal_accuracy"]
                assert len(off[1]) == 80 and abs(off[0] - o["acc_mean"]) < 1e-9, (t, off[0], o["acc_mean"])
        ci = l_["d_vs_ref_ci95"]
        out[sname] = {"B0": b, "L": l_, "by_timing": {"B0": per_timing(b, "acc_per_prompt"),
                                                      "L": per_timing(l_, "acc_per_prompt"),
                                                      "d": per_timing(l_, "d_vs_ref_per_prompt")}}
        md += [f"## {sname} ({l_['n']} prompts)", "",
               "| run | accuracy | Δ vs B0 [95 % CI] | better / worse (sign p) | absent ok | present ok | CLIP Δ |",
               "|---|---|---|---|---|---|---|",
               f"| B0 | {b['acc_mean']:.3f} | — | — | {b['absent_rate']:.2f} | {b['present_rate']:.2f} | — |",
               f"| L(2,2) | {l_['acc_mean']:.3f} | {l_['d_vs_ref_mean']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] | "
               f"{l_['n_better']} / {l_['n_worse']} ({l_['sign_p_vs_ref']:.4f}) | {l_['absent_rate']:.2f} | "
               f"{l_['present_rate']:.2f} | {l_.get('clip_d_vs_ref', float('nan')):+.4f} |", "",
               f"By timing (2nd / 3rd / 4th / last): B0 {out[sname]['by_timing']['B0']}; L {out[sname]['by_timing']['L']}; "
               f"paired Δ {out[sname]['by_timing']['d']}", "",
               f"Wall time / peak memory: from the {l_.get('n_timed')} step-4 videos that have phase-4 rows "
               f"(part-1 videos were timed in part 1): L overhead {100 * (l_['overhead_vs_ref_mean'] or 0):+.2f} %, "
               f"peak {l_['peak_gb_max']} GB.", ""]
        if sname == "all80":
            md.append("Note: 8 of the 80 (the pilot ids) were used to choose β in part 1.")
            md.append("")
    b80 = out["all80"]["B0"]["acc_mean"]
    comparable = abs(b80 - TC_B0) <= 0.05
    out["gate"] = {"B0_all80": b80, "TempoControl_B0": TC_B0, "diff_pts": 100 * (b80 - TC_B0),
                   "verdict": "comparable" if comparable else "not comparable"}
    l80 = out["all80"]["L"]["acc_mean"]
    d80 = out["all80"]["L"]["d_vs_ref_mean"]
    read = (f"L's absolute accuracy on all 80, {100 * l80:.1f} %, is read directly against TempoControl's {100 * TC_L:.1f} %"
            if comparable else
            f"both baselines reported (ours {100 * b80:.1f} %, theirs {100 * TC_B0:.1f} %); gains compared: "
            f"L − B0 = {100 * d80:+.1f} pts vs their +{100 * (TC_L - TC_B0):.1f} pts; reason: stated in the report "
            f"(the protocol matches theirs, so the difference is attributed to hardware/software and seed draws)")
    out["Q3"] = {"met": comparable}
    d72 = out["np72"]["L"]["d_vs_ref_mean"]
    ci72 = out["np72"]["L"]["d_vs_ref_ci95"]
    out["Q4"] = {"d72": d72, "ci72": ci72, "met": bool(d72 >= 0.12 and ci72[0] > 0)}
    md += [f"**Calibration gate (Q3):** B0 on all 80 = {100 * b80:.1f} % vs 63.9 % ({out['gate']['diff_pts']:+.1f} pts) → "
           f"**{out['gate']['verdict']}**; Q3 {'met' if comparable else 'MISSED'}. Reading: {read}.", "",
           f"**Q4** (L ≥ +12 pts on the 72 non-pilot, 95 % CI excluding 0): Δ = {100 * d72:+.1f} pts "
           f"[{100 * ci72[0]:+.1f}, {100 * ci72[1]:+.1f}] → **{'met' if out['Q4']['met'] else 'MISSED'}**", ""]
    json.dump(out, open(os.path.join(TEMPO, "results", "step4_summary.json"), "w"), indent=1, default=float)
    open(os.path.join(TEMPO, "results", "step4_summary.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
