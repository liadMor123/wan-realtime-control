#!/usr/bin/env python3
"""Part 2, step 3b summary (CPU): runs paired_accuracy_analysis.py on every pre-registered set, then adds timing breakdowns, the P3b
verdicts and the prompt-switch latency. Writes results/step3b_summary.{md,json}.

Sets (pre-registered protocol, part 2, step 3b): seeds 42 and 43 separately x {all 80, 72 non-pilot}, Self-Forcing B0 vs L
(fa2kv); prompt switching (seed 42, 20 held-out) vs B0 and vs L.
"""
import glob
import json
import os
import statistics as st
import subprocess
import sys

TEMPO = os.environ.get("TEMPO_ROOT", os.path.expanduser("~/tempo"))
PY = sys.executable
ALL80, NP72, HO20 = "0-79", "2-19,22-39,42-59,62-79", "2-6,22-26,42-46,62-66"
TIMING = {"2nd": range(0, 20), "3rd": range(20, 40), "4th": range(40, 60), "last": range(60, 80)}


def run_paired_analysis(ids, ref, tags, name):
    cmd = [PY, os.path.join(TEMPO, "scripts", "paired_accuracy_analysis.py"), "--phase", "23", "--ids", ids, "--ref", ref,
           "--tags", ",".join(tags), "--name", name]
    r = subprocess.run(cmd, cwd=TEMPO, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"analyze failed for {name}:\n{r.stdout[-2000:]}{r.stderr[-2000:]}")
    return json.load(open(os.path.join(TEMPO, "results", f"{name}.json")))


def by_timing(o):
    out = {}
    for k, rg in TIMING.items():
        v = [o["acc_per_prompt"][str(i)] if str(i) in o["acc_per_prompt"] else o["acc_per_prompt"].get(i)
             for i in rg if (str(i) in o["acc_per_prompt"] or i in o["acc_per_prompt"])]
        if v:
            out[k] = round(st.mean(v), 3)
    return out


def d_by_timing(o):
    """Mean paired Δ vs the reference per timing class."""
    out = {}
    for k, rg in TIMING.items():
        v = [o["d_vs_ref_per_prompt"][str(i)] for i in rg if str(i) in o["d_vs_ref_per_prompt"]]
        if v:
            out[k] = round(st.mean(v), 3)
    return out


def table_row(t, o, ref=False):
    if ref:
        return (f"| {t} | {o['acc_mean']:.3f} | — | — | {o['absent_rate']:.2f} | {o['present_rate']:.2f} | — |")
    ci = o["d_vs_ref_ci95"]
    return (f"| {t} | {o['acc_mean']:.3f} | {o['d_vs_ref_mean']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] | "
            f"{o['n_better']} / {o['n_worse']} ({o['sign_p_vs_ref']:.4f}) | {o['absent_rate']:.2f} | "
            f"{o['present_rate']:.2f} | {o.get('clip_d_vs_ref', float('nan')):+.4f} |")


def main():
    out, md = {}, ["# Step 3b summary (Self-Forcing, fa2kv path; official metric)", ""]
    hdr = ["| run | accuracy | Δ vs ref [95 % CI] | better / worse (sign p) | absent ok | present ok | CLIP Δ |",
           "|---|---|---|---|---|---|---|"]
    for seed in (42, 43):
        b0, l_ = f"SF_B0_fa2kv_s{seed}", f"SF_L_b2g2_fa2kv_s{seed}"
        for setname, ids in (("all80", ALL80), ("np72", NP72)):
            name = f"step3b_s{seed}_{setname}"
            r = run_paired_analysis(ids, b0, [b0, l_], name)
            out[name] = {"B0": r[b0], "L": r[l_], "by_timing": {"B0": by_timing(r[b0]), "L": by_timing(r[l_]),
                                                                "d_L": d_by_timing(r[l_])}}
            md += [f"## seed {seed}, {setname} ({r[l_]['n']} prompts), reference B0", ""] + hdr + \
                  [table_row(b0, r[b0], ref=True), table_row(l_, r[l_])] + \
                  ["", f"By timing (2nd / 3rd / 4th / last): B0 {out[name]['by_timing']['B0']}; "
                       f"L {out[name]['by_timing']['L']}; paired Δ {out[name]['by_timing']['d_L']}", ""]
    d42 = out["step3b_s42_np72"]["L"]["d_vs_ref_mean"]
    d43 = out["step3b_s43_np72"]["L"]["d_vs_ref_mean"]
    out["P3b_1"] = {"d42_np72": d42, "d43_np72": d43, "met": bool(d42 >= 0.12 and d43 > 0)}
    md += [f"**P3b-1** (L ≥ +12 pts on the 72 non-pilot at seed 42, same sign at seed 43): Δ42 = {100 * d42:+.1f} pts, "
           f"Δ43 = {100 * d43:+.1f} pts → **{'met' if out['P3b_1']['met'] else 'MISSED'}**", ""]

    # prompt switching (seed 42, 20 held-out)
    tags = ["SF_B0_fa2kv_s42", "SF_L_b2g2_fa2kv_s42", "SF_PS_s42"]
    rb = run_paired_analysis(HO20, "SF_B0_fa2kv_s42", tags, "step3b_ps_vs_B0")
    rl = run_paired_analysis(HO20, "SF_L_b2g2_fa2kv_s42", tags, "step3b_ps_vs_L")
    out["ps"] = {"vs_B0": rb["SF_PS_s42"], "vs_L": rl["SF_PS_s42"], "L_vs_B0": rb["SF_L_b2g2_fa2kv_s42"],
                 "by_timing": {t: by_timing(rb[t]) for t in tags}}
    md += ["## Prompt switching (seed 42, 20 held-out)", "", "Reference B0:", ""] + hdr + \
          [table_row("SF_B0_fa2kv_s42", rb["SF_B0_fa2kv_s42"], ref=True), table_row("SF_L_b2g2_fa2kv_s42", rb["SF_L_b2g2_fa2kv_s42"]),
           table_row("SF_PS_s42", rb["SF_PS_s42"]), "", "Reference L:", ""] + hdr + \
          [table_row("SF_PS_s42", rl["SF_PS_s42"]), "", f"By timing: {out['ps']['by_timing']}", ""]
    ps = rb["SF_PS_s42"]
    lat = {}
    for p in sorted(glob.glob(os.path.join(TEMPO, "results", "step3", "sfps_*.json"))):
        j = json.load(open(p))
        if j.get("done"):
            lat = {"switch_latency": j["switch_latency"], "t5_ms": j["t5_full_prompt_ms_median"],
                   "machinery_check": j["check_switch_at_0_equals_plain_full"], "file": os.path.basename(p)}
    out["ps"]["latency"] = lat
    worst = max(v["overhead"] for v in lat["switch_latency"].values()) if lat else None
    out["P3b_2"] = {"i_absent_ge_0.90": bool(ps["absent_rate"] >= 0.90), "ii_beats_B0": bool(ps["d_vs_ref_mean"] > 0),
                    "iii_switch_overhead_le_5pct": ("not measured" if worst is None else bool(worst <= 0.05)),
                    "worst_switch_overhead": worst,
                    "iv": "no prediction (PS vs L)"}
    if lat:
        md += ["Switch-chunk latency (eager; median chunk ms at the onset position, switch videos vs normal):", "",
               "| position | switch ms | normal ms | overhead | n switch / normal |", "|---|---|---|---|---|"]
        for p, v in sorted(lat["switch_latency"].items(), key=lambda kv: int(kv[0])):
            md.append(f"| {p} | {v['switch_median_ms']:.1f} | {v['normal_median_ms']:.1f} | {100 * v['overhead']:+.2f} % | "
                      f"{v['n_switch']} / {v['n_normal']} |")
        md += ["", f"T5 encode of the full prompt (median): {lat['t5_ms']:.0f} ms (not inside the chunk times). "
                   f"Machinery check: {lat['machinery_check']}", ""]
    md += [f"**P3b-2**: (i) absent ≥ 0.90: {out['P3b_2']['i_absent_ge_0.90']} ({ps['absent_rate']:.2f}); "
           f"(ii) PS > B0: {out['P3b_2']['ii_beats_B0']} ({100 * ps['d_vs_ref_mean']:+.1f} pts); "
           f"(iii) switch overhead ≤ +5 %: {out['P3b_2']['iii_switch_overhead_le_5pct']} "
           f"(worst {'n/a' if worst is None else f'{100 * worst:+.2f} %'}); "
           f"(iv) no prediction. PS − L = {100 * rl['SF_PS_s42']['d_vs_ref_mean']:+.1f} pts "
           f"[{100 * rl['SF_PS_s42']['d_vs_ref_ci95'][0]:+.1f}, {100 * rl['SF_PS_s42']['d_vs_ref_ci95'][1]:+.1f}]", ""]
    json.dump(out, open(os.path.join(TEMPO, "results", "step3b_summary.json"), "w"), indent=1, default=float)
    open(os.path.join(TEMPO, "results", "step3b_summary.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
