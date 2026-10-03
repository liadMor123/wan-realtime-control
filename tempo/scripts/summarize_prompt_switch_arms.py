#!/usr/bin/env python3
"""Part 2, step 3c summary (CPU): every prompt-switching arm against Self-Forcing B0 and against L (fa2kv, seed 42),
on all 80 and on the 72 non-pilot prompts (paired_accuracy_analysis.py: paired bootstrap CI, sign test, absent/present, CLIP), by
timing, the switch-chunk latency, and the substitute-object check (3c-5): per-arm counts and absent-frame accuracy
as scored and corrected (every absent frame of a counted video treated as failed), plus the corrected overall accuracy
(secondary, derived). Writes results/step3c_summary.{md,json}. Run after blinded_substitute_object_count.py unblind.
"""
import glob
import json
import os
import statistics as st
import subprocess
import sys

import numpy as np

TEMPO = os.environ.get("TEMPO_ROOT", os.path.expanduser("~/tempo"))
sys.path.insert(0, os.path.join(TEMPO, "scripts"))
sys.path.insert(0, os.path.join(TEMPO, "src"))
from paired_accuracy_analysis import bootstrap_ci, metric_by_prompt, sign_test  # noqa: E402
from generate_benchmark_videos import parse_ids  # noqa: E402
from tempo_ctrl import benchmark  # noqa: E402

PY = sys.executable
SETS = {"all80": "0-79", "np72": "2-19,22-39,42-59,62-79"}
TIMING = {"2nd": range(0, 20), "3rd": range(20, 40), "4th": range(40, 60), "last": range(60, 80)}
B0, L = "SF_B0_fa2kv_s42", "SF_L_b2g2_fa2kv_s42"
ARMS = [("SF_PSRF_fa2kv_s42", "PS-RF"), ("SF_PSLL_fa2kv_s42", "PS-LongLive-style"), ("SF_PSRFL_fa2kv_s42", "PS-RF+L")]
NAME = {B0: "B0", L: "L", **dict(ARMS)}


def run_paired_analysis(ids, ref, tags, name):
    cmd = [PY, os.path.join(TEMPO, "scripts", "paired_accuracy_analysis.py"), "--phase", "23", "--ids", ids, "--ref", ref,
           "--tags", ",".join(tags), "--name", name]
    r = subprocess.run(cmd, cwd=TEMPO, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"analyze failed for {name}:\n{r.stdout[-2000:]}{r.stderr[-2000:]}")
    return json.load(open(os.path.join(TEMPO, "results", f"{name}.json")))


def per_timing(d):
    return {k: round(st.mean(d[str(i)] for i in rg if str(i) in d), 3) for k, rg in TIMING.items()
            if any(str(i) in d for i in rg)}


def table_row(o, ref=False):
    if ref:
        return f"{o['acc_mean']:.3f} | — | — | {o['absent_rate']:.2f} | {o['present_rate']:.2f} | —"
    ci = o["d_vs_ref_ci95"]
    return (f"{o['acc_mean']:.3f} | {100 * o['d_vs_ref_mean']:+.1f} [{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}] | "
            f"{o['n_better']} / {o['n_worse']} ({o['sign_p_vs_ref']:.4f}) | {o['absent_rate']:.2f} | "
            f"{o['present_rate']:.2f} | {o.get('clip_d_vs_ref', float('nan')):+.4f}")


def corrected(tag, ids, counted, prompts):
    """Absent-frame accuracy as scored and corrected; per-prompt overall accuracy corrected (3c-5)."""
    per = metric_by_prompt(tag, ids, prompts, 23)
    ab_s = sum(per[i]["object_absent_successes"] for i in ids)
    ab_n = sum(per[i]["absent_frames"] for i in ids)
    ab_c = sum(0 if i in counted else per[i]["object_absent_successes"] for i in ids)
    acc_c = {}
    for i in ids:
        v = per[i]
        n = v["absent_frames"] + v["present_frames"]
        acc_c[i] = (v["object_present_successes"] + (0 if i in counted else v["object_absent_successes"])) / n
    return {"absent_scored": ab_s / ab_n, "absent_corrected": ab_c / ab_n, "acc_corrected": acc_c,
            "n_counted_in_set": len([i for i in ids if i in counted])}


def main():
    have = [(t, n) for t, n in ARMS if glob.glob(os.path.join(TEMPO, "results", "metric", "phase23", t, "*.json"))]
    if not have:
        raise SystemExit("no scored 3c arm")
    prompts = benchmark.load_one_object()
    tags = [B0, L] + [t for t, _ in have]
    out, md = {"arms": [n for _, n in have]}, ["# Step 3c summary: prompt-switching baselines (Self-Forcing, fa2kv, "
                                              "seed 42; official metric)", ""]
    hdr = ["| run | accuracy | Δ vs ref, pts [95 % CI] | better / worse (sign p) | absent ok | present ok | CLIP Δ |",
           "|---|---|---|---|---|---|---|"]
    for sname, ids in SETS.items():
        rb = run_paired_analysis(ids, B0, tags, f"step3c_{sname}_vsB0")
        rl = run_paired_analysis(ids, L, tags, f"step3c_{sname}_vsL")
        out[sname] = {"vs_B0": {NAME[t]: rb[t] for t in tags}, "vs_L": {NAME[t]: rl[t] for t in tags},
                      "by_timing_acc": {NAME[t]: per_timing(rb[t]["acc_per_prompt"]) for t in tags},
                      "by_timing_d_vs_B0": {NAME[t]: per_timing(rb[t]["d_vs_ref_per_prompt"]) for t in tags if t != B0},
                      "by_timing_d_vs_L": {NAME[t]: per_timing(rl[t]["d_vs_ref_per_prompt"]) for t in tags if t != L}}
        n = rb[B0]["n"]
        md += [f"## {sname} ({n} prompts)", "", "Reference B0:", ""] + hdr + [f"| B0 | {table_row(rb[B0], True)} |"] + \
              [f"| {NAME[t]} | {table_row(rb[t])} |" for t in tags if t != B0] + ["", "Reference L:", ""] + hdr + \
              [f"| L | {table_row(rl[L], True)} |"] + [f"| {NAME[t]} | {table_row(rl[t])} |" for t in tags if t not in (B0, L)]
        md += ["", "Accuracy by timing (2nd / 3rd / 4th / last): " +
               "; ".join(f"{k} {v}" for k, v in out[sname]["by_timing_acc"].items()), ""]
        if len(have) > 1 and "SF_PSLL_fa2kv_s42" in [t for t, _ in have]:
            rr = run_paired_analysis(ids, "SF_PSRF_fa2kv_s42", ["SF_PSRF_fa2kv_s42", "SF_PSLL_fa2kv_s42"], f"step3c_{sname}_LLvsRF")
            out[sname]["LL_vs_RF"] = rr["SF_PSLL_fa2kv_s42"]
            md += [f"PS-LongLive-style − PS-RF (information only): {table_row(rr['SF_PSLL_fa2kv_s42'])}", ""]
    # latency
    out["latency"] = {}
    md += ["## Switch-chunk latency (eager; median chunk ms at the onset position: 20 switch videos vs 60 normal)", "",
           "| arm | position | switch ms | normal ms | overhead | re-cache ms (share of overhead) |",
           "|---|---|---|---|---|---|"]
    for p in sorted(glob.glob(os.path.join(TEMPO, "results", "step3c", "sfsw_*.json"))):
        j = json.load(open(p))
        if not j.get("done"):
            continue
        out["latency"][j["arm"]] = {"file": os.path.basename(p), "checks": j["checks"], "lat": j["switch_latency"],
                                    "t5_ms": j["t5_full_prompt_ms_median"],
                                    "vs_3b2": j.get("vs_3b2_SF_PS_mp4_bytes_equal", {}).get("n_equal")}
        for pos, v in sorted(j["switch_latency"].items(), key=lambda kv: int(kv[0])):
            rc = (f"{v['recache_median_ms']:.1f} ({v['recache_share_of_overhead']:.2f})"
                  if v.get("recache_median_ms") is not None and v.get("recache_share_of_overhead") is not None else "—")
            md.append(f"| {j['arm']} | {pos} | {v['switch_median_ms']:.1f} | {v['normal_median_ms']:.1f} | "
                      f"{100 * v['overhead']:+.2f} % | {rc} |")
    md += ["", "T5 encode of the full prompt (median ms, outside the chunk times): " +
           "; ".join(f"{a} {v['t5_ms']:.0f}" for a, v in out["latency"].items()), ""]
    # substitute-object check
    ub = os.path.join(TEMPO, "results", "blind3c", "unblinded.json")
    if os.path.isfile(ub):
        u = json.load(open(ub))
        out["substitutes"] = {}
        md += ["## Substitute objects (blinded, rule 3b-3) and absent-frame accuracy, scored and corrected", "",
               "| set | arm | counted | ambiguous | absent (scored) | absent (corrected) | acc corrected | "
               "Δ corr. vs B0 [CI] (b/w) | Δ corr. vs L [CI] (b/w) |", "|---|---|---|---|---|---|---|---|---|"]
        for sname, ids_s in SETS.items():
            ids = parse_ids(ids_s)
            cor = {t: corrected(t, ids, set(u[t]["counted_ids"]) if t in u else set(), prompts) for t in tags}
            for t in tags:
                c = cor[t]
                e = {"counted": c["n_counted_in_set"],
                     "ambiguous": len([i for i in (u[t]["ambiguous_ids"] if t in u else []) if i in ids]),
                     "absent_scored": c["absent_scored"], "absent_corrected": c["absent_corrected"],
                     "acc_corrected": float(np.mean([c["acc_corrected"][i] for i in ids]))}
                cells = []
                for ref in (B0, L):
                    if t == ref:
                        cells.append("—")
                        continue
                    d = np.array([c["acc_corrected"][i] - cor[ref]["acc_corrected"][i] for i in ids])
                    ci = bootstrap_ci(d)
                    e[f"d_corr_vs_{NAME[ref]}"] = {"mean": float(d.mean()), "ci95": ci, "better": int((d > 0).sum()),
                                                   "worse": int((d < 0).sum()), "sign_p": sign_test(d)}
                    cells.append(f"{100 * d.mean():+.1f} [{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}] "
                                 f"({int((d > 0).sum())}/{int((d < 0).sum())})")
                out["substitutes"].setdefault(sname, {})[NAME[t]] = e
                md.append(f"| {sname} | {NAME[t]} | {e['counted']} | {e['ambiguous']} | {e['absent_scored']:.3f} | "
                          f"{e['absent_corrected']:.3f} | {e['acc_corrected']:.3f} | {cells[0]} | {cells[1]} |")
        md += ["", f"Agreement with 3b-3 on re-judged videos: "
                   f"{ {k: {'n': v['n'], 'same': v['same']} for k, v in u['_agreement_with_3b3'].items()} }", ""]
    else:
        md += ["(substitute-object check not yet unblinded)", ""]
    json.dump(out, open(os.path.join(TEMPO, "results", "step3c_summary.json"), "w"), indent=1, default=float)
    open(os.path.join(TEMPO, "results", "step3c_summary.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
