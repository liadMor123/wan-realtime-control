#!/usr/bin/env python3
"""Per-arm summary for one phase: official temporal accuracy (from each videos/<tag>/ JSON written by
run_official_temporal_accuracy.py), absent/present success, paired per-prompt differences vs a reference tag,
wall-time overhead vs the reference on the same prompt (same node: shards are by prompt), peak memory, CLIP.

  paired_accuracy_analysis.py --phase 1 --ids 0,1,20,21,40,41,60,61 --ref B0_s42 [--seedctl B0_s43]
  -> results/phase1_summary.json, results/phase1_summary.md
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

TEMPO = os.environ.get("TEMPO_ROOT", os.path.expanduser("~/tempo"))
sys.path.insert(0, os.path.join(TEMPO, "src"))
from tempo_ctrl import benchmark  # noqa: E402
from generate_benchmark_videos import parse_ids  # noqa: E402


def load_phase_rows(phase):
    rows = []
    for p in sorted(glob.glob(os.path.join(TEMPO, "results", "rows", f"phase{phase}_*.jsonl"))):
        rows += [json.loads(l) for l in open(p)]
    return rows


def metric_by_prompt(tag, ids, prompts, phase):
    res = json.load(open(os.path.join(TEMPO, "results", "metric", f"phase{phase}", tag,
                                      "temporal_accuracy_one_object.json")))
    per = {}
    for v in res["temporal_accuracy"][1]:
        name = os.path.basename(v["video_path"])
        pid = next((i for i in ids if benchmark.video_name(prompts[i]["prompt"]) == name), None)
        if pid is not None:                       # a subset analysis ignores the other scored videos
            per[pid] = v
    missing = set(ids) - set(per)
    if missing:
        raise RuntimeError(f"{tag}: metric missing prompt ids {sorted(missing)}")
    return per


def bootstrap_ci(d, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.choice(d, size=(n, len(d)), replace=True).mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def sign_test(d):
    """Two-sided exact sign test p-value (ties dropped)."""
    from math import comb
    k, n = int((d > 0).sum()), int((d != 0).sum())
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--ref", default="B0_s42")
    ap.add_argument("--seedctl", default=None, help="second-seed B0 tag: noise floor for paired diffs")
    ap.add_argument("--tags", default=None, help="restrict to these run tags (comma list)")
    ap.add_argument("--name", default=None, help="output basename (default phase<N>_summary)")
    a = ap.parse_args()
    ids = parse_ids(a.ids)
    prompts = benchmark.load_one_object()
    rows = load_phase_rows(a.phase)
    tags = sorted({r["run_tag"] for r in rows})
    if a.tags:
        tags = [t for t in tags if t in a.tags.split(",")]
    name = a.name or f"phase{a.phase}_summary"
    by = {(r["run_tag"], r["prompt_id"]): r for r in rows}   # last row wins on re-runs
    acc = {t: metric_by_prompt(t, ids, prompts, a.phase) for t in tags}
    clip = {}
    for t in tags:
        q = os.path.join(TEMPO, "results", "quality", f"phase{a.phase}", f"{t}.json")
        if os.path.isfile(q):
            clip[t] = {int(k): v["clip_mean"] for k, v in json.load(open(q)).items()}
            if not set(ids) <= set(clip[t]):
                del clip[t]
    ref = a.ref
    out = {}
    for t in tags:
        va = np.array([acc[t][i]["video_results"] for i in ids])
        d = va - np.array([acc[ref][i]["video_results"] for i in ids])
        ab = sum(acc[t][i]["object_absent_successes"] for i in ids) / sum(acc[t][i]["absent_frames"] for i in ids)
        pr = sum(acc[t][i]["object_present_successes"] for i in ids) / sum(acc[t][i]["present_frames"] for i in ids)
        wall = [by[(t, i)]["wall_s"] for i in ids if (t, i) in by]
        ovh = [by[(t, i)]["wall_s"] / by[(ref, i)]["wall_s"] - 1 for i in ids if (t, i) in by and (ref, i) in by]
        peak = [by[(t, i)]["peak_gb"] for i in ids if (t, i) in by]
        o = {"n": len(ids), "acc_mean": va.mean(), "acc_per_prompt": dict(zip(ids, va.round(3).tolist())),
             "d_vs_ref_mean": d.mean(), "d_vs_ref_per_prompt": dict(zip(ids, d.round(3).tolist())),
             "n_better": int((d > 0).sum()), "n_worse": int((d < 0).sum()),
             "d_vs_ref_ci95": bootstrap_ci(d), "sign_p_vs_ref": sign_test(d),
             "absent_rate": ab, "present_rate": pr,
             "wall_s_mean": float(np.mean(wall)) if wall else None, "n_timed": len(ovh),
             "overhead_vs_ref_mean": float(np.mean(ovh)) if ovh else None,
             "peak_gb_max": float(np.max(peak)) if peak else None}
        if t in clip and ref in clip:
            o["clip_mean"] = float(np.mean([clip[t][i] for i in ids]))
            o["clip_d_vs_ref"] = float(np.mean([clip[t][i] - clip[ref][i] for i in ids]))
        out[t] = o
    if a.seedctl and a.seedctl in out:
        sd = np.array([out[a.seedctl]["d_vs_ref_per_prompt"][i] for i in ids])
        out["_noise_floor"] = {"seed_ctl": a.seedctl, "mean_abs_paired_diff": float(np.abs(sd).mean()),
                               "mean_paired_diff": float(sd.mean()), "sd_paired_diff": float(sd.std(ddof=1))}
        for t in tags:
            if t in (ref, a.seedctl):
                continue
            dd = np.array([acc[t][i]["video_results"] - acc[a.seedctl][i]["video_results"] for i in ids])
            out[t]["d_vs_seedctl_mean"] = float(dd.mean())
            out[t]["d_vs_seedctl_ci95"] = bootstrap_ci(dd)
            out[t]["sign_p_vs_seedctl"] = sign_test(dd)
    os.makedirs(os.path.join(TEMPO, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(TEMPO, "results", f"{name}.json"), "w"), indent=1, default=float)
    lines = [f"# {name} (phase {a.phase}; prompt ids {a.ids}; reference {ref})", "",
             "| run | acc | Δ vs ref [95% CI] | better/worse (sign p) | absent ok | present ok | CLIP Δ | wall s | overhead | peak GB |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for t in sorted(tags, key=lambda t: -out[t]["acc_mean"]):
        o = out[t]
        ci = o["d_vs_ref_ci95"]
        lines.append(f"| {t} | {o['acc_mean']:.3f} | {o['d_vs_ref_mean']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] | "
                     f"{o['n_better']}/{o['n_worse']} ({o['sign_p_vs_ref']:.3f}) | "
                     f"{o['absent_rate']:.2f} | {o['present_rate']:.2f} | "
                     f"{o.get('clip_d_vs_ref', float('nan')):+.4f} | {o['wall_s_mean'] or float('nan'):.1f} | "
                     f"{(o['overhead_vs_ref_mean'] or 0) * 100:+.1f}% | {o['peak_gb_max'] or float('nan'):.2f} |")
    if "_noise_floor" not in out:
        lines += ["", "Per-prompt accuracy:", "", "| run | " + " | ".join(f"p{i}" for i in ids) + " |",
                  "|---|" + "---|" * len(ids)]
    if "_noise_floor" in out:
        nf = out["_noise_floor"]
        lines += ["", f"Noise floor ({nf['seed_ctl']} − {ref}, paired): mean {nf['mean_paired_diff']:+.3f}, "
                      f"mean |Δ| {nf['mean_abs_paired_diff']:.3f}, sd {nf['sd_paired_diff']:.3f}", "",
                  "| run | Δ vs seed control [95% CI] | sign p |", "|---|---|---|"]
        for t in tags:
            if "d_vs_seedctl_mean" in out[t]:
                c = out[t]["d_vs_seedctl_ci95"]
                lines.append(f"| {t} | {out[t]['d_vs_seedctl_mean']:+.3f} [{c[0]:+.3f}, {c[1]:+.3f}] | "
                             f"{out[t]['sign_p_vs_seedctl']:.3f} |")
        lines += ["", "Per-prompt accuracy:", "", "| run | " + " | ".join(f"p{i}" for i in ids) + " |",
                  "|---|" + "---|" * len(ids)]
    for t in sorted(tags, key=lambda t: -out[t]["acc_mean"]):
        lines.append(f"| {t} | " + " | ".join(f"{out[t]['acc_per_prompt'][i]:.2f}" for i in ids) + " |")
    open(os.path.join(TEMPO, "results", f"{name}.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
