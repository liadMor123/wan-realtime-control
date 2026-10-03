#!/usr/bin/env python3
"""Part 2, step 5a summary (CPU; brief §5): Wan2.1 B0 vs L(2,2) with K = 2 objects, seed 42, explicit path,
on the 20 two-object pairs of data/step5a_ids.txt (the first 20 scorable ones in file order).

Official two-object metric JSONs in results/metric/phase5/<tag>/ (run_official_temporal_accuracy.py --bench two-object).
Per arm: mean accuracy; L - B0 paired per pair with a 10,000-resample bootstrap 95 % CI, better/worse counts and a
two-sided sign test (paired_accuracy_analysis.py's bootstrap_ci / sign_test); off-frame accuracy (control_signal1 = 0: static seen and
temporal not) and on-frame accuracy (= 1: both seen), pooled over pairs; CLIP delta; wall time and peak memory from
the phase-5 rows. Q5: L gains >= +8 pts over B0 on the 20 pairs. No absolute calibration gate (brief §5); the gain is
printed next to TempoControl's +15.7 pts (37.5 -> 53.2, all 82 pairs). Writes results/step5a_summary.{md,json}.
"""
import json
import os
import sys

import numpy as np

TEMPO = os.environ.get("TEMPO_ROOT", os.path.expanduser("~/tempo"))
sys.path.insert(0, os.path.join(TEMPO, "src"))
from tempo_ctrl import benchmark  # noqa: E402
from paired_accuracy_analysis import bootstrap_ci, load_phase_rows, sign_test  # noqa: E402
from generate_benchmark_videos import parse_ids  # noqa: E402

PHASE = 5
REF, ARM = "B0_2obj_s42", "L_b2g2_2obj_s42"
TC_B0, TC_L = 0.375, 0.532                              # TempoControl paper, two objects, all 82 pairs
Q5_MIN = 0.08


def load_pair_metric(tag, ids, pairs, metric_dir=f"phase{PHASE}"):
    """Per-pair official results of `tag` from results/metric/<metric_dir>/ (step 5b passes phase5_gpu)."""
    p = os.path.join(TEMPO, "results", "metric", metric_dir, tag, "temporal_accuracy_two_objects.json")
    res = json.load(open(p))
    per = {}
    for v in res["temporal_accuracy"][1]:
        name = os.path.basename(v["video_path"])
        pid = next((i for i in ids if benchmark.row_video_name(pairs[i]) == name), None)
        if pid is not None:
            per[pid] = v
    if set(per) != set(ids):
        raise RuntimeError(f"{tag}: metric missing pair ids {sorted(set(ids) - set(per))}")
    if len(res["temporal_accuracy"][1]) == len(ids):   # our mean must equal the official script's own mean
        assert abs(res["temporal_accuracy"][0] - np.mean([per[i]["video_results"] for i in ids])) < 1e-9, tag
    return per, res


def arm_stats(tag, ids, pairs, rows, metric_dir=f"phase{PHASE}", quality_dir=f"phase{PHASE}"):
    per, res = load_pair_metric(tag, ids, pairs, metric_dir)
    acc = np.array([per[i]["video_results"] for i in ids])
    n_off = np.array([int((pairs[i]["mask"] == 0).sum()) for i in ids])
    n_on = np.array([int((pairs[i]["mask"] == 1).sum()) for i in ids])
    off = np.array([per[i]["static_object_success_rate"] for i in ids])
    on = np.array([per[i]["temp_object_success_rate"] for i in ids])
    by = {r["prompt_id"]: r for r in rows if r["run_tag"] == tag}   # last row wins on re-runs
    o = {"n": len(ids), "acc_mean": float(acc.mean()), "acc_per_pair": dict(zip(ids, acc.round(3).tolist())),
         "off_frame_acc": float((off * n_off).sum() / n_off.sum()),
         "on_frame_acc": float((on * n_on).sum() / n_on.sum()),
         "off_per_pair": dict(zip(ids, off.round(3).tolist())), "on_per_pair": dict(zip(ids, on.round(3).tolist())),
         "official_mean": res["temporal_accuracy"][0], "official_n": len(res["temporal_accuracy"][1]),
         "n_rows": sum(i in by for i in ids), "wall_per_pair": {i: by[i]["wall_s"] for i in ids if i in by},
         "wall_s_mean": float(np.mean([by[i]["wall_s"] for i in ids if i in by])) if by else None,
         "peak_gb_max": float(np.max([by[i]["peak_gb"] for i in ids if i in by])) if by else None}
    q = os.path.join(TEMPO, "results", "quality", quality_dir, f"{tag}.json")
    if os.path.isfile(q):
        c = {int(k): v["clip_mean"] for k, v in json.load(open(q)).items()}
        if set(ids) <= set(c):
            o["clip_per_pair"] = {i: c[i] for i in ids}
            o["clip_mean"] = float(np.mean([c[i] for i in ids]))
    return o, acc


def add_paired_stats(b, l_, acc_b, acc_l, ids):
    """L - B0 per pair (arm_stats outputs of the same ids): adds the paired statistics to l_; returns (d, ci)."""
    d = acc_l - acc_b
    ci = bootstrap_ci(d)
    ovh = [l_["wall_per_pair"][i] / b["wall_per_pair"][i] - 1         # same pair, same node (shards are by pair)
           for i in ids if i in l_["wall_per_pair"] and i in b["wall_per_pair"]]
    l_.update({"d_vs_ref_mean": float(d.mean()), "d_vs_ref_per_pair": dict(zip(ids, d.round(3).tolist())),
               "d_vs_ref_ci95": ci, "n_better": int((d > 0).sum()), "n_worse": int((d < 0).sum()),
               "n_tied": int((d == 0).sum()), "sign_p_vs_ref": sign_test(d),
               "d_off_frame": l_["off_frame_acc"] - b["off_frame_acc"],
               "d_on_frame": l_["on_frame_acc"] - b["on_frame_acc"],
               "n_timed": len(ovh), "overhead_vs_ref_mean": float(np.mean(ovh)) if ovh else None})
    if "clip_mean" in b and "clip_mean" in l_:
        l_["clip_d_vs_ref"] = float(np.mean([l_["clip_per_pair"][i] - b["clip_per_pair"][i] for i in ids]))
    return d, ci


def main():
    ids = parse_ids(open(os.path.join(TEMPO, "data", "step5a_ids.txt")).read().strip())
    assert len(ids) == 20, ids
    pairs = benchmark.load_two_objects()
    rows = load_phase_rows(PHASE)
    b, acc_b = arm_stats(REF, ids, pairs, rows)
    l_, acc_l = arm_stats(ARM, ids, pairs, rows)
    d, ci = add_paired_stats(b, l_, acc_b, acc_l, ids)
    met = bool(d.mean() >= Q5_MIN)
    out = {"ids": ids, "B0": b, "L": l_,
           "TempoControl": {"B0_all82": TC_B0, "TempoControl_all82": TC_L, "gain_pts": 100 * (TC_L - TC_B0)},
           "Q5": {"d": float(d.mean()), "ci95": ci, "threshold": Q5_MIN, "met": met}}
    md = ["# Step 5a summary (Wan2.1-T2V-1.3B, explicit path, seed 42, two objects, K = 2; official metric)", "",
          f"20 pairs (first 20 scorable in file order): {', '.join(map(str, ids))}", "",
          "| run | accuracy | Δ vs B0 [95 % CI] | better / worse / tied (sign p) | off-frame ok | on-frame ok | CLIP Δ |",
          "|---|---|---|---|---|---|---|",
          f"| B0 | {b['acc_mean']:.3f} | — | — | {b['off_frame_acc']:.3f} | {b['on_frame_acc']:.3f} | — |",
          f"| L(2,2), K = 2 | {l_['acc_mean']:.3f} | {l_['d_vs_ref_mean']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] | "
          f"{l_['n_better']} / {l_['n_worse']} / {l_['n_tied']} ({l_['sign_p_vs_ref']:.4f}) | "
          f"{l_['off_frame_acc']:.3f} | {l_['on_frame_acc']:.3f} | {l_.get('clip_d_vs_ref', float('nan')):+.4f} |", "",
          "Off-frame: control_signal1 = 0 frames (10 per pair), static object detected and temporal object not. "
          "On-frame: the 11 frames with control_signal1 = 1, both detected.", "",
          f"**Gain vs TempoControl:** L − B0 = {100 * d.mean():+.1f} pts on our 20 pairs vs TempoControl's "
          f"+{100 * (TC_L - TC_B0):.1f} pts ({100 * TC_B0:.1f} → {100 * TC_L:.1f} %, all 82 pairs). No absolute "
          "comparison: 20 of 82 pairs, scorable only (brief §5). Caveat: their +15.7 includes the 17 pairs whose "
          "score is capped whatever the method does; if those gained nothing, their gain on the 65 scorable pairs is "
          "≈ +19.8 pts (15.7 × 82 / 65). With 20 pairs the standard error of our Δ is several points, so a Q5 verdict "
          "near +8 is within noise. Off-frame success mixes a missed static object with a leaked temporal object; it "
          "is not a leakage rate.", "",
          f"**Q5** (L ≥ +8 pts over B0 on the 20 scorable pairs): Δ = {100 * d.mean():+.1f} pts "
          f"[{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}] → **{'met' if met else 'MISSED'}**", "",
          f"Wall time / peak memory ({l_['n_rows']} L and {b['n_rows']} B0 phase-5 rows): L overhead "
          f"{100 * (l_['overhead_vs_ref_mean'] or 0):+.2f} %, peak {l_['peak_gb_max']} GB.", "",
          "Per pair (static / temporal: B0 → L):", "",
          "| pair | static | temporal | B0 | L | Δ |", "|---|---|---|---|---|---|"]
    for k, i in enumerate(ids):
        md.append(f"| {i} | {pairs[i]['static_object']} | {pairs[i]['temp_object']} | {acc_b[k]:.2f} | "
                  f"{acc_l[k]:.2f} | {d[k]:+.2f} |")
    json.dump(out, open(os.path.join(TEMPO, "results", "step5a_summary.json"), "w"), indent=1, default=float)
    open(os.path.join(TEMPO, "results", "step5a_summary.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
