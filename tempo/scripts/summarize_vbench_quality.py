#!/usr/bin/env python3
"""Step 6 summary (brief §6, pre-registered question Q6): VBench imaging_quality / subject_consistency.

  summarize_vbench_quality.py [--all80 results/vbench/step6_all80.json] [--seed43 results/vbench/step6_seed43.json]
  -> results/step6_summary.json, results/step6_summary.md

Per dimension:
  * paired L - B0 on all 80 prompts (L_b2g2_s42 vs B0_s42): mean, 10,000-resample bootstrap 95% CI, exact sign test
    (paired_accuracy_analysis.py bootstrap_ci / sign_test);
  * seed spread on the 20 held-out prompts (data/heldout_ids.txt): mean |B0_s43 - B0_s42|;
  * Q6 verdict as pre-registered: "within the seed spread" only if BOTH |mean paired delta (all 80)| <= seed spread
    AND the 95% CI of the paired delta includes 0.
  Secondary (not part of the verdict): L - B0 restricted to the 20 held-out prompts.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

TEMPO = os.environ.get("TEMPO_ROOT", os.path.expanduser("~/tempo"))
sys.path.insert(0, os.path.join(TEMPO, "scripts"))
from paired_accuracy_analysis import bootstrap_ci, sign_test  # noqa: E402
from run_vbench import DIMS, parse_ids  # noqa: E402

REF, ARM, SEEDCTL = "B0_s42", "L_b2g2_s42", "B0_s43"


def scores(res, tag, ids, dim):
    if tag not in res["tags"]:
        raise KeyError(f"{tag} not in {sorted(res['tags'])}")
    pv = res["tags"][tag]["per_video"]
    missing = [i for i in ids if str(i) not in pv]
    if missing:
        raise KeyError(f"{tag}: ids missing {missing}")
    x = np.array([pv[str(i)][dim] for i in ids], dtype=float)
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{tag}/{dim}: non-finite scores")
    if not np.all((x >= 0) & (x <= 1.0001)):
        raise ValueError(f"{tag}/{dim}: score outside [0, 1] (scale error?) {x.min()} .. {x.max()}")
    return x


def paired_stats(d):
    return {"n": int(len(d)), "mean": float(d.mean()), "ci95": bootstrap_ci(d), "sign_p": sign_test(d),
            "n_pos": int((d > 0).sum()), "n_neg": int((d < 0).sum()), "n_tie": int((d == 0).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all80", default="results/vbench/step6_all80.json")
    ap.add_argument("--seed43", default="results/vbench/step6_seed43.json")
    ap.add_argument("--ids", default="@" + os.path.join(TEMPO, "data", "all80_ids.txt"))
    ap.add_argument("--heldout", default="@" + os.path.join(TEMPO, "data", "heldout_ids.txt"))
    ap.add_argument("--name", default="step6_summary")
    a = ap.parse_args()
    p80 = a.all80 if os.path.isabs(a.all80) else os.path.join(TEMPO, a.all80)
    p43 = a.seed43 if os.path.isabs(a.seed43) else os.path.join(TEMPO, a.seed43)
    r80, r43 = json.load(open(p80)), json.load(open(p43))
    ids, held = parse_ids(a.ids), parse_ids(a.heldout)
    if len(ids) != 80 or len(held) != 20 or not set(held) <= set(ids):
        raise ValueError(f"expected 80 ids and 20 held-out ids within them, got {len(ids)} / {len(held)}")
    hpos = [ids.index(i) for i in held]
    for k in ["versions", "imaging_quality_preprocessing_mode", "dims", "videos_root", "device", "gpu", "mode"]:
        if r80["meta"].get(k) != r43["meta"].get(k):
            raise ValueError(f"inputs not comparable: meta[{k!r}] {r80['meta'].get(k)!r} != {r43['meta'].get(k)!r}")

    out = {"ref": REF, "arm": ARM, "seedctl": SEEDCTL, "ids": ids, "heldout": held,
           "inputs": {"all80": p80, "seed43": p43}, "meta_all80": r80["meta"], "meta_seed43": r43["meta"], "dims": {}}
    for dim in DIMS:
        b0, l = scores(r80, REF, ids, dim), scores(r80, ARM, ids, dim)
        s43 = scores(r43, SEEDCTL, held, dim)
        d = l - b0
        seed_d = s43 - b0[hpos]
        spread = float(np.abs(seed_d).mean())
        pd = paired_stats(d)
        ph = paired_stats(d[hpos])
        within_spread = abs(pd["mean"]) <= spread
        ci_has_0 = pd["ci95"][0] <= 0.0 <= pd["ci95"][1]
        out["dims"][dim] = {
            "mean_B0": float(b0.mean()), "mean_L": float(l.mean()),
            "paired_L_minus_B0_all80": pd,
            "seed_spread_heldout20": {"n": len(held), "mean_abs_s43_minus_s42": spread,
                                      "mean_s43_minus_s42": float(seed_d.mean()),
                                      "mean_B0_s42": float(b0[hpos].mean()), "mean_B0_s43": float(s43.mean())},
            "secondary_L_minus_B0_heldout20": ph,
            "q6": {"reading": "delta and CI on all 80; spread on the 20 held-out (brief §6)",
                   "abs_mean_delta_le_spread": bool(within_spread), "ci95_includes_0": bool(ci_has_0),
                   "within_seed_spread": bool(within_spread and ci_has_0)},
            "q6_alt_heldout20": {"reading": "delta, CI and spread all on the 20 held-out (secondary)",
                                 "abs_mean_delta_le_spread": bool(abs(ph["mean"]) <= spread),
                                 "ci95_includes_0": bool(ph["ci95"][0] <= 0.0 <= ph["ci95"][1]),
                                 "within_seed_spread": bool(abs(ph["mean"]) <= spread
                                                            and ph["ci95"][0] <= 0.0 <= ph["ci95"][1])},
            "per_prompt": {str(i): {"B0_s42": float(b0[k]), "L_b2g2_s42": float(l[k])} for k, i in enumerate(ids)},
        }
        for k, i in enumerate(held):
            out["dims"][dim]["per_prompt"][str(i)]["B0_s43"] = float(s43[k])
    mj = os.path.join(TEMPO, "results", "metric", "phase4", REF, "temporal_accuracy_one_object.json")
    if os.path.isfile(mj):
        sys.path.insert(0, os.path.join(TEMPO, "src"))
        from tempo_ctrl import benchmark
        prompts = benchmark.load_one_object()
        name2id = {benchmark.video_name(prompts[i]["prompt"]): i for i in ids}
        clean = sorted(name2id[os.path.basename(v["video_path"])] for v in json.load(open(mj))["temporal_accuracy"][1]
                       if os.path.basename(v["video_path"]) in name2id
                       and v["object_absent_successes"] == v["absent_frames"])
        cpos = [ids.index(i) for i in clean]
        for dim in DIMS:
            b0, l = scores(r80, REF, ids, dim), scores(r80, ARM, ids, dim)
            out["dims"][dim]["diag_B0_absent_all_correct"] = {"ids": clean, **(paired_stats((l - b0)[cpos]) if len(cpos) > 1
                                                                                 else {"n": len(cpos)})}
    for dim, r in out["dims"].items():
        if not all(math.isfinite(v) for v in [r["paired_L_minus_B0_all80"]["mean"],
                                              r["seed_spread_heldout20"]["mean_abs_s43_minus_s42"]]):
            raise ValueError(f"{dim}: non-finite summary")
    os.makedirs(os.path.join(TEMPO, "results"), exist_ok=True)
    jp = os.path.join(TEMPO, "results", f"{a.name}.json")
    json.dump(out, open(jp, "w"), indent=1)

    f = lambda x: f"{x:.4f}"  # noqa: E731
    lines = [f"# Step 6: VBench quality (vbench {r80['meta']['versions']['vbench']}, custom_input mode)", "",
             f"L = `{ARM}`, B0 = `{REF}`, seed control = `{SEEDCTL}` (20 held-out prompts). "
             f"Bootstrap: 10,000 resamples of the 80 paired differences (paired_accuracy_analysis.bootstrap_ci, seed 0); sign test: exact, "
             f"two-sided, ties dropped. imaging_quality = MUSIQ-SPAQ / 100 (preprocessing "
             f"'{r80['meta']['imaging_quality_preprocessing_mode']}'); subject_consistency = DINO ViT-B/16 frame "
             f"similarity. GPU: {r80['meta'].get('gpu')}.", "",
             "| dimension | B0 mean | L mean | L − B0 (80) | 95 % CI | sign p (+/−/0) | seed spread (20) | "
             "L − B0 (20 held-out) [95 % CI] | Q6: within seed spread? | alt. reading (all on 20) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for dim, r in out["dims"].items():
        p, s, q, h = (r["paired_L_minus_B0_all80"], r["seed_spread_heldout20"], r["q6"],
                      r["secondary_L_minus_B0_heldout20"])
        verdict = "**yes**" if q["within_seed_spread"] else (
            f"**no** (\\|Δ\\| ≤ spread: {'yes' if q['abs_mean_delta_le_spread'] else 'no'}; "
            f"CI ∋ 0: {'yes' if q['ci95_includes_0'] else 'no'})")
        lines.append(f"| {dim} | {f(r['mean_B0'])} | {f(r['mean_L'])} | {p['mean']:+.4f} | "
                     f"[{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}] | {p['sign_p']:.3g} ({p['n_pos']}/{p['n_neg']}/"
                     f"{p['n_tie']}) | {f(s['mean_abs_s43_minus_s42'])} | {h['mean']:+.4f} "
                     f"[{h['ci95'][0]:+.4f}, {h['ci95'][1]:+.4f}] | {verdict} | "
                     f"{'yes' if r['q6_alt_heldout20']['within_seed_spread'] else 'no'}"
                     f"{' (**readings disagree**)' if r['q6_alt_heldout20']['within_seed_spread'] != q['within_seed_spread'] else ''} |")
    lines += ["", "Q6 (pre-registered): L − B0 counts as within the seed spread only if both hold: |mean paired Δ| on "
              "all 80 ≤ mean |B0_s43 − B0_s42| on the 20 held-out prompts, and the 95 % CI of the paired Δ includes 0. "
              "The held-out-only L − B0 column and the alternative reading (Δ, CI and spread all on the 20 held-out) "
              "are secondary, not the verdict. Caveat: the spread is a per-prompt quantity (≈ 0.8 sd) while |mean Δ| "
              "shrinks like sd/√80, so with n = 80 the first condition rarely binds and Q6 effectively reduces to "
              "\"CI ∋ 0\"; a yes means no detectable shift, not proof of zero quality cost.",
              "", "**Interpretation caveat (subject_consistency).** VBench's score averages each frame's similarity to "
              "the previous frame and to the first frame; in these videos the first-frame term dominates. L keeps the "
              "object out of the early frames while B0 often shows it from frame 0, so the video changes more against "
              "its first frame under L *because the timing control works*. A negative Δ here is therefore not by "
              "itself a quality cost. Diagnostic: Δ on the prompts where B0's absent frames were all correct too "
              "(below). imaging_quality is milder: empty-scene frames vs object frames also differ in content.", "",
              "Diagnostic, prompts where B0 also kept every absent frame empty: " + "; ".join(
                  f"{dim}: n = {r['diag_B0_absent_all_correct']['n']}" + (
                      f", Δ {r['diag_B0_absent_all_correct']['mean']:+.4f} [{r['diag_B0_absent_all_correct']['ci95'][0]:+.4f}, "
                      f"{r['diag_B0_absent_all_correct']['ci95'][1]:+.4f}]" if 'mean' in r['diag_B0_absent_all_correct'] else "")
                  for dim, r in out["dims"].items() if "diag_B0_absent_all_correct" in r), "",
              "Provenance: B0_s42 and B0_s43 on the 20 held-out prompts come from part-1 phase 2 (same jobs, same "
              "path); of the other 60 B0_s42 / L_b2g2_s42 videos, 8 (pilot ids) are from part-1 phase 1 and 52 from "
              "step 4, all on the explicit path (step-4 repro checked byte-identical in job 163093). VBench 0.1.5 sets "
              "only `model.training = False` on MUSIQ, so its submodules stay in training mode; its dropout rates are 0, "
              "so scores are deterministic (seeds are also fixed before evaluation).", ""]
    mp = os.path.join(TEMPO, "results", f"{a.name}.md")
    open(mp, "w").write("\n".join(lines))
    print("\n".join(lines))
    print(f"wrote {jp} {mp}")


if __name__ == "__main__":
    main()
