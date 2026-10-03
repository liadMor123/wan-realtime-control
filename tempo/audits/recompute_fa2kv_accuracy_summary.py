#!/usr/bin/env python3
"""Independent recomputation of the fa2kv accuracy check (phase 24) from the raw result files.

Audit script: re-derives, without importing any project code, the numbers that summarize_fa2kv_accuracy.py reports
for B0 on Wan's flash path vs L(2,2) on the fa2kv path over the 20 held-out prompts: per-video accuracy from the
official metric JSONs, pooled and macro absent/present rates, the paired delta with sign test and bootstrap CI
(pure-Python and NumPy resamplers), the CLIP delta, wall-time and peak-memory statistics from the phase-24 rows, and
MD5 checks that the video files of the two arms differ from each other and from the explicit-path videos.

Run from a checkout that holds results/, videos/ and ext/TempoControl (TEMPO_ROOT overrides ~/tempo).
"""
import csv
import glob
import hashlib
import json
import math
import os
import random
import statistics as st

import numpy as np

os.chdir(os.environ.get("TEMPO_ROOT", os.path.expanduser("~/tempo")))
rows_csv = list(csv.DictReader(open("ext/TempoControl/data/one_object.csv", newline="")))
ids = [2, 3, 4, 5, 6, 22, 23, 24, 25, 26, 42, 43, 44, 45, 46, 62, 63, 64, 65, 66]


def load(p):
    res = json.load(open(p))["temporal_accuracy"]
    by = {}
    for v in res[1]:
        name = os.path.basename(v["video_path"])[:-6]
        by[name] = v
    return res[0], by


out = {}
for arm, path in (("B0", "results/metric/phase24/B0_flash_s42/temporal_accuracy_one_object.json"),
                  ("L", "results/metric/phase24/L_b2g2_fa2kv_s42/temporal_accuracy_one_object.json")):
    m, by = load(path)
    assert len(by) == 20, len(by)
    per = {}
    for i in ids:
        v = by[rows_csv[i]["prompt"]]
        per[i] = v
    acc = [per[i]["video_results"] for i in ids]
    recon = [per[i]["success_frame_count"] / per[i]["frame_count"] for i in ids]
    assert all(abs(a - b) < 1e-12 for a, b in zip(acc, recon))
    out[arm] = dict(official_mean=m, mean=sum(acc) / 20, per=dict(zip(ids, acc)),
                    absent=sum(per[i]["object_absent_successes"] for i in ids) / sum(per[i]["absent_frames"] for i in ids),
                    present=sum(per[i]["object_present_successes"] for i in ids) / sum(per[i]["present_frames"] for i in ids),
                    absent_macro=sum(per[i]["object_absent_success_rate"] for i in ids) / 20,
                    present_macro=sum(per[i]["object_present_success_rate"] for i in ids) / 20)
for a in out:
    print(a, "official mean", out[a]["official_mean"], "our mean", out[a]["mean"], "absent", out[a]["absent"], out[a]["absent_macro"], "present", out[a]["present"], out[a]["present_macro"])
d = [out["L"]["per"][i] - out["B0"]["per"][i] for i in ids]
md = sum(d) / 20
print("paired delta", md, "per", [round(x, 2) for x in d])
b = sum(x > 1e-9 for x in d); w = sum(x < -1e-9 for x in d); t = 20 - b - w
n = b + w
p = min(1.0, 2 * sum(math.comb(n, k) for k in range(0, min(b, w) + 1)) / 2 ** n)
print("better/worse/tied", b, w, t, "sign p two-sided", p)
for seed in (0, 1, 12345):
    rng = random.Random(seed)
    bs = sorted(sum(rng.choice(d) for _ in range(20)) / 20 for _ in range(10000))
    print("boot seed", seed, "CI", bs[249], bs[9749], "(pct 2.5/97.5 idx)")
rng = np.random.default_rng(0)
dd = np.array(d); bs = dd[rng.integers(0, 20, (10000, 20))].mean(1)
print("numpy boot CI", np.percentile(bs, [2.5, 97.5]))
# CLIP
cl = {}
for arm, tag in (("B0", "B0_flash_s42"), ("L", "L_b2g2_fa2kv_s42")):
    q = json.load(open(f"results/quality/phase24/{tag}.json"))
    assert set(map(int, q)) == set(ids), sorted(q)
    cl[arm] = {i: q[str(i)]["clip_mean"] for i in ids}
    print(arm, "clip mean", sum(cl[arm].values()) / 20, "n frames", {len(q[str(i)]["clip_per_frame"]) for i in ids})
cd = [cl["L"][i] - cl["B0"][i] for i in ids]
print("CLIP delta", sum(cd) / 20, "better/worse", sum(x > 0 for x in cd), sum(x < 0 for x in cd))
# rows
rows = [json.loads(l) for f in sorted(glob.glob("results/rows/phase24_*.jsonl")) for l in open(f)]
wall = {(r["run_tag"], r["prompt_id"]): r for r in rows}
wb = [wall[("B0_flash_s42", i)]["wall_s"] for i in ids]
wl = [wall[("L_b2g2_fa2kv_s42", i)]["wall_s"] for i in ids]
print("wall B0", st.mean(wb), "L", st.mean(wl), "ratio-of-means overhead", st.mean(wl) / st.mean(wb) - 1,
      "mean paired ratio", st.mean([l / b_ - 1 for l, b_ in zip(wl, wb)]),
      "min/max paired", min(l / b_ - 1 for l, b_ in zip(wl, wb)), max(l / b_ - 1 for l, b_ in zip(wl, wb)))
print("wall range B0", min(wb), max(wb), "L", min(wl), max(wl))
db = [wall[("B0_flash_s42", i)]["denoise_s"] for i in ids]; dl = [wall[("L_b2g2_fa2kv_s42", i)]["denoise_s"] for i in ids]
print("denoise overhead", st.mean(dl) / st.mean(db) - 1)
print("peak B0", sorted(round(wall[("B0_flash_s42", i)]["peak_gb"], 3) for i in ids))
print("peak L", sorted(round(wall[("L_b2g2_fa2kv_s42", i)]["peak_gb"], 3) for i in ids))


# video hashes
def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


same_arm = 0; same_expl_B0 = 0; same_expl_L = 0; missing = []
for i in ids:
    n = rows_csv[i]["prompt"] + "-0.mp4"
    hb = md5(f"videos/B0_flash_s42/{n}"); hl = md5(f"videos/L_b2g2_fa2kv_s42/{n}")
    same_arm += hb == hl
    for tag, h, ctr in (("B0_s42", hb, "B0"), ("L_b2g2_s42", hl, "L")):
        p = f"videos/{tag}/{n}"
        if not os.path.exists(p):
            missing.append(p); continue
        if md5(p) == h:
            if ctr == "B0": same_expl_B0 += 1
            else: same_expl_L += 1
print("identical B0flash==Lfa2kv", same_arm, "B0flash==B0explicit", same_expl_B0, "Lfa2kv==Lexplicit", same_expl_L, "missing", missing)
print("unique md5 within tags", len({md5(p) for p in glob.glob("videos/B0_flash_s42/*.mp4")}), len({md5(p) for p in glob.glob("videos/L_b2g2_fa2kv_s42/*.mp4")}))
print("sizes", sorted(os.path.getsize(p) for p in glob.glob("videos/*_s42/*.mp4") if "flash" in p or "fa2kv" in p)[:3])
