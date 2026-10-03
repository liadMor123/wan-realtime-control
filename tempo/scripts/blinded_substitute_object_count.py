#!/usr/bin/env python3
"""Step 3c-5: extended blinded substitute-object check (pre-registered protocol, part 2, step 3c).

  blinded_substitute_object_count.py make TAG1,TAG2,...   -> results/blind3c/sheets/<id>.png (random ids, one shuffled pool over all tags x 80
                                     prompts), results/blind3c/batches/batch_NN.json (~40 ids each, pool order) and
                                     results/blind3c/key.json (hidden key; not opened before all judgements are committed)
  blinded_substitute_object_count.py unblind              -> joins results/blind3c/judgements/batch_NN.json with the key: per-arm counts,
                                     results/blind3c/unblinded.json, agreement with 3b-3 on the re-judged videos, and
                                     the case folder results/blind3c/cases/ (+ index.md) of every counted / ambiguous case

Sheets are rendered exactly as in 3b-3 (scripts/blinded_substitute_object_sheets.py): the target's name and the 16 frames at output
indices linspace(0, 80, 16), each with a bar, red = OFF, green = ON (mask at latent frame (k + 3) // 4). No arm shown.
"""
import glob
import json
import os
import random
import shutil
import sys

TEMPO = os.environ.get("TEMPO_ROOT", os.path.expanduser("~/tempo"))
sys.path.insert(0, os.path.join(TEMPO, "src"))
sys.path.insert(0, os.path.join(TEMPO, "scripts"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from blinded_substitute_object_sheets import read_frames  # noqa: E402
from tempo_ctrl import benchmark  # noqa: E402

OUT = os.path.join(TEMPO, "results", "blind3c")
IDS = list(range(80))
BATCH = 40
ARM = {"SF_B0_fa2kv_s42": "B0", "SF_L_b2g2_fa2kv_s42": "L", "SF_PSRF_fa2kv_s42": "PS-RF",
       "SF_PSLL_fa2kv_s42": "PS-LongLive-style", "SF_PSRFL_fa2kv_s42": "PS-RF+L"}
OLD3B3 = {"SF_B0_fa2kv_s42": "SF_B0_fa2kv_s42", "SF_L_b2g2_fa2kv_s42": "SF_L_b2g2_fa2kv_s42",
          "SF_PSRF_fa2kv_s42": "SF_PS_s42"}          # 3c tag -> the 3b-3 tag of the same (or byte-identical) videos


def sheet(path, row, sid):
    """Identical rendering to blinded_substitute_object_sheets.make()."""
    v = read_frames(path)
    m = row["mask"]
    idx = np.linspace(0, 80, 16).round().astype(int)
    tiles = []
    for k in idx:
        f = cv2.resize(v[k], (208, 120))
        on = int(m[(k + 3) // 4]) == 1
        bar = np.zeros((10, 208, 3), np.uint8)
        bar[:] = (0, 160, 0) if on else (0, 0, 200)
        cv2.putText(bar, f"{'ON' if on else 'OFF'} f{k}", (4, 9), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        tiles.append(np.vstack([bar, f]))
    grid = np.vstack([np.hstack(tiles[r * 4:(r + 1) * 4]) for r in range(4)])
    head = np.full((28, grid.shape[1], 3), 255, np.uint8)
    cv2.putText(head, f"sheet {sid}   target: {row['temp_object']}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 0), 1)
    return np.vstack([head, grid])


def make(tags):
    if os.path.exists(os.path.join(OUT, "key.json")):
        raise SystemExit("results/blind3c/key.json exists: the pool was already made")
    rows = benchmark.load_one_object()
    items = [(t, i) for t in tags for i in IDS]
    for t, i in items:
        p = os.path.join(TEMPO, "videos", t, benchmark.video_name(rows[i]["prompt"]))
        if not os.path.isfile(p):
            raise SystemExit(f"missing video {p}")
    rng = random.SystemRandom()                  # OS entropy: the mapping exists only in key.json
    rng.shuffle(items)
    os.makedirs(os.path.join(OUT, "sheets"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "batches"), exist_ok=True)
    key, order = {}, []
    for t, i in items:
        sid = f"{rng.getrandbits(24):06x}"
        while sid in key:
            sid = f"{rng.getrandbits(24):06x}"
        key[sid] = {"tag": t, "prompt_id": i}
        order.append(sid)
        img = sheet(os.path.join(TEMPO, "videos", t, benchmark.video_name(rows[i]["prompt"])), rows[i], sid)
        cv2.imwrite(os.path.join(OUT, "sheets", f"{sid}.png"), img)
    nb = -(-len(order) // BATCH)
    for b in range(nb):
        json.dump(order[b::nb], open(os.path.join(OUT, "batches", f"batch_{b:02d}.json"), "w"), indent=1)
    json.dump(key, open(os.path.join(OUT, "key.json"), "w"), indent=1)
    json.dump(sorted(key), open(os.path.join(OUT, "sheet_ids.json"), "w"), indent=1)
    print(f"{len(key)} sheets in {nb} batches")


def unblind():
    key = json.load(open(os.path.join(OUT, "key.json")))
    jd = {}
    for p in sorted(glob.glob(os.path.join(OUT, "judgements", "batch_*.json"))):
        for sid, j in json.load(open(p)).items():
            if sid in jd:
                raise SystemExit(f"sheet {sid} judged twice")
            jd[sid] = j
    missing, extra = set(key) - set(jd), set(jd) - set(key)
    if missing or extra:
        raise SystemExit(f"{len(missing)} sheets not judged, {len(extra)} unknown ids")
    rows = benchmark.load_one_object()
    tags = sorted({k["tag"] for k in key.values()})
    res = {t: {"arm": ARM[t], "count": 0, "ambiguous": 0, "group_flags": 0, "counted_ids": [], "ambiguous_ids": [],
               "videos": []} for t in tags}
    for sid, j in jd.items():
        t, i = key[sid]["tag"], key[sid]["prompt_id"]
        amb, cnt = bool(j.get("ambiguous", False)), bool(j["counts"])
        if amb and cnt:
            raise SystemExit(f"sheet {sid}: ambiguous cases are listed, not counted")
        res[t]["videos"].append({"sheet": sid, "prompt_id": i, **j})
        res[t]["count"] += int(cnt)
        res[t]["ambiguous"] += int(amb)
        res[t]["group_flags"] += int(bool(j.get("group_flag", False)))
        if cnt:
            res[t]["counted_ids"].append(i)
        if amb:
            res[t]["ambiguous_ids"].append(i)
    for t in tags:
        res[t]["counted_ids"].sort()
        res[t]["ambiguous_ids"].sort()
    # agreement with 3b-3 on the 20 held-out videos judged there
    old = json.load(open(os.path.join(TEMPO, "results", "blind", "unblinded.json")))
    agree = {}
    for t, ot in OLD3B3.items():
        if t not in res or ot not in old:
            continue
        o = {v["prompt_id"]: ("amb" if v.get("ambiguous") else ("count" if v["counts"] else "no")) for v in old[ot]["videos"]}
        n = {v["prompt_id"]: ("amb" if v.get("ambiguous") else ("count" if v["counts"] else "no"))
             for v in res[t]["videos"] if v["prompt_id"] in o}
        agree[t] = {"n": len(o), "same": sum(o[i] == n[i] for i in o),
                    "diff": {i: {"3b3": o[i], "3c": n[i]} for i in o if o[i] != n[i]}}
    res["_agreement_with_3b3"] = agree
    json.dump(res, open(os.path.join(OUT, "unblinded.json"), "w"), indent=1)
    # case folder
    cd = os.path.join(OUT, "cases")
    os.makedirs(cd, exist_ok=True)
    lines = ["# Step 3c substitute-object check: every counted or ambiguous case", "",
             "Rule: pre-registered protocol, part 2, 3b-3 (S1 / S2); pool and protocol: 3c-5. Judged blind, in batches of "
             "about 40 sheets by separate subagents; verdicts committed before unblinding.", "",
             "| file | prompt | target | arm | verdict | reason |", "|---|---|---|---|---|---|"]
    ents = []
    for t in tags:
        for v in res[t]["videos"]:
            if not (v["counts"] or v.get("ambiguous")):
                continue
            verdict = f"counted {v.get('criterion', '?')}" if v["counts"] else "ambiguous"
            fn = f"{ARM[t]}_p{v['prompt_id']:02d}_{'counted' if v['counts'] else 'ambiguous'}.png"
            shutil.copy2(os.path.join(OUT, "sheets", f"{v['sheet']}.png"), os.path.join(cd, fn))
            ents.append((v["prompt_id"], ARM[t], fn, rows[v["prompt_id"]]["temp_object"], verdict,
                         str(v.get("reason", "")).replace("|", "/").replace("\n", " ")))
    for pid, arm, fn, obj, verdict, reason in sorted(ents):
        lines.append(f"| [{fn}]({fn}) | {pid} | {obj} | {arm} | {verdict} | {reason} |")
    lines += ["", "Per arm (80 prompts): " + "; ".join(f"{ARM[t]} {res[t]['count']} counted, {res[t]['ambiguous']} "
                                                       f"ambiguous" for t in tags)]
    open(os.path.join(cd, "index.md"), "w").write("\n".join(lines) + "\n")
    for t in tags:
        print(f"{ARM[t]:>20}: counted {res[t]['count']:2d} / 80, ambiguous {res[t]['ambiguous']:2d}, "
              f"group-flagged {res[t]['group_flags']:2d}")
    print("agreement with 3b-3:", json.dumps({t: {k: v for k, v in a.items() if k != 'diff'} for t, a in agree.items()}))


if __name__ == "__main__":
    if sys.argv[1] == "make":
        make(sys.argv[2].split(","))
    elif sys.argv[1] == "unblind":
        unblind()
    else:
        raise SystemExit(__doc__)
