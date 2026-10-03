#!/usr/bin/env python3
"""Step 3b-3: blinded 16-frame contact sheets for the substitute-object count (pre-registered protocol, part 2, step 3b).

  blinded_substitute_object_sheets.py make   -> results/blind/sheets/<id>.png (random ids) and results/blind/key.json (hidden key)
  blinded_substitute_object_sheets.py unblind -> joins results/blind/judgements.json (written before unblinding) with the key

Each sheet shows the target object's name and the 16 frames at output indices linspace(0, 80, 16), each with a bar:
red = OFF (the benchmark mask says the object must be absent), green = ON. The arm is not shown.
"""
import json
import os
import random
import sys

TEMPO = os.environ.get("TEMPO_ROOT", os.path.expanduser("~/tempo"))
sys.path.insert(0, os.path.join(TEMPO, "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from tempo_ctrl import benchmark  # noqa: E402

TAGS = ["SF_B0_fa2kv_s42", "SF_L_b2g2_fa2kv_s42", "SF_PS_s42"]
IDS = [2, 3, 4, 5, 6, 22, 23, 24, 25, 26, 42, 43, 44, 45, 46, 62, 63, 64, 65, 66]
OUT = os.path.join(TEMPO, "results", "blind")


def read_frames(path):
    cap = cv2.VideoCapture(path)
    v = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        v.append(f)
    cap.release()
    if len(v) != 81:
        raise RuntimeError(f"{path}: {len(v)} frames")
    return v


def make():
    rows = benchmark.load_one_object()
    items = [(t, i) for t in TAGS for i in IDS]
    rng = random.SystemRandom()                  # OS entropy: the mapping exists only in key.json (not reproducible)
    rng.shuffle(items)
    os.makedirs(os.path.join(OUT, "sheets"), exist_ok=True)
    key = {}
    idx = np.linspace(0, 80, 16).round().astype(int)
    for t, i in items:
        sid = f"{rng.getrandbits(24):06x}"
        while sid in key:
            sid = f"{rng.getrandbits(24):06x}"
        key[sid] = {"tag": t, "prompt_id": i}
        v = read_frames(os.path.join(TEMPO, "videos", t, benchmark.video_name(rows[i]["prompt"])))
        m = rows[i]["mask"]
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
        cv2.putText(head, f"sheet {sid}   target: {rows[i]['temp_object']}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 1)
        cv2.imwrite(os.path.join(OUT, "sheets", f"{sid}.png"), np.vstack([head, grid]))
    json.dump(key, open(os.path.join(OUT, "key.json"), "w"), indent=1)
    json.dump(sorted(key), open(os.path.join(OUT, "sheet_ids.json"), "w"), indent=1)
    print(f"{len(key)} sheets written")


def unblind():
    key = json.load(open(os.path.join(OUT, "key.json")))
    jd = json.load(open(os.path.join(OUT, "judgements.json")))
    missing = set(key) - set(jd)
    if missing:
        raise SystemExit(f"{len(missing)} sheets not judged")
    res = {t: {"count": 0, "ambiguous": 0, "videos": []} for t in TAGS}
    for sid, j in jd.items():
        t = key[sid]["tag"]
        res[t]["videos"].append({"sheet": sid, "prompt_id": key[sid]["prompt_id"], **j})
        amb = bool(j.get("ambiguous", False))
        if amb and j["counts"]:
            raise SystemExit(f"sheet {sid}: ambiguous cases are listed, not counted")
        res[t]["count"] += int(j["counts"] and not amb)
        res[t]["ambiguous"] += int(amb)
    json.dump(res, open(os.path.join(OUT, "unblinded.json"), "w"), indent=1)
    for t in TAGS:
        mp = os.path.join(TEMPO, "results", "metric", "phase23", t, "temporal_accuracy_one_object.json")
        absent = None
        if os.path.isfile(mp):
            per = json.load(open(mp))["temporal_accuracy"][1]
            rows = benchmark.load_one_object()
            names = {benchmark.video_name(rows[i]["prompt"]): i for i in IDS}
            sel = [v for v in per if os.path.basename(v["video_path"]) in names]
            absent = sum(v["object_absent_successes"] for v in sel) / sum(v["absent_frames"] for v in sel)
            res[t]["absent_rate_20"] = absent
        print(t, "count", res[t]["count"], "/ 20; ambiguous", res[t]["ambiguous"], "; absent-frame accuracy", absent)
    json.dump(res, open(os.path.join(OUT, "unblinded.json"), "w"), indent=1)


if __name__ == "__main__":
    {"make": make, "unblind": unblind}[sys.argv[1]]()
