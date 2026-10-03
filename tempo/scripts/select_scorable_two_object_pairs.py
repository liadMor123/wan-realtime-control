#!/usr/bin/env python3
"""Part 2, step 5a pair selection (CPU; brief §5): the first 20 two-object pairs in file order among the 65
scorable ones -> data/step5a_ids.txt.

A pair is unscorable when an object name, normalised as the official metric does (str(x).strip().lower()), is not a
class name of the metric's detector: then that object can never be "detected". The class list is
data/yolov10x_names.json, dumped on CPU from the checkpoint the metric loads (YOLOv10.from_pretrained
('jameslahm/yolov10x'), cache/hf) with `--dump-names` under venv_metric. The script fails unless the unscorable set is
exactly the brief's 17 ids.

  select_scorable_two_object_pairs.py                # check + write data/step5a_ids.txt (venv)
  select_scorable_two_object_pairs.py --dump-names   # re-dump data/yolov10x_names.json (venv_metric; HF_HOME=cache/hf)
"""
import json
import os
import sys

TEMPO = os.environ.get("TEMPO_ROOT", os.path.expanduser("~/tempo"))
NAMES = os.path.join(TEMPO, "data", "yolov10x_names.json")
IDS = os.path.join(TEMPO, "data", "step5a_ids.txt")
UNSCORABLE = (1, 2, 12, 14, 15, 16, 17, 30, 31, 33, 34, 51, 52, 56, 60, 62, 79)   # brief §5, 0-based
N_PAIRS = 20


def yolo_names():
    return set(json.load(open(NAMES)).values())


def unscorable(rows, names):
    return [r["prompt_id"] for r in rows
            if any(str(r[k]).strip().lower() not in names for k in ("temp_object", "static_object"))]


def select_first_scorable(rows, names):
    bad = unscorable(rows, names)
    if tuple(bad) != UNSCORABLE:
        raise SystemExit(f"unscorable pairs {bad} != brief's {list(UNSCORABLE)}")
    return [r["prompt_id"] for r in rows if r["prompt_id"] not in bad][:N_PAIRS]


def compress_id_list(ids):
    """[0, 3, 4, 5] -> '0,3-5' (the data/*_ids.txt format read by generate_benchmark_videos.parse_ids)."""
    out, i = [], 0
    while i < len(ids):
        j = i
        while j + 1 < len(ids) and ids[j + 1] == ids[j] + 1:
            j += 1
        out.append(str(ids[i]) if j == i else f"{ids[i]}-{ids[j]}")
        i = j + 1
    return ",".join(out)


def dump_names():
    from ultralytics import YOLOv10
    m = YOLOv10.from_pretrained("jameslahm/yolov10x")
    json.dump({str(k): v for k, v in m.names.items()}, open(NAMES, "w"))
    print(f"wrote {NAMES}: {len(m.names)} classes")


def main():
    sys.path.insert(0, os.path.join(TEMPO, "src"))
    from tempo_ctrl import benchmark
    rows = benchmark.load_two_objects()
    names = yolo_names()
    assert len(rows) == 82 and len(names) == 80, (len(rows), len(names))
    ids = select_first_scorable(rows, names)
    s = compress_id_list(ids)
    if os.path.isfile(IDS) and open(IDS).read().strip() != s:
        raise SystemExit(f"{IDS} exists with different ids: {open(IDS).read().strip()!r} vs {s!r}")
    open(IDS, "w").write(s + "\n")
    print(f"{len(rows) - len(UNSCORABLE)} scorable pairs; first {N_PAIRS}: {s}")
    for i in ids:
        print(f"  {i:2d}  static {rows[i]['static_object']!r:16} temporal {rows[i]['temp_object']!r}")


if __name__ == "__main__":
    dump_names() if "--dump-names" in sys.argv else main()
