#!/usr/bin/env python3
"""Run TempoControl's official, unmodified temporal_accuracy.py (one-object or two-object) on a videos dir.

Runs in ~/tempo/venv_metric. Writes a subset CSV (rows of the requested prompt ids,
file order, columns and text unchanged) so the script scores only those videos, then
checks the output format and returns the parsed JSON.

  run_official_temporal_accuracy.py --videos DIR --ids 0,1,2      -> DIR/temporal_accuracy_one_object.json
  run_official_temporal_accuracy.py --preflight                   -> synthetic positive/negative controls
  --bench two-object (with either form)                  -> data/two_objects.csv, pair ids, videos named
                                                            f"{original_prompt}-0.mp4", temporal_accuracy_two_objects.json
  --require-cuda (with either form)                      -> GPU scoring: fail unless the detector ran on CUDA
  --prompt-set showcase --ids 1,2                        -> ids of data/showcase_prompts.csv (rows of --bench); the
                                                            subset CSV is written in the official file's columns

Device. The official scripts call YOLOv10.from_pretrained(...) and model.predict(source=frame, save=False) with no
device argument, so ultralytics 8.1.34 uses its default device=None: select_device(None) -> "" -> cuda:0 whenever
torch.cuda.is_available(), else CPU (ultralytics/utils/torch_utils.py select_device; engine/predictor.py
setup_model). So the same unmodified script runs on the GPU on a GPU node and on the CPU on a CPU node. With
--require-cuda the official script is started through DEVICE_WRAPPER, which only records (it changes no argument and
no value): torch.cuda.is_available(), the TF32 settings, the device of every predictor setup_model (the predictor's
device and the network's parameter device) and the device of every tensor that reaches the network's forward, into
<out>/_metric_device.json; the run then fails unless every one of them is CUDA.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile

TEMPO = os.environ.get("TEMPO_ROOT", os.path.expanduser("~/tempo"))
TC = os.path.join(TEMPO, "ext", "TempoControl")
CSV = os.path.join(TC, "data", "one_object.csv")
PY = sys.executable
# per benchmark: (csv, subset csv name, result json, column naming the video)
BENCH = {"one-object": (CSV, "_subset_one_object.csv", "temporal_accuracy_one_object.json", "prompt"),
         "two-object": (os.path.join(TC, "data", "two_objects.csv"), "_subset_two_objects.csv",
                        "temporal_accuracy_two_objects.json", "original_prompt")}


def write_subset_csv(ids, path, csv_path=CSV):
    with open(csv_path, newline="") as f:
        rd = csv.DictReader(f)
        rows = list(rd)
        fields = rd.fieldnames
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for i in ids:
            wr.writerow(rows[i])
    return [rows[i] for i in ids]


SHOWCASE_CSV = os.path.join(TEMPO, "data", "showcase_prompts.csv")
# official column order of data/one_object.csv / data/two_objects.csv
OFFICIAL_COLS = {"one-object": ["prompt", "temp_object", "control_signal1"],
                 "two-object": ["prompt", "original_prompt", "Object 1", "Object 2", "temp_object", "static_object",
                                "control_signal1", "control_signal2"]}


def write_showcase_subset_csv(ids, path, bench):
    """Subset CSV of showcase rows (data/showcase_prompts.csv) in the official benchmark's columns. Two-object
    original_prompt is "a <static> and a <temporal> (website)", so Object 1 = static_object, Object 2 = temp_object
    (the official script reads only original_prompt, temp_object, static_object and control_signal1)."""
    name = {"one-object": "one_object", "two-object": "two_objects"}[bench]
    with open(SHOWCASE_CSV, newline="") as f:
        by_id = {int(r["id"]): r for r in csv.DictReader(f)}
    rows = []
    for i in ids:
        r = by_id[i]
        if r["bench"] != name:
            raise ValueError(f"showcase id {i} is {r['bench']}, not {name}")
        rows.append({c: r[{"Object 1": "static_object", "Object 2": "temp_object"}.get(c, c)]
                     for c in OFFICIAL_COLS[bench]})
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=OFFICIAL_COLS[bench])
        wr.writeheader()
        wr.writerows(rows)
    return rows


# Started as `python -c DEVICE_WRAPPER <log.json> temporal_accuracy.py <args>` in the TempoControl dir (sys.path[0] = cwd,
# as for `python temporal_accuracy.py`). It wraps two ultralytics methods with pure recorders, then runs the official
# entry point unmodified as __main__.
DEVICE_WRAPPER = r"""
import atexit, json, os, runpy, sys
log_path = sys.argv[1]
sys.argv = sys.argv[2:]
import torch
from ultralytics.engine import predictor as _pred
from ultralytics.nn import autobackend as _ab
rec = {"torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
       "cuda_device_count": torch.cuda.device_count(),
       "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
       "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
       "NVIDIA_TF32_OVERRIDE": os.environ.get("NVIDIA_TF32_OVERRIDE"),
       "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32, "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
       "setup_model": [], "forward_calls": 0, "forward_input_devices": {}, "forward_param_devices": {}}
_setup, _fwd = _pred.BasePredictor.setup_model, _ab.AutoBackend.forward
def setup_model(self, *a, **k):
    r = _setup(self, *a, **k)
    rec["setup_model"].append({"args_device": repr(self.args.device), "predictor_device": str(self.device),
                               "param_device": str(next(self.model.parameters()).device), "fp16": bool(self.args.half)})
    return r
def forward(self, im, *a, **k):
    rec["forward_calls"] += 1
    d = rec["forward_input_devices"]; d[str(im.device)] = d.get(str(im.device), 0) + 1
    pd = str(next(self.parameters()).device); rec["forward_param_devices"][pd] = rec["forward_param_devices"].get(pd, 0) + 1
    return _fwd(self, im, *a, **k)
_pred.BasePredictor.setup_model, _ab.AutoBackend.forward = setup_model, forward
def dump():
    if torch.cuda.is_available():
        rec["max_memory_allocated_mb"] = torch.cuda.max_memory_allocated() / 2**20
    json.dump(rec, open(log_path, "w"), indent=1)
atexit.register(dump)
runpy.run_path(sys.argv[0], run_name="__main__")
"""


def check_device_log(log, n_frames_min=1):
    """Raise unless the recorded run used CUDA everywhere: CUDA available, every predictor on cuda, every network
    input and parameter on cuda, and at least n_frames_min forward calls (one per scored frame, + GPU warmup)."""
    bad = []
    if not log.get("cuda_available"):
        bad.append("torch.cuda.is_available() is False")
    if not log.get("setup_model"):
        bad.append("no predictor was set up")
    for s in log.get("setup_model", []):
        if not (s["predictor_device"].startswith("cuda") and s["param_device"].startswith("cuda")):
            bad.append(f"predictor on {s['predictor_device']} / params on {s['param_device']}")
    for key in ("forward_input_devices", "forward_param_devices"):
        non = {d: n for d, n in log.get(key, {}).items() if not d.startswith("cuda")}
        if non:
            bad.append(f"{key}: {non}")
    if log.get("forward_calls", 0) < n_frames_min:
        bad.append(f"{log.get('forward_calls', 0)} forward calls < {n_frames_min} frames")
    if bad:
        raise RuntimeError("metric did NOT run on CUDA: " + "; ".join(bad))
    return True


def run_official(videos, ids, out=None, bench="one-object", require_cuda=False, prompt_set=None):
    out = out or videos
    os.makedirs(out, exist_ok=True)
    csv_path, sub_name, res_name, col = BENCH[bench]
    sub = os.path.join(out, sub_name)
    if prompt_set == "showcase":
        rows = write_showcase_subset_csv(ids, sub, bench)
    elif prompt_set is None:
        rows = write_subset_csv(ids, sub, csv_path)
    else:
        raise ValueError(f"unknown prompt set {prompt_set!r}")
    missing = [r[col] for r in rows if not os.path.isfile(os.path.join(videos, f"{r[col]}-0.mp4"))]
    if missing:
        raise FileNotFoundError(f"{len(missing)} videos missing in {videos}, e.g. {missing[0]!r}")
    env = dict(os.environ, HF_HOME=os.path.join(TEMPO, "cache", "hf"), YOLO_VERBOSE="False")
    cmd = [PY, "temporal_accuracy.py", "--benchmark", bench, "--videos_path", os.path.abspath(videos),
           "--output_path", os.path.abspath(out), "--csv_file", os.path.abspath(sub)]
    dev_log = os.path.join(os.path.abspath(out), "_metric_device.json")
    if require_cuda:
        if os.path.exists(dev_log):
            os.remove(dev_log)
        cmd = [PY, "-c", DEVICE_WRAPPER, dev_log] + cmd[1:]
    r = subprocess.run(cmd, cwd=TC, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout[-3000:] + r.stderr[-3000:])
        raise RuntimeError(f"temporal_accuracy.py failed rc={r.returncode}")
    # ~/tempo/runs/ holds our own run configs; ultralytics training writes runs/detect/train*
    if any(os.path.isdir(os.path.join(d, "runs", "detect")) for d in (TC, TEMPO)):
        raise RuntimeError("ultralytics created a runs/ dir: YOLOv10.from_pretrained trained instead of "
                           "loading (huggingface_hub too new; venv_metric pins 0.23.4)")
    res = json.load(open(os.path.join(out, res_name)))
    (check_one_object_format if bench == "one-object" else check_two_object_format)(res, len(ids))
    if require_cuda:
        log = json.load(open(dev_log))
        n_frames = sum(v["frame_count"] for v in res["temporal_accuracy"][1])
        check_device_log(log, n_frames)
        print(f"[metric] device check PASS: {log['gpu']}, predictor {log['setup_model'][0]['predictor_device']}, "
              f"{log['forward_calls']} forward calls on {log['forward_input_devices']} for {n_frames} frames, "
              f"TF32 override {log['NVIDIA_TF32_OVERRIDE']}", file=sys.stderr, flush=True)
    return res


def check_one_object_format(res, n):
    assert set(res) == {"temporal_accuracy", "global_frame_metrics"}, res.keys()
    mean, per = res["temporal_accuracy"]
    assert isinstance(mean, float) and len(per) == n, (mean, len(per))
    for v in per:
        assert v["frame_count"] == 20, v
        assert v["absent_frames"] + v["present_frames"] == 20, v
        assert 0.0 <= v["video_results"] <= 1.0


def check_two_object_format(res, n):
    """metrics/temporal_accuracy_two_objects.py: 21 frames (len(control_signal1)); static_object_success_rate is over
    the control_signal1 = 0 frames (static seen, temporal not), temp_object_success_rate over the = 1 frames (both)."""
    assert set(res) == {"temporal_accuracy", "global_absent_object_success_rate",
                        "global_present_object_success_rate"}, res.keys()
    mean, per = res["temporal_accuracy"]
    assert isinstance(mean, float) and len(per) == n, (mean, len(per))
    for v in per:
        assert v["frame_count"] == 21 and v["video_path"], v
        assert abs(v["video_results"] - v["success_frame_count"] / 21) < 1e-12, v
        for k in ("video_results", "static_object_success_rate", "temp_object_success_rate"):
            assert 0.0 <= v[k] <= 1.0, v


def preflight_two_object(require_cuda=False):
    """Two-object wiring check on pair 78 ('a person and a toilet': static person, temporal toilet, control_signal1 =
    10 zeros then 11 ones). All-gray video -> 0. The letterboxed bus.jpg (persons, no toilet) on every frame -> the
    10 "off" frames succeed (person seen, toilet absent), the 11 "on" frames fail (no toilet): 10/21."""
    import cv2
    import numpy as np
    import ultralytics
    csv_path = BENCH["two-object"][0]
    with open(csv_path, newline="") as f:
        row = list(csv.DictReader(f))[78]
    assert (row["static_object"], row["temp_object"]) == ("person", "toilet"), row
    assert row["control_signal1"].split() == ["0"] * 10 + ["1"] * 11, row["control_signal1"]
    bus = cv2.imread(os.path.join(os.path.dirname(ultralytics.__file__), "assets", "bus.jpg"))
    assert bus is not None, "ultralytics/assets/bus.jpg missing"
    gray = np.full((480, 832, 3), 127, np.uint8)
    h, w = bus.shape[:2]
    sc = min(832 / w, 480 / h)
    small = cv2.resize(bus, (int(w * sc), int(h * sc)))
    frame = gray.copy()
    y0, x0 = (480 - small.shape[0]) // 2, (832 - small.shape[1]) // 2
    frame[y0:y0 + small.shape[0], x0:x0 + small.shape[1]] = small
    out = {}
    for name, img in (("negative_all_gray", gray), ("static_only_bus", frame)):
        d = tempfile.mkdtemp(prefix=f"tempo_preflight2_{name}_")
        vw = cv2.VideoWriter(os.path.join(d, f"{row['original_prompt']}-0.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                             16, (832, 480))
        for _ in range(81):
            vw.write(img)
        vw.release()
        out[name] = run_official(d, [78], bench="two-object", require_cuda=require_cuda)["temporal_accuracy"][1][0]
    print(json.dumps(out, indent=1))
    assert out["negative_all_gray"]["video_results"] == 0.0, out
    v = out["static_only_bus"]
    assert abs(v["video_results"] - 10 / 21) < 1e-9 and v["static_object_success_rate"] == 1.0 \
        and v["temp_object_success_rate"] == 0.0, out
    print(json.dumps({"preflight_two_object": "PASS", **{k: v["video_results"] for k, v in out.items()}}))
    return out


def preflight_one_object(require_cuda=False):
    import cv2
    import numpy as np
    import ultralytics
    bus = cv2.imread(os.path.join(os.path.dirname(ultralytics.__file__), "assets", "bus.jpg"))
    assert bus is not None, "ultralytics/assets/bus.jpg missing"
    gray = np.full((480, 832, 3), 127, np.uint8)
    h, w = bus.shape[:2]                          # letterbox (keep aspect) into 832x480
    sc = min(832 / w, 480 / h)
    small = cv2.resize(bus, (int(w * sc), int(h * sc)))
    frame = gray.copy()
    y0, x0 = (480 - small.shape[0]) // 2, (832 - small.shape[1]) // 2
    frame[y0:y0 + small.shape[0], x0:x0 + small.shape[1]] = small
    bus = frame
    with open(CSV, newline="") as f:
        row0 = next(csv.DictReader(f))
    assert row0["temp_object"] == "person"
    out = {}
    for name, first_obj_frame in (("negative_all_gray", None), ("positive_bus_from_16", 16)):
        d = tempfile.mkdtemp(prefix=f"tempo_preflight_{name}_")
        vw = cv2.VideoWriter(os.path.join(d, f"{row0['prompt']}-0.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 16, (832, 480))
        for k in range(81):
            vw.write(bus if first_obj_frame is not None and k >= first_obj_frame else gray)
        vw.release()
        res = run_official(d, [0], require_cuda=require_cuda)
        out[name] = res["temporal_accuracy"][1][0]
    # mask[1:] of row 0 has 4 zeros of 20; the script samples frames linspace(0, 80, 20) -> 0,4,8,12,16,...
    print(json.dumps(out, indent=1))
    assert abs(out["negative_all_gray"]["video_results"] - 0.2) < 1e-9, out
    assert abs(out["positive_bus_from_16"]["video_results"] - 1.0) < 1e-9, out
    print(json.dumps({"preflight": "PASS", **{k: v["video_results"] for k, v in out.items()}}))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos")
    ap.add_argument("--ids")
    ap.add_argument("--out")
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--bench", default="one-object", choices=sorted(BENCH))
    ap.add_argument("--require-cuda", action="store_true", help="GPU scoring: fail unless the detector ran on CUDA")
    ap.add_argument("--prompt-set", choices=("showcase",), default=None)
    a = ap.parse_args()
    if a.preflight:
        preflight_one_object(a.require_cuda) if a.bench == "one-object" else preflight_two_object(a.require_cuda)
    else:
        res = run_official(a.videos, [int(x) for x in a.ids.split(",")], a.out, a.bench, a.require_cuda, a.prompt_set)
        print(json.dumps({"videos": a.videos, "mean": res["temporal_accuracy"][0],
                          "per_video": [v["video_results"] for v in res["temporal_accuracy"][1]]}))
