#!/usr/bin/env python3
"""GPU-vs-CPU consistency gate for the official temporal-accuracy metric (and CLIP), final runs (step2acc, 5b).

All part-1/part-2 scores so far were computed on CPU nodes. The final runs score on the GPU (slurm/score_gpu.sbatch),
with the same unmodified official scripts; the same GPU job re-scores already CPU-scored videos:

  one-object : B0_s42 and L_b2g2_s42 on the 20 held-out ids, results/metric/phase2_gpu/ vs results/metric/phase2/ (CPU)
  two-object : B0_2obj_s42 and L_b2g2_2obj_s42 on the 20 step-5a pairs, results/metric/phase5_gpu/ vs .../phase5/ (CPU)

Per video: the official video_results and the frame-level success counts, CPU vs GPU, exact equality and |delta|.
CLIP (clip_mean per video) is compared the same way, from results/quality/<dir>/. The device records written by
run_official_temporal_accuracy.py --require-cuda (_metric_device.json) and clip_similarity_and_contact_sheets.py --require-cuda (_clip_device.json) are
collected as evidence. A section whose GPU files do not exist yet is reported as not scored. Nothing here fails on a
disagreement: it is reported (summaries print it next to their numbers).

  gpu_vs_cpu_metric_check.py   -> results/gpu_vs_cpu_metric_check.json (both sections, from whatever exists)
"""
import glob
import json
import os
import sys

import numpy as np

TEMPO = os.environ.get("TEMPO_ROOT", os.path.expanduser("~/tempo"))
sys.path.insert(0, os.path.join(TEMPO, "src"))
sys.path.insert(0, os.path.join(TEMPO, "scripts"))
from tempo_ctrl import benchmark  # noqa: E402
from generate_benchmark_videos import parse_ids  # noqa: E402

OUT = os.path.join(TEMPO, "results", "gpu_vs_cpu_metric_check.json")
GATES = {
    "one_object": {"bench": "one_object", "res": "temporal_accuracy_one_object.json",
                   "ids_file": "data/heldout_ids.txt", "tags": ("B0_s42", "L_b2g2_s42"),
                   "cpu": "phase2", "gpu": "phase2_gpu",
                   "fields": ("video_results", "success_frame_count", "frame_count", "object_absent_successes",
                              "object_present_successes")},
    "two_object": {"bench": "two_objects", "res": "temporal_accuracy_two_objects.json",
                   "ids_file": "data/step5a_ids.txt", "tags": ("B0_2obj_s42", "L_b2g2_2obj_s42"),
                   "cpu": "phase5", "gpu": "phase5_gpu",
                   "fields": ("video_results", "success_frame_count", "frame_count", "static_object_success_rate",
                              "temp_object_success_rate")},
}
GPU_DIRS = ("phase24", "phase2_gpu", "phase5_gpu")


def read_ids(rel):
    return parse_ids(open(os.path.join(TEMPO, rel)).read().strip())


def per_video(path, ids, rows):
    """{id: official per-video dict} for the ids, matched by video file name (a result may hold more videos)."""
    res = json.load(open(path))
    name_to_id = {benchmark.row_video_name(rows[i]): i for i in ids}
    per = {}
    for v in res["temporal_accuracy"][1]:
        if v.get("video_path") and os.path.basename(v["video_path"]) in name_to_id:
            per[name_to_id[os.path.basename(v["video_path"])]] = v
    missing = sorted(set(ids) - set(per))
    if missing:
        raise RuntimeError(f"{path}: no result for ids {missing}")
    return per


def clip_per_video(qdir, tag, ids):
    p = os.path.join(TEMPO, "results", "quality", qdir, f"{tag}.json")
    if not os.path.isfile(p):
        return None
    c = {int(k): v["clip_mean"] for k, v in json.load(open(p)).items()}
    return c if set(ids) <= set(c) else None


def compare(section):
    """Compare GPU and CPU scores of one gate section. Returns a JSON-able dict (status 'not_scored_yet' if the GPU
    files are missing)."""
    g = GATES[section]
    rows = benchmark.load_benchmark(g["bench"])
    ids = read_ids(g["ids_file"])
    out = {"ids": ids, "cpu_dir": f"results/metric/{g['cpu']}", "gpu_dir": f"results/metric/{g['gpu']}", "tags": {}}
    for tag in g["tags"]:
        pc = os.path.join(TEMPO, "results", "metric", g["cpu"], tag, g["res"])
        pg = os.path.join(TEMPO, "results", "metric", g["gpu"], tag, g["res"])
        if not os.path.isfile(pg):
            out["tags"][tag] = {"status": "not_scored_yet"}
            continue
        cpu, gpu = per_video(pc, ids, rows), per_video(pg, ids, rows)
        pv = []
        for i in ids:
            fe = {f: cpu[i].get(f) == gpu[i].get(f) for f in g["fields"]}
            d = abs(gpu[i]["video_results"] - cpu[i]["video_results"])
            pv.append({"id": i, "cpu": cpu[i]["video_results"], "gpu": gpu[i]["video_results"],
                       "abs_diff": d, "equal": all(fe.values()), "fields_equal": fe})
        diffs = np.array([p["abs_diff"] for p in pv])
        t = {"status": "compared", "n": len(ids), "n_equal": sum(p["equal"] for p in pv),
             "all_equal": all(p["equal"] for p in pv), "max_abs_diff": float(diffs.max()),
             "mean_abs_diff": float(diffs.mean()),
             "cpu_mean": float(np.mean([p["cpu"] for p in pv])), "gpu_mean": float(np.mean([p["gpu"] for p in pv])),
             "per_video": pv}
        cc, cg = clip_per_video(g["cpu"], tag, ids), clip_per_video(g["gpu"], tag, ids)
        if cc and cg:
            cd = np.array([abs(cg[i] - cc[i]) for i in ids])
            t["clip"] = {"max_abs_diff": float(cd.max()), "mean_abs_diff": float(cd.mean()),
                         "n_bitwise_equal": int(sum(cg[i] == cc[i] for i in ids))}
        out["tags"][tag] = t
    done = [t for t in out["tags"].values() if t["status"] == "compared"]
    out["status"] = "compared" if len(done) == len(g["tags"]) else ("partial" if done else "not_scored_yet")
    if done:
        out["n_videos"] = sum(t["n"] for t in done)
        out["n_equal"] = sum(t["n_equal"] for t in done)
        out["all_equal"] = all(t["all_equal"] for t in done)
        out["max_abs_diff"] = max(t["max_abs_diff"] for t in done)
    return out


def device_records():
    rec = {}
    for d in GPU_DIRS:
        for p in sorted(glob.glob(os.path.join(TEMPO, "results", "metric", d, "*", "_metric_device.json"))):
            r = json.load(open(p))
            rec[os.path.relpath(p, TEMPO)] = {
                "cuda_available": r.get("cuda_available"), "gpu": r.get("gpu"),
                "predictor_devices": sorted({s["predictor_device"] for s in r.get("setup_model", [])}),
                "param_devices": sorted({s["param_device"] for s in r.get("setup_model", [])}),
                "forward_input_devices": r.get("forward_input_devices"), "forward_calls": r.get("forward_calls"),
                "NVIDIA_TF32_OVERRIDE": r.get("NVIDIA_TF32_OVERRIDE"),
                "max_memory_allocated_mb": r.get("max_memory_allocated_mb")}
        p = os.path.join(TEMPO, "results", "quality", d, "_clip_device.json")
        if os.path.isfile(p):
            rec[os.path.relpath(p, TEMPO)] = json.load(open(p))
    return rec


def gate_one_line(sec):
    """Human-readable gate line for a summary."""
    if sec.get("status") != "compared":
        return f"GPU-vs-CPU gate: {sec.get('status', 'not run')}"
    s = (f"GPU-vs-CPU gate: {sec['n_equal']}/{sec['n_videos']} re-scored videos identical "
         f"(max |Δ| {sec['max_abs_diff']:.3f})")
    clips = [t["clip"]["max_abs_diff"] for t in sec["tags"].values() if "clip" in t]
    if clips:
        s += f"; CLIP max |Δ| {max(clips):.2e}"
    return s


def main():
    out = {"what": "official metric (and CLIP) on the GPU vs the earlier CPU scores of the same videos",
           "one_object": compare("one_object"), "two_object": compare("two_object"), "devices": device_records()}
    done = [out[s] for s in ("one_object", "two_object") if out[s]["status"] != "not_scored_yet"]
    out["all_equal"] = all(s["all_equal"] for s in done) if done else None
    tmp = OUT + ".tmp"
    json.dump(out, open(tmp, "w"), indent=1)
    os.replace(tmp, OUT)
    for s in ("one_object", "two_object"):
        print(f"[gate] {s}: {gate_one_line(out[s])}")
        for tag, t in out[s]["tags"].items():
            if t["status"] == "compared" and not t["all_equal"]:
                bad = [(p["id"], p["cpu"], p["gpu"]) for p in t["per_video"] if not p["equal"]]
                print(f"### GPU != CPU for {tag} on {len(bad)} videos (id, cpu, gpu): {bad}")
    print(f"[gate] wrote {OUT}")


if __name__ == "__main__":
    main()
