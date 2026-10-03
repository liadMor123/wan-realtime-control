#!/usr/bin/env python3
"""Showcase page data (pre-registered, "Part 2 · showcase"): for each TempoControl-website prompt, the B0 and L
videos (same prompt, same seed), re-encoded for the web, plus the timing mask and the official per-video accuracy.

  build_showcase_manifest.py  -> results/showcase/media/<slug>__{b0,l}.mp4 and results/showcase/manifest.json

Sections (website order): Wan2.1 single object (4 benchmark prompts reused + the new "dog, last second"),
Self-Forcing single object (the same 4 benchmark prompts), Wan2.1 two objects (5 pairs in the website's wording).
"""
import json
import os
import re
import subprocess
import sys

TEMPO = os.environ.get("TEMPO_ROOT", os.path.expanduser("~/tempo"))
sys.path.insert(0, os.path.join(TEMPO, "src"))
from tempo_ctrl import benchmark  # noqa: E402

OUT = os.path.join(TEMPO, "results", "showcase")
FFMPEG = __import__("imageio_ffmpeg").get_ffmpeg_exe()


def load_metric_by_video(phase, tag, kind):
    p = os.path.join(TEMPO, "results", "metric", phase, tag, f"temporal_accuracy_{kind}.json")
    res = json.load(open(p))
    return {os.path.basename(v["video_path"]): v for v in res["temporal_accuracy"][1]}


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:60]


def encode_for_web(src, dst):
    if os.path.isfile(dst):
        return
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", src, "-c:v", "libx264", "-preset", "slow", "-crf", "24",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", dst], check=True)


def main():
    os.makedirs(os.path.join(OUT, "media"), exist_ok=True)
    one = benchmark.load_one_object()
    sc1, sc2 = benchmark.load_showcase("one_object"), benchmark.load_showcase("two_objects")
    m = {"wan": {t: load_metric_by_video("phase4", t, "one_object") for t in ("B0_s42", "L_b2g2_s42")},
         "sf": {t: load_metric_by_video("phase23", t, "one_object") for t in ("SF_B0_fa2kv_s42", "SF_L_b2g2_fa2kv_s42")},
         "sc1": {t: load_metric_by_video("phase25_gpu", t, "one_object") for t in ("showcase_B0_s42", "showcase_L_b2g2_s42")},
         "sc2": {t: load_metric_by_video("phase25_gpu", t, "two_objects")
                 for t in ("showcase_B0_2obj_s42", "showcase_L_b2g2_2obj_s42")}}
    sections = []

    def showcase_entry(row, tags, mkey, label, source):
        name = benchmark.row_video_name(row) if "static_object" in row else benchmark.video_name(row["prompt"])
        e = {"prompt": row["prompt"], "temp_object": row["temp_object"], "static_object": row.get("static_object"),
             "mask": [int(x) for x in row["mask"]], "source": source, "label": label, "arms": {}}
        for arm, tag in zip(("b0", "l"), tags):
            src = os.path.join(TEMPO, "videos", tag, name)
            if not os.path.isfile(src):
                raise SystemExit(f"missing {src}")
            fn = f"{slug(label)}__{arm}.mp4"
            encode_for_web(src, os.path.join(OUT, "media", fn))
            v = m[mkey][tag][name]
            e["arms"][arm] = {"file": f"media/{fn}", "tag": tag, "accuracy": v["video_results"],
                              "src_bytes": os.path.getsize(src),
                              "web_bytes": os.path.getsize(os.path.join(OUT, "media", fn))}
        return e

    # website order: dog last, apple 4th, umbrella 3rd, dog 2nd, skateboard 3rd
    wan = []
    for key in ("dog-last", 47, 25, 16, 36):
        if key == "dog-last":
            row = next(iter(sc1.values()))
            wan.append(showcase_entry(row, ("showcase_B0_s42", "showcase_L_b2g2_s42"), "sc1", "dog, last second",
                             "website prompt, not in the benchmark: generated for this page"))
        else:
            r = dict(one[key], mask=one[key]["mask"])
            when = re.search(r"during the (\w+) second", r["prompt"]).group(1)
            wan.append(showcase_entry(r, ("B0_s42", "L_b2g2_s42"), "wan", f"{r['temp_object']}, {when} second",
                             f"benchmark prompt {key} (step 4)"))
    sections.append({"id": "wan-one", "title": "Wan2.1-T2V-1.3B, one object", "model": "Wan2.1-T2V-1.3B",
                     "entries": wan})
    sf = []
    for key in (47, 25, 16, 36):
        r = one[key]
        when = re.search(r"during the (\w+) second", r["prompt"]).group(1)
        sf.append(showcase_entry(r, ("SF_B0_fa2kv_s42", "SF_L_b2g2_fa2kv_s42"), "sf", f"{r['temp_object']}, {when} second (SF)",
                        f"benchmark prompt {key} (step 3b)"))
    sections.append({"id": "sf-one", "title": "Self-Forcing (streaming), one object", "model": "Self-Forcing",
                     "entries": sf})
    pairs = []
    for row in sc2.values():                                   # file order = website order
        pairs.append(showcase_entry(row, ("showcase_B0_2obj_s42", "showcase_L_b2g2_2obj_s42"), "sc2",
                           f"{row['static_object']} then {row['temp_object']}",
                           f"website wording of benchmark pair {row['source_ids'][0]}: generated for this page"))
    sections.append({"id": "wan-two", "title": "Wan2.1-T2V-1.3B, two objects", "model": "Wan2.1-T2V-1.3B",
                     "entries": pairs})
    json.dump({"sections": sections, "fps": 16, "frames": 81}, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
    tot = sum(a["web_bytes"] for s in sections for e in s["entries"] for a in e["arms"].values())
    print(f"{sum(len(s['entries']) for s in sections)} examples, web media {tot / 2**20:.1f} MB")


if __name__ == "__main__":
    main()
