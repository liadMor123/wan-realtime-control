#!/usr/bin/env python3
"""Descriptive quality: CLIP prompt-frame cosine similarity (mean of 16 evenly spaced frames,
open_clip ViT-L-14-quickgelu / openai — the OpenAI weights use QuickGELU) and a 4x4 contact sheet per video.

  clip_similarity_and_contact_sheets.py --tags B0_s42,L_b2g2_s42 --ids 0-7   -> results/quality/phase<N>/<tag>.json, results/contact/<tag>/p<id>.png
  --bench two_objects: pair ids; CLIP text = the full prompt the model was given (`prompt` column), video named by
  original_prompt.
  --out-name phase5_gpu: write results/quality/phase5_gpu/<tag>.json instead of results/quality/phase<N>/ (GPU
  re-scoring must not overwrite the CPU-scored files). --require-cuda: fail unless CLIP runs on CUDA; the device is
  recorded in results/quality/<dir>/_clip_device.json. Contact sheets that already exist are never rewritten.
  --prompt-set showcase: ids of data/showcase_prompts.csv (benchmark.load_showcase) instead of the benchmark's.
"""
import argparse
import json
import os
import sys

TEMPO = os.environ.get("TEMPO_ROOT", os.path.expanduser("~/tempo"))
sys.path.insert(0, os.path.join(TEMPO, "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from tempo_ctrl import benchmark  # noqa: E402
from generate_benchmark_videos import parse_ids  # noqa: E402


def sample_16_frames(path):
    cap = cv2.VideoCapture(path)
    v = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        v.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    if len(v) != 81:
        raise RuntimeError(f"{path}: {len(v)} frames, expected 81")
    idx = np.linspace(0, len(v) - 1, 16).round().astype(int)
    return [Image.fromarray(v[i]) for i in idx], idx


def contact_sheet(frames, path):
    w, h = frames[0].size
    tw, th = w // 4, h // 4
    sheet = Image.new("RGB", (4 * tw, 4 * th))
    for i, f in enumerate(frames):
        sheet.paste(f.resize((tw, th)), ((i % 4) * tw, (i // 4) * th))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sheet.save(path)


def main():
    import open_clip
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--phase", type=int)
    ap.add_argument("--out-name", default=None, help="results/quality/<out-name>/ (default phase<N>)")
    ap.add_argument("--bench", default="one_object", choices=("one_object", "two_objects"))
    ap.add_argument("--require-cuda", action="store_true")
    ap.add_argument("--prompt-set", choices=("showcase",), default=None)
    a = ap.parse_args()
    if a.phase is None and a.out_name is None:
        ap.error("--phase or --out-name is required")
    qdir = os.path.join(TEMPO, "results", "quality", a.out_name or f"phase{a.phase}")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if a.require_cuda and dev != "cuda":
        raise SystemExit("### --require-cuda: torch.cuda.is_available() is False")
    model, _, pre = open_clip.create_model_and_transforms("ViT-L-14-quickgelu", pretrained="openai", device=dev)
    tokz = open_clip.get_tokenizer("ViT-L-14-quickgelu")
    model.eval()
    pdev = str(next(model.parameters()).device)
    info = {"device": dev, "param_device": pdev, "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "NVIDIA_TF32_OVERRIDE": os.environ.get("NVIDIA_TF32_OVERRIDE"),
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32, "tags": a.tags.split(",")}
    print("[clip] device", json.dumps(info), flush=True)
    if a.require_cuda and not pdev.startswith("cuda"):
        raise SystemExit(f"### --require-cuda: CLIP parameters are on {pdev}")
    rows = benchmark.load_showcase(a.bench) if a.prompt_set else benchmark.load_benchmark(a.bench)
    for tag in a.tags.split(","):
        out = {}
        for i in parse_ids(a.ids):
            r = rows[i]
            p = os.path.join(TEMPO, "videos", tag, benchmark.row_video_name(r))
            frames, idx = sample_16_frames(p)
            with torch.no_grad():
                x = torch.stack([pre(f) for f in frames]).to(dev)
                if a.require_cuda and not x.is_cuda:
                    raise SystemExit(f"### --require-cuda: CLIP input on {x.device}")
                im = model.encode_image(x)
                tx = model.encode_text(tokz([r["prompt"]]).to(dev))
                im = im / im.norm(dim=-1, keepdim=True)
                tx = tx / tx.norm(dim=-1, keepdim=True)
                sims = (im @ tx.T).squeeze(1).float().cpu().numpy()
            out[i] = {"clip_mean": float(sims.mean()), "clip_per_frame": sims.round(4).tolist(),
                      "frame_idx": idx.tolist()}
            cpath = os.path.join(TEMPO, "results", "contact", tag, f"p{i:02d}.png")
            if a.out_name is None or not os.path.exists(cpath):   # default runs rewrite, as before
                contact_sheet(frames, cpath)
        os.makedirs(qdir, exist_ok=True)
        json.dump(out, open(os.path.join(qdir, f"{tag}.json"), "w"), indent=1)
        print(tag, "clip mean", np.mean([v["clip_mean"] for v in out.values()]).round(4), flush=True)
    if a.require_cuda:
        info["max_memory_allocated_mb"] = torch.cuda.max_memory_allocated() / 2**20
        json.dump(info, open(os.path.join(qdir, "_clip_device.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
