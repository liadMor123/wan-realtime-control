#!/usr/bin/env python3
"""
Tier-3 descriptive quality metrics for the J9 videos.

For every <arm>_p<i>.npy in --vid_dir computes CLIP prompt similarity
(openai/clip-vit-base-patch32, mean over frames), CLIP temporal consistency
(mean cosine between consecutive frame embeddings) and LPIPS between
consecutive frames (AlexNet backbone), and saves a 4x4 contact sheet per video
to --sheets. Descriptive only; no pass/fail.

Writes <out>/j9_metrics.json (per-video values, per-arm mean/min/max, metric
versions) and <sheets>/j9_<arm>_p<i>.png.
"""
import argparse, glob, json, os
import numpy as np, torch
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--vid_dir", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--sheets", required=True)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True); os.makedirs(args.sheets, exist_ok=True)
dev = torch.device("cuda"); torch.set_grad_enabled(False)

from transformers import CLIPModel, CLIPProcessor
import lpips, transformers
from importlib.metadata import version as _pkgver


def package_version(mod, name):
    return getattr(mod, "__version__", None) or (
        _pkgver(name) if name else "unknown")


LPIPS_VER = package_version(lpips, "lpips")


def extract_features(out):
    """transformers 5.x returns a model-output object from get_*_features;
    older versions return a tensor. Accept both."""
    if torch.is_tensor(out):
        return out
    for attr in ("image_embeds", "text_embeds", "pooler_output"):
        v = getattr(out, attr, None)
        if torch.is_tensor(v):
            return v
    v = getattr(out, "last_hidden_state", None)
    if torch.is_tensor(v):
        return v[:, 0]
    raise TypeError(f"cannot extract features from {type(out).__name__}")


CLIP_ID = "openai/clip-vit-base-patch32"
clip = CLIPModel.from_pretrained(CLIP_ID).to(dev).eval()
proc = CLIPProcessor.from_pretrained(CLIP_ID)
lpips_model = lpips.LPIPS(net="alex").to(dev).eval()
prompts = json.load(open(os.path.join(args.vid_dir, "j9_prompts.json")))["prompts"]
print(f"[metrics] CLIP {CLIP_ID} | lpips {LPIPS_VER} net=alex | "
      f"transformers {transformers.__version__}", flush=True)


def contact_sheet(arr, path):
    idx = np.linspace(0, len(arr) - 1, 16).astype(int)
    h, w = arr.shape[1] // 3, arr.shape[2] // 3
    tiles = [np.array(Image.fromarray(arr[i]).resize((w, h))) for i in idx]
    rows = [np.concatenate(tiles[r * 4:(r + 1) * 4], axis=1) for r in range(4)]
    Image.fromarray(np.concatenate(rows, axis=0)).save(path)


res = {}
for f in sorted(glob.glob(os.path.join(args.vid_dir, "*_p*.npy"))):
    base = os.path.basename(f)[:-4]; arm, pi = base.rsplit("_p", 1); pi = int(pi)
    arr = np.load(f)
    contact_sheet(arr, os.path.join(args.sheets, f"j9_{arm}_p{pi}.png"))
    ims = [Image.fromarray(x) for x in arr]
    embs = []
    for k in range(0, len(ims), 27):
        b = proc(images=ims[k:k + 27], return_tensors="pt").to(dev)
        e = extract_features(clip.get_image_features(**b))
        embs.append(torch.nn.functional.normalize(e, dim=-1))
    E = torch.cat(embs)
    t = proc(text=[prompts[pi]], return_tensors="pt", padding=True,
             truncation=True, max_length=77).to(dev)
    T = torch.nn.functional.normalize(extract_features(clip.get_text_features(**t)), dim=-1)
    clip_sim = float((E @ T.T).squeeze(-1).mean())
    clip_tc = float((E[:-1] * E[1:]).sum(-1).mean())
    x = torch.from_numpy(arr).to(dev).permute(0, 3, 1, 2).float() / 127.5 - 1.0
    ls = [float(lpips_model(x[i:i + 1], x[i + 1:i + 2]).mean()) for i in range(0, len(x) - 1)]
    res.setdefault(arm, []).append({"prompt": pi, "clip_prompt_sim": clip_sim,
                                    "clip_temporal_cos": clip_tc,
                                    "lpips_consecutive": float(np.mean(ls))})
    print(f"  {arm} p{pi}: CLIP-sim {clip_sim:.4f}  CLIP-TC {clip_tc:.4f}  "
          f"LPIPS {np.mean(ls):.4f}", flush=True)
    del arr, x, E
    torch.cuda.empty_cache()

LBL = {"eager_a": "eager-original (seed A)", "final": "final (seed A)",
       "eager_b": "eager-original (seed B)"}
print(f"\n{'arm':<28}{'CLIP prompt-sim':>26}{'CLIP temporal':>26}{'LPIPS consecutive':>26}")
print(f"{'':<28}{'mean [min, max]':>26}{'mean [min, max]':>26}{'mean [min, max]':>26}")
summary = {}
for arm in ("eager_a", "final", "eager_b"):
    if arm not in res: continue
    r = res[arm]
    def mean_min_max(k):
        v = [x[k] for x in r]; return float(np.mean(v)), min(v), max(v)
    a1, a2, a3 = (mean_min_max("clip_prompt_sim"), mean_min_max("clip_temporal_cos"),
                  mean_min_max("lpips_consecutive"))
    summary[arm] = {"clip_prompt_sim": a1, "clip_temporal_cos": a2, "lpips_consecutive": a3,
                    "n": len(r)}
    print(f"{LBL[arm]:<28}{a1[0]:>10.4f} [{a1[1]:.4f}, {a1[2]:.4f}]"
          f"{a2[0]:>10.4f} [{a2[1]:.4f}, {a2[2]:.4f}]"
          f"{a3[0]:>10.4f} [{a3[1]:.4f}, {a3[2]:.4f}]")
json.dump({"per_video": res, "summary": summary, "clip_checkpoint": CLIP_ID,
           "lpips_backbone": "alex", "lpips_version": LPIPS_VER,
           "transformers_version": transformers.__version__},
          open(os.path.join(args.out, "j9_metrics.json"), "w"), indent=2)
print("\n[metrics] written")
