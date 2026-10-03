#!/usr/bin/env python3
"""
Per-frame PSNR between two uint8 videos saved as .npy (frames, H, W, 3).

Usage: frame_psnr.py A.npy B.npy out.json
Used for the J2 nondeterminism envelope (two identical runs in separate
processes) and for the J3 mode-vs-eager comparisons. Frames that are bitwise
identical are counted separately (PSNR would be infinite).

Writes out.json with frame count, identical-frame count, min/median/max PSNR
over the non-identical frames and the per-frame list.
"""
import json
import sys

import numpy as np

a = np.load(sys.argv[1]).astype(np.float64)
b = np.load(sys.argv[2]).astype(np.float64)
if a.shape != b.shape:
    sys.exit(f"shape mismatch {a.shape} vs {b.shape}")

psnr, identical = [], 0
for i in range(a.shape[0]):
    mse = np.mean((a[i] - b[i]) ** 2)
    if mse == 0:
        identical += 1
        psnr.append(float("inf"))
    else:
        psnr.append(20.0 * np.log10(255.0 / np.sqrt(mse)))

finite = [p for p in psnr if np.isfinite(p)]
out = {
    "frames": int(a.shape[0]),
    "identical_frames": identical,
    "bitwise_identical": identical == a.shape[0],
    "psnr_min": min(finite) if finite else None,
    "psnr_median": float(np.median(finite)) if finite else None,
    "psnr_max": max(finite) if finite else None,
    "per_frame": [None if not np.isfinite(p) else round(p, 3) for p in psnr],
}
print(json.dumps({k: v for k, v in out.items() if k != "per_frame"}, indent=2))
with open(sys.argv[3], "w") as f:
    json.dump(out, f, indent=2)
