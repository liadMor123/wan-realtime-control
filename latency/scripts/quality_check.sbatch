#!/bin/bash
#SBATCH --job-name=j9
#SBATCH --partition=a100-public
#SBATCH --account=mendelson_prj
#SBATCH --qos=12h_4g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=/home/USER/proj/logs/j9-%j.out
#SBATCH --error=/home/USER/proj/logs/j9-%j.out
set -uo pipefail
PROJ=$HOME/proj; CACHE=$PROJ/cache; VENV=$PROJ/venv; RUN=$TMPDIR/run
module load python/3.11.15-x86-7sp4zgy
PY="$VENV/bin/python"
export TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor"; mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
export HF_HOME="$CACHE/hf"; mkdir -p "$HF_HOME"
VIDS="$CACHE/j9_vids"; mkdir -p "$VIDS"
mkdir -p "$RUN/out" "$RUN/sheets"
cp -a "$PROJ/Self-Forcing" "$RUN/repo"; cp -a "$CACHE/wan_models" "$RUN/repo/wan_models"
mkdir -p "$RUN/repo/checkpoints"; cp -a "$CACHE/checkpoints/self_forcing_dmd.pt" "$RUN/repo/checkpoints/"
cp "$PROJ/src/harness/quality_generate_videos.py" "$PROJ/src/harness/quality_metrics.py" "$RUN/repo/"
cd "$RUN/repo"
echo "=== J9 on $(hostname) job $SLURM_JOB_ID; patch @ $(git rev-parse --short HEAD) ==="

# environment is built inside the allocation, never on the login node
echo "--- installing pinned metric models ---"
$PY -m pip install -q lpips 2>&1 | tail -2
# preflight: fail fast BEFORE generating 30 videos if the metric stack is broken
$PY - <<'PRE' || { echo "J9 PREFLIGHT FAILED"; exit 9; }
import torch, lpips, transformers
from importlib.metadata import version
from transformers import CLIPModel, CLIPProcessor
import os
cid="openai/clip-vit-base-patch32"
m=CLIPModel.from_pretrained(cid); p=CLIPProcessor.from_pretrained(cid)
l=lpips.LPIPS(net="alex")
x=torch.rand(1,3,64,64)*2-1
_=l(x,x)
# exercise the ACTUAL feature APIs the metrics use -- a model that loads is not
# a model whose outputs are tensors (transformers 5.x returns output objects)
from PIL import Image
import numpy as np
im=[Image.fromarray(np.zeros((64,64,3),dtype=np.uint8))]
ib=p(images=im, return_tensors="pt"); tb=p(text=["x"], return_tensors="pt", padding=True)
def features_of(o):
    if torch.is_tensor(o): return o
    for a in ("image_embeds","text_embeds","pooler_output"):
        v=getattr(o,a,None)
        if torch.is_tensor(v): return v
    raise TypeError(type(o).__name__)
assert features_of(m.get_image_features(**ib)).dim()==2, "image features not 2-D"
assert features_of(m.get_text_features(**tb)).dim()==2, "text features not 2-D"
print(f"PREFLIGHT OK: CLIP {cid} | lpips {version('lpips')} | transformers {transformers.__version__}")
PRE

# skip generation for any arm already cached (a metrics-only rerun is then free)
have () { [ "$(ls "$VIDS"/$1_p*.npy 2>/dev/null | wc -l)" -ge 10 ]; }
echo; echo "########## arm A: eager-original, seed 1000+i ##########"
have eager_a && echo "  [cached] eager_a, skipping generation" || $PY quality_generate_videos.py --arm eager_a --out "$VIDS" 2>&1 | grep -E "^\[j9"
echo; echo "########## arm C: eager-original, seed 2000+i (seed control) ##########"
have eager_b && echo "  [cached] eager_b, skipping generation" || $PY quality_generate_videos.py --arm eager_b --out "$VIDS" 2>&1 | grep -E "^\[j9"
echo; echo "########## arm B: final, seed 1000+i (alone) ##########"
have final && echo "  [cached] final, skipping generation" || $PY quality_generate_videos.py --arm final --out "$VIDS" 2>&1 | grep -E "^\[j9"

echo; echo "########## metrics ##########"
$PY quality_metrics.py --vid_dir "$VIDS" --out "$RUN/out" --sheets "$RUN/sheets" 2>&1 \
  | grep -vE "^W09|^E09|Downloading|%\|"

mkdir -p "$PROJ/figures/j9"
cp -f "$RUN/out"/j9_*.json "$PROJ/results/" 2>/dev/null || true
cp -f "$VIDS/j9_prompts.json" "$PROJ/results/" 2>/dev/null || true
cp -f "$RUN/sheets"/*.png "$PROJ/figures/j9/" 2>/dev/null || true
du -sh "$PROJ/figures/j9"
echo; echo "=== J9 done ==="
