#!/bin/bash
#SBATCH --job-name=tempo_vbenv
#SBATCH --partition=a100-public
#SBATCH --account=mendelson_prj
#SBATCH --qos=4h_0g
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/USER/tempo/logs/vbenv_%j.out
#SBATCH --error=/home/USER/tempo/logs/vbenv_%j.out
# Step 6 (brief §6): separate pinned VBench env ~/tempo/venv_vbench (CPU-only QoS: 0 GPU-hours).
# vbench==0.1.5 is installed with --no-deps: its full requirement list pulls deps of dimensions we do not run
# (GRiT/detectron: lvis, fairscale, fvcore, pycocoevalcap; transformers==4.33.2 for tag2text/UMT). We install only what
# vbench/__init__.py, vbench/utils.py, imaging_quality.py (pyiqa MUSIQ) and subject_consistency.py (DINO via torch.hub)
# import. pyiqa==0.1.10 is the last release without a transformers==4.37.2 pin (VBench declares transformers==4.33.2).
set -euo pipefail
T=$HOME/tempo
module load python/3.11.15-x86-7sp4zgy
export PIP_CACHE_DIR=${TMPDIR:-/tmp}/pipcache
echo "=== vbench env build on $(hostname) job ${SLURM_JOB_ID:-none} $(date -Is) ==="
[ -x $T/venv_vbench/bin/python ] || python3 -m venv $T/venv_vbench
P=$T/venv_vbench/bin/python
$P -m pip install -q --upgrade pip
$P -m pip install -q torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
cat > ${TMPDIR:-/tmp}/vb_constraints.txt <<'EOF'
torch==2.7.1
torchvision==0.22.1
numpy==1.26.4
EOF
$P -m pip install -q -c ${TMPDIR:-/tmp}/vb_constraints.txt "vbench==0.1.5" --no-deps
$P -m pip install -q -c ${TMPDIR:-/tmp}/vb_constraints.txt "numpy==1.26.4" "decord==0.6.0" "pyiqa==0.1.10" \
    "timm==0.9.16" "opencv-python==4.11.0.86" "scipy" "scikit-image" "easydict" "pyyaml" "Pillow" "requests" "tqdm" \
    "matplotlib"
$P - <<'EOF'
import importlib.metadata as md
import torch, torchvision, numpy, decord, timm, pyiqa
import vbench, vbench.imaging_quality, vbench.subject_consistency
from pyiqa.archs.musiq_arch import MUSIQ
print("venv_vbench:", {k: md.version(k) for k in ["vbench", "torch", "torchvision", "numpy", "decord", "pyiqa", "timm",
                                                  "opencv-python"]}, "cuda", torch.version.cuda)
EOF
$P -m pip freeze > $T/setup/venv_vbench_freeze.txt
echo "=== done $(date -Is) ==="
