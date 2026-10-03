#!/bin/bash
#SBATCH --job-name=tempo_env
#SBATCH --partition=a100-public
#SBATCH --account=mendelson_prj
#SBATCH --qos=4h_0g
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=/home/USER/tempo/logs/env_%j.out
#SBATCH --error=/home/USER/tempo/logs/env_%j.out
# Build the two isolated environments inside an allocation (CPU-only QoS: no GPU-hours).
#   venv        : Wan2.1 generation (torch 2.7.1+cu126, prebuilt flash-attn 2.8.3), CLIP metric
#   venv_metric : TempoControl's temporal_accuracy.py (THU-MIG yolov10 @453c6e38)
set -euo pipefail
T=$HOME/tempo
module load python/3.11.15-x86-7sp4zgy
echo "=== build on $(hostname) job ${SLURM_JOB_ID:-none} $(date -Is) ==="

if [ ! -x $T/venv/bin/python ]; then python3 -m venv $T/venv; fi
P=$T/venv/bin/python
$P -m pip install -q --upgrade pip
$P -m pip install -q torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
$P -m pip install -q "numpy==1.26.4" "diffusers==0.31.0" "transformers==4.51.3" "tokenizers>=0.20.3" \
    "accelerate==1.6.0" tqdm imageio imageio-ffmpeg easydict ftfy "opencv-python-headless==4.11.0.86" \
    "safetensors" "einops" "sentencepiece" "protobuf" "pandas" "matplotlib" "pytest" "open_clip_torch==2.32.0"
$P $T/setup/install_flash_attn.py
$P - <<'EOF'
import torch, flash_attn, transformers, diffusers, numpy, open_clip
print("venv:", "torch", torch.__version__, "cuda", torch.version.cuda, "fa", flash_attn.__version__,
      "transformers", transformers.__version__, "diffusers", diffusers.__version__, "numpy", numpy.__version__,
      "open_clip", open_clip.__version__)
EOF
$P -m pip freeze > $T/setup/venv.freeze.txt

if [ ! -x $T/venv_metric/bin/python ]; then python3 -m venv $T/venv_metric; fi
M=$T/venv_metric/bin/python
$M -m pip install -q --upgrade pip
$M -m pip install -q torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
$M -m pip install -q "numpy<2" safetensors "huggingface_hub==0.23.4" pandas opencv-python tqdm huggingface_hub \
    "git+https://github.com/THU-MIG/yolov10.git@453c6e38"
$M -c "import ultralytics, torch, cv2, numpy; from ultralytics import YOLOv10; print('venv_metric: ultralytics', ultralytics.__version__, 'torch', torch.__version__, 'numpy', numpy.__version__)"
$M -m pip freeze > $T/setup/venv_metric.freeze.txt
echo "=== done $(date -Is) ==="
