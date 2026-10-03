#!/bin/bash
#SBATCH --job-name=j1-selfforcing
#SBATCH --partition=a100-public
#SBATCH --account=mendelson_prj
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
# NOTE: %h is NOT a Slurm home-dir pattern (it expands literally). Use an
# absolute path; Slurm does not expand $HOME in #SBATCH lines either.
#SBATCH --output=/home/USER/proj/logs/j1-%j.out
#SBATCH --error=/home/USER/proj/logs/j1-%j.out
set -euo pipefail

PROJ=$HOME/proj
CACHE=$PROJ/cache
VENV=$PROJ/venv
RUN=$TMPDIR/run

echo "=== J1 on $(hostname), job $SLURM_JOB_ID, $(date -Is) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version,mig.mode.current --format=csv,noheader
echo "TMPDIR=$TMPDIR"; df -h "$TMPDIR" | tail -1

module load python/3.11.15-x86-7sp4zgy
echo "python: $(python3 --version) at $(command -v python3)"

# ---------- 1. environment (built inside the allocation, never on the login node) ----------
# torch is PINNED. The unpinned install resolved to 2.14.0+cu130, for which no
# prebuilt flash-attn wheel exists; the model's cross-attention hard-asserts on
# FLASH_ATTN_2_AVAILABLE (wan/modules/attention.py:118) and has no fallback.
TORCH_VER=2.7.1
TV_VER=0.22.1
TORCH_IDX=https://download.pytorch.org/whl/cu126

NEED=0
[ -x "$VENV/bin/python" ] || NEED=1
[ "${J1_FORCE_REBUILD:-0}" = "1" ] && NEED=1
if [ "$NEED" = "1" ]; then
  echo "=== building pinned venv (torch $TORCH_VER cu126) ==="
  rm -rf "$VENV"
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip wheel setuptools
  "$VENV/bin/pip" install -q --index-url "$TORCH_IDX" "torch==$TORCH_VER" "torchvision==$TV_VER"
  "$VENV/bin/pip" install -q \
      omegaconf tqdm einops imageio imageio-ffmpeg "av==13.1.0" easydict ftfy \
      sentencepiece transformers tokenizers accelerate "diffusers==0.31.0" \
      "numpy<2" opencv-python-headless scikit-image matplotlib huggingface_hub \
      pydantic Pillow lmdb regex
else
  echo "=== venv already present, reusing ==="
fi
PY="$VENV/bin/python"
echo "torch: $($PY -c 'import torch;print(torch.__version__, torch.version.cuda)')"

# flash-attn is REQUIRED by the cross-attention path; no silent fallback.
if ! $PY -c "import flash_attn" >/dev/null 2>&1; then
  echo "=== installing prebuilt flash-attn wheel ==="
  $PY "$PROJ/scripts/install_flash_attn.py"
fi
$PY -c "import flash_attn, torch; print('flash_attn', flash_attn.__version__, '| torch', torch.__version__)" \
  || { echo "J1 FAILURE: flash_attn unavailable; cross-attention would assert."; exit 3; }

# ---------- 2. weights (cold cache in $HOME, staged to scratch) ----------
mkdir -p "$CACHE"
if [ ! -f "$CACHE/wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth" ]; then
  echo "=== downloading Wan2.1-T2V-1.3B ==="
  $PY - <<'PYD'
from huggingface_hub import snapshot_download
import os
p = snapshot_download("Wan-AI/Wan2.1-T2V-1.3B",
                      local_dir=os.path.expanduser("~/proj/cache/wan_models/Wan2.1-T2V-1.3B"),
                      max_workers=8)
print("downloaded to", p)
PYD
fi
if [ ! -f "$CACHE/checkpoints/self_forcing_dmd.pt" ]; then
  echo "=== downloading self_forcing_dmd.pt ==="
  $PY - <<'PYD'
from huggingface_hub import hf_hub_download
import os
p = hf_hub_download("gdhe17/Self-Forcing", "checkpoints/self_forcing_dmd.pt",
                    local_dir=os.path.expanduser("~/proj/cache"))
print("downloaded to", p)
PYD
fi

echo "=== cached artifact sizes ==="
du -sh "$CACHE/wan_models/Wan2.1-T2V-1.3B" "$CACHE/checkpoints/self_forcing_dmd.pt"
ls -l "$CACHE/wan_models/Wan2.1-T2V-1.3B" | awk '{print "  ", $5, $9}'

SHAFILE="$CACHE/checkpoints/self_forcing_dmd.pt.sha256"
if [ ! -f "$SHAFILE" ]; then
  echo "=== hashing checkpoint (once) ==="
  sha256sum "$CACHE/checkpoints/self_forcing_dmd.pt" | cut -d' ' -f1 > "$SHAFILE"
fi
export J1_CKPT_SHA=$(cat "$SHAFILE")
export J1_REPO_REV=$(git -C "$PROJ/Self-Forcing" rev-parse HEAD)
export J1_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
echo "ckpt sha256: $J1_CKPT_SHA"
echo "repo rev:    $J1_REPO_REV"

# ---------- 3. stage working copies onto node-local scratch ----------
echo "=== staging to $RUN ==="
mkdir -p "$RUN"
cp -a "$PROJ/Self-Forcing" "$RUN/repo"
cp -a "$CACHE/wan_models" "$RUN/repo/wan_models"
mkdir -p "$RUN/repo/checkpoints"
cp -a "$CACHE/checkpoints/self_forcing_dmd.pt" "$RUN/repo/checkpoints/"
cp "$PROJ/src/harness/environment_memory_probe.py" "$RUN/repo/"
cd "$RUN/repo"
du -sh "$RUN/repo" | sed 's/^/  staged: /'

mkdir -p "$PROJ/results" "$PROJ/videos" "$RUN/out"
JSONL="$PROJ/results/runs.jsonl"

# ---------- 4. the two J1 runs ----------
RC_OFF=0; RC_ON=0
set +e
echo; echo "############ J1 run A: offload OFF (text encoder resident) ############"
$PY environment_memory_probe.py --offload off --output_folder "$RUN/out" \
   --run_id "j1-offloadoff-$SLURM_JOB_ID" --jsonl "$JSONL" \
   --memjson "$RUN/out/mem-offloadoff.json" 2>&1
RC_OFF=$?
echo "### run A exit: $RC_OFF ###"

echo; echo "############ J1 run B: offload ON (repo low_memory path) ############"
$PY environment_memory_probe.py --offload on --output_folder "$RUN/out" \
   --run_id "j1-offloadon-$SLURM_JOB_ID" --jsonl "$JSONL" \
   --memjson "$RUN/out/mem-offloadon.json" 2>&1
RC_ON=$?
echo "### run B exit: $RC_ON ###"
set -e

# ---------- 5. reduced results home (no traces, ever) ----------
cp -f "$RUN"/out/*.json "$PROJ/results/" 2>/dev/null || true
for f in "$RUN"/out/*.mp4; do [ -e "$f" ] && cp -f "$f" "$PROJ/videos/" && break; done
ls -l "$PROJ/videos" | tail -3

echo; echo "=== J1 SUMMARY: offload_off_rc=$RC_OFF offload_on_rc=$RC_ON ==="
if [ "$RC_ON" -ne 0 ]; then
  echo "J1 FAILED: the decisive configuration (text-encoder offload ON) did not pass."
  exit "$RC_ON"
fi
echo "J1 PASSED on the decisive configuration (offload ON)."
