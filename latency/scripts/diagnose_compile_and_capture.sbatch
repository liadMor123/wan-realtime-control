#!/bin/bash
#SBATCH --job-name=j3-diag
#SBATCH --partition=a100-public
#SBATCH --account=mendelson_prj
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/USER/proj/logs/j3diag-%j.out
#SBATCH --error=/home/USER/proj/logs/j3diag-%j.out
set -uo pipefail
PROJ=$HOME/proj; CACHE=$PROJ/cache; VENV=$PROJ/venv; RUN=$TMPDIR/run
module load python/3.11.15-x86-7sp4zgy
PY="$VENV/bin/python"
export TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor"
mkdir -p "$RUN/out"
cp -a "$PROJ/Self-Forcing" "$RUN/repo"
cp -a "$CACHE/wan_models" "$RUN/repo/wan_models"
mkdir -p "$RUN/repo/checkpoints"
cp -a "$CACHE/checkpoints/self_forcing_dmd.pt" "$RUN/repo/checkpoints/"
cp "$PROJ/scripts/diagnose_compile_and_capture.py" "$PROJ/scripts/compare_eager_compiled_frames.py" "$RUN/repo/"
cd "$RUN/repo"

echo "########## (a) numerics: eager vs compiled, single pass ##########"
$PY diagnose_compile_and_capture.py --part numerics --out "$RUN/out/j3_diag_numerics.json"

echo; echo "########## (b) capture blockers (CUDA_LAUNCH_BLOCKING=1) ##########"
CUDA_LAUNCH_BLOCKING=1 $PY diagnose_compile_and_capture.py --part capture --out "$RUN/out/j3_diag_capture.json"

echo; echo "########## (c) visual check: frame grids ##########"
$PY compare_eager_compiled_frames.py --out "$RUN/out"

cp -f "$RUN"/out/j3_diag_*.json "$PROJ/results/" 2>/dev/null || true
cp -f "$RUN"/out/*.png "$PROJ/figures/" 2>/dev/null || true
echo "=== diag done ==="
