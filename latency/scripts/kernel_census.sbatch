#!/bin/bash
#SBATCH --job-name=j54
#SBATCH --partition=a100-public
#SBATCH --account=mendelson_prj
#SBATCH --qos=12h_4g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/USER/proj/logs/j54-%j.out
#SBATCH --error=/home/USER/proj/logs/j54-%j.out
set -uo pipefail
PROJ=$HOME/proj; CACHE=$PROJ/cache; VENV=$PROJ/venv; RUN=$TMPDIR/run
module load python/3.11.15-x86-7sp4zgy
PY="$VENV/bin/python"
export TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor"; mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
mkdir -p "$RUN/out"
cp -a "$PROJ/Self-Forcing" "$RUN/repo"
cp -a "$CACHE/wan_models" "$RUN/repo/wan_models"
mkdir -p "$RUN/repo/checkpoints"
cp -a "$CACHE/checkpoints/self_forcing_dmd.pt" "$RUN/repo/checkpoints/"
cp "$PROJ/src/harness/kernel_census.py" "$RUN/repo/"
cd "$RUN/repo"
echo "=== J5+J4 on $(hostname) job $SLURM_JOB_ID; patch @ $(git rev-parse --short HEAD) ==="
git log --oneline -1
$PY kernel_census.py --out "$RUN/out" --positions 3,6 --try_compile_cg; RC=$?
cp -f "$RUN/out/j54.json" "$PROJ/results/" 2>/dev/null || true
echo "=== j54 exit: $RC ==="
