#!/bin/bash
#SBATCH --job-name=j3-probe
#SBATCH --partition=a100-public
#SBATCH --account=mendelson_prj
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/home/USER/proj/logs/j3probe-%j.out
#SBATCH --error=/home/USER/proj/logs/j3probe-%j.out
set -euo pipefail
PROJ=$HOME/proj; CACHE=$PROJ/cache; VENV=$PROJ/venv; RUN=$TMPDIR/run
module load python/3.11.15-x86-7sp4zgy
PY="$VENV/bin/python"
mkdir -p "$RUN/out"
cp -a "$PROJ/Self-Forcing" "$RUN/repo"
cp -a "$CACHE/wan_models" "$RUN/repo/wan_models"
mkdir -p "$RUN/repo/checkpoints"
cp -a "$CACHE/checkpoints/self_forcing_dmd.pt" "$RUN/repo/checkpoints/"
cp "$PROJ/scripts/cuda_graph_capture_probe.py" "$RUN/repo/"
cd "$RUN/repo"
export J3_PROBE_OUT="$RUN/out/j3_probe.json"
echo "=== J3 capture probe on $(hostname) job $SLURM_JOB_ID ==="
$PY cuda_graph_capture_probe.py
cp -f "$RUN/out/j3_probe.json" "$PROJ/results/" 2>/dev/null || true
echo "=== probe done ==="
