#!/bin/bash
#SBATCH --job-name=j10
#SBATCH --partition=a100-public
#SBATCH --account=mendelson_prj
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/USER/proj/logs/j10-%j.out
#SBATCH --error=/home/USER/proj/logs/j10-%j.out
set -uo pipefail
PROJ=$HOME/proj; CACHE=$PROJ/cache; VENV=$PROJ/venv; RUN=$TMPDIR/run
module load python/3.11.15-x86-7sp4zgy
PY="$VENV/bin/python"
mkdir -p "$RUN/out"
cp -a "$PROJ/Self-Forcing" "$RUN/repo"
cp "$PROJ/src/harness/splitkv_position_rule.py" "$RUN/repo/"
cd "$RUN/repo"
echo "=== J10 on $(hostname) job $SLURM_JOB_ID; patch @ $(git rev-parse --short HEAD) ==="
export J10_OUT="$RUN/out/j10_rule.json"
$PY splitkv_position_rule.py 2>&1 | grep -vE "^W09|^E09"
cp -f "$RUN/out/j10_rule.json" "$PROJ/results/" 2>/dev/null || true
echo "=== J10 done ==="
