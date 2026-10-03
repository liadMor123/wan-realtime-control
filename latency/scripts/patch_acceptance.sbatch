#!/bin/bash
#SBATCH --job-name=j3b-accept
#SBATCH --partition=a100-public
#SBATCH --account=mendelson_prj
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/USER/proj/logs/j3baccept-%j.out
#SBATCH --error=/home/USER/proj/logs/j3baccept-%j.out
set -uo pipefail
PROJ=$HOME/proj; CACHE=$PROJ/cache; VENV=$PROJ/venv; RUN=$TMPDIR/run
module load python/3.11.15-x86-7sp4zgy
PY="$VENV/bin/python"
mkdir -p "$RUN/out"
cp -a "$PROJ/Self-Forcing" "$RUN/repo"
cp -a "$CACHE/wan_models" "$RUN/repo/wan_models"
mkdir -p "$RUN/repo/checkpoints"
cp -a "$CACHE/checkpoints/self_forcing_dmd.pt" "$RUN/repo/checkpoints/"
cp "$PROJ/scripts/patch_acceptance_bitwise.py" "$RUN/repo/"
cd "$RUN/repo"
echo "=== patch branch: $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD) ==="
git log --oneline 33593df3..HEAD
export J3B_ACCEPT_OUT="$RUN/out/j3b_accept.json"
echo; echo "=== acceptance: eager-patched vs eager-original, 5 prompts ==="
$PY patch_acceptance_bitwise.py; RC=$?
cp -f "$RUN/out/j3b_accept.json" "$PROJ/results/" 2>/dev/null || true
echo "=== acceptance exit: $RC ==="
exit $RC
