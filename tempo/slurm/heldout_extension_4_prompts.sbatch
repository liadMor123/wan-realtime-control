#!/bin/bash
#SBATCH --job-name=tempo_p2x
#SBATCH --partition=a100-public
#SBATCH --account=mendelson_prj
#SBATCH --qos=12h_4g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:40:00
#SBATCH --array=0
#SBATCH --output=/home/USER/tempo/logs/phase2x_%A_%a.out
#SBATCH --error=/home/USER/tempo/logs/phase2x_%A_%a.out
# Phase 2 extension A: held-out ids 6,26,46,66 x the 5 Phase-2 runs = 20 videos (1 GPU).
set -uo pipefail
T=$HOME/tempo
module load python/3.11.15-x86-7sp4zgy
export HF_HOME=$T/cache/hf PYTHONUNBUFFERED=1
cd $T
echo "=== phase2 shard $SLURM_ARRAY_TASK_ID on $(hostname) job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} commit $(git rev-parse --short HEAD) $(date -Is) ==="
$T/venv/bin/python scripts/generate_benchmark_videos.py --runs runs/phase2.json --ids "$(cat data/heldout_ext4_ids.txt)" \
    --shard 0/1 --phase 2
echo "### rc=$? $(date -Is)"
