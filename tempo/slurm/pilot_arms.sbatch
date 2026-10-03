#!/bin/bash
#SBATCH --job-name=tempo_p1
#SBATCH --partition=a100-public
#SBATCH --account=mendelson_prj
#SBATCH --qos=12h_4g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:15:00
#SBATCH --array=0-3
#SBATCH --output=/home/USER/tempo/logs/phase1_%A_%a.out
#SBATCH --error=/home/USER/tempo/logs/phase1_%A_%a.out
# Phase 1 pilot (pre-registered): 8 pilot prompts x 7 arms, 4 shards x 2 prompts x 7 arms = 14 videos each.
set -uo pipefail
T=$HOME/tempo
module load python/3.11.15-x86-7sp4zgy
export HF_HOME=$T/cache/hf PYTHONUNBUFFERED=1
cd $T
echo "=== phase1 shard $SLURM_ARRAY_TASK_ID on $(hostname) job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} commit $(git rev-parse --short HEAD) $(date -Is) ==="
$T/venv/bin/python scripts/generate_benchmark_videos.py --runs runs/phase1.json --ids "$(cat data/pilot_ids.txt)" \
    --shard ${SLURM_ARRAY_TASK_ID}/4 --phase 1
echo "### rc=$? $(date -Is)"
