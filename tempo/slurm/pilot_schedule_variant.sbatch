#!/bin/bash
#SBATCH --job-name=tempo_p1k
#SBATCH --partition=a100-public
#SBATCH --account=mendelson_prj
#SBATCH --qos=12h_4g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --array=0-1
#SBATCH --output=/home/USER/tempo/logs/phase1k10_%A_%a.out
#SBATCH --error=/home/USER/tempo/logs/phase1k10_%A_%a.out
# Phase 1 pilot (pre-registered): schedule variant: best pilot arm L(2,2) edited only in steps < 10; 8 pilot prompts, 2 shards x 4.
set -uo pipefail
T=$HOME/tempo
module load python/3.11.15-x86-7sp4zgy
export HF_HOME=$T/cache/hf PYTHONUNBUFFERED=1
cd $T
echo "=== phase1 shard $SLURM_ARRAY_TASK_ID on $(hostname) job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} commit $(git rev-parse --short HEAD) $(date -Is) ==="
$T/venv/bin/python scripts/generate_benchmark_videos.py --runs runs/phase1_k10.json --ids "$(cat data/pilot_ids.txt)" \
    --shard ${SLURM_ARRAY_TASK_ID}/2 --phase 1
echo "### rc=$? $(date -Is)"
