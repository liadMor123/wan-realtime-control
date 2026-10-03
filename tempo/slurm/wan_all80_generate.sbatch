#!/bin/bash
#SBATCH --job-name=tempo_s4
#SBATCH --partition=a100-public
#SBATCH --account=mendelson_prj
#SBATCH --qos=12h_4g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-3
#SBATCH --output=/home/USER/tempo/logs/step4_%A_%a.out
#SBATCH --error=/home/USER/tempo/logs/step4_%A_%a.out
# Part 2 step 4 (brief §4, pre-registered): Wan2.1 B0 and L(2,2), seed 42, part 1's explicit path, on the
# 52 single-object prompts not yet run (data/step4_ids.txt). 4 shards x 13 prompts x 2 runs = 26 videos each.
# Tags B0_s42 / L_b2g2_s42 = part 1's, so the 28 existing videos per tag complete all 80.
set -uo pipefail
T=$HOME/tempo
module load python/3.11.15-x86-7sp4zgy
export HF_HOME=$T/cache/hf PYTHONUNBUFFERED=1
cd $T
echo "=== step4 shard $SLURM_ARRAY_TASK_ID on $(hostname) job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} commit $(git rev-parse --short HEAD) $(date -Is) ==="
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1); echo "GPU $GPU"
[[ "$GPU" == *A100-SXM4-40GB* ]] || { echo "### wrong GPU $GPU"; exit 2; }
$T/venv/bin/python scripts/generate_benchmark_videos.py --runs runs/step4.json --ids "$(cat data/step4_ids.txt)" \
    --shard ${SLURM_ARRAY_TASK_ID}/4 --phase 4
RC=$?; echo "### rc=$RC $(date -Is)"; exit $RC
