#!/bin/bash
#SBATCH --job-name=tempo_s5a
#SBATCH --partition=a100-public
#SBATCH --account=mendelson_prj
#SBATCH --qos=12h_4g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#SBATCH --output=/home/USER/tempo/logs/step5a_%A_%a.out
#SBATCH --error=/home/USER/tempo/logs/step5a_%A_%a.out
# Part 2 step 5a (brief §5): Wan2.1 B0 and L(2,2) with K = 2 objects, seed 42, part 1's explicit path, on the
# 20 two-object pairs of data/step5a_ids.txt. 4 shards x 5 pairs x 2 runs = 10 videos each, into the new dirs
# videos/B0_2obj_s42 and videos/L_b2g2_2obj_s42. Each shard first T5-encodes any two-object prompt missing from
# cache/text_enc_two.pt (same call as cache/text_enc.pt, checked against it).
set -uo pipefail
T=$HOME/tempo
module load python/3.11.15-x86-7sp4zgy
export HF_HOME=$T/cache/hf PYTHONUNBUFFERED=1
cd $T
echo "=== step5a shard $SLURM_ARRAY_TASK_ID on $(hostname) job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} commit $(git rev-parse --short HEAD) $(date -Is) ==="
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1); echo "GPU $GPU"
[[ "$GPU" == *A100-SXM4-40GB* ]] || { echo "### wrong GPU $GPU"; exit 2; }
$T/venv/bin/python scripts/generate_benchmark_videos.py --runs runs/step5a.json --ids "$(cat data/step5a_ids.txt)" \
    --shard ${SLURM_ARRAY_TASK_ID}/4 --phase 5
RC=$?; echo "### rc=$RC $(date -Is)"; exit $RC
