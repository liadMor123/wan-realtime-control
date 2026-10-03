#!/bin/bash
#SBATCH --job-name=tempo_s5b
#SBATCH --partition=a100-public
#SBATCH --account=mendelson_prj
#SBATCH --qos=12h_4g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:40:00
#SBATCH --array=0-3
#SBATCH --output=/home/USER/tempo/logs/step5b_%A_%a.out
#SBATCH --error=/home/USER/tempo/logs/step5b_%A_%a.out
# Part 2 step 5b (brief §5): the remaining 62 two-object pairs (data/step5b_ids.txt = all 82 minus the 20 of
# data/step5a_ids.txt), exactly step 5a's protocol: runs/step5a.json (B0 and L(2,2) with K = 2, seed 42, part 1's
# explicit path), tags B0_2obj_s42 / L_b2g2_2obj_s42 (5a's dirs; 5b extends them to all 82), phase-5 rows.
# 4 shards over the 62 pairs (16/16/15/15 pairs x 2 arms). Each shard T5-encodes its id list into its own cache file,
# cache/text_enc_two_5b_<task>.pt (TEMPO_TEXT_CACHE_TAG; 5a's text_enc_two_<task>.pt files are not touched), after
# checking the T5 call against cache/text_enc.pt, as in 5a. Scoring: slurm/score_gpu.sbatch (afterany).
set -uo pipefail
T=$HOME/tempo
module load python/3.11.15-x86-7sp4zgy
export HF_HOME=$T/cache/hf PYTHONUNBUFFERED=1 TEMPO_TEXT_CACHE_TAG=5b_${SLURM_ARRAY_TASK_ID}
cd $T
echo "=== step5b shard $SLURM_ARRAY_TASK_ID on $(hostname) job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} commit $(git rev-parse --short HEAD) $(date -Is) ==="
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1); echo "GPU $GPU"
[[ "$GPU" == *A100-SXM4-40GB* ]] || { echo "### wrong GPU $GPU"; exit 2; }
$T/venv/bin/python -c "
import sys, torch
ok = torch.cuda.is_available()
print('torch', torch.__version__, 'torch.cuda.is_available()', ok, torch.cuda.get_device_name(0) if ok else '-')
sys.exit(0 if ok else 3)" || { echo "### CUDA NOT AVAILABLE"; exit 3; }
$T/venv/bin/python scripts/generate_benchmark_videos.py --runs runs/step5a.json --ids "$(cat data/step5b_ids.txt)" \
    --shard ${SLURM_ARRAY_TASK_ID}/4 --phase 5
RC=$?; echo "### rc=$RC $(date -Is)"; exit $RC
