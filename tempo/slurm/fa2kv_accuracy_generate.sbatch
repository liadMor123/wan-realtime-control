#!/bin/bash
#SBATCH --job-name=tempo_s2acc
#SBATCH --partition=a100-public
#SBATCH --account=mendelson_prj
#SBATCH --qos=12h_4g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=0-3
#SBATCH --output=/home/USER/tempo/logs/step2acc_%A_%a.out
#SBATCH --error=/home/USER/tempo/logs/step2acc_%A_%a.out
# Step 2's optional accuracy check (brief §2): Wan2.1-T2V-1.3B, seed 42, the 20 held-out prompts
# (data/heldout_ids.txt), two arms (runs/step2acc.json): B0 on Wan's unmodified flash path (tag B0_flash_s42) and
# L(2,2) on the production fa2kv path (tag L_b2g2_fa2kv_s42). Phase 24. 4 shards x 5 prompts x 2 arms = 10 videos
# each, into the new dirs videos/B0_flash_s42 and videos/L_b2g2_fa2kv_s42 (videos/step2/ holds step 2's timing videos of
# prompts 0-1 under the same tag names; not used here). Scoring: slurm/score_gpu.sbatch (afterany).
set -uo pipefail
T=$HOME/tempo
module load python/3.11.15-x86-7sp4zgy
export HF_HOME=$T/cache/hf PYTHONUNBUFFERED=1
cd $T
echo "=== step2acc shard $SLURM_ARRAY_TASK_ID on $(hostname) job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} commit $(git rev-parse --short HEAD) $(date -Is) ==="
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1); echo "GPU $GPU"
[[ "$GPU" == *A100-SXM4-40GB* ]] || { echo "### wrong GPU $GPU"; exit 2; }
$T/venv/bin/python -c "
import sys, torch, flash_attn
ok = torch.cuda.is_available()
print('torch', torch.__version__, 'torch.cuda.is_available()', ok, torch.cuda.get_device_name(0) if ok else '-', 'flash_attn', flash_attn.__version__)
sys.exit(0 if ok else 3)" || { echo "### CUDA NOT AVAILABLE (or flash_attn import failed)"; exit 3; }
$T/venv/bin/python scripts/generate_benchmark_videos.py --runs runs/step2acc.json --ids "$(cat data/heldout_ids.txt)" \
    --shard ${SLURM_ARRAY_TASK_ID}/4 --phase 24
RC=$?; echo "### rc=$RC $(date -Is)"; exit $RC
