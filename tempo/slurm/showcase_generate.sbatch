#!/bin/bash
#SBATCH --job-name=tempo_show
#SBATCH --partition=a100-public
#SBATCH --account=mendelson_prj
#SBATCH --qos=12h_4g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=0-1
#SBATCH --output=/home/USER/tempo/logs/showcase_%A_%a.out
#SBATCH --error=/home/USER/tempo/logs/showcase_%A_%a.out
# Showcase (phase 25): TempoControl's project-page prompts, not in the benchmark (data/showcase_prompts.csv), for
# side-by-side B0 vs L(2,2) videos. Wan2.1-T2V-1.3B, seed 42, part 1's explicit path, exactly the step-4 (one object)
# and step-5a (two objects, K = 2) protocols: runs/showcase.json = runs/step4.json and runs/showcase_2obj.json =
# runs/step5a.json, each run + "set": "showcase" (tag prefix only). 12 videos into new dirs:
#   showcase id 0 (one object, dog, last-second mask)  -> videos/showcase_B0_s42, videos/showcase_L_b2g2_s42
#   showcase ids 1-5 (two objects, second-half masks)   -> videos/showcase_B0_2obj_s42, videos/showcase_L_b2g2_2obj_s42
# 2 shards x 6 videos: task 0 = pairs 1,3,5; task 1 = pairs 2,4 + the one-object prompt 0. Each generate_benchmark_videos.py call
# T5-encodes its prompts into cache/text_enc_showcase_<task>.pt (generate_benchmark_videos.load_or_build_text_cache: the call that wrote
# cache/text_enc.pt, checked against it). Rows: results/rows/phase25_*.jsonl. Scoring: slurm/showcase_score.sbatch.
set -uo pipefail
T=$HOME/tempo
module load python/3.11.15-x86-7sp4zgy
export HF_HOME=$T/cache/hf PYTHONUNBUFFERED=1
cd $T
echo "=== showcase shard $SLURM_ARRAY_TASK_ID on $(hostname) job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} commit $(git rev-parse --short HEAD) $(date -Is) ==="
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1); echo "GPU $GPU"
[[ "$GPU" == *A100-SXM4-40GB* ]] || { echo "### wrong GPU $GPU"; exit 2; }
$T/venv/bin/python -c "
import sys, torch
ok = torch.cuda.is_available()
print('torch', torch.__version__, 'torch.cuda.is_available()', ok, torch.cuda.get_device_name(0) if ok else '-')
sys.exit(0 if ok else 3)" || { echo "### CUDA NOT AVAILABLE"; exit 3; }
RC=0
$T/venv/bin/python scripts/generate_benchmark_videos.py --prompt-set showcase --runs runs/showcase_2obj.json --ids 1-5 \
    --shard ${SLURM_ARRAY_TASK_ID}/2 --phase 25 || RC=1
if [ "$SLURM_ARRAY_TASK_ID" = 1 ]; then
  $T/venv/bin/python scripts/generate_benchmark_videos.py --prompt-set showcase --runs runs/showcase.json --ids 0 \
      --shard 0/1 --phase 25 || RC=1
fi
echo "### rc=$RC $(date -Is)"; exit $RC
