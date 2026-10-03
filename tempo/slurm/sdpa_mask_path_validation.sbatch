#!/bin/bash
#SBATCH --job-name=tempo_s2
#SBATCH --partition=a100-public
#SBATCH --account=mendelson_prj
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:15:00
#SBATCH --output=/home/USER/tempo/logs/step2_%j.out
#SBATCH --error=/home/USER/tempo/logs/step2_%j.out
# Part 2 step 2 (pre-registered): fused-mask L in Wan2.1 -- preflight, fp64 probes, timing of 3 paths x 2 prompts.
set -uo pipefail
T=$HOME/tempo
module load python/3.11.15-x86-7sp4zgy
export HF_HOME=$T/cache/hf PYTHONUNBUFFERED=1
cd $T
echo "=== step2 on $(hostname) job $SLURM_JOB_ID commit $(git rev-parse --short HEAD) $(date -Is) ==="
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv,noheader
$T/venv/bin/python scripts/sdpa_mask_path_validation.py
RC=$?
echo "### rc=$RC $(date -Is)"
exit $RC
