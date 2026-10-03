#!/bin/bash
#SBATCH --job-name=tempo_s2k
#SBATCH --partition=a100-public
#SBATCH --account=mendelson_prj
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:15:00
#SBATCH --output=/home/USER/tempo/logs/step2_fa2kv_%j.out
#SBATCH --error=/home/USER/tempo/logs/step2_fa2kv_%j.out
# Part 2 step 2 re-run with the step-2b choice (fa2kv): preflight, fp64 probes (+ noise and exact-rounding floors), timing.
set -uo pipefail
T=$HOME/tempo
module load python/3.11.15-x86-7sp4zgy
export HF_HOME=$T/cache/hf PYTHONUNBUFFERED=1
cd $T
echo "=== step2_fa2kv on $(hostname) job $SLURM_JOB_ID commit $(git rev-parse --short HEAD) $(date -Is) ==="
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv,noheader
$T/venv/bin/python scripts/fa2kv_path_validation.py
RC=$?
echo "### rc=$RC $(date -Is)"
exit $RC
