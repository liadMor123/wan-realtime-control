#!/bin/bash
#SBATCH --job-name=tempo_s2b
#SBATCH --partition=a100-public
#SBATCH --account=mendelson_prj
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/home/USER/tempo/logs/step2b_%j.out
#SBATCH --error=/home/USER/tempo/logs/step2b_%j.out
# Part 2 step 2b (pre-registered decision 14): kernel bake-off on pilot prompt 1's probes; no videos.
set -uo pipefail
T=$HOME/tempo
module load python/3.11.15-x86-7sp4zgy
export HF_HOME=$T/cache/hf PYTHONUNBUFFERED=1 TORCHINDUCTOR_CACHE_DIR=$T/cache/inductor
cd $T
echo "=== step2b on $(hostname) job $SLURM_JOB_ID commit $(git rev-parse --short HEAD) $(date -Is) ==="
$T/venv/bin/python scripts/fused_kernel_bakeoff.py
RC=$?; echo "### rc=$RC $(date -Is)"; exit $RC
