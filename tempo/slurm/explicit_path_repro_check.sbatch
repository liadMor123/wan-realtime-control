#!/bin/bash
#SBATCH --job-name=tempo_s4r
#SBATCH --partition=a100-public
#SBATCH --account=mendelson_prj
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/home/USER/tempo/logs/step4_repro_%j.out
#SBATCH --error=/home/USER/tempo/logs/step4_repro_%j.out
# Step-4 preflight: current code reproduces part-1 B0_s42 / L_b2g2_s42 prompt 2 bitwise (2 videos).
set -uo pipefail
T=$HOME/tempo
module load python/3.11.15-x86-7sp4zgy
export HF_HOME=$T/cache/hf PYTHONUNBUFFERED=1
cd $T
echo "=== step4_repro on $(hostname) job $SLURM_JOB_ID commit $(git rev-parse --short HEAD) $(date -Is) ==="
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1); [[ "$GPU" == *A100-SXM4-40GB* ]] || { echo "### wrong GPU $GPU"; exit 2; }
$T/venv/bin/python scripts/explicit_path_repro_check.py 2>&1 | grep -v FutureWarning | grep -v "amp.autocast"
RC=${PIPESTATUS[0]}; echo "### rc=$RC $(date -Is)"; exit $RC
