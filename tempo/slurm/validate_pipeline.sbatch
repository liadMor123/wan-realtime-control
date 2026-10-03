#!/bin/bash
#SBATCH --job-name=tempo_p0
#SBATCH --partition=a100-public
#SBATCH --account=mendelson_prj
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:15:00
#SBATCH --output=/home/USER/tempo/logs/phase0_%j.out
#SBATCH --error=/home/USER/tempo/logs/phase0_%j.out
# Phase 0: 6 videos + equivalence/correctness checks, then metric preflight
# and the official metric on the Phase-0 videos. The protocol is committed before sbatch.
set -uo pipefail
T=$HOME/tempo
module load python/3.11.15-x86-7sp4zgy
export HF_HOME=$T/cache/hf PYTHONUNBUFFERED=1
cd $T
echo "=== phase0 on $(hostname) job $SLURM_JOB_ID commit $(git rev-parse --short HEAD) $(date -Is) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
$T/venv/bin/python scripts/validate_pipeline_equivalence.py; RC=$?
echo "### validate_pipeline_equivalence.py rc=$RC"
$T/venv_metric/bin/python scripts/run_official_temporal_accuracy.py --preflight; echo "### preflight rc=$?"
for v in v1 v2 v3 v4; do $T/venv_metric/bin/python scripts/run_official_temporal_accuracy.py --videos videos/phase0/$v --ids 0; done
for v in v5 v6; do $T/venv_metric/bin/python scripts/run_official_temporal_accuracy.py --videos videos/phase0/$v --ids 1; done
echo "=== end $(date -Is) ==="
exit $RC
