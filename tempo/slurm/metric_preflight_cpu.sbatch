#!/bin/bash
#SBATCH --job-name=tempo_mpf
#SBATCH --partition=a100-public
#SBATCH --account=mendelson_prj
#SBATCH --qos=4h_0g
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=/home/USER/tempo/logs/mpf_%j.out
#SBATCH --error=/home/USER/tempo/logs/mpf_%j.out
# Metric preflight + Phase-0 videos on CPU (no GPU-hours): synthetic negative (expect 0.2) and positive (expect 1.0).
module load python/3.11.15-x86-7sp4zgy
cd $HOME/tempo && venv_metric/bin/python scripts/run_official_temporal_accuracy.py --preflight; echo "### preflight rc=$?"
for v in v1 v2 v3 v4; do venv_metric/bin/python scripts/run_official_temporal_accuracy.py --videos videos/phase0/$v --ids 0; done
for v in v5 v6; do venv_metric/bin/python scripts/run_official_temporal_accuracy.py --videos videos/phase0/$v --ids 1; done
