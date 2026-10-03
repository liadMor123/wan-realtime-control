#!/bin/bash
#SBATCH --job-name=tempo_s6
#SBATCH --partition=a100-public
#SBATCH --account=mendelson_prj
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/USER/tempo/logs/step6_%j.out
#SBATCH --error=/home/USER/tempo/logs/step6_%j.out
# Part 2 step 6 (brief §6, pre-registered question Q6): VBench 0.1.5 imaging_quality + subject_consistency
# (custom_input mode) on the step-4 videos: B0_s42 and L_b2g2_s42 on all 80 prompts, B0_s43 on the 20 held-out;
# then the paired / seed-spread summary. No generation. Env ~/tempo/venv_vbench, weights pre-downloaded to
# ~/tempo/cache/vbench (login node); runs fully offline. Resumable per tag (results/vbench/*.json.parts/).
set -uo pipefail
T=$HOME/tempo
module load python/3.11.15-x86-7sp4zgy
export VBENCH_CACHE_DIR=$T/cache/vbench TORCH_HOME=$T/cache/vbench/torch HF_HOME=$T/cache/vbench/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
export http_proxy=http://127.0.0.1:9 https_proxy=http://127.0.0.1:9 HTTP_PROXY=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9
cd $T
echo "=== step6 vbench on $(hostname) job ${SLURM_JOB_ID} commit $(git rev-parse --short HEAD) $(date -Is) ==="
nvidia-smi --query-gpu=name --format=csv,noheader | grep -q "A100-SXM4-40GB" \
    || { echo "### wrong GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"; exit 2; }
P=$T/venv_vbench/bin/python
$P scripts/run_vbench.py --tags B0_s42,L_b2g2_s42 --ids @data/all80_ids.txt --out results/vbench/step6_all80.json \
    || { echo "### all80 rc=$? $(date -Is)"; exit 3; }
$P scripts/run_vbench.py --tags B0_s43 --ids @data/heldout_ids.txt --out results/vbench/step6_seed43.json \
    || { echo "### seed43 rc=$? $(date -Is)"; exit 4; }
$P scripts/summarize_vbench_quality.py
RC=$?; echo "### rc=$RC $(date -Is)"; exit $RC
