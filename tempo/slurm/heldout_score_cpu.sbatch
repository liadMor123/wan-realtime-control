#!/bin/bash
#SBATCH --job-name=tempo_fin
#SBATCH --partition=a100-public
#SBATCH --account=mendelson_prj
#SBATCH --qos=4h_0g
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:30:00
#SBATCH --output=/home/USER/tempo/logs/finalize_%j.out
#SBATCH --error=/home/USER/tempo/logs/finalize_%j.out
# Phase-2 scoring and analysis on CPU (0 GPU-h).
#   primary : 20 held-out ids, 5 seed-42/43 runs, ref B0_s42, noise floor B0_s43
#   subset  : the frozen 16-prompt subset, same runs
#   seed 43 : L arms at seed 43 vs B0 seed 43, 20 ids
set -uo pipefail
T=$HOME/tempo
module load python/3.11.15-x86-7sp4zgy
export HF_HOME=$T/cache/hf
cd $T
echo "=== finalize on $(hostname) job $SLURM_JOB_ID commit $(git rev-parse --short HEAD) $(date -Is) ==="
IDS20=$(cat data/heldout_ids.txt); IDS16=$(cat data/heldout16_ids.txt)
EXP20=$($T/venv/bin/python -c "import sys;sys.path.insert(0,'scripts');from generate_benchmark_videos import parse_ids;print(','.join(map(str,parse_ids('$IDS20'))))")
MAIN=B0_s42,B0_s43,L_b2g2_s42,L_b4g4_s42,L_b2g2_k10_s42
S43=B0_s43,L_b2g2_s43,L_b2g2_k10_s43
# score 4 run tags concurrently, 2 threads each (8 CPUs)
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
score() { $T/venv_metric/bin/python scripts/run_official_temporal_accuracy.py --videos videos/$1 --ids $EXP20 \
      --out results/metric/phase2/$1 > logs/finalize_metric_$1.log 2>&1 || echo "### METRIC FAILED $1"; echo "scored $1"; }
N=0
for t in B0_s42 B0_s43 L_b2g2_s42 L_b4g4_s42 L_b2g2_k10_s42 L_b2g2_s43 L_b2g2_k10_s43; do
  echo "videos/$t: $(ls videos/$t/*.mp4 2>/dev/null | wc -l) mp4 (pilot + held-out)"
  score $t & N=$((N+1)); [ $((N % 4)) -eq 0 ] && wait
done
wait
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
$T/venv/bin/python scripts/clip_similarity_and_contact_sheets.py --tags $MAIN,L_b2g2_s43,L_b2g2_k10_s43 --ids $IDS20 --phase 2 || echo "### QUALITY FAILED"
$T/venv/bin/python scripts/paired_accuracy_analysis.py --phase 2 --ids $IDS20 --ref B0_s42 --seedctl B0_s43 --tags $MAIN --name phase2_summary
$T/venv/bin/python scripts/paired_accuracy_analysis.py --phase 2 --ids $IDS16 --ref B0_s42 --seedctl B0_s43 --tags $MAIN --name phase2_summary_16
$T/venv/bin/python scripts/paired_accuracy_analysis.py --phase 2 --ids $IDS20 --ref B0_s43 --tags $S43 --name phase2_seed43_summary
$T/venv/bin/python scripts/plot_pilot_heldout_figure.py --summary results/phase2_summary.json --pilot results/phase1_summary.json \
    --out results/findings_figure.png
echo "### rc=$? $(date -Is)"
