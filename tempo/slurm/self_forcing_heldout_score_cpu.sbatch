#!/bin/bash
#SBATCH --job-name=tempo_s3c
#SBATCH --partition=a100-public
#SBATCH --account=mendelson_prj
#SBATCH --qos=4h_0g
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home/USER/tempo/logs/step3_score_%j.out
#SBATCH --error=/home/USER/tempo/logs/step3_score_%j.out
# Part 2 step 3 scoring on CPU (0 GPU-h): official one-object metric (unmodified) on both Self-Forcing arms,
# CLIP + contact sheets, paired analysis vs SF B0 on the 20 held-out prompts.
set -uo pipefail
T=$HOME/tempo
module load python/3.11.15-x86-7sp4zgy
export HF_HOME=$T/cache/hf OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd $T
IDS=$(cat data/heldout_ids.txt)
EXP=$($T/venv/bin/python -c "import sys;sys.path.insert(0,'scripts');from generate_benchmark_videos import parse_ids;print(','.join(map(str,parse_ids('$IDS'))))")
TAGS=SF_B0_fa2kv_s42,SF_L_b2g2_fa2kv_s42
echo "=== step3 scoring on $(hostname) job $SLURM_JOB_ID commit $(git rev-parse --short HEAD) $(date -Is) ==="
FAIL=0; PIDS=""
for t in ${TAGS//,/ }; do
  echo "videos/$t: $(ls videos/$t/*.mp4 2>/dev/null | wc -l) mp4"
  $T/venv_metric/bin/python scripts/run_official_temporal_accuracy.py --videos videos/$t --ids $EXP \
      --out results/metric/phase23/$t > logs/step3_metric_$t.log 2>&1 &
  PIDS="$PIDS $!"
done
for p in $PIDS; do wait $p || FAIL=1; done
for t in ${TAGS//,/ }; do tail -1 logs/step3_metric_$t.log; done
[ $FAIL -eq 0 ] || { echo "### METRIC FAILED"; exit 1; }
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
$T/venv/bin/python scripts/clip_similarity_and_contact_sheets.py --tags $TAGS --ids $IDS --phase 23 || echo "### QUALITY FAILED"
$T/venv/bin/python scripts/paired_accuracy_analysis.py --phase 23 --ids $IDS --ref SF_B0_fa2kv_s42 --tags $TAGS --name step3_sf_summary
RC=$?; echo "### rc=$RC $(date -Is)"; exit $RC
