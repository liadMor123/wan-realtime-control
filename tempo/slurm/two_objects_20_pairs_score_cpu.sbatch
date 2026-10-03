#!/bin/bash
#SBATCH --job-name=tempo_s5ac
#SBATCH --partition=a100-public
#SBATCH --account=mendelson_prj
#SBATCH --qos=4h_0g
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/home/USER/tempo/logs/step5a_score_%j.out
#SBATCH --error=/home/USER/tempo/logs/step5a_score_%j.out
# Part 2 step 5a scoring on CPU (0 GPU-h): two-object metric preflight, the official two-object metric (unmodified) on
# the 20 pairs of B0_2obj_s42 and L_b2g2_2obj_s42 into results/metric/phase5/<tag>, CLIP vs the full prompt, then
# summarize_two_objects_20_pairs.py.
set -uo pipefail
T=$HOME/tempo
module load python/3.11.15-x86-7sp4zgy
export HF_HOME=$T/cache/hf OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd $T
IDS=$($T/venv/bin/python -c "import sys;sys.path.insert(0,'scripts');from generate_benchmark_videos import parse_ids;print(','.join(map(str,parse_ids(open('data/step5a_ids.txt').read().strip()))))")
echo "=== step5a scoring on $(hostname) job $SLURM_JOB_ID commit $(git rev-parse --short HEAD) $(date -Is) ==="
echo "pairs $IDS"
$T/venv_metric/bin/python scripts/run_official_temporal_accuracy.py --preflight --bench two-object \
    > logs/step5a_preflight.log 2>&1 || { echo "### TWO-OBJECT PREFLIGHT FAILED (logs/step5a_preflight.log)"; exit 1; }
tail -1 logs/step5a_preflight.log
FAIL=0; PIDS=""
for t in B0_2obj_s42 L_b2g2_2obj_s42; do
  n=$(ls videos/$t/*.mp4 | wc -l); echo "videos/$t: $n mp4"; [ "$n" -eq 20 ] || FAIL=1
  $T/venv_metric/bin/python scripts/run_official_temporal_accuracy.py --bench two-object --videos videos/$t --ids $IDS \
      --out results/metric/phase5/$t > logs/step5a_metric_$t.log 2>&1 &
  PIDS="$PIDS $!"
done
for p in $PIDS; do wait $p || FAIL=1; done
[ $FAIL -eq 0 ] || { echo "### METRIC FAILED or videos missing"; exit 1; }
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
$T/venv/bin/python scripts/clip_similarity_and_contact_sheets.py --bench two_objects --tags B0_2obj_s42,L_b2g2_2obj_s42 --ids $IDS --phase 5 \
    || { echo "### QUALITY FAILED"; exit 1; }
$T/venv/bin/python scripts/summarize_two_objects_20_pairs.py
RC=$?; echo "### rc=$RC $(date -Is)"; exit $RC
