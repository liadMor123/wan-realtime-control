#!/bin/bash
#SBATCH --job-name=tempo_s3cc
#SBATCH --partition=a100-public
#SBATCH --account=mendelson_prj
#SBATCH --qos=4h_0g
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/home/USER/tempo/logs/step3c_score_%j.out
#SBATCH --error=/home/USER/tempo/logs/step3c_score_%j.out
# Part 2 step 3c scoring on CPU (0 GPU-h): official one-object metric (unmodified) and CLIP + contact sheets for the
# given 3c tags, all 80 prompts. The summary (scripts/summarize_prompt_switch_arms.py) runs only after the blinded substitute check
# is judged and unblinded (3c-5); this job's outputs are not opened before that.
# Usage: sbatch --export=ALL,TAGS=SF_PSRF_fa2kv_s42[:TAG2:...] slurm/prompt_switch_arms_score_cpu.sbatch
# (colon-separated: sbatch --export splits on commas and would silently drop every tag after the first)
set -uo pipefail
TAGS=${TAGS//:/,}
T=$HOME/tempo
module load python/3.11.15-x86-7sp4zgy
export HF_HOME=$T/cache/hf OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
cd $T
ALL=$($T/venv/bin/python -c "import sys;sys.path.insert(0,'scripts');from generate_benchmark_videos import parse_ids;print(','.join(map(str,parse_ids('0-79'))))")
echo "=== step3c scoring TAGS=$TAGS on $(hostname) job $SLURM_JOB_ID commit $(git rev-parse --short HEAD) $(date -Is) ==="
FAIL=0; PIDS=""
for t in ${TAGS//,/ }; do
  n=$(ls videos/$t/*.mp4 2>/dev/null | wc -l); echo "videos/$t: $n mp4"
  [ "$n" -eq 80 ] || { echo "### $t has $n videos, expected 80"; exit 1; }
  $T/venv_metric/bin/python scripts/run_official_temporal_accuracy.py --videos videos/$t --ids $ALL \
      --out results/metric/phase23/$t > logs/step3c_metric_$t.log 2>&1 &
  PIDS="$PIDS $!"
done
for p in $PIDS; do wait $p || FAIL=1; done
[ $FAIL -eq 0 ] || { echo "### METRIC FAILED"; exit 1; }
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
$T/venv/bin/python scripts/clip_similarity_and_contact_sheets.py --tags "$TAGS" --ids 0-79 --phase 23 > logs/step3c_quality_${SLURM_JOB_ID}.log 2>&1 \
    || { echo "### QUALITY FAILED"; exit 1; }
echo "### rc=0 $(date -Is)"
