#!/bin/bash
#SBATCH --job-name=tempo_s3bc
#SBATCH --partition=a100-public
#SBATCH --account=mendelson_prj
#SBATCH --qos=4h_0g
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/home/USER/tempo/logs/step3b_score_%j.out
#SBATCH --error=/home/USER/tempo/logs/step3b_score_%j.out
# Part 2 step 3b scoring on CPU (0 GPU-h): official one-object metric (unmodified) on the 4 Self-Forcing benchmark tags
# (all 80 ids) and prompt switching (20 held-out), CLIP + per-tag contact sheets, then scripts/summarize_self_forcing_accuracy.py.
# The blinded substitute-object sheets are made separately (scripts/blinded_substitute_object_sheets.py) and judged before this job's
# outputs are opened.
set -uo pipefail
T=$HOME/tempo
module load python/3.11.15-x86-7sp4zgy
export HF_HOME=$T/cache/hf OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
cd $T
EXP() { $T/venv/bin/python -c "import sys;sys.path.insert(0,'scripts');from generate_benchmark_videos import parse_ids;print(','.join(map(str,parse_ids('$1'))))"; }
ALL=$(EXP 0-79); HO=$(EXP "$(cat data/heldout_ids.txt)")
echo "=== step3b scoring on $(hostname) job $SLURM_JOB_ID commit $(git rev-parse --short HEAD) $(date -Is) ==="
# keep step 3's 20-prompt metric outputs; after re-scoring all 80, the 20 held-out results must be unchanged
BK=results/metric/phase23_step3_backup; mkdir -p $BK
for t in SF_B0_fa2kv_s42 SF_L_b2g2_fa2kv_s42; do [ -d $BK/$t ] || cp -r results/metric/phase23/$t $BK/$t; done
FAIL=0; PIDS=""
for spec in SF_B0_fa2kv_s42:$ALL SF_L_b2g2_fa2kv_s42:$ALL SF_B0_fa2kv_s43:$ALL SF_L_b2g2_fa2kv_s43:$ALL SF_PS_s42:$HO; do
  t=${spec%%:*}; ids=${spec#*:}
  echo "videos/$t: $(ls videos/$t/*.mp4 2>/dev/null | wc -l) mp4"
  $T/venv_metric/bin/python scripts/run_official_temporal_accuracy.py --videos videos/$t --ids $ids \
      --out results/metric/phase23/$t > logs/step3b_metric_$t.log 2>&1 &
  PIDS="$PIDS $!"
done
for p in $PIDS; do wait $p || FAIL=1; done
for t in SF_B0_fa2kv_s42 SF_L_b2g2_fa2kv_s42 SF_B0_fa2kv_s43 SF_L_b2g2_fa2kv_s43 SF_PS_s42; do tail -1 logs/step3b_metric_$t.log | cut -c1-200; done
[ $FAIL -eq 0 ] || { echo "### METRIC FAILED"; exit 1; }
$T/venv/bin/python - <<'PYC' || { echo "### STEP-3 METRIC DID NOT REPRODUCE"; exit 1; }
import json, os
for t in ("SF_B0_fa2kv_s42", "SF_L_b2g2_fa2kv_s42"):
    old = {os.path.basename(v["video_path"]): v["video_results"] for v in
           json.load(open(f"results/metric/phase23_step3_backup/{t}/temporal_accuracy_one_object.json"))["temporal_accuracy"][1]}
    new = {os.path.basename(v["video_path"]): v["video_results"] for v in
           json.load(open(f"results/metric/phase23/{t}/temporal_accuracy_one_object.json"))["temporal_accuracy"][1]}
    assert len(old) == 20 and all(new[k] == old[k] for k in old), t
    print(t, "step-3 metric reproduced on the 20 held-out")
PYC
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
$T/venv/bin/python scripts/clip_similarity_and_contact_sheets.py --tags SF_B0_fa2kv_s42,SF_L_b2g2_fa2kv_s42,SF_B0_fa2kv_s43,SF_L_b2g2_fa2kv_s43 \
    --ids 0-79 --phase 23 || { echo "### QUALITY FAILED"; exit 1; }
$T/venv/bin/python scripts/clip_similarity_and_contact_sheets.py --tags SF_PS_s42 --ids "$(cat data/heldout_ids.txt)" --phase 23 || { echo "### QUALITY FAILED"; exit 1; }
$T/venv/bin/python scripts/summarize_self_forcing_accuracy.py
RC=$?; echo "### rc=$RC $(date -Is)"; exit $RC
