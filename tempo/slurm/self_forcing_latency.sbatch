#!/bin/bash
#SBATCH --job-name=tempo_s3l
#SBATCH --partition=a100-public
#SBATCH --account=mendelson_prj
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/USER/tempo/logs/step3_lat_%j.out
#SBATCH --error=/home/USER/tempo/logs/step3_lat_%j.out
# Part 2 step 3 latency (pre-registered decision 9): ~/proj per-position protocol, unmodified Self-Forcing (a5deb2f +
# verbatim j3b_run.py) vs L (tempo-L + self_forcing_latency_harness.py --tempo L). Every mode alone in its own process;
# `final` in ABBA order, then the eager pair.
# Usage: sbatch --export=ALL,SFREV=<tempo-L commit> slurm/self_forcing_latency.sbatch
set -uo pipefail
T=$HOME/tempo; ORIG=a5deb2f61007e41a3ca306661e1c04637fd3b52c
module load python/3.11.15-x86-7sp4zgy
export HF_HOME=$T/cache/hf PYTHONUNBUFFERED=1 TORCHINDUCTOR_CACHE_DIR=$T/cache/inductor
PY=$T/venv/bin/python
RUN=$TMPDIR/sflat; OUT=$RUN/out; mkdir -p $OUT
for R in orig L; do
  mkdir -p $RUN/$R/checkpoints
  REV=$([ $R = orig ] && echo $ORIG || echo $SFREV)
  git -C $T/ext/Self-Forcing archive "$REV" | tar -x -C $RUN/$R
  ln -s $HOME/proj/cache/wan_models $RUN/$R/wan_models                    # read-only use of ~/proj
  ln -s $HOME/proj/cache/checkpoints/self_forcing_dmd.pt $RUN/$R/checkpoints/self_forcing_dmd.pt
done
cp $T/scripts/self_forcing_latency_harness_original.py $RUN/orig/j3b_run.py
cp $T/scripts/self_forcing_latency_harness.py $RUN/L/self_forcing_latency_harness.py
echo "=== step3_lat on $(hostname) job $SLURM_JOB_ID tempo $(git -C $T rev-parse --short HEAD) SF $SFREV orig $ORIG $(date -Is) ==="
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1); echo "GPU $GPU"
[[ "$GPU" == *A100-SXM4-40GB* ]] || { echo "### wrong GPU $GPU"; exit 2; }
M1=$(md5sum < $RUN/orig/j3b_run.py); M2=$(md5sum < $HOME/proj/src/harness/j3b_run.py)
[ "$M1" = "$M2" ] || { echo "### harness copy differs from ~/proj/src/harness/j3b_run.py"; exit 2; }
grep -q "tempo_bias" $RUN/orig/wan/modules/model.py && { echo "### orig tree contains the tempo hook"; exit 2; }
grep -q "tempo_bias" $RUN/L/wan/modules/model.py || { echo "### L tree lacks the tempo hook"; exit 2; }
DST=$T/results/step3/lat_$SLURM_JOB_ID; mkdir -p $DST
CLK=$OUT/clocks.csv; echo "timestamp,clocks_sm_mhz,clocks_mem_mhz,power_w,temp_c" > $CLK
( while true; do nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu \
   --format=csv,noheader,nounits >> $CLK 2>/dev/null || true; sleep 1; done ) & CLKPID=$!
trap 'kill $CLKPID 2>/dev/null || true' EXIT
FAIL=0
run() {  # name dir script mode extra  -- full log kept; console shows [J3b] lines and anything failing
  echo; echo "########## $1 ($4) $(date -Is) ##########"
  ( cd $RUN/$2 && $PY $3 --modes $4 --perf per-mode --out $OUT/$1 --jsonl $OUT/unused.jsonl --n_videos 5 $5 ) \
    > $DST/$1.log 2>&1
  local rc=$?
  grep -E "^\[J3b\]|Traceback|Error|FAILED" $DST/$1.log | grep -v "^W09" | tail -40
  cp -f $OUT/$1/j3b.json $DST/$1.json 2>/dev/null || { echo "### $1 produced no j3b.json"; rc=1; }
  grep -q '"failures": {}' $DST/$1.json 2>/dev/null || { echo "### $1 reports mode failures"; rc=1; }
  echo "### $1 rc=$rc"; [ $rc -eq 0 ] || FAIL=1
}
# headline `final` mode in ABBA order (drift control); the eager pair after it
run orig_final   orig j3b_run.py        final ""
run L_final      L    self_forcing_latency_harness.py final "--tempo L"
run L_final_b    L    self_forcing_latency_harness.py final "--tempo L"
run orig_final_b orig j3b_run.py        final ""
run orig_eager   orig j3b_run.py        eager-patched ""
run L_eager      L    self_forcing_latency_harness.py eager-patched "--tempo L"
kill $CLKPID 2>/dev/null || true
cp -f $CLK $DST/
$PY $T/scripts/summarize_self_forcing_latency.py $DST || FAIL=1
echo "### done FAIL=$FAIL $(date -Is)"
exit $FAIL
