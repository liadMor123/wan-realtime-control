#!/bin/bash
#SBATCH --job-name=tempo_scg
#SBATCH --partition=a100-public
#SBATCH --account=mendelson_prj
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/home/USER/tempo/logs/score_gpu_%j.out
#SBATCH --error=/home/USER/tempo/logs/score_gpu_%j.out
# GPU scoring for the final runs (owner: all metric evaluation on the GPU). Runs the unmodified official temporal-
# accuracy metric (scripts/run_official_temporal_accuracy.py, venv_metric) and CLIP (scripts/clip_similarity_and_contact_sheets.py, venv) on the GPU, then
# the GPU-vs-CPU consistency gate (scripts/gpu_vs_cpu_metric_check.py -> results/gpu_vs_cpu_metric_check.json) and a summary.
#
# Device. The official scripts pass no device, so ultralytics uses its default (device=None -> cuda:0 when
# torch.cuda.is_available()); --require-cuda records the device of the predictor, the network parameters and every
# network input (results/metric/<OUT>/<tag>/_metric_device.json) and fails unless all are CUDA. CLIP: clip_similarity_and_contact_sheets.py picks
# cuda when available; --require-cuda fails otherwise (results/quality/<OUT>/_clip_device.json).
# NVIDIA_TF32_OVERRIDE=0 keeps cuDNN/cuBLAS in full fp32 (PyTorch enables TF32 for cuDNN convolutions by default), the
# closest GPU arithmetic to the fp32 CPU scoring used so far. It is an environment setting; no script is changed.
#
# Parameters (sbatch --export splits on commas, so lists use ':'):
#   BENCH         one_object | two_objects
#   TAGS          tags to score, e.g. B0_flash_s42:L_b2g2_fa2kv_s42
#   IDS           an ids file (data/heldout_ids.txt) or a comma-free id spec (0-81; ':' stands for ',')
#   OUT           results/metric/<OUT>/<tag>/ and results/quality/<OUT>/<tag>.json; must be new: phase24 or *_gpu
#   RESCORE_TAGS  optional: already CPU-scored tags to re-score on the GPU on the same ids (consistency gate)
#   RESCORE_OUT   their dir name (e.g. phase2_gpu)
#   SUMMARY       optional: script run at the end (scripts/summarize_fa2kv_accuracy.py | scripts/summarize_two_objects_all82.py)
set -uo pipefail
T=$HOME/tempo
module load python/3.11.15-x86-7sp4zgy
export HF_HOME=$T/cache/hf PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 NVIDIA_TF32_OVERRIDE=0
cd $T
: "${BENCH:?BENCH not set}" "${TAGS:?TAGS not set}" "${IDS:?IDS not set}" "${OUT:?OUT not set}"
RESCORE_TAGS=${RESCORE_TAGS:-}; RESCORE_OUT=${RESCORE_OUT:-}; SUMMARY=${SUMMARY:-}
J=${SLURM_JOB_ID:-local}
echo "=== GPU scoring on $(hostname) job $J commit $(git rev-parse --short HEAD) $(date -Is) ==="
echo "BENCH=$BENCH TAGS=$TAGS IDS=$IDS OUT=$OUT RESCORE_TAGS=$RESCORE_TAGS RESCORE_OUT=$RESCORE_OUT SUMMARY=$SUMMARY"
case $BENCH in one_object) MB=one-object;; two_objects) MB=two-object;; *) echo "### bad BENCH=$BENCH"; exit 2;; esac
new_dir() { case "$1" in phase24|*_gpu) ;; *) echo "### '$1': GPU scores go to new dirs (phase24 or *_gpu), never to the CPU-scored ones"; exit 2;; esac; }
new_dir "$OUT"; [ -z "$RESCORE_TAGS" ] || new_dir "$RESCORE_OUT"
if [ -f "$IDS" ]; then SPEC=$(tr -d ' \n' < "$IDS"); else SPEC=${IDS//:/,}; fi
ID_LIST=$($T/venv/bin/python -c "import sys;sys.path.insert(0,'scripts');from generate_benchmark_videos import parse_ids;print(','.join(map(str,parse_ids('$SPEC'))))") \
    || { echo "### bad IDS=$IDS"; exit 2; }
echo "ids ($(echo $ID_LIST | tr ',' '\n' | wc -l)): $ID_LIST"

GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1); echo "GPU $GPU"
[[ "$GPU" == *A100-SXM4-40GB* ]] || { echo "### wrong GPU $GPU"; exit 2; }
for V in venv venv_metric; do
  $T/$V/bin/python -c "
import sys, torch
ok = torch.cuda.is_available()
print('$V: torch', torch.__version__, 'torch.cuda.is_available()', ok, torch.cuda.get_device_name(0) if ok else '-')
sys.exit(0 if ok else 3)" || { echo "### CUDA NOT AVAILABLE in $V"; exit 3; }
done

# 1. metric preflight on the GPU (synthetic positive/negative controls through the same wrapper)
$T/venv_metric/bin/python scripts/run_official_temporal_accuracy.py --preflight --bench $MB --require-cuda \
    > logs/score_gpu_${J}_preflight.log 2>&1 || { echo "### GPU METRIC PREFLIGHT FAILED (logs/score_gpu_${J}_preflight.log)"; exit 1; }
grep -h "device check PASS" logs/score_gpu_${J}_preflight.log | head -1; tail -1 logs/score_gpu_${J}_preflight.log

# 2. official metric on the GPU, one process per tag
FAIL=0; PIDS=""
score() {
  local t=$1 o=$2 n
  n=$(ls videos/$t/*.mp4 2>/dev/null | wc -l); echo "videos/$t: $n mp4 -> results/metric/$o/$t"
  $T/venv_metric/bin/python scripts/run_official_temporal_accuracy.py --bench $MB --videos videos/$t --ids $ID_LIST \
      --out results/metric/$o/$t --require-cuda > logs/score_gpu_${J}_${o}_$t.log 2>&1
}
for t in ${TAGS//:/ }; do score $t $OUT & PIDS="$PIDS $!"; done
for t in ${RESCORE_TAGS//:/ }; do score $t $RESCORE_OUT & PIDS="$PIDS $!"; done
for p in $PIDS; do wait $p || FAIL=1; done
grep -H "device check PASS" logs/score_gpu_${J}_*_*.log | grep -v preflight
[ $FAIL -eq 0 ] || { echo "### GPU METRIC FAILED, not on CUDA, or videos missing (logs/score_gpu_${J}_*.log)"; exit 1; }

# 3. CLIP on the GPU (contact sheets only where missing)
$T/venv/bin/python scripts/clip_similarity_and_contact_sheets.py --bench $BENCH --tags ${TAGS//:/,} --ids $ID_LIST --out-name $OUT --require-cuda \
    || { echo "### CLIP FAILED"; exit 1; }
if [ -n "$RESCORE_TAGS" ]; then
  $T/venv/bin/python scripts/clip_similarity_and_contact_sheets.py --bench $BENCH --tags ${RESCORE_TAGS//:/,} --ids $ID_LIST \
      --out-name $RESCORE_OUT --require-cuda || { echo "### CLIP (re-score) FAILED"; exit 1; }
fi

# 4. GPU-vs-CPU gate (reports, never hides; does not fail on a disagreement), then the summary
$T/venv/bin/python scripts/gpu_vs_cpu_metric_check.py || { echo "### GATE SCRIPT FAILED"; exit 1; }
RC=0
if [ -n "$SUMMARY" ]; then $T/venv/bin/python $SUMMARY; RC=$?; fi
echo "### rc=$RC $(date -Is)"; exit $RC
