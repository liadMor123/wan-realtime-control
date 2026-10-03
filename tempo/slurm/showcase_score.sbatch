#!/bin/bash
#SBATCH --job-name=tempo_shsc
#SBATCH --partition=a100-public
#SBATCH --account=mendelson_prj
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/home/USER/tempo/logs/showcase_score_%j.out
#SBATCH --error=/home/USER/tempo/logs/showcase_score_%j.out
# GPU scoring of the showcase videos (slurm/showcase_generate.sbatch), as slurm/score_gpu.sbatch does for the benchmark runs,
# which cannot take another prompt set: the unmodified official temporal-accuracy metric through
# scripts/run_official_temporal_accuracy.py --prompt-set showcase (subset CSV built from data/showcase_prompts.csv in the
# official columns; one-object metric for showcase id 0, two-object metric for ids 1-5) and CLIP (scripts/clip_similarity_and_contact_sheets.py
# --prompt-set showcase), all with --require-cuda and NVIDIA_TF32_OVERRIDE=0 as in score_gpu.sbatch. Outputs (new):
# results/metric/phase25_gpu/<tag>/, results/quality/phase25_gpu/<tag>.json, results/contact/<tag>/p<id>.png.
# Refuses to start if any of those result files exist. No GPU-vs-CPU gate (no CPU scores for these prompts).
set -uo pipefail
T=$HOME/tempo
module load python/3.11.15-x86-7sp4zgy
export HF_HOME=$T/cache/hf PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 NVIDIA_TF32_OVERRIDE=0
cd $T
OUT=phase25_gpu
J=${SLURM_JOB_ID:-local}
ONE="showcase_B0_s42 showcase_L_b2g2_s42"; TWO="showcase_B0_2obj_s42 showcase_L_b2g2_2obj_s42"
echo "=== showcase GPU scoring on $(hostname) job $J commit $(git rev-parse --short HEAD) $(date -Is) ==="
for t in $ONE $TWO; do
  for f in results/metric/$OUT/$t/temporal_accuracy_one_object.json results/metric/$OUT/$t/temporal_accuracy_two_objects.json \
           results/quality/$OUT/$t.json; do
    [ ! -e "$f" ] || { echo "### $f exists: showcase scores are never overwritten"; exit 2; }
  done
done
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1); echo "GPU $GPU"
[[ "$GPU" == *A100-SXM4-40GB* ]] || { echo "### wrong GPU $GPU"; exit 2; }
for V in venv venv_metric; do
  $T/$V/bin/python -c "
import sys, torch
ok = torch.cuda.is_available()
print('$V: torch', torch.__version__, 'torch.cuda.is_available()', ok, torch.cuda.get_device_name(0) if ok else '-')
sys.exit(0 if ok else 3)" || { echo "### CUDA NOT AVAILABLE in $V"; exit 3; }
done

# 1. metric preflight on the GPU, both benchmarks (synthetic positive/negative controls through the same wrapper)
for MB in one-object two-object; do
  $T/venv_metric/bin/python scripts/run_official_temporal_accuracy.py --preflight --bench $MB --require-cuda \
      > logs/showcase_score_${J}_preflight_$MB.log 2>&1 || { echo "### GPU METRIC PREFLIGHT FAILED ($MB)"; exit 1; }
  grep -h "device check PASS" logs/showcase_score_${J}_preflight_$MB.log | head -1; tail -1 logs/showcase_score_${J}_preflight_$MB.log
done

# 2. official metric on the GPU, one process per tag
FAIL=0; PIDS=""
score() {
  local t=$1 mb=$2 ids=$3
  echo "videos/$t: $(ls videos/$t/*.mp4 2>/dev/null | wc -l) mp4 -> results/metric/$OUT/$t ($mb, showcase ids $ids)"
  $T/venv_metric/bin/python scripts/run_official_temporal_accuracy.py --prompt-set showcase --bench $mb --videos videos/$t \
      --ids $ids --out results/metric/$OUT/$t --require-cuda > logs/showcase_score_${J}_$t.log 2>&1
}
for t in $ONE; do score $t one-object 0 & PIDS="$PIDS $!"; done
for t in $TWO; do score $t two-object 1,2,3,4,5 & PIDS="$PIDS $!"; done
for p in $PIDS; do wait $p || FAIL=1; done
grep -H "device check PASS" logs/showcase_score_${J}_showcase_*.log
tail -qn1 logs/showcase_score_${J}_showcase_*.log
[ $FAIL -eq 0 ] || { echo "### GPU METRIC FAILED, not on CUDA, or videos missing (logs/showcase_score_${J}_*.log)"; exit 1; }

# 3. CLIP on the GPU (contact sheets only where missing)
$T/venv/bin/python scripts/clip_similarity_and_contact_sheets.py --prompt-set showcase --bench one_object --tags ${ONE// /,} --ids 0 \
    --out-name $OUT --require-cuda || { echo "### CLIP (one object) FAILED"; exit 1; }
mv results/quality/$OUT/_clip_device.json results/quality/$OUT/_clip_device_one_object.json
$T/venv/bin/python scripts/clip_similarity_and_contact_sheets.py --prompt-set showcase --bench two_objects --tags ${TWO// /,} --ids 1-5 \
    --out-name $OUT --require-cuda || { echo "### CLIP (two objects) FAILED"; exit 1; }
mv results/quality/$OUT/_clip_device.json results/quality/$OUT/_clip_device_two_objects.json

# 4. per-video table
$T/venv/bin/python - <<PY
import json, os
T = os.path.expanduser("~/tempo"); O = "$OUT"
for tags, res in ((("$ONE").split(), "temporal_accuracy_one_object.json"), (("$TWO").split(), "temporal_accuracy_two_objects.json")):
    for t in tags:
        m = json.load(open(f"{T}/results/metric/{O}/{t}/{res}"))
        q = json.load(open(f"{T}/results/quality/{O}/{t}.json"))
        for v, (i, c) in zip(m["temporal_accuracy"][1], q.items()):
            print(f"{t:28s} id {i}  TA {v['video_results']:.3f}  CLIP {c['clip_mean']:.4f}  {os.path.basename(v.get('video_path', ''))}")
        print(f"{t:28s} mean TA {m['temporal_accuracy'][0]:.3f}")
PY
echo "### rc=0 $(date -Is)"
