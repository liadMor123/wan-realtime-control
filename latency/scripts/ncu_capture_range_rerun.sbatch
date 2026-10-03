#!/bin/bash
#SBATCH --job-name=j6a2
#SBATCH --partition=a100-public
#SBATCH --account=mendelson_prj
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/home/USER/proj/logs/j6a2-%j.out
#SBATCH --error=/home/USER/proj/logs/j6a2-%j.out
set -uo pipefail
PROJ=$HOME/proj; CACHE=$PROJ/cache; VENV=$PROJ/venv; RUN=$TMPDIR/run
module load python/3.11.15-x86-7sp4zgy
PY="$VENV/bin/python"
export TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor"; mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
mkdir -p "$RUN/out"
cp -a "$PROJ/Self-Forcing" "$RUN/repo"; cp -a "$CACHE/wan_models" "$RUN/repo/wan_models"
mkdir -p "$RUN/repo/checkpoints"; cp -a "$CACHE/checkpoints/self_forcing_dmd.pt" "$RUN/repo/checkpoints/"
cp "$PROJ/src/harness/profile_single_pass.py" "$RUN/repo/"
cd "$RUN/repo"
echo "=== J6a2: ncu with --profile-from-start off, so the capture region is honoured ==="
# profile_single_pass.py already brackets the profiled chunk with cudaProfilerStart/Stop.
for POS in 3 6; do
  for KRX in "flash_fwd_kernel" "triton_tem_fused_addmm_7" "triton_tem_fused_addmm_5"; do
    echo "--- pos $POS $KRX ---"
    timeout 900 ncu --set full -c 1 --kernel-name "regex:$KRX" \
      --profile-from-start off --csv --log-file "$RUN/out/ncu2_${POS}_${KRX}.csv" \
      --target-processes all \
      $PY profile_single_pass.py --position $POS --mode compile_nocg --chunks 1 >/dev/null 2>&1
    echo "    exit $? ; $(grep -c '^"0"' "$RUN/out/ncu2_${POS}_${KRX}.csv" 2>/dev/null || echo 0) metric rows"
  done
done
mkdir -p "$PROJ/results/ncu2" && cp -f "$RUN/out"/ncu2_*.csv "$PROJ/results/ncu2/" 2>/dev/null || true
echo "=== J6a2 done ==="
