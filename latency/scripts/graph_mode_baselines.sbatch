#!/bin/bash
#SBATCH --job-name=j3-baselines
#SBATCH --partition=a100-public
#SBATCH --account=mendelson_prj
#SBATCH --qos=12h_4g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/USER/proj/logs/j3-%j.out
#SBATCH --error=/home/USER/proj/logs/j3-%j.out
set -uo pipefail

PROJ=$HOME/proj; CACHE=$PROJ/cache; VENV=$PROJ/venv; RUN=$TMPDIR/run
echo "=== J3 on $(hostname) job $SLURM_JOB_ID $(date -Is) ==="
module load python/3.11.15-x86-7sp4zgy
PY="$VENV/bin/python"

# Inductor autotune cache persists across processes AND jobs: max-autotune is
# expensive and both compile modes share most kernels.
export TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"

mkdir -p "$RUN/out"
cp -a "$PROJ/Self-Forcing" "$RUN/repo"
cp -a "$CACHE/wan_models" "$RUN/repo/wan_models"
mkdir -p "$RUN/repo/checkpoints"
cp -a "$CACHE/checkpoints/self_forcing_dmd.pt" "$RUN/repo/checkpoints/"
cp "$PROJ/src/harness/graph_mode_baselines.py" "$RUN/repo/"
cd "$RUN/repo"

CLK="$RUN/out/j3_clocks.csv"
echo "timestamp,clocks_sm_mhz,clocks_mem_mhz,power_w,temp_c" > "$CLK"
( while true; do nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu \
    --format=csv,noheader,nounits >> "$CLK" 2>/dev/null || true; sleep 1; done ) &
CLKPID=$!
trap 'kill $CLKPID 2>/dev/null || true' EXIT

JSONL="$PROJ/results/runs.jsonl"
declare -A RC

echo; echo "########## eager reference video (for PSNR) ##########"
$PY graph_mode_baselines.py --mode eager_ref --n_measured 5 --out "$RUN/out" --jsonl "$JSONL" \
     --save_video "$RUN/out/ref_eager.npy"; RC[eager_ref]=$?

for M in manual_graph compile_nocg compile_cg; do
  echo; echo "########## mode: $M ##########"
  $PY graph_mode_baselines.py --mode "$M" --n_measured 5 --out "$RUN/out" --jsonl "$JSONL" \
       --save_video "$RUN/out/vid_$M.npy"
  RC[$M]=$?
  echo "### $M exit: ${RC[$M]} ###"
done

echo; echo "########## PSNR vs eager reference ##########"
for M in manual_graph compile_nocg compile_cg; do
  if [ -f "$RUN/out/vid_$M.npy" ]; then
    echo "--- $M ---"
    $PY "$PROJ/scripts/frame_psnr.py" "$RUN/out/ref_eager.npy" "$RUN/out/vid_$M.npy" \
        "$RUN/out/j3_psnr_$M.json" || true
  else
    echo "--- $M: no video produced ---"
  fi
done

kill $CLKPID 2>/dev/null || true
echo; echo "########## clocks (active only) ##########"
$PY - "$CLK" <<'PYC'
import csv,sys,statistics as st
rows=[r for r in csv.reader(open(sys.argv[1])) if r and not r[0].startswith("timestamp")]
def v(r,i):
    try: return float(r[i])
    except Exception: return None
act=[r for r in rows if (v(r,3) or 0)>=100.0]
print(f"  {len(rows)} samples, {len(act)} active")
for name,i in (("clocks.sm MHz",1),("power W",3),("temp C",4)):
    x=[v(r,i) for r in act if v(r,i) is not None]
    if x:
        rng=100*(max(x)-min(x))/st.median(x) if st.median(x) else 0
        print(f"  {name:15s} min={min(x):7.1f} median={st.median(x):7.1f} max={max(x):7.1f} range={rng:5.2f}%")
PYC

cp -f "$RUN"/out/j3_*.json "$PROJ/results/" 2>/dev/null || true
cp -f "$CLK" "$PROJ/results/" 2>/dev/null || true
echo; echo "=== J3 exit codes: eager_ref=${RC[eager_ref]} manual_graph=${RC[manual_graph]} compile_nocg=${RC[compile_nocg]} compile_cg=${RC[compile_cg]} ==="
