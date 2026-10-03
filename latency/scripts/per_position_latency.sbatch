#!/bin/bash
#SBATCH --job-name=j2-timing
#SBATCH --partition=a100-public
#SBATCH --account=mendelson_prj
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/USER/proj/logs/j2-%j.out
#SBATCH --error=/home/USER/proj/logs/j2-%j.out
set -euo pipefail

PROJ=$HOME/proj
CACHE=$PROJ/cache
VENV=$PROJ/venv
RUN=$TMPDIR/run

echo "=== J2 on $(hostname), job $SLURM_JOB_ID, $(date -Is) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
module load python/3.11.15-x86-7sp4zgy
PY="$VENV/bin/python"
[ -x "$PY" ] || { echo "J2 FAILURE: venv missing; run J1 first."; exit 3; }
$PY -c "import torch,flash_attn;print('torch',torch.__version__,'fa',flash_attn.__version__)"

export J1_CKPT_SHA=$(cat "$CACHE/checkpoints/self_forcing_dmd.pt.sha256")
export J1_REPO_REV=$(git -C "$PROJ/Self-Forcing" rev-parse HEAD)

echo "=== staging to $RUN ==="
mkdir -p "$RUN/out"
cp -a "$PROJ/Self-Forcing" "$RUN/repo"
cp -a "$CACHE/wan_models" "$RUN/repo/wan_models"
mkdir -p "$RUN/repo/checkpoints"
cp -a "$CACHE/checkpoints/self_forcing_dmd.pt" "$RUN/repo/checkpoints/"
cp "$PROJ/src/harness/measure_per_position_latency.py" "$RUN/repo/"
cd "$RUN/repo"

# ---------- clocks / power trace, 1 Hz (protocol: clocks and power logged alongside timing) ----------
CLK="$RUN/out/j2_clocks.csv"
echo "timestamp,clocks_sm_mhz,clocks_mem_mhz,power_w,temp_c" > "$CLK"
( while true; do
    nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu \
               --format=csv,noheader,nounits >> "$CLK" 2>/dev/null || true
    sleep 1
  done ) &
CLKPID=$!
trap 'kill $CLKPID 2>/dev/null || true' EXIT

JSONL="$PROJ/results/runs.jsonl"
RC=0
set +e

echo; echo "############ J2 sweep: 1 warmup + 5 measured videos ############"
$PY measure_per_position_latency.py --mode sweep --n_measured 5 --out "$RUN/out" --jsonl "$JSONL"
RC=$?
echo "### sweep exit: $RC ###"

echo; echo "############ J2 envelope: two separate processes ############"
$PY measure_per_position_latency.py --mode envelope --envelope_tag a --out "$RUN/out" --jsonl "$JSONL"
RA=$?
$PY measure_per_position_latency.py --mode envelope --envelope_tag b --out "$RUN/out" --jsonl "$JSONL"
RB=$?
if [ "$RA" -eq 0 ] && [ "$RB" -eq 0 ]; then
  $PY "$PROJ/scripts/frame_psnr.py" "$RUN/out/envelope_a.npy" "$RUN/out/envelope_b.npy" \
      "$RUN/out/j2_envelope.json"
else
  echo "J2 WARNING: envelope runs failed ($RA/$RB)"; RC=$(( RC == 0 ? 4 : RC ))
fi
set -e

kill $CLKPID 2>/dev/null || true

echo; echo "=== clocks / power summary ==="
$PY - "$CLK" <<'PYC'
import csv, sys, statistics as st
rows=[r for r in csv.reader(open(sys.argv[1])) if r and not r[0].startswith("timestamp")]
def val(r,i):
    try: return float(r[i])
    except Exception: return None
# Whole-job stats are dominated by idle staging/model-load and are NOT
# meaningful; condition on the GPU actually working (power >= 100 W).
act=[r for r in rows if (val(r,3) or 0) >= 100.0]
print(f"  samples: {len(rows)} total, {len(act)} active (power>=100W)")
for label,src in (("ALL",rows),("ACTIVE",act)):
    for name,i in (("clocks.sm MHz",1),("clocks.mem MHz",2),("power W",3),("temp C",4)):
        v=[val(r,i) for r in src if val(r,i) is not None]
        if v:
            rng=100*(max(v)-min(v))/st.median(v) if st.median(v) else 0
            print(f"  [{label:6s}] {name:16s} min={min(v):8.1f} median={st.median(v):8.1f} max={max(v):8.1f}  range={rng:6.2f}%")
PYC

# ---------- reduced results home; no .npy, no traces ----------
cp -f "$RUN/out/j2_sweep.json" "$RUN/out/j2_envelope.json" "$PROJ/results/" 2>/dev/null || true
cp -f "$CLK" "$PROJ/results/" 2>/dev/null || true
ls -l "$PROJ/results/" | tail -6

echo; echo "=== J2 exit $RC ==="
exit $RC
