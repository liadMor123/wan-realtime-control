#!/bin/bash
#SBATCH --job-name=j3b
#SBATCH --partition=a100-public
#SBATCH --account=mendelson_prj
#SBATCH --qos=12h_4g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/USER/proj/logs/j3b-%j.out
#SBATCH --error=/home/USER/proj/logs/j3b-%j.out
set -uo pipefail
PROJ=$HOME/proj; CACHE=$PROJ/cache; VENV=$PROJ/venv; RUN=$TMPDIR/run
module load python/3.11.15-x86-7sp4zgy
PY="$VENV/bin/python"
export TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor"; mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
mkdir -p "$RUN/out"
cp -a "$PROJ/Self-Forcing" "$RUN/repo"
cp -a "$CACHE/wan_models" "$RUN/repo/wan_models"
mkdir -p "$RUN/repo/checkpoints"
cp -a "$CACHE/checkpoints/self_forcing_dmd.pt" "$RUN/repo/checkpoints/"
cp "$PROJ/src/harness/mode_latency_benchmark.py" "$PROJ/scripts/fp32_band_numerics.py" "$RUN/repo/"
cd "$RUN/repo"
echo "=== J3b on $(hostname) job $SLURM_JOB_ID; patch @ $(git rev-parse --short HEAD) ==="

CLK="$RUN/out/j3b_clocks.csv"
echo "timestamp,clocks_sm_mhz,clocks_mem_mhz,power_w,temp_c" > "$CLK"
( while true; do nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu \
    --format=csv,noheader,nounits >> "$CLK" 2>/dev/null || true; sleep 1; done ) & CLKPID=$!
trap 'kill $CLKPID 2>/dev/null || true' EXIT

# Process isolation, forced by J3b run 147741: manual CUDA graphs and Inductor
# cannot coexist in one process here -- after a torch.compile warmup (and a
# failed Inductor-cudagraph attempt), graph REPLAY raised an illegal memory
# access. Each group below is interleaved by video internally, so the
# comparisons that matter are drift-cancelled within their own process.
echo; echo "########## A: eager-original | eager-patched | compile_nocg (interleaved) ##########"
$PY mode_latency_benchmark.py --modes eager-original,eager-patched,compile_nocg \
   --out "$RUN/out/A" --jsonl "$PROJ/results/runs.jsonl" --n_videos 5; RA=$?
echo "### A exit: $RA ###"

# Graph modes CANNOT be interleaved: in run 147772 a full eager generation
# between capture and replay disturbed the caching allocator and graph replay
# raised an illegal memory access. Each graph mode therefore runs ALONE.
echo; echo "########## B: manual_graph over eager-patched kernels (alone) ##########"
$PY mode_latency_benchmark.py --modes manual_graph \
   --out "$RUN/out/B" --jsonl "$PROJ/results/runs.jsonl" --n_videos 5; RB=$?
echo "### B exit: $RB ###"

echo; echo "########## C: compile_cg, kv cache marked static (alone) ##########"
$PY mode_latency_benchmark.py --modes compile_cg \
   --out "$RUN/out/C" --jsonl "$PROJ/results/runs.jsonl" --n_videos 5; RCG=$?
echo "### C exit: $RCG ###"

echo; echo "########## D: fallback -- manual graphs over compile_nocg kernels (alone) ##########"
$PY mode_latency_benchmark.py --modes manual_graph_compiled \
   --out "$RUN/out/D" --jsonl "$PROJ/results/runs.jsonl" --n_videos 5; RD=$?
echo "### D exit: $RD ###"
RC=$(( RA != 0 ? RA : 0 ))
for d in A B C D; do cp -f "$RUN/out/$d/j3b.json" "$RUN/out/j3b_$d.json" 2>/dev/null || true; done
for d in A B C D; do cp -f "$RUN/out/$d"/vid_*.npy "$RUN/out/" 2>/dev/null || true; done

echo; echo "########## Tier-1 bitwise checks ##########"
$PY - "$RUN/out" <<'PYB'
import sys, os, hashlib, json, glob
d=sys.argv[1]; out={}
import numpy as np
h={}
for f in sorted(glob.glob(os.path.join(d,"vid_*.npy"))):
    m=os.path.basename(f)[4:-4]
    h[m]=hashlib.sha256(np.load(f).tobytes()).hexdigest()
for m,x in h.items(): print(f"  {m:16s} {x[:16]}")
pairs=[("eager-patched","eager-original"),("manual_graph","eager-patched")]
for a,b in pairs:
    if a in h and b in h:
        ok=h[a]==h[b]; out[f"{a}_vs_{b}"]={"bitwise_identical":ok}
        print(f"  TIER1 {a} vs {b}: {'IDENTICAL' if ok else 'DIFFERS'}")
json.dump({"hashes":h,"tier1":out}, open(os.path.join(d,"j3b_tier1.json"),"w"), indent=2)
PYB

echo; echo "########## Tier-2 fp32 band ##########"
export J3B_FP32_OUT="$RUN/out/j3b_fp32.json"
$PY fp32_band_numerics.py

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
cp -f "$RUN"/out/j3b*.json "$PROJ/results/" 2>/dev/null || true
cp -f "$CLK" "$PROJ/results/" 2>/dev/null || true
echo "=== J3b done, exit $RC ==="
