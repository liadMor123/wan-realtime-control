#!/bin/bash
#SBATCH --job-name=j6a
#SBATCH --partition=a100-public
#SBATCH --account=mendelson_prj
#SBATCH --qos=12h_4g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/USER/proj/logs/j6a-%j.out
#SBATCH --error=/home/USER/proj/logs/j6a-%j.out
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
cp "$PROJ/src/harness/profile_single_pass.py" "$PROJ/src/harness/triton_fence_and_kv_fidelity.py" \
   "$PROJ/src/harness/mode_latency_benchmark.py" "$RUN/repo/"
cd "$RUN/repo"
echo "=== J6a on $(hostname) job $SLURM_JOB_ID; patch @ $(git rev-parse --short HEAD) ==="

# ---------------- (f) gate item 6 + outstanding J4 items ----------------
echo; echo "########## (f) Triton release/acquire + J4 leftovers ##########"
export J6_EXTRAS_OUT="$RUN/out/j6_extras.json"
$PY triton_fence_and_kv_fidelity.py 2>&1 | grep -vE "^W0914|^E0914"

# ---------------- (b) nsys inter-kernel gaps ----------------
echo; echo "########## (b) nsys: inter-kernel gaps, graphed-compiled, 3 chunks ##########"
nsys profile -t cuda,nvtx --cuda-memory-usage=false -f true \
  -o "$RUN/out/j6_gaps" --capture-range=cudaProfilerApi --capture-range-end=stop \
  $PY profile_single_pass.py --position 6 --mode compile_nocg --chunks 3 --nvtx 2>&1 | tail -5
echo "--- reducing trace in-job (no .nsys-rep goes home) ---"
nsys stats --report cuda_gpu_trace --format csv \
  -o "$RUN/out/gaps" "$RUN/out/j6_gaps.nsys-rep" 2>&1 | tail -3
$PY - "$RUN/out" <<'PYG'
import csv, glob, json, os, sys
d = sys.argv[1]
f = None
for c in glob.glob(os.path.join(d, "gaps*cuda_gpu_trace*.csv")): f = c
if not f:
    print("  no cuda_gpu_trace csv produced"); raise SystemExit
rows = list(csv.DictReader(open(f)))
def num(x):
    try: return float(x.replace(",", ""))
    except Exception: return None
ev = []
for r in rows:
    s = num(r.get("Start (ns)") or r.get("Start(ns)") or "")
    dur = num(r.get("Duration (ns)") or r.get("Duration(ns)") or "")
    if s is None or dur is None: continue
    ev.append((s, s + dur, (r.get("Name") or "")[:60]))
ev.sort()
busy = 0.0; gap = 0.0; prev_end = None; gaps = []
for s, e, n in ev:
    busy += e - s
    if prev_end is not None and s > prev_end:
        gap += s - prev_end; gaps.append((s - prev_end, n))
    prev_end = max(prev_end or e, e)
span = ev[-1][1] - ev[0][0] if ev else 0
out = {"n_events": len(ev), "span_ms": span/1e6, "busy_ms": busy/1e6,
       "gap_ms": gap/1e6, "gap_pct_of_span": 100*gap/span if span else None,
       "gap_ms_per_pass": gap/1e6/15, "top_gaps_us": sorted(gaps, reverse=True)[:10]}
print(json.dumps({k: v for k, v in out.items() if k != "top_gaps_us"}, indent=2))
print("  largest gaps (us, next kernel):")
for g, n in out["top_gaps_us"]: print(f"    {g/1e3:8.2f}  {n}")
json.dump(out, open(os.path.join(d, "j6_gaps.json"), "w"), indent=2, default=str)
PYG
rm -f "$RUN/out"/*.nsys-rep "$RUN/out"/*.sqlite

# ---------------- (a) ncu tail imbalance ----------------
echo; echo "########## (a) ncu --set full on critical kernels ##########"
for POS in 3 6; do
  for KRX in "triton_tem_fused_addmm_7" "triton_tem_fused_addmm_5" \
             "triton_tem_fused_addmm_0" "triton_tem_fused_addmm_2" \
             "ampere_bf16_s16816gemm" "flash_fwd_kernel" \
             "elementwise_kernel" "unrolled_elementwise_kernel" "direct_copy_kernel"; do
    echo "--- pos $POS kernel $KRX ---"
    timeout 900 ncu --set full -c 1 --kernel-name "regex:$KRX" \
      --csv --log-file "$RUN/out/ncu_${POS}_${KRX}.csv" \
      --target-processes all \
      $PY profile_single_pass.py --position $POS --mode compile_nocg --chunks 1 >/dev/null 2>&1
    echo "    exit $? ; $(wc -l < "$RUN/out/ncu_${POS}_${KRX}.csv" 2>/dev/null || echo 0) csv lines"
  done
done

# ---------------- (g) compile_cg over the five-video protocol ----------------
echo; echo "########## (g) compile_cg timing, five videos ##########"
$PY mode_latency_benchmark.py --modes compile_cg --out "$RUN/out/G" \
   --jsonl "$PROJ/results/runs.jsonl" --n_videos 5 2>&1 | grep -E "^\[J3b\]|Traceback|Error" | tail -20
cp -f "$RUN/out/G/j3b.json" "$RUN/out/j6_compile_cg.json" 2>/dev/null || true

cp -f "$RUN/out"/j6_*.json "$PROJ/results/" 2>/dev/null || true
mkdir -p "$PROJ/results/ncu" && cp -f "$RUN/out"/ncu_*.csv "$PROJ/results/ncu/" 2>/dev/null || true
echo; echo "=== J6a done ==="
