#!/bin/bash
#SBATCH --job-name=j6c
#SBATCH --partition=a100-public
#SBATCH --account=mendelson_prj
#SBATCH --qos=12h_4g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/USER/proj/logs/j6c-%j.out
#SBATCH --error=/home/USER/proj/logs/j6c-%j.out
set -uo pipefail
PROJ=$HOME/proj; CACHE=$PROJ/cache; VENV=$PROJ/venv; RUN=$TMPDIR/run
module load python/3.11.15-x86-7sp4zgy
PY="$VENV/bin/python"
export TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor"; mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
mkdir -p "$RUN/out"
cp -a "$PROJ/Self-Forcing" "$RUN/repo"; cp -a "$CACHE/wan_models" "$RUN/repo/wan_models"
mkdir -p "$RUN/repo/checkpoints"; cp -a "$CACHE/checkpoints/self_forcing_dmd.pt" "$RUN/repo/checkpoints/"
cp "$PROJ/src/harness/mode_latency_benchmark.py" "$PROJ/scripts/cache_write_acceptance_bitwise.py" "$RUN/repo/"
cd "$RUN/repo"
echo "=== J6c on $(hostname) job $SLURM_JOB_ID; patch @ $(git rev-parse --short HEAD) ==="
git log --oneline -3

echo; echo "########## acceptance: byte-identity of the cache-write intervention ##########"
export J6C_OUT="$RUN/out/j6c_accept.json"
$PY cache_write_acceptance_bitwise.py; RA=$?
echo "### acceptance exit: $RA ###"

echo; echo "########## timing A/B: eager-patched vs eager-perf (interleaved) ##########"
$PY mode_latency_benchmark.py --modes eager-patched,eager-patched-perf --perf per-mode \
   --out "$RUN/out/E" --jsonl "$PROJ/results/runs.jsonl" --n_videos 5 2>&1 | grep -E "^\[J3b\]" 
cp -f "$RUN/out/E/j3b.json" "$RUN/out/j6c_eager.json" 2>/dev/null || true

echo; echo "########## compile_nocg WITH perf patches (flag fixed at process start) ##########"
$PY mode_latency_benchmark.py --modes compile_nocg --perf on \
   --out "$RUN/out/C" --jsonl "$PROJ/results/runs.jsonl" --n_videos 5 2>&1 | grep -E "^\[J3b\]"
cp -f "$RUN/out/C/j3b.json" "$RUN/out/j6c_compile_perf.json" 2>/dev/null || true

cp -f "$RUN/out"/j6c_*.json "$PROJ/results/" 2>/dev/null || true
echo; echo "=== J6c done (acceptance $RA) ==="
