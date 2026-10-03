#!/bin/bash
#SBATCH --job-name=j11
#SBATCH --partition=a100-public
#SBATCH --account=mendelson_prj
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/USER/proj/logs/j11-%j.out
#SBATCH --error=/home/USER/proj/logs/j11-%j.out
set -uo pipefail
PROJ=$HOME/proj; CACHE=$PROJ/cache; VENV=$PROJ/venv; RUN=$TMPDIR/run
module load python/3.11.15-x86-7sp4zgy
PY="$VENV/bin/python"
export TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor"; mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
mkdir -p "$RUN/out"
cp -a "$PROJ/Self-Forcing" "$RUN/repo"; cp -a "$CACHE/wan_models" "$RUN/repo/wan_models"
mkdir -p "$RUN/repo/checkpoints"; cp -a "$CACHE/checkpoints/self_forcing_dmd.pt" "$RUN/repo/checkpoints/"
cp "$PROJ/src/harness/mode_latency_benchmark.py" "$RUN/repo/"
cd "$RUN/repo"
echo "=== J11 (bounded, one attempt) on $(hostname) job $SLURM_JOB_ID ==="
echo "remedy under test: torch.compiler.cudagraph_mark_step_begin() before each pass"
$PY mode_latency_benchmark.py --modes compile_cg --perf on --out "$RUN/out/CG" \
   --jsonl "$PROJ/results/runs.jsonl" --n_videos 5 2>&1 \
   | grep -E "^\[J3b\]|FAILED|accessing tensor|Error" | head -30
cp -f "$RUN/out/CG/j3b.json" "$PROJ/results/j11_compile_cg.json" 2>/dev/null || true
echo "=== J11 done ==="
