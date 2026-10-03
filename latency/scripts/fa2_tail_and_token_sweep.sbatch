#!/bin/bash
#SBATCH --job-name=j6d
#SBATCH --partition=a100-public
#SBATCH --account=mendelson_prj
#SBATCH --qos=12h_4g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=/home/USER/proj/logs/j6d-%j.out
#SBATCH --error=/home/USER/proj/logs/j6d-%j.out
set -uo pipefail
PROJ=$HOME/proj; CACHE=$PROJ/cache; VENV=$PROJ/venv; RUN=$TMPDIR/run
module load python/3.11.15-x86-7sp4zgy
PY="$VENV/bin/python"
export TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor"; mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
mkdir -p "$RUN/out"
cp -a "$PROJ/Self-Forcing" "$RUN/repo"; cp -a "$CACHE/wan_models" "$RUN/repo/wan_models"
mkdir -p "$RUN/repo/checkpoints"; cp -a "$CACHE/checkpoints/self_forcing_dmd.pt" "$RUN/repo/checkpoints/"
cp "$PROJ/src/harness/mode_latency_benchmark.py" "$PROJ/src/harness/token_count_sweep.py" \
   "$PROJ/src/harness/fa2_tail_head_sweep.py" "$RUN/repo/"
cd "$RUN/repo"
echo "=== J6d on $(hostname) job $SLURM_JOB_ID; patch @ $(git rev-parse --short HEAD) ==="

echo; echo "########## 13: FA2 wave-quantization head sweep ##########"
export J6_FA2_OUT="$RUN/out/j6_fa2sweep.json"
$PY fa2_tail_head_sweep.py 2>&1 | grep -vE "^W09|^E09"

echo; echo "########## (g) compile_cg over the five-video protocol ##########"
$PY mode_latency_benchmark.py --modes compile_cg --perf on --out "$RUN/out/G" \
   --jsonl "$PROJ/results/runs.jsonl" --n_videos 5 2>&1 | grep -E "^\[J3b\]"
cp -f "$RUN/out/G/j3b.json" "$RUN/out/j6_compile_cg.json" 2>/dev/null || true

echo; echo "########## (e) matched-K token-count sweep ##########"
$PY token_count_sweep.py --out "$RUN/out" --n_videos 3 2>&1 | grep -vE "^W09|^E09"

cp -f "$RUN/out"/j6_*.json "$PROJ/results/" 2>/dev/null || true
echo; echo "=== J6d done ==="
