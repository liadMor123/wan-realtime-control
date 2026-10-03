#!/bin/bash
#SBATCH --job-name=j7a
#SBATCH --partition=a100-public
#SBATCH --account=mendelson_prj
#SBATCH --qos=12h_4g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/USER/proj/logs/j7a-%j.out
#SBATCH --error=/home/USER/proj/logs/j7a-%j.out
set -uo pipefail
PROJ=$HOME/proj; CACHE=$PROJ/cache; VENV=$PROJ/venv; RUN=$TMPDIR/run
module load python/3.11.15-x86-7sp4zgy
PY="$VENV/bin/python"
export TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor"; mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
mkdir -p "$RUN/out"
cp -a "$PROJ/Self-Forcing" "$RUN/repo"; cp -a "$CACHE/wan_models" "$RUN/repo/wan_models"
mkdir -p "$RUN/repo/checkpoints"; cp -a "$CACHE/checkpoints/self_forcing_dmd.pt" "$RUN/repo/checkpoints/"
cp "$PROJ/src/harness/splitkv_attention_benchmark.py" "$PROJ/src/harness/mode_latency_benchmark.py" "$RUN/repo/"
cd "$RUN/repo"
echo "=== J7a on $(hostname) job $SLURM_JOB_ID; patch @ $(git rev-parse --short HEAD) ==="
git log --oneline -1

echo; echo "########## 14.5 compile_cg over five videos (with fix 7b) ##########"
$PY mode_latency_benchmark.py --modes compile_cg --perf on --out "$RUN/out/G" \
   --jsonl "$PROJ/results/runs.jsonl" --n_videos 5 2>&1 | grep -E "^\[J3b\]|FAILED"
cp -f "$RUN/out/G/j3b.json" "$RUN/out/j7_compile_cg.json" 2>/dev/null || true

echo; echo "########## 14.1-14.2 conventional attention arms ##########"
export J7_OUT="$RUN/out/j7_attn.json"
$PY splitkv_attention_benchmark.py 2>&1 | grep -vE "^W09|^E09"

cp -f "$RUN/out"/j7_*.json "$PROJ/results/" 2>/dev/null || true
echo; echo "=== J7a done ==="
