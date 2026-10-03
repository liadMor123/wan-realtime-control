#!/bin/bash
#SBATCH --job-name=j7b
#SBATCH --partition=a100-public
#SBATCH --account=mendelson_prj
#SBATCH --qos=12h_4g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=/home/USER/proj/logs/j7b-%j.out
#SBATCH --error=/home/USER/proj/logs/j7b-%j.out
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
echo "=== J7b on $(hostname) job $SLURM_JOB_ID; patch @ $(git rev-parse --short HEAD) ==="

echo; echo "########## (a) staircase at the chosen split + (b) Q probe ##########"
export J7_OUT="$RUN/out/j7_attn2.json"
$PY splitkv_attention_benchmark.py 2>&1 | grep -vE "^W09|^E09" | sed -n '/head staircase/,$p'

echo; echo "########## (c) integration: compile_nocg with split-KV (num_splits=4) ##########"
$PY mode_latency_benchmark.py --modes compile_nocg --perf on --attn_splits 4 \
   --out "$RUN/out/S4" --jsonl "$PROJ/results/runs.jsonl" --n_videos 5 2>&1 | grep -E "^\[J3b\]"
cp -f "$RUN/out/S4/j3b.json" "$RUN/out/j7_integrated_s4.json" 2>/dev/null || true

echo; echo "########## (c) control: same build, split-KV OFF ##########"
$PY mode_latency_benchmark.py --modes compile_nocg --perf on --attn_splits 0 \
   --out "$RUN/out/S0" --jsonl "$PROJ/results/runs.jsonl" --n_videos 5 2>&1 | grep -E "^\[J3b\]"
cp -f "$RUN/out/S0/j3b.json" "$RUN/out/j7_integrated_s0.json" 2>/dev/null || true

cp -f "$RUN/out"/j7_*.json "$PROJ/results/" 2>/dev/null || true
echo; echo "=== J7b done ==="
