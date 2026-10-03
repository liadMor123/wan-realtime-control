#!/bin/bash
#SBATCH --job-name=tempo_s3c
#SBATCH --partition=a100-public
#SBATCH --account=mendelson_prj
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/home/USER/tempo/logs/step3c_sw_%j.out
#SBATCH --error=/home/USER/tempo/logs/step3c_sw_%j.out
# Part 2 step 3c (pre-registered): prompt-switching baselines in Self-Forcing on the fa2kv path, all 80 prompts,
# seed 42. One arm per job: MODE=rf (PS-RF), ll (PS-LongLive-style), rfL (PS-RF + L).
# Usage: sbatch --export=ALL,SFREV=<tempo-L commit>,MODE=<rf|ll|rfL> slurm/self_forcing_prompt_switch_arms.sbatch
set -uo pipefail
T=$HOME/tempo
module load python/3.11.15-x86-7sp4zgy
export HF_HOME=$T/cache/hf PYTHONUNBUFFERED=1 TORCHINDUCTOR_CACHE_DIR=$T/cache/inductor
RUN=$TMPDIR/sfsw; mkdir -p $RUN/repo/checkpoints
git -C $T/ext/Self-Forcing archive "$SFREV" | tar -x -C $RUN/repo
ln -s $HOME/proj/cache/wan_models $RUN/repo/wan_models                      # read-only use of ~/proj
ln -s $HOME/proj/cache/checkpoints/self_forcing_dmd.pt $RUN/repo/checkpoints/self_forcing_dmd.pt
cd $RUN/repo
echo "=== step3c_sw MODE=$MODE on $(hostname) job $SLURM_JOB_ID tempo $(git -C $T rev-parse --short HEAD) SF $SFREV $(date -Is) ==="
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
grep -q "tempo_bias" wan/modules/model.py || { echo "### staged repo is not tempo-L"; exit 2; }
# interleave the onset groups (0,20,40,60,1,21,...) so switch and normal chunks at a position are spread over the job
IDS=$(python3 -c "print(','.join(str(g * 20 + k) for k in range(20) for g in range(4)))")
$T/venv/bin/python $T/scripts/self_forcing_prompt_switch_arms.py --mode "$MODE" --ids "$IDS" --seed 42
RC=$?
echo "### rc=$RC $(date -Is)"
exit $RC
