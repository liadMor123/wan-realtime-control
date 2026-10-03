#!/bin/bash
#SBATCH --job-name=tempo_s3g
#SBATCH --partition=a100-public
#SBATCH --account=mendelson_prj
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/home/USER/tempo/logs/step3_gen_%j.out
#SBATCH --error=/home/USER/tempo/logs/step3_gen_%j.out
# Part 2 step 3 (pre-registered): Self-Forcing B0 (zero tables) vs L(2,2), fa2kv path, 20 held-out prompts, seed 42.
# Usage: sbatch --export=ALL,SFREV=<tempo-L commit>[,IDS_FILE=data/all80_ids.txt,SEED=43] slurm/self_forcing_generate.sbatch
set -uo pipefail
T=$HOME/tempo
module load python/3.11.15-x86-7sp4zgy
export HF_HOME=$T/cache/hf PYTHONUNBUFFERED=1 TORCHINDUCTOR_CACHE_DIR=$T/cache/inductor
RUN=$TMPDIR/sfgen; mkdir -p $RUN/repo/checkpoints
git -C $T/ext/Self-Forcing archive "$SFREV" | tar -x -C $RUN/repo
ln -s $HOME/proj/cache/wan_models $RUN/repo/wan_models                      # read-only use of ~/proj
ln -s $HOME/proj/cache/checkpoints/self_forcing_dmd.pt $RUN/repo/checkpoints/self_forcing_dmd.pt
cd $RUN/repo
echo "=== step3_gen ids ${IDS_FILE:-data/heldout_ids.txt} seed ${SEED:-42} on $(hostname) job $SLURM_JOB_ID tempo $(git -C $T rev-parse --short HEAD) SF $SFREV $(date -Is) ==="
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
grep -q "tempo_bias" wan/modules/model.py || { echo "### staged repo is not tempo-L"; exit 2; }
$T/venv/bin/python $T/scripts/self_forcing_generate_videos.py --ids "$(cat $T/${IDS_FILE:-data/heldout_ids.txt})" --seed ${SEED:-42}
RC=$?
echo "### rc=$RC $(date -Is)"
exit $RC
