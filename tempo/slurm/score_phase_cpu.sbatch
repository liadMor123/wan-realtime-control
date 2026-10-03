#!/bin/bash
#SBATCH --job-name=tempo_met
#SBATCH --partition=a100-public
#SBATCH --account=mendelson_prj
#SBATCH --qos=4h_0g
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/home/USER/tempo/logs/metrics_%j.out
#SBATCH --error=/home/USER/tempo/logs/metrics_%j.out
# CPU-only scoring (0 GPU-h): official temporal accuracy + CLIP/contact sheets for every run tag of a phase,
# then the per-arm summary.  Usage: sbatch --export=ALL,PHASE=1,IDS_FILE=data/pilot_ids.txt,REF=B0_s42[,SEEDCTL=B0_s43] score_phase_cpu.sbatch
set -uo pipefail
T=$HOME/tempo
module load python/3.11.15-x86-7sp4zgy
export HF_HOME=$T/cache/hf
cd $T
IDS=$(cat $IDS_FILE)
TAGS=$(cat results/rows/phase${PHASE}_*.jsonl | $T/venv/bin/python -c "import sys,json;print(','.join(sorted({json.loads(l)['run_tag'] for l in sys.stdin})))")
echo "=== metrics phase $PHASE ids $IDS tags $TAGS $(date -Is) ==="
EXP=$($T/venv/bin/python -c "import sys;sys.path.insert(0,'scripts');from generate_benchmark_videos import parse_ids;print(','.join(map(str,parse_ids('$IDS'))))")
for t in ${TAGS//,/ }; do
  $T/venv_metric/bin/python scripts/run_official_temporal_accuracy.py --videos videos/$t --ids $EXP \
      --out results/metric/phase$PHASE/$t || echo "### METRIC FAILED $t"
done
$T/venv/bin/python scripts/clip_similarity_and_contact_sheets.py --tags $TAGS --ids $IDS --phase $PHASE || echo "### QUALITY FAILED"
$T/venv/bin/python scripts/paired_accuracy_analysis.py --phase $PHASE --ids $IDS --ref ${REF:-B0_s42} ${SEEDCTL:+--seedctl $SEEDCTL}
echo "### rc=$? $(date -Is)"
