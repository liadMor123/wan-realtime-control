#!/bin/bash
#SBATCH --job-name=tempo_s4c
#SBATCH --partition=a100-public
#SBATCH --account=mendelson_prj
#SBATCH --qos=4h_0g
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/home/USER/tempo/logs/step4_score_%j.out
#SBATCH --error=/home/USER/tempo/logs/step4_score_%j.out
# Part 2 step 4 scoring on CPU (0 GPU-h): official one-object metric (unmodified) on all 80 videos of B0_s42 and
# L_b2g2_s42 into results/metric/phase4/<tag>, a check that part 1's 28 scores reproduce, CLIP, then summarize_wan_all80_accuracy.py.
set -uo pipefail
T=$HOME/tempo
module load python/3.11.15-x86-7sp4zgy
export HF_HOME=$T/cache/hf OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd $T
ALL=$($T/venv/bin/python -c "import sys;sys.path.insert(0,'scripts');from generate_benchmark_videos import parse_ids;print(','.join(map(str,parse_ids('0-79'))))")
echo "=== step4 scoring on $(hostname) job $SLURM_JOB_ID commit $(git rev-parse --short HEAD) $(date -Is) ==="
FAIL=0; PIDS=""
for t in B0_s42 L_b2g2_s42; do
  n=$(ls videos/$t/*.mp4 | wc -l); echo "videos/$t: $n mp4"; [ "$n" -eq 80 ] || FAIL=1
  $T/venv_metric/bin/python scripts/run_official_temporal_accuracy.py --videos videos/$t --ids $ALL \
      --out results/metric/phase4/$t > logs/step4_metric_$t.log 2>&1 &
  PIDS="$PIDS $!"
done
for p in $PIDS; do wait $p || FAIL=1; done
[ $FAIL -eq 0 ] || { echo "### METRIC FAILED or videos missing"; exit 1; }
$T/venv/bin/python - <<'PYC' || { echo "### PART-1 METRIC DID NOT REPRODUCE"; exit 1; }
import json, os
def per(p):
    return {os.path.basename(v["video_path"]): v["video_results"] for v in json.load(open(p))["temporal_accuracy"][1]}
for t in ("B0_s42", "L_b2g2_s42"):
    new = per(f"results/metric/phase4/{t}/temporal_accuracy_one_object.json")
    n = 0
    for ph in ("phase1", "phase2"):
        p = f"results/metric/{ph}/{t}/temporal_accuracy_one_object.json"
        if os.path.isfile(p):
            old = per(p)
            assert all(new[k] == v for k, v in old.items()), (t, ph)
            n += len(old)
    assert n == 28, (t, n)                                     # 8 pilot (phase 1) + 20 held-out (phase 2), all CPU-scored
    print(t, "part-1 metric reproduced on", n, "videos")
PYC
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
$T/venv/bin/python scripts/clip_similarity_and_contact_sheets.py --tags B0_s42,L_b2g2_s42 --ids 0-79 --phase 4 || { echo "### QUALITY FAILED"; exit 1; }
$T/venv/bin/python scripts/summarize_wan_all80_accuracy.py
RC=$?; echo "### rc=$RC $(date -Is)"; exit $RC
