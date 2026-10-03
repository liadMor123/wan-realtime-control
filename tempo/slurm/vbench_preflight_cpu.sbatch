#!/bin/bash
#SBATCH --job-name=tempo_s6pf
#SBATCH --partition=a100-public
#SBATCH --account=mendelson_prj
#SBATCH --qos=4h_0g
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=/home/USER/tempo/logs/step6_preflight_%j.out
#SBATCH --error=/home/USER/tempo/logs/step6_preflight_%j.out
# Step 6 preflight, CPU only (0 GPU-hours), offline exactly as the GPU job (dead proxy: any download attempt fails):
#  1. run_vbench.py on 2 real videos (B0_s42 ids 0,1), then again to exercise the per-tag resume;
#  2. synthetic videos: static mid-gray (subject_consistency should be ~1) and per-frame uniform noise (should be low);
#  3. summarize_vbench_quality.py on dummy score files (80 + 20 ids) to exercise the statistics and the Q6 verdict code;
#  -> results/vbench/preflight.json
set -uo pipefail
T=$HOME/tempo
module load python/3.11.15-x86-7sp4zgy
export VBENCH_CACHE_DIR=$T/cache/vbench TORCH_HOME=$T/cache/vbench/torch HF_HOME=$T/cache/vbench/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
export http_proxy=http://127.0.0.1:9 https_proxy=http://127.0.0.1:9 HTTP_PROXY=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9
cd $T
echo "=== step6 preflight on $(hostname) job ${SLURM_JOB_ID:-none} $(date -Is) ==="
P=$T/venv_vbench/bin/python
PF=$T/results/vbench/preflight_work; mkdir -p $PF; rm -rf $PF/*.parts
SYN=${TMPDIR:-/tmp}/vbsyn
set -e
t0=$(date +%s)
$P scripts/run_vbench.py --device cpu --tags B0_s42 --ids 0,1 --out $PF/real.json
t1=$(date +%s)
cp $PF/real.json $PF/real_first.json
$P scripts/run_vbench.py --device cpu --tags B0_s42 --ids 0,1 --out $PF/real.json | tee $PF/resume.log
grep -q "reusing" $PF/resume.log
cmp <($P -c "import json;print(json.load(open('$PF/real_first.json'))['tags'])") \
    <($P -c "import json;print(json.load(open('$PF/real.json'))['tags'])")
# diagnostic for interpreting subject_consistency: DINO (VBench's model + transform) adjacent vs first-frame similarity
$P - <<'EOF'
import os, sys, json
import torch, torch.nn.functional as F
sys.path.insert(0, os.path.expanduser("~/tempo/src"))
from tempo_ctrl import benchmark
from vbench.utils import load_video, dino_transform, init_submodules
sub = init_submodules(["subject_consistency"], local=True)["subject_consistency"]
sub = {k: v for k, v in sub.items() if k != "read_frame"}
m = torch.hub.load(**sub).eval()
P = benchmark.load_one_object()
out = {}
for i in (0, 1):
    x = dino_transform(224)(load_video(os.path.expanduser(f"~/tempo/videos/B0_s42/{benchmark.video_name(P[i]['prompt'])}")))
    with torch.no_grad():
        f = F.normalize(m(x), dim=-1)
    adj = (f[1:] * f[:-1]).sum(-1).clamp(min=0); fir = (f[1:] * f[:1]).sum(-1).clamp(min=0)
    out[i] = {"prompt": P[i]["prompt"], "mean_adjacent": adj.mean().item(), "min_adjacent": adj.min().item(),
              "mean_to_first": fir.mean().item(), "vbench_style": ((adj + fir) / 2).mean().item()}
json.dump(out, open(os.path.expanduser("~/tempo/results/vbench/preflight_work/dino_diag.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
EOF
$P - "$SYN" <<'EOF'
import os, sys
import cv2, numpy as np
sys.path.insert(0, os.path.expanduser("~/tempo/src"))
from tempo_ctrl import benchmark
name = benchmark.video_name(benchmark.load_one_object()[0]["prompt"])
rng = np.random.default_rng(0)
for tag, gen in [("SYN_gray", lambda t: np.full((480, 832, 3), 128, np.uint8)),
                 ("SYN_noise", lambda t: rng.integers(0, 256, (480, 832, 3), dtype=np.uint8))]:
    d = os.path.join(sys.argv[1], tag); os.makedirs(d, exist_ok=True)
    w = cv2.VideoWriter(os.path.join(d, name), cv2.VideoWriter_fourcc(*"mp4v"), 16, (832, 480))
    assert w.isOpened()
    for t in range(81):
        w.write(gen(t))
    w.release()
    print("wrote", tag, os.path.getsize(os.path.join(d, name)))
EOF
$P scripts/run_vbench.py --device cpu --videos_root $SYN --tags SYN_gray,SYN_noise --ids 0 --out $PF/syn.json
$P - "$PF" <<'EOF'
import json, os, sys
import numpy as np
pf = sys.argv[1]
rng = np.random.default_rng(1)
sys.path.insert(0, os.path.expanduser("~/tempo/scripts"))
from run_vbench import parse_ids
ids, held = list(range(80)), parse_ids("@" + os.path.expanduser("~/tempo/data/heldout_ids.txt"))
assert len(held) == 20
def tag(ids, base, shift):
    return {"per_video": {str(i): {"imaging_quality": base[i] + shift, "subject_consistency": 0.95 + 0.01 * rng.standard_normal()} for i in ids}}
base = 0.6 + 0.03 * rng.standard_normal(80)
meta = json.load(open(f"{pf}/real.json"))["meta"]
json.dump({"meta": meta, "tags": {"B0_s42": tag(ids, base, 0.0), "L_b2g2_s42": tag(ids, base, -0.02)}},
          open(f"{pf}/dummy80.json", "w"))
json.dump({"meta": meta, "tags": {"B0_s43": tag(held, base + 0.01 * rng.standard_normal(80), 0.0)}},
          open(f"{pf}/dummy43.json", "w"))
# a mismatched-meta pair must be rejected by summarize_vbench_quality.py
json.dump({"meta": dict(meta, imaging_quality_preprocessing_mode="shorter"),
           "tags": {"B0_s43": tag(held, base, 0.0)}}, open(f"{pf}/dummy43_badmeta.json", "w"))
EOF
$P scripts/summarize_vbench_quality.py --all80 $PF/dummy80.json --seed43 $PF/dummy43.json --name vbench/preflight_work/summary_dummy
if $P scripts/summarize_vbench_quality.py --all80 $PF/dummy80.json --seed43 $PF/dummy43_badmeta.json \
    --name vbench/preflight_work/summary_badmeta > $PF/badmeta.log 2>&1; then echo "### bad meta accepted"; exit 1; fi
grep -q "inputs not comparable" $PF/badmeta.log
set +e
$P - "$PF" "$((t1 - t0))" <<'EOF'
import json, os, sys
pf, t_real = sys.argv[1], int(sys.argv[2])
real, syn = json.load(open(f"{pf}/real.json")), json.load(open(f"{pf}/syn.json"))
dummy = json.load(open(f"{pf}/summary_dummy.json"))
g = syn["tags"]["SYN_gray"]["per_video"]["0"]; n = syn["tags"]["SYN_noise"]["per_video"]["0"]
checks = {"real_finite": all(0 < v < 1.0001 for r in real["tags"]["B0_s42"]["per_video"].values() for v in r.values()),
          "gray_subject_consistency_ge_0.99": g["subject_consistency"] >= 0.99,
          "noise_subject_consistency_lt_gray": n["subject_consistency"] < g["subject_consistency"],
          "dummy_iq_delta_negative_detected": dummy["dims"]["imaging_quality"]["paired_L_minus_B0_all80"]["ci95"][1] < 0,
          "dummy_iq_q6_not_within": not dummy["dims"]["imaging_quality"]["q6"]["within_seed_spread"],
          "mismatched_meta_rejected": True,
          "id_mapping_matches_independent_dino": all(
              abs(json.load(open(f"{pf}/dino_diag.json"))[str(i)]["vbench_style"]
                  - real["tags"]["B0_s42"]["per_video"][str(i)]["subject_consistency"]) < 1e-4 for i in (0, 1))}
out = {"job": os.environ.get("SLURM_JOB_ID"), "host": os.uname().nodename, "device": "cpu", "offline": True,
       "meta": real["meta"], "real_B0_s42_ids_0_1": real["tags"]["B0_s42"],
       "cpu_seconds_2_videos_incl_model_load": t_real, "resume_reused_identical": True,
       "dino_diag": json.load(open(f"{pf}/dino_diag.json")),
       "synthetic": {"SYN_gray": g, "SYN_noise": n}, "dummy_summary_q6": {d: v["q6"] for d, v in dummy["dims"].items()},
       "dummy_summary_iq_paired": dummy["dims"]["imaging_quality"]["paired_L_minus_B0_all80"],
       "checks": checks, "pass": all(checks.values())}
json.dump(out, open(os.path.expanduser("~/tempo/results/vbench/preflight.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
sys.exit(0 if out["pass"] else 1)
EOF
RC=$?; echo "### preflight rc=$RC $(date -Is)"; exit $RC
