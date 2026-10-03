#!/bin/bash
#SBATCH --job-name=j8
#SBATCH --partition=a100-public
#SBATCH --account=mendelson_prj
#SBATCH --qos=12h_4g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/USER/proj/logs/j8-%j.out
#SBATCH --error=/home/USER/proj/logs/j8-%j.out
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
echo "=== J8 on $(hostname) job $SLURM_JOB_ID; patch @ $(git rev-parse --short HEAD) ==="

CLK="$RUN/out/j8_clocks.csv"
echo "timestamp,clocks_sm_mhz,clocks_mem_mhz,power_w,temp_c" > "$CLK"
( while true; do nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu \
   --format=csv,noheader,nounits >> "$CLK" 2>/dev/null || true; sleep 1; done ) & CLKPID=$!
trap 'kill $CLKPID 2>/dev/null || true' EXIT

# non-graph modes interleave; every graph mode runs alone (graph replay after an
# unrelated eager generation disturbed the caching allocator in earlier runs)
echo; echo "########## A: eager-original | final_no_graphs (interleaved) ##########"
$PY mode_latency_benchmark.py --modes eager-original,final_no_graphs --perf per-mode \
   --out "$RUN/out/A" --jsonl "$PROJ/results/runs.jsonl" --n_videos 5 2>&1 | grep -E "^\[J3b\]"
for M in final final_no_perf final_no_split final_no_inductor final_rule; do
  echo; echo "########## $M (alone) ##########"
  $PY mode_latency_benchmark.py --modes "$M" --out "$RUN/out/$M" \
     --jsonl "$PROJ/results/runs.jsonl" --n_videos 5 2>&1 | grep -E "^\[J3b\]|FAILED"
done

echo; echo "########## 15.7 frame-level bridge: final at nfpb=1 ##########"
export J8_NFPB1="$RUN/out/j8_nfpb1.json"
$PY - <<'PYN' 2>&1 | grep -vE "^W09|^E09|AUTOTUNE|triton_|^  mm |^  addmm |SingleProcess|convolution|bias_addmm"
import contextlib, io, json, math, os, statistics as st, sys, time
import numpy as np, torch
sys.path.insert(0, os.getcwd())
from omegaconf import OmegaConf
from pipeline import CausalInferencePipeline
from demo_utils.memory import gpu, DynamicSwapInstaller
from utils.misc import set_seed
from wan import patch_flags
from wan.modules.model import prebuild_sinusoid_cache
import torch._dynamo as dynamo
dev=torch.device("cuda"); torch.set_grad_enabled(False)
patch_flags.set_enabled(True); patch_flags.set_perf(True); patch_flags.set_attn_splits(4)
dynamo.config.cache_size_limit=256; dynamo.config.accumulated_cache_size_limit=1024
cfg=OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"),
                    OmegaConf.load("configs/self_forcing_dmd.yaml"))
cfg.num_frame_per_block=1
pipe=CausalInferencePipeline(cfg, device=dev)
sd=torch.load("checkpoints/self_forcing_dmd.pt", map_location="cpu")
pipe.generator.load_state_dict(sd["generator_ema"]); del sd
pipe=pipe.to(dtype=torch.bfloat16)
DynamicSwapInstaller.install_model(pipe.text_encoder, device=gpu)
pipe.generator.to(device=gpu); pipe.vae.to(device=gpu); prebuild_sinusoid_cache(256, dev)
pipe.generator.model=torch.compile(pipe.generator.model, mode="max-autotune-no-cudagraphs", dynamic=False)
first=float(pipe.denoising_step_list.tolist()[0]); timing={"c":-1,"i":0,"t0":None,"tc":[]}
original_forward=pipe.generator.forward
def timed_forward(*a,**kw):
    ts=kw.get("timestep"); tv=float(ts.flatten()[0].item()) if ts is not None else float("nan")
    if tv==first: timing["c"]+=1; timing["i"]=0
    if timing["t0"] is None: torch.cuda.synchronize(); timing["t0"]=time.perf_counter()
    timing["i"]+=1; o=original_forward(*a,**kw)
    if timing["i"]==5: torch.cuda.synchronize(); timing["tc"].append(time.perf_counter())
    return o
pipe.generator.forward=timed_forward
with open("prompts/MovieGenVideoBench_extended.txt",encoding="utf-8") as f:
    prompts=[f.readline().rstrip() for _ in range(3)]
def one_video(p,seed):
    set_seed(seed); timing.update({"c":-1,"i":0,"t0":None,"tc":[]})
    noise=torch.randn([1,21,16,60,104],device=dev,dtype=torch.bfloat16)
    with contextlib.redirect_stdout(io.StringIO()):
        pipe.inference(noise=noise,text_prompts=[p],return_latents=True,initial_latent=None,low_memory=True)
    torch.cuda.synchronize(); out=[];prev=timing["t0"]
    for t in timing["tc"]: out.append((t-prev)*1e3); prev=t
    return out
one_video(prompts[0],999)
vids=[one_video(prompts[i],1000+i) for i in range(3)]
n=min(len(v) for v in vids); med=[st.median([v[i] for v in vids]) for i in range(n)]
K=[(i+1)*1560 for i in range(n)]
A=np.vstack([np.ones(n),np.array(K,float)]).T
(a_,b_),*_=np.linalg.lstsq(A,np.array(med,float),rcond=None)
tiles=math.ceil(1560/128); cta=tiles*12
print(f"[nfpb1] chunks={n} a={a_:.2f} ms b={b_*1000:.4f} ms/1k-K")
print(f"[nfpb1] CTA/wave: Q=1560 tiles={tiles} CTAs={cta} waves={cta/108:.2f} launched={math.ceil(cta/108)} "
      f"waste={100*(math.ceil(cta/108)-cta/108)/math.ceil(cta/108):.1f}%")
json.dump({"chunks":n,"median_ms":med,"K":K,"a_ms":float(a_),"b_ms_per_1k_K":float(b_*1000),
           "ctas":cta,"waves":cta/108,"launched":math.ceil(cta/108)},
          open(os.environ.get("J8_NFPB1","j8_nfpb1.json"),"w"), indent=2)
PYN

kill $CLKPID 2>/dev/null || true
echo; echo "########## clocks (active only) ##########"
$PY - "$CLK" <<'PYC'
import csv,sys,statistics as st
rows=[r for r in csv.reader(open(sys.argv[1])) if r and not r[0].startswith("timestamp")]
def v(r,i):
    try: return float(r[i])
    except Exception: return None
act=[r for r in rows if (v(r,3) or 0)>=100.0]
print(f"  {len(rows)} samples, {len(act)} active")
for name,i in (("clocks.sm MHz",1),("power W",3),("temp C",4)):
    x=[v(r,i) for r in act if v(r,i) is not None]
    if x:
        rng=100*(max(x)-min(x))/st.median(x) if st.median(x) else 0
        print(f"  {name:15s} min={min(x):7.1f} median={st.median(x):7.1f} max={max(x):7.1f} range={rng:5.2f}%")
PYC
for d in A final final_no_perf final_no_split final_no_inductor final_rule; do
  cp -f "$RUN/out/$d/j3b.json" "$RUN/out/j8_$d.json" 2>/dev/null || true
done
cp -f "$RUN/out"/j8_*.json "$CLK" "$PROJ/results/" 2>/dev/null || true
echo; echo "=== J8 done ==="
