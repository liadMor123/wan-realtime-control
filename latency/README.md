# Part A: latency attribution of Self-Forcing (Wan2.1-1.3B) on one A100

This directory holds the code, Slurm jobs, raw results and figures of Part A of the
project: where a streaming, chunk-wise autoregressive video DiT (Self-Forcing, built on
Wan2.1-T2V-1.3B) spends its time on a single NVIDIA A100-SXM4-40GB, and how much of
that time conventional interventions recover. The study proceeds by controlled
intervention over thirteen pre-registered jobs (J1-J13): a reproduction and memory
probe, a frozen per-position timing harness, graph-mode baselines (hand-captured CUDA
graphs, `torch.compile` with and without Inductor cudagraphs), a capture-enabling patch
accepted only if byte-identical, a kernel census, `ncu`/`nsys` attribution of the
FlashAttention-2 wave-quantization tail, an off-the-shelf split-KV attention arm with a
position-dependent split rule, a leave-one-out waterfall of the integrated
configuration, a descriptive quality check, and a Triton kernel that fuses the split-KV
merge with the output projection. Every threshold was frozen before the job that tested
it; every number in the report traces to a file in `results/`.

## Layout

```
src/harness/    timing harness, census, profiling targets, attention benchmarks, quality jobs
src/kernels/    the J13 Triton kernel, its correctness test and its ncu target
src/plotting/   regenerates figures/fig1-fig8 from results/*.json (no GPU)
scripts/        the exact Slurm jobs that produced every number, plus small helper scripts
results/        per-job JSON, nvidia-smi clock traces, ncu CSVs
figures/        fig1_*.png ... fig8_*.png and the J3 frame grids
```

The harness scripts are copied next to a staged checkout of Self-Forcing and run from
its root (they import `pipeline`, `wan`, `utils`, `demo_utils` from the current
directory). Cluster home paths appear as `/home/USER`.

## Scripts and modules

### `src/harness/`

| file | purpose | writes |
|---|---|---|
| `environment_memory_probe.py` | J1. Reproduces `inference.py` exactly, instruments generator / text encoder / VAE, records the five pre-registered memory points, passes per chunk and the latent-to-frame mapping; gates on the 36 GB steady-state ceiling | one row in `runs.jsonl`, `--memjson`, one MP4 |
| `measure_per_position_latency.py` | J2. The frozen per-position timing protocol on eager Self-Forcing: 1 warmup + 5 measured videos (`--mode sweep`), or one fixed-seed video dumped for the nondeterminism envelope (`--mode envelope`); gates on chunk-6 spread < 3 % | `j2_sweep.json`, `envelope_<tag>.npy`, `runs.jsonl` row |
| `graph_mode_baselines.py` | J3. Same protocol under `eager_ref`, `manual_graph` (one CUDA graph per chunk/pass with two numerically identical capture shims), `compile_nocg`, `compile_cg`; capture proof by kernel-launch count; linear fit `t = a + b*K` | `j3_<mode>.json`, optional uint8 video, `runs.jsonl` row |
| `mode_latency_benchmark.py` | J3b / J6 / J7 / J8 / J11. The main harness on the patched model: any set of modes (`eager-original`, `eager-patched`, `*-perf`, `manual_graph`, `manual_graph_compiled`, `compile_nocg`, `compile_cg`, `final` and its leave-one-out ablations `final_no_*`, `final_rule`), interleaved by video; `--attn_splits` routes self-attention through `flash_attn_with_kvcache` | `<out>/j3b.json`, `<out>/vid_<mode>.npy` |
| `kernel_census.py` | J5 + J4. Profiles one generator pass per position under `compile_nocg`, assigns every CUDA kernel to a pipeline stage, reconciles against the J3b fit; checks KV-cache index, attention K length and zero region at every position | `j54.json` |
| `profile_single_pass.py` | Profiler target: advances the cache to a position, then runs N chunks inside `cudaProfilerStart/Stop` (optional NVTX per pass) for `ncu` / `nsys` | nothing (the Slurm job collects the trace) |
| `triton_fence_and_kv_fidelity.py` | J6 (f). Triton cross-program release/acquire test (gate item 6); KV-cache prefix stability across chunks; FA2 vs dense fp32 attention on the real cache contents | `j6_extras.json` |
| `fa2_tail_head_sweep.py` | J6 (13). FA2 kernel time vs head count H = 9..24 at positions 3 and 6: the wave-quantization staircase | `j6_fa2sweep.json` |
| `token_count_sweep.py` | J6 (e). Chunk time vs K for `num_frame_per_block` in {1, 2, 3}, compared at matched K | `j6_sweep.json` |
| `splitkv_attention_benchmark.py` | J7 (a)+(b). `flash_attn_with_kvcache` with `num_splits` in {1, 2, 4, 8, heuristic} vs the varlen call at the real shapes: speed-up, combine share, partial memory, bitwise / fp32-band agreement; head staircase at the best split; query-length probe | `j7_attn.json` (J7a), `j7_attn2.json` (J7b) |
| `splitkv_position_rule.py` | J10. Measures attention time for s = 1..8 at all seven positions, extracts combine(s) and the wave-quantization floor, and derives the closed-form rule s(pos) used by `final_rule` | `j10_rule.json` |
| `quality_generate_videos.py` | J9. Generates the ten quality-check videos of one arm (`eager_a`, `final`, `eager_b` = seed control) | `<arm>_p<i>.npy`, `j9_prompts.json` |
| `quality_metrics.py` | J9. CLIP prompt similarity, CLIP temporal consistency, LPIPS between consecutive frames, 4x4 contact sheets; descriptive only | `j9_metrics.json`, `j9_<arm>_p<i>.png` |

### `src/kernels/`

| file | purpose | writes |
|---|---|---|
| `fused_merge_outproj.py` | J13. Triton kernel fusing the FA2 split-KV merge (`exp(lse_s - LSE)`-weighted sum of fp32 partials, rounded to bf16 at FA2's rounding point) with the out-projection GEMM; `merge_partials_fp32` is the fp32 reference | library module |
| `fused_merge_outproj_test.py` | J13 Step 1/2. Correctness vs an fp32 merge + fp32 GEMM within 2x the PyTorch-bf16 band at all positions and splits; `--bench` times unfused vs fused and each component alone | `--result` JSON, optional `runs.jsonl` row |
| `fused_merge_outproj_ncu_target.py` | One unfused and one fused launch pair inside a profiler range at position-6 shapes, for `ncu` | nothing |

The J13 scripts need a flash-attn 2.8.3 build patched to expose `return_partials`
(the partial outputs and log-sum-exps of the split-KV kernel without the combine); the
patch is not part of this repository.

### `src/plotting/`

| file | purpose | writes |
|---|---|---|
| `make_figures.py` | Regenerates every Part-A figure from `results/*.json` | `figures/fig1_*.png` ... `fig8_*.png` |

### `scripts/` (helper scripts)

| file | purpose | writes |
|---|---|---|
| `install_flash_attn.py` | Resolves and installs the prebuilt flash-attn wheel matching the installed torch / Python / ABI from the GitHub release assets; fails loudly otherwise | nothing |
| `frame_psnr.py` | Per-frame PSNR between two uint8 `.npy` videos, counting bitwise-identical frames separately | JSON given on the command line |
| `cuda_graph_capture_probe.py` | J3 probe: eager kernel launches per pass, one `CUDAGraph` capture attempt on the unpatched model, `.item()` sites on the inference path | `j3_probe.json` |
| `diagnose_compile_and_capture.py` | J3 diagnostics: single-pass eager vs compiled numerics (`--part numerics`); capture attempt with shims and list of capture-hostile constructs (`--part capture`) | `j3_diag_numerics.json`, `j3_diag_capture.json` |
| `compare_eager_compiled_frames.py` | Frame grids of one video eager and under `compile_nocg`, plus their PSNR | `j3_frames_eager.png`, `j3_frames_compile_nocg.png` |
| `patch_acceptance_bitwise.py` | Tier-1 acceptance of the capture-enabling patch: `eager-patched` byte-identical to `eager-original` on five prompts | `j3b_accept.json` |
| `fp32_band_numerics.py` | Tier-2 numerics: single-pass error of `compile_nocg` vs an fp32 eager reference, judged against 2x the eager-bf16 band | `j3b_fp32.json` |
| `cache_write_acceptance_bitwise.py` | Tier-1 acceptance of the cache-write (perf) intervention: perf-patched byte-identical to eager-patched on five prompts | `j6c_accept.json` |

### `scripts/` (Slurm jobs, in the order they were run)

| job file | job | what it runs |
|---|---|---|
| `environment_memory_probe.sbatch` | J1 | builds the pinned venv, downloads weights, runs `environment_memory_probe.py` with text-encoder offload off and on |
| `per_position_latency.sbatch` | J2 | `measure_per_position_latency.py` sweep and two envelope runs, `frame_psnr.py`, 1 Hz clock/power trace |
| `cuda_graph_capture_probe.sbatch` | J3 probe | `cuda_graph_capture_probe.py` |
| `diagnose_compile_and_capture.sbatch` | J3 diag | `diagnose_compile_and_capture.py` (both parts), `compare_eager_compiled_frames.py` |
| `graph_mode_baselines.sbatch` | J3 | `graph_mode_baselines.py` in all four modes, PSNR of each vs the eager reference |
| `patch_acceptance.sbatch` | J3b accept | `patch_acceptance_bitwise.py` |
| `interleaved_modes_and_numerics.sbatch` | J3b | `mode_latency_benchmark.py` in four process groups (interleaved eager/compile group; each graph mode alone), Tier-1 hashes, `fp32_band_numerics.py` |
| `kernel_census.sbatch` | J5 + J4 | `kernel_census.py` at positions 3 and 6 with the `compile_cg` retry |
| `ncu_nsys_profiling.sbatch` | J6a | `triton_fence_and_kv_fidelity.py`; `nsys` inter-kernel gaps over 3 chunks (reduced in-job to `j6_gaps.json`); `ncu --set full` on the critical kernels at positions 3 and 6; `compile_cg` timing |
| `ncu_capture_range_rerun.sbatch` | J6a2 | `ncu` rerun with `--profile-from-start off` so the profiler range is honoured (`results/ncu2/`) |
| `cache_write_intervention.sbatch` | J6c | `cache_write_acceptance_bitwise.py`; eager-patched vs eager-patched-perf A/B; `compile_nocg` with perf patches |
| `fa2_tail_and_token_sweep.sbatch` | J6d | `fa2_tail_head_sweep.py`, `compile_cg` timing, `token_count_sweep.py` |
| `splitkv_attention_arms.sbatch` | J7a | `compile_cg` with fix 7b; `splitkv_attention_benchmark.py` (num_splits sweep, fp32 band) |
| `splitkv_integration.sbatch` | J7b | `splitkv_attention_benchmark.py` (staircase and Q probe); `compile_nocg` with `--attn_splits 4` and the `--attn_splits 0` control |
| `final_ablation_waterfall.sbatch` | J8 | `final` and every leave-one-out ablation, each graph mode alone; the `num_frame_per_block = 1` bridge |
| `quality_check.sbatch` | J9 | `quality_generate_videos.py` for the three arms, `quality_metrics.py` |
| `splitkv_position_rule.sbatch` | J10 | `splitkv_position_rule.py` |
| `compile_cg_mark_step_retry.sbatch` | J11 | one bounded attempt at `compile_cg` with `torch.compiler.cudagraph_mark_step_begin()` before each pass |

J13 (`src/kernels/`) was run from an ad-hoc allocation with `PYTHONPATH` pointing at the
patched flash-attn build; no `.sbatch` is kept for it.

## Regenerating the figures

```
python src/plotting/make_figures.py
```

No GPU is needed; the script reads `results/*.json` and writes `figures/fig1_*.png`
through `fig8_*.png`. The GPU jobs themselves expect a Slurm cluster with
A100-SXM4-40GB nodes and the environment pinned below.

## Environment

| component | pin |
|---|---|
| GPU | NVIDIA A100-SXM4-40GB (108 SMs), one per job |
| PyTorch | 2.7.1+cu126 (torchvision 0.22.1) |
| Triton | 3.3.1 |
| flash-attn | 2.8.3.post1 (prebuilt wheel; J13 uses a source build of 2.8.3 with the `return_partials` patch) |
| Self-Forcing | revision `33593df3`, checkpoint `self_forcing_dmd.pt`, Wan2.1-T2V-1.3B weights |
| Python | 3.11 |

Model: Wan2.1-T2V-1.3B as distilled by Self-Forcing; 832x480, 21 latent frames in
7 chunks of `num_frame_per_block = 3`, 1560 tokens per latent frame (4680 per chunk),
5 generator passes per chunk (4 denoising steps plus the clean-context cache-update
pass), history window 32,760 tokens, bf16, text encoder offloaded. The configuration,
shapes and step list are never changed by any job.

## Protocols

**Per-position timing.** `t_gen` is a host `perf_counter()` taken after
`torch.cuda.synchronize()` immediately before the first generator pass of a video
(after text encoding, so neither it nor the model move is charged to chunk 0).
`t_chunk[i]` is a synchronized host timestamp immediately after the 5th pass of chunk i
returns. Chunk time `T[i]` is the difference of consecutive boundaries; per-pass times
are CUDA-event intervals read only after generation ends. Exactly one synchronization
per chunk boundary. The VAE decode (one call at the end) is timed as its own stage and
never included in a chunk. Each job runs 1 warmup video (excluded) and 5 measured videos
(first five prompts of `MovieGenVideoBench_extended.txt`, seed `1000 + i`); statistics
are per chunk position 0-6 (median, min, max, IQR, `spread = 100*(max-min)/median`),
with the full curve always reported and positions 3 and 6 as headlines. Chunk time is
fitted as `t = a + b*K` over the history length K to separate K-independent cost (a)
from the per-token attention cost (b). Modes that compare against each other are
interleaved by video within one process; CUDA-graph modes run alone because graph
replay after an unrelated eager generation disturbed the allocator.

**Three-tier numerics.** Tier 1, bitwise identity (sha256 of the uint8 video on five
prompts), required wherever identity is expected: the capture-enabling patch vs the
original, manual graphs vs the eager kernels they wrap, the cache-write intervention,
and `num_splits = 1` vs the varlen call. A Tier-1 failure rejects the mode. Tier 2, for
kernel-changing modes (compilation, split-KV, the fused kernel): one pass on identical
inputs and cache state in fp32 eager is the reference; the eager-bf16 error vs that
reference is the model's inherent band, measured in the same job; a mode passes if its
own error is within 2x the band on both `max_abs` and `mean_abs`. Tier 3, descriptive
only and never a gate: ten prompts, CLIP prompt similarity, CLIP temporal consistency
and LPIPS, with a seed change as the control for how far these metrics move on their
own.

## Results files

| file | produced by | content |
|---|---|---|
| `results/j2_sweep.json`, `j2_envelope.json`, `j2_clocks.csv` | J2 | eager per-position statistics; PSNR of two identical runs (nondeterminism envelope); 1 Hz SM/memory clocks, power, temperature |
| `results/j3_eager_ref.json`, `j3_manual_graph.json`, `j3_compile_nocg.json`, `j3_compile_cg.json`, `j3_psnr_compile_nocg.json` | J3 | per-mode per-position statistics, fit, capture proof; PSNR vs eager |
| `results/j3_probe.json`, `j3_diag_numerics.json`, `j3_diag_capture.json` | J3 probe / diag | capture attempt on the unpatched model, launch count, numerics of compile vs eager |
| `results/j3b_A.json` ... `j3b_D.json`, `j3b_accept.json`, `j3b_tier1.json`, `j3b_fp32.json` | J3b | process groups A (eager-original, eager-patched, compile_nocg), B (manual_graph), C (compile_cg), D (manual_graph_compiled); patch acceptance; Tier-1 hashes; Tier-2 band |
| `results/j54.json` | J5 + J4 | kernel census per stage at positions 3 and 6, reconciliation, KV-cache checks, measured HBM bandwidth |
| `results/j6_extras.json`, `j6_gaps.json`, `j6_fa2sweep.json`, `j6_sweep.json`, `j6_tails.json`, `j6_tails_final.json`, `j6_compile_cg.json` | J6 | Triton fence test and cache fidelity; nsys inter-kernel gaps; FA2 head staircase; token-count sweep; per-kernel tail imbalance reduced from the ncu CSVs; compile_cg timing |
| `results/j6c_accept.json`, `j6c_eager.json`, `j6c_compile_perf.json` | J6c | cache-write acceptance; eager A/B with the perf patches; compile_nocg with perf patches |
| `results/ncu/*.csv`, `results/ncu2/*.csv` | J6a / J6a2 | `ncu --set full` metrics, one file per (position, kernel regex) |
| `results/j7_attn2.json`, `j7_integrated_s4.json`, `j7_integrated_s0.json`, `j7_compile_cg.json` | J7 | split-KV benchmark (splits, correctness, staircase, Q probe); integrated compile_nocg with and without split-KV; compile_cg |
| `results/j8_A.json`, `j8_final.json`, `j8_final_no_perf.json`, `j8_final_no_split.json`, `j8_final_no_inductor.json`, `j8_final_rule.json`, `j8_nfpb1.json` | J8 | the waterfall: eager-original and final_no_graphs interleaved, then each ablation alone; the num_frame_per_block = 1 bridge |
| `results/j9_prompts.json`, `j9_metrics.json` | J9 | the ten prompts; per-video and per-arm quality metrics |
| `results/j10_rule.json` | J10 | measured attention times per (position, splits), combine(s), waves/waste, the rule and its totals |

File names keep the job prefix (`j2_`, `j3b_`, ...) because the Slurm jobs, the
harness scripts and the figure script all refer to them; the job column above maps
each prefix to the job that produced it.
