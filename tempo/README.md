# Part B: forward-only control of *when* an object appears

This directory holds the code, Slurm jobs, run configurations and result files of Part B of the project: a
forward-only method for controlling the moment an object appears in a text-to-video diffusion model, measured on
TempoControl's benchmark with TempoControl's own, unmodified metric, in Wan2.1-T2V-1.3B and in its real-time streaming
distillation Self-Forcing. The method, called arm **L** throughout the code, adds a per-latent-frame bias to the logits
of the object's text tokens in every cross-attention layer: +beta on the frames where the object should be visible and
-gamma everywhere else (beta = gamma = 2, applied in the conditional CFG branch only; the unconditional branch, weights,
prompt, sampler and step count are untouched). On the production path, "fa2kv", the bias rides inside Wan's own
FlashAttention-2 call: the query and key vectors are extended by 8 head dimensions (128 -> 136) that carry the per-frame
scalar and three bf16 constants summing to 1/scale, so the kernel adds exactly the bias to the object-token logits and
exact zeros to every other logit; no custom kernel, no extra pass, no gradient. The other arms (U, S, P) and the SDPA
mask path are pilot-stage alternatives that lost to L and fa2kv under pre-registered rules and are kept for the record.

The pre-registration documents (brief, experiment spec, decisions, findings write-ups) that the docstrings refer to as
"brief §N", "decision N", "Q1..Q6", "P3b-1", "PA", "PB" are summarised in the project report at the repository root
(`report/report.pdf`); they are not duplicated here. "Part 1" / "Part 2" and "phase" / "step" are the internal
chronology of this study (Part 1 = phases 0-2: validation, pilot, held-out; Part 2 = steps 2-6: fused path,
Self-Forcing, full benchmark, two objects, VBench) and are kept in the names of result files and run tags.

## Layout

```
src/tempo_ctrl/   the library: controller, attention paths, sampler, masks, token lookup, benchmark I/O
scripts/          generation, scoring, analysis, figures, showcase (every script is a CLI; see the table)
slurm/            the exact Slurm jobs that produced every number (A100-SXM4-40GB, Technion Athena cluster)
setup/            environment builds (three pinned virtualenvs) and the flash-attn wheel resolver
tests/            CPU unit tests (pytest)
runs/             every run configuration (JSON lists of arm / parameters / seed / path / benchmark)
data/             prompt-id lists of every stage, the showcase prompts, the detector's class list
results/          official metric JSONs, CLIP scores, one JSONL row per video, analysis JSONs, figures
audits/           independent recomputation of the final-run headline numbers from the raw files
```

Everything resolves paths from `TEMPO_ROOT` (default `~/tempo`), which on the cluster held this directory plus the
upstream checkouts in `ext/` (Wan2.1, TempoControl, Self-Forcing), the weights in `cache/`, the generated videos in
`videos/` and the virtualenvs `venv`, `venv_metric`, `venv_vbench`. Cluster home paths in the Slurm files are written
as `/home/USER`. Videos, weights, the upstream checkouts and the Self-Forcing patch series are not in this repository.

## Library: `src/tempo_ctrl/`

| module | purpose |
|---|---|
| `cross_attention_arms.py` | The controller (`CTRL`) and the explicit cross-attention path: matmul -> edit -> softmax -> matmul in fp32 on bf16-cast inputs, with arms B0 (none), U (temporal-phrase bias), L (per-frame object bias, K objects), S (exact object-share shift), P (object keys masked off-frame). `install_cross_attention_patch(model)` swaps the forward of every `WanT2VCrossAttention` and dispatches on `CTRL.path` to the flash, explicit, fused or fa2kv path. |
| `fused_attention.py` | Arm L through fused kernels: `masked_attention` (torch SDPA with a per-query-token additive mask) and `fa2kv_attention` (Wan's FlashAttention-2 call with the bias carried in 8 extra head dims). Frame/token table builders, the fp64 reference, kernel-name helpers. |
| `sampler.py` | `Sampler`: the Wan2.1-T2V-1.3B sampling loop mirroring `WanT2V.generate` line for line (832x480, 81 frames, 50 UniPC steps, shift 3, guide 6), minus T5 (cached encodings), plus `CTRL.step` / `CTRL.branch` set before every DiT forward. |
| `masks.py` | Output-frame <-> latent-frame mapping (81 -> 21), interval-to-mask, the temporal-phrase parser that reproduces every benchmark mask, the benchmark mask parser. |
| `tokens.py` | umT5 token positions of the timed object (whole-word, identical to TempoControl's substring lookup on the benchmark) and of the temporal phrase, in the 512-slot context exactly as `WanModel` sees it. |
| `benchmark.py` | Benchmark CSV loaders (one object, two objects, showcase prompts), video naming as the official metric expects, run tags (`B0_s42`, `L_b2g2_s42`, `L_b2g2_fa2kv_s42`, `B0_2obj_s42`, ...), `configure_controller(run, row, tok)`, video and JSONL writers. |

## Scripts: `scripts/`

Generation and validation (GPU):

| script | purpose | writes |
|---|---|---|
| `validate_pipeline_equivalence.py` | Phase 0: T5 text cache; our sampler on the flash path is byte-identical to `WanT2V.generate`; explicit path vs fp64 at 9 probes; L(0,0) byte-identical to B0; explicit-path overhead; S exactness; 3-step smoke test of every arm. | `cache/text_enc.pt`, `results/phase0/phase0_<job>.json`, `results/rows/phase0_<job>.jsonl`, `videos/phase0/v1..v6/` |
| `generate_benchmark_videos.py` | Benchmark videos for a run file x id list, sharded by prompt (resumable). Used by every Wan2.1 stage: pilot, held-out, full 80, two objects, fa2kv accuracy check, showcase. | `videos/<run_tag>/<prompt>-0.mp4`, `results/rows/phase<N>_<job>.jsonl` |
| `sdpa_mask_path_validation.py` | Step 2, first candidate: SDPA additive-mask path. GPU preflight vs fp64, CUDA-graph capture, kernel names, 9 fp64 probes on a real video, timing of flash / SDPA-mask / explicit on prompts 0-1. | `results/step2/step2_<job>.json`, `results/rows/phase22_<job>.jsonl`, `videos/step2/` |
| `fused_kernel_bakeoff.py` | Step 2b: SDPA mem-efficient, cuDNN, FlexAttention, FA2 key augmentation and an fp32 floor, scored at the same 9 probes and timed at Wan and Self-Forcing shapes; picks the fastest passing candidate (fa2kv). | `results/step2/step2b_<job>.json` |
| `fa2kv_path_validation.py` | Step 2 re-run with fa2kv: preflight, probes with noise and exact-rounding floors, defect gate, timing of flash B0 / fa2kv L / explicit L. | `results/step2/step2_fa2kv_<job>.json`, `results/rows/phase22_<job>.jsonl` |
| `explicit_path_repro_check.py` | Before extending Part 1's video directories: the current code regenerates prompt 2 of B0 and L bitwise. | `results/step4_repro.json`, `videos/step4_repro/` |
| `self_forcing_common.py` | Self-Forcing helpers: build the `generator_ema` pipeline, install or clear the `tempo_bias` tables on all 30 cross-attention modules, L tables for a benchmark row. Imported from a staged Self-Forcing checkout on branch `tempo-L`. | - |
| `self_forcing_generate_videos.py` | Step 3 / 3b: Self-Forcing B0 (zero tables) and L(2,2) on the fa2kv path, with tokenizer, kernel, compile and call-pattern preflights. | `videos/SF_{B0,L_b2g2}_fa2kv_s<seed>/`, `results/rows/phase23_<job>.jsonl`, `results/step3/sfgen_<job>_s<seed>.json` |
| `self_forcing_latency_harness.py` | Part A's per-position latency harness plus a `--tempo L` flag that installs static L tables before any warmup, compile or CUDA-graph capture. | `<out>/j3b.json` |
| `self_forcing_latency_harness_original.py` | The unmodified Part A harness (md5 `6c334eced726b5b1a8d5c1b33740cac8`), kept so the diff to the `--tempo` version is in the tree and the latency job can verify the copy. Not edited. | - |
| `self_forcing_prompt_switch_heldout.py` | Step 3b-2: prompt-switching baseline on the unmodified path, 20 held-out prompts ("An empty scene." until the onset chunk, then the full prompt; cross-attention cache re-computed). | `videos/SF_PS_s<seed>/`, `results/rows/phase23_<job>_ps.jsonl`, `results/step3/sfps_<job>.json` |
| `self_forcing_prompt_switch_arms.py` | Step 3c: prompt-switching arms on the fa2kv path, all 80 prompts: PS-RF (cross-attention re-cache only), PS-LongLive-style (both caches rebuilt as in LongLive v1), PS-RF+L; machinery checks M0-M5 first. | `videos/SF_PS{RF,LL,RFL}_fa2kv_s42/`, `results/rows/phase23_<job>_<mode>.jsonl`, `results/step3c/sfsw_<job>.json` |

Scoring (official metric, CLIP, VBench):

| script | purpose | writes |
|---|---|---|
| `run_official_temporal_accuracy.py` | Runs TempoControl's `temporal_accuracy.py` unmodified (one- or two-object) on a videos directory for a list of prompt ids, checks the output format, optionally records and enforces that the detector ran on CUDA; `--preflight` runs synthetic positive/negative controls. | `results/metric/<dir>/<tag>/temporal_accuracy_{one_object,two_objects}.json`, `_subset_*.csv`, `_metric_device.json` |
| `clip_similarity_and_contact_sheets.py` | Descriptive quality: open_clip ViT-L/14 prompt-frame cosine similarity over 16 frames and a 4x4 contact sheet per video. | `results/quality/<dir>/<tag>.json`, `results/contact/<tag>/p<id>.png`, `_clip_device.json` |
| `run_vbench.py` | VBench 0.1.5 `imaging_quality` and `subject_consistency` in custom-input mode, offline, resumable per tag. | `results/vbench/<name>.json` |
| `gpu_vs_cpu_metric_check.py` | Consistency gate: the same videos scored on GPU and CPU must agree per video (metric fields and CLIP); never fails, only reports. | `results/gpu_vs_cpu_metric_check.json` |

Analysis (CPU, from the files above):

| script | purpose | writes |
|---|---|---|
| `paired_accuracy_analysis.py` | Per-arm summary for one phase: mean accuracy, paired delta vs a reference tag with 10,000-resample bootstrap 95 % CI and exact sign test, absent/present success, CLIP delta, wall overhead on the same node, peak memory; optional seed-control noise floor. Called by every `summarize_*` script. | `results/<name>.json` (+ a Markdown twin) |
| `summarize_fa2kv_accuracy.py` | Step 2 accuracy check: L on fa2kv vs B0 on Wan's flash path, 20 held-out prompts, GPU-scored; agreement with the explicit path. | `results/phase24_summary.json`, `results/step2acc_summary.json` |
| `summarize_self_forcing_latency.py` | Step 3 latency: per chunk position, median chunk time of L vs unmodified Self-Forcing (`final` and `eager-patched` modes, ABBA pooled). | `results/step3/lat_<job>/summary.json` |
| `summarize_self_forcing_accuracy.py` | Step 3b: Self-Forcing B0 vs L at seeds 42 and 43 on all 80 and the 72 non-pilot prompts, prompt switching vs both, by onset timing, switch-chunk latency. | `results/step3b_*.json`, `results/step3b_summary.json` |
| `summarize_prompt_switch_arms.py` | Step 3c: every prompt-switching arm vs B0 and vs L, switch-chunk latency, substitute-object counts and corrected accuracies. | `results/step3c_*.json`, `results/step3c_summary.json` |
| `summarize_wan_all80_accuracy.py` | Step 4: Wan2.1 B0 vs L on all 80 and the 72 non-pilot prompts, calibration gate against TempoControl's baseline, Q3/Q4. | `results/step4_{all80,np72}.json`, `results/step4_summary.json` |
| `select_scorable_two_object_pairs.py` | Step 5a pair selection: the first 20 pairs whose object names are detector classes; `--dump-names` re-dumps the class list. | `data/step5a_ids.txt`, `data/yolov10x_names.json` |
| `summarize_two_objects_20_pairs.py` | Step 5a: two-object B0 vs L(2,2) with K = 2 on the 20 pairs (CPU-scored), Q5. | `results/step5a_summary.json` |
| `summarize_two_objects_all82.py` | Step 5b: all 82 pairs, GPU-scored, with the 65-scorable / 62-new / 20-of-5a readings and the GPU-vs-CPU gate. | `results/step5b_summary.json` |
| `summarize_vbench_quality.py` | Step 6: paired VBench deltas on all 80, seed spread on the 20 held-out, Q6 verdict. | `results/step6_summary.json` |
| `blinded_substitute_object_sheets.py` | Step 3b-3: blinded 16-frame sheets (random ids, hidden key) for the substitute-object count on the 20 held-out Self-Forcing videos; `unblind` joins judgements with the key. | `results/blind/` |
| `blinded_substitute_object_count.py` | Step 3c-5: the same, over all five Self-Forcing arms x 80 prompts in batches; `unblind` writes per-arm counts, agreement with 3b-3 and a case folder. | `results/blind3c/` |
| `collect_run_rows.py` | Concatenates every `results/rows/*.jsonl` into one file with a `source_file` field and an index. | `results/runs.jsonl` |

Figures and project page:

| script | purpose | writes |
|---|---|---|
| `plot_pilot_heldout_figure.py` | Part 1 figure: held-out accuracy per arm with paired CIs and the seed control, wall overhead per arm. | `results/findings_figure.png` |
| `plot_deliverable_figures.py` | The three Part B figures: accuracy (Wan, Self-Forcing with prompt switching, two objects), time and memory per attention path, Self-Forcing per-chunk cost. | `results/figures/fig_{accuracy,time_memory,sf_block_cost}.{png,pdf}` |
| `make_comparison_gifs.py` | Side-by-side B0 / L GIFs with the timing mask under each clip, chosen by a committed rule. | `results/gifs/` |
| `build_showcase_manifest.py` | Web-encoded B0 / L video pairs and per-video accuracies for the project page's showcase prompts. | `results/showcase/manifest.json`, `results/showcase/media/` |
| `make_showcase_page.py` | Static HTML page from the manifest. | `results/showcase/index.html` |

`audits/recompute_fa2kv_accuracy_summary.py` recomputes the fa2kv accuracy-check numbers from the raw metric JSONs,
rows and video files without importing any project code.

## Tests

```
pip install pytest torch numpy   # CPU is enough
TEMPO_ROOT=/path/to/this/directory python -m pytest tests/ -q
```

62 CPU tests: latent-frame mapping and the mask parser against every benchmark mask; token lookup against
TempoControl's own logic; the fused and fa2kv tables, absolute-frame indexing of 3-frame blocks, casts and failure
modes; an fp64 emulation of the fa2kv head augmentation; K-object tables for the two-object benchmark; the prompt-switch
`Switcher` on a mock pipeline; and the id lists, run files and Slurm jobs of the final runs. The tests that read the
benchmark CSVs or the umT5 tokenizer need `ext/TempoControl` and `ext/Wan2.1` under `TEMPO_ROOT` (28 of the 62);
the other 34 run anywhere.

## Running the benchmark

All jobs are `sbatch slurm/<job>.sbatch` from `TEMPO_ROOT` after `setup/build_env.sbatch` (and
`setup/build_vbench_env.sbatch` for VBench). The Part 2 GPU scripts (and the Part 2 generation jobs) refuse to run on anything but an A100-SXM4-40GB. In the order
they were run:

| stage | job | what it runs |
|---|---|---|
| validation | `validate_pipeline.sbatch`, `metric_preflight_cpu.sbatch` | `validate_pipeline_equivalence.py`, then the metric preflight and the official metric on the six Phase-0 videos |
| pilot (8 prompts, 7 arms) | `pilot_arms.sbatch`, `pilot_schedule_variant.sbatch`, `score_phase_cpu.sbatch` | `generate_benchmark_videos.py` with `runs/phase1.json`, `runs/phase1_k10.json`; scoring + `paired_accuracy_analysis.py` |
| held-out (20 prompts, 5 runs, seeds 42/43) | `heldout_generate.sbatch`, `heldout_extension_4_prompts.sbatch`, `heldout_seed43.sbatch`, `heldout_score_cpu.sbatch` (or `_gpu`) | `runs/phase2.json`, `runs/phase2_seed43.json`; scoring, CLIP, analysis, `plot_pilot_heldout_figure.py` |
| fused path | `sdpa_mask_path_validation.sbatch`, `fused_kernel_bakeoff.sbatch`, `fa2kv_path_validation.sbatch` | the three step-2 scripts above |
| Self-Forcing | `self_forcing_generate.sbatch` (`--export SFREV=<tempo-L commit>[,IDS_FILE,SEED]`), `self_forcing_latency.sbatch`, `self_forcing_heldout_score_cpu.sbatch`, `self_forcing_all80_score_cpu.sbatch` | `self_forcing_generate_videos.py`, the latency harness pair + `summarize_self_forcing_latency.py`, scoring + `summarize_self_forcing_accuracy.py` |
| prompt switching | `self_forcing_prompt_switch_heldout.sbatch`, `self_forcing_prompt_switch_arms.sbatch` (`--export MODE=rf|ll|rfL`), `prompt_switch_arms_score_cpu.sbatch` | the two prompt-switch scripts; scoring, then the blinded count and `summarize_prompt_switch_arms.py` |
| full 80 (Wan2.1) | `explicit_path_repro_check.sbatch`, `wan_all80_generate.sbatch`, `wan_all80_score_cpu.sbatch` | repro gate, `runs/step4.json` on `data/step4_ids.txt`, scoring + `summarize_wan_all80_accuracy.py` |
| fa2kv accuracy check | `fa2kv_accuracy_generate.sbatch`, `score_gpu.sbatch` (`--export BENCH=one_object,TAGS=B0_flash_s42:L_b2g2_fa2kv_s42,IDS=data/heldout_ids.txt,OUT=phase24,RESCORE_TAGS=B0_s42:L_b2g2_s42,RESCORE_OUT=phase2_gpu,SUMMARY=scripts/summarize_fa2kv_accuracy.py`) | `runs/step2acc.json`; GPU scoring with the device check and the GPU-vs-CPU gate |
| two objects | `two_objects_20_pairs_generate.sbatch`, `two_objects_20_pairs_score_cpu.sbatch`, `two_objects_remaining_62_generate.sbatch`, `score_gpu.sbatch` (`BENCH=two_objects`, `OUT=phase5_gpu`, `SUMMARY=scripts/summarize_two_objects_all82.py`) | `runs/step5a.json` on `data/step5a_ids.txt` then `data/step5b_ids.txt`; `summarize_two_objects_20_pairs.py`, `summarize_two_objects_all82.py` |
| VBench | `vbench_preflight_cpu.sbatch`, `vbench_quality.sbatch` | `run_vbench.py` on B0/L (all 80) and B0 seed 43 (held-out), `summarize_vbench_quality.py` |
| showcase | `showcase_generate.sbatch`, `showcase_score.sbatch` | `runs/showcase*.json` on `data/showcase_prompts.csv`; GPU scoring with `--prompt-set showcase` |

Prompt-id lists (`data/*_ids.txt`, read by `generate_benchmark_videos.parse_ids`): pilot `0,1,20,21,40,41,60,61`;
held-out `2-6,22-26,42-46,62-66` (the first 16 plus the extension `6,26,46,66`); step 4 the remaining 52; two objects
`step5a_ids.txt` (first 20 scorable pairs) and `step5b_ids.txt` (the other 62).

## Environment

| component | pin |
|---|---|
| GPU | NVIDIA A100-SXM4-40GB, one per job |
| Wan2.1 | commit `9737cba`, unmodified (the cross-attention forward is replaced at runtime by `install_cross_attention_patch`) |
| TempoControl | commit `ac28c443`, data CSVs and `temporal_accuracy.py` only, invoked unmodified |
| Self-Forcing | upstream `a5deb2f` for the unmodified latency arm; branch `tempo-L` (the `tempo_bias` hook) for L |
| PyTorch | 2.7.1+cu126, torchvision 0.22.1, Python 3.11 |
| flash-attn | 2.8.3 (prebuilt wheel resolved by `setup/install_flash_attn.py`) |
| metric detector | THU-MIG YOLOv10 fork `453c6e38` with `huggingface_hub==0.23.4` (`venv_metric`) |
| CLIP | open_clip_torch 2.32.0, ViT-L-14-quickgelu / openai |
| VBench | 0.1.5 with `--no-deps`, pyiqa 0.1.10, timm 0.9.16, decord 0.6.0 (`venv_vbench`; `setup/venv_vbench_freeze.txt`) |

## Results files

| path | content |
|---|---|
| `results/metric/<dir>/<tag>/temporal_accuracy_*.json` | the official metric's own output per run tag; `<dir>` is `phase1` (pilot), `phase2` (held-out), `phase23` (Self-Forcing, all arms), `phase4` (Wan all 80), `phase5` (two objects, 20 pairs), `phase24`, `phase2_gpu`, `phase5_gpu` (GPU-scored), `phase23_step3_backup` (the step-3 scores kept before re-scoring all 80); `_subset_*.csv` is the id subset the script was given, `_metric_device.json` the CUDA record |
| `results/quality/<dir>/<tag>.json` | CLIP per video (16 frames) |
| `results/rows/phase<N>_<job>.jsonl`, `results/runs.jsonl` | one row per generated video: arm, parameters, prompt, seed, wall and denoise time, peak memory, token slots, job and host |
| `results/phase*_summary.json`, `results/step*_summary.json`, `results/step3b_*.json`, `results/step3c_*.json`, `results/step4_*.json` | outputs of the analysis scripts above (phase 1 pilot; phase 2 held-out with the seed-43 noise floor and the 16-prompt subset; phase 24 fa2kv check; steps 2acc, 3, 3b, 3c, 4, 5a, 5b, 6) |
| `results/step3c/sfsw_<job>.json` | prompt-switch arms: machinery checks, per-video chunk timings, switch-chunk latency |
| `results/step4_repro.json`, `results/gpu_vs_cpu_metric_check.json` | the repro gate and the GPU-vs-CPU gate (80/80 videos identical) |
| `results/phase0/` | Phase-0 report and the recorded per-layer attention shares of prompt 0 |
| `results/figures/` | the three deliverable figures |
| `data/yolov10x_names.json` | the detector's class list, used to define scorable two-object pairs |

The analysis scripts also write a Markdown twin next to each JSON; the JSONs are the files kept here. Headline
numbers: Wan2.1 all 80 prompts B0 0.695 -> L 0.850 (+15.5, 95 % CI [+11.1, +20.2]); Self-Forcing all 80, seed 42 / 43,
B0 0.617 / 0.601 -> L 0.875 / 0.843; two objects, all 82 pairs, 0.376 -> 0.472 (+9.6 [+4.2, +15.0]); cost +0.8 % wall
time on the explicit path and +1.58 % inside Wan's FA2 kernel, with no measurable memory change.
