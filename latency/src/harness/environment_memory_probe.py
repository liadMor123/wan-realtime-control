#!/usr/bin/env python3
"""
Environment, memory-fit and reproduction probe for Self-Forcing on A100-SXM4-40GB (J1).

Reproduces inference.py's setup exactly (same config merge, same dtype, same
device placement), instruments the generator, text encoder and VAE, and records
the five pre-registered memory points, passes per chunk and the latent-to-frame
mapping. It does NOT change model shapes, the history window,
num_frame_per_block, or the denoising step list.

Writes: one row to --jsonl (run metadata), a --memjson file (memory points,
per-pass wall times, live config) and one MP4 of the generated video.
Wall-clock times recorded here are reproduction evidence, not latency results.

The memory ceiling is pre-registered and is not read from the command line, so
it cannot be loosened after seeing a failure.
"""
import argparse
import json
import os
import socket
import sys
import time
from datetime import datetime, timezone

import torch

GB = 1024 ** 3
CEILING_GB = 36.0  # pre-registered steady-state memory ceiling


def fail(msg, code=2):
    print(f"\n!!! J1 FAILURE: {msg}\n", flush=True)
    sys.exit(code)


def peak_memory_bytes():
    return (torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved())


def to_gb(peak):
    return {"allocated_gb": round(peak[0] / GB, 3), "reserved_gb": round(peak[1] / GB, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_path", default="configs/self_forcing_dmd.yaml")
    ap.add_argument("--checkpoint_path", default="checkpoints/self_forcing_dmd.pt")
    ap.add_argument("--data_path", default="prompts/MovieGenVideoBench_extended.txt")
    ap.add_argument("--output_folder", required=True)
    ap.add_argument("--offload", choices=["auto", "on", "off"], required=True,
                    help="text-encoder offload (repo's low_memory path)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_output_frames", type=int, default=21)
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--memjson", required=True)
    args = ap.parse_args()

    from omegaconf import OmegaConf
    from einops import rearrange
    from pipeline import CausalInferencePipeline
    from utils.misc import set_seed
    from demo_utils.memory import gpu, get_cuda_free_memory_gb

    device = torch.device("cuda")
    set_seed(args.seed)
    torch.set_grad_enabled(False)

    free_gb = get_cuda_free_memory_gb(gpu)
    auto_low = free_gb < 40
    low_memory = {"auto": auto_low, "on": True, "off": False}[args.offload]
    print(f"[J1] free VRAM {free_gb:.2f} GB | repo auto low_memory={auto_low} "
          f"| offload={args.offload} -> low_memory={low_memory}", flush=True)

    # ---- versions -------------------------------------------------------
    import torchvision
    import triton
    import triton.language as tl
    import inspect
    try:
        sem_ok = "sem" in inspect.signature(tl.atomic_add).parameters
    except (ValueError, TypeError):
        sem_ok = None
    print(f"[J1] torch={torch.__version__} torchvision={torchvision.__version__} "
          f"triton={triton.__version__} "
          f"tl.atomic_add has sem= : {sem_ok}", flush=True)

    cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"),
                          OmegaConf.load(args.config_path))

    torch.cuda.reset_peak_memory_stats()
    t_load0 = time.perf_counter()
    pipeline = CausalInferencePipeline(cfg, device=device)
    sd = torch.load(args.checkpoint_path, map_location="cpu")
    pipeline.generator.load_state_dict(sd["generator_ema"])
    del sd
    pipeline = pipeline.to(dtype=torch.bfloat16)

    if low_memory:
        from demo_utils.memory import DynamicSwapInstaller
        DynamicSwapInstaller.install_model(pipeline.text_encoder, device=gpu)
    else:
        pipeline.text_encoder.to(device=gpu)
    pipeline.generator.to(device=gpu)
    pipeline.vae.to(device=gpu)
    torch.cuda.synchronize()
    t_load = time.perf_counter() - t_load0

    points = {}
    points["P1_after_model_load"] = to_gb(peak_memory_bytes())
    torch.cuda.reset_peak_memory_stats()

    # ---- live config facts (measured, not from the paper) ---------------
    nfpb = pipeline.num_frame_per_block
    dsl = [float(x) for x in pipeline.denoising_step_list.tolist()]
    ctx_noise = int(getattr(cfg, "context_noise", 0))
    local_attn = pipeline.local_attn_size
    kv_tokens = 32760 if local_attn == -1 else local_attn * pipeline.frame_seq_length
    live = {
        "num_frame_per_block": nfpb,
        "denoising_step_list_raw": list(cfg.denoising_step_list),
        "denoising_step_list_warped": dsl,
        "warp_denoising_step": bool(cfg.warp_denoising_step),
        "context_timestep": ctx_noise,
        "num_transformer_blocks": pipeline.num_transformer_blocks,
        "frame_seq_length": pipeline.frame_seq_length,
        "local_attn_size": local_attn,
        "history_window_tokens": kv_tokens,
        "tokens_per_chunk": nfpb * pipeline.frame_seq_length,
        "height": int(cfg.height), "width": int(cfg.width),
        "num_frames_cfg": int(cfg.num_frames),
    }
    print("[J1] live config:", json.dumps(live, indent=2), flush=True)

    # ---- instrumentation -------------------------------------------------
    calls = []           # one record per generator forward
    state = {"chunk": -1}
    first_ts = dsl[0]

    gen = pipeline.generator
    original_generator_forward = gen.forward

    def instrumented_generator_forward(*a, **kw):
        ts = kw.get("timestep")
        tsv = float(ts.flatten()[0].item()) if ts is not None else float("nan")
        shp = tuple(kw["noisy_image_or_video"].shape) if "noisy_image_or_video" in kw else None
        if tsv == first_ts:                      # a new chunk begins
            state["chunk"] += 1
            c = state["chunk"]
            if c == 1:
                points["P3_after_first_chunk"] = to_gb(peak_memory_bytes())
                torch.cuda.reset_peak_memory_stats()
            if c == 5:                            # steady state begins
                torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        out = original_generator_forward(*a, **kw)
        torch.cuda.synchronize()
        calls.append({"i": len(calls), "chunk": state["chunk"], "timestep": tsv,
                      "shape": shp, "wall_ms": (time.perf_counter() - t0) * 1e3})
        return out

    gen.forward = instrumented_generator_forward

    te = pipeline.text_encoder
    original_text_encoder_forward = te.forward

    def instrumented_text_encoder_forward(*a, **kw):
        out = original_text_encoder_forward(*a, **kw)
        torch.cuda.synchronize()
        points["P2_after_text_encoding"] = to_gb(peak_memory_bytes())
        torch.cuda.reset_peak_memory_stats()
        return out

    te.forward = instrumented_text_encoder_forward

    vae = pipeline.vae
    original_decode_to_pixel = vae.decode_to_pixel

    def instrumented_decode_to_pixel(*a, **kw):
        torch.cuda.synchronize()
        points["P4_steady_state"] = to_gb(peak_memory_bytes())   # peak over chunks >= 5
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        out = original_decode_to_pixel(*a, **kw)
        torch.cuda.synchronize()
        points["_vae_decode_ms"] = (time.perf_counter() - t0) * 1e3
        points["P5_after_vae_decode"] = to_gb(peak_memory_bytes())
        torch.cuda.reset_peak_memory_stats()
        return out

    vae.decode_to_pixel = instrumented_decode_to_pixel

    # ---- run -------------------------------------------------------------
    # Same file and same first entry TextDataset would yield; read directly so
    # utils.dataset (which imports lmdb at module scope) is not a dependency.
    with open(args.data_path, encoding="utf-8") as f:
        prompt = f.readline().rstrip()
    if not prompt:
        fail(f"empty first prompt in {args.data_path}")
    print(f"[J1] prompt[:120]: {prompt[:120]!r}", flush=True)

    noise = torch.randn([1, args.num_output_frames, 16, 60, 104],
                        device=device, dtype=torch.bfloat16)

    t0 = time.perf_counter()
    video, latents = pipeline.inference(noise=noise, text_prompts=[prompt],
                                        return_latents=True, initial_latent=None,
                                        low_memory=low_memory)
    torch.cuda.synchronize()
    wall_s = time.perf_counter() - t0

    # ---- measured passes per chunk and frame mapping ---------------------
    per_chunk = {}
    for c in calls:
        per_chunk.setdefault(c["chunk"], []).append(c["timestep"])
    passes = {k: len(v) for k, v in sorted(per_chunk.items())}
    uniq = sorted(set(passes.values()))
    n_chunks = len(per_chunk)

    out_frames = int(video.shape[1])
    lat_frames = int(latents.shape[1])
    mapping = {
        "latent_frames_total": lat_frames,
        "output_frames_total": out_frames,
        "chunks": n_chunks,
        "latent_frames_per_chunk": nfpb,
        "output_frames_per_chunk_mean": round(out_frames / n_chunks, 3),
        "formula_check_4x_plus_1": (lat_frames - 1) * 4 + 1 == out_frames,
        "video_shape": list(video.shape),
        "latents_shape": list(latents.shape),
        "fps_written": 16,
    }

    print(f"\n[J1] passes per chunk (measured): {passes}")
    print(f"[J1] distinct passes-per-chunk values: {uniq}")
    print(f"[J1] timestep pattern chunk0: {per_chunk.get(0)}")
    print(f"[J1] frame mapping: {json.dumps(mapping, indent=2)}", flush=True)

    # ---- save video ------------------------------------------------------
    os.makedirs(args.output_folder, exist_ok=True)
    # torchvision.io.write_video was removed in current torchvision; encode with
    # imageio-ffmpeg instead. This changes only MP4 encoding, never the model.
    import imageio.v2 as iio
    vid = (255.0 * rearrange(video, "b t c h w -> b t h w c").cpu().float())[0]
    vid = vid.clamp(0, 255).to(torch.uint8).numpy()
    mp4 = os.path.join(args.output_folder, f"{args.run_id}.mp4")
    iio.mimwrite(mp4, list(vid), fps=16, quality=8, macro_block_size=1)
    pipeline.vae.model.clear_cache()
    print(f"[J1] wrote {mp4} ({os.path.getsize(mp4)/1e6:.2f} MB)", flush=True)

    for k in ("P1_after_model_load", "P2_after_text_encoding",
              "P3_after_first_chunk", "P4_steady_state", "P5_after_vae_decode"):
        if k not in points:
            fail(f"memory point {k} was never captured -- instrumentation did not fire")

    print("\n[J1] memory points:")
    for k in ("P1_after_model_load", "P2_after_text_encoding", "P3_after_first_chunk",
              "P4_steady_state", "P5_after_vae_decode"):
        print(f"   {k:26s} allocated {points[k]['allocated_gb']:7.3f} GB | "
              f"reserved {points[k]['reserved_gb']:7.3f} GB")

    # ---- records ---------------------------------------------------------
    gname = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    row = {
        "run_id": args.run_id,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "node": socket.gethostname(),
        "gpu_name": gname,
        "gpu_uuid": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_total_gb": round(props.total_memory / GB, 3),
        "sm_count": props.multi_processor_count,
        "driver": os.environ.get("J1_DRIVER"),
        "cuda_runtime": torch.version.cuda,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "triton_version": triton.__version__,
        "triton_atomic_add_has_sem": sem_ok,
        "repo_revision": os.environ.get("J1_REPO_REV"),
        "checkpoint_sha256": os.environ.get("J1_CKPT_SHA"),
        "resolution": f"{int(cfg.height)}x{int(cfg.width)}",
        "num_frame_per_block": nfpb,
        "denoise_steps": live["denoising_step_list_warped"],
        "context_timestep": ctx_noise,
        "history_window_tokens": kv_tokens,
        "passes_per_chunk": uniq[0] if len(uniq) == 1 else passes,
        "passes_per_chunk_all": passes,
        "precision": "bfloat16",
        "attention_backend": "NOT_MEASURED_IN_J1",
        "graph_mode": "eager",
        "fusion_mode": "none",
        "warmup_excluded": False,
        "profiler_attached": False,
        "n_chunks": n_chunks,
        "t_per_pass_ms_median": None,
        "t_per_pass_ms_p95": None,
        "t_per_chunk_ms_median": None,
        "t_per_chunk_ms_p95": None,
        "t_first_chunk_ms": None,
        "j1_wall_s_inference": round(wall_s, 3),
        "j1_wall_s_model_load": round(t_load, 3),
        "j1_vae_decode_ms": round(points["_vae_decode_ms"], 2),
        "max_memory_allocated_bytes": None,
        "max_memory_reserved_bytes": None,
        "memory_points": {k: v for k, v in points.items() if not k.startswith("_")},
        "frame_mapping": mapping,
        "live_config": live,
        "offload_mode": args.offload,
        "low_memory": low_memory,
        "output_paths": {"mp4": mp4},
        "notes": "J1 eager reproduction probe; wall times are NOT latency results",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    os.makedirs(os.path.dirname(args.jsonl), exist_ok=True)
    with open(args.jsonl, "a") as f:
        f.write(json.dumps(row) + "\n")
    with open(args.memjson, "w") as f:
        json.dump({"points": points, "passes": passes, "mapping": mapping,
                   "live": live, "calls": calls}, f, indent=2)

    # ---- pre-registered gate --------------------------------------------
    steady = points["P4_steady_state"]["reserved_gb"]
    print(f"\n[J1] GATE: steady-state reserved {steady:.3f} GB vs ceiling {CEILING_GB} GB "
          f"(offload={args.offload})", flush=True)
    if steady > CEILING_GB:
        fail(f"steady-state max_memory_reserved {steady:.3f} GB exceeds the "
            f"pre-registered {CEILING_GB} GB ceiling (offload={args.offload}). "
            f"Model, resolution, window and num_frame_per_block were NOT reduced.")
    print(f"[J1] PASS for offload={args.offload}", flush=True)


if __name__ == "__main__":
    main()
