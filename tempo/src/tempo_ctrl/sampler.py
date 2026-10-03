"""Wan2.1-T2V-1.3B sampling loop that mirrors wan.text2video.WanT2V.generate line for line,
minus T5 (text encodings come from a cache written by the same T5 call), plus two hooks:
CTRL.step and CTRL.branch are set before every DiT forward.

Protocol (TempoControl one-object benchmark): 832x480, 81 frames, 50 UniPC steps,
shift 3.0, guide scale 6.0, Wan's default negative prompt.
"""
import math
import os
import time

import torch
import torch.cuda.amp as amp

from .cross_attention_arms import CTRL, install_cross_attention_patch

SIZE = (832, 480)
FRAME_NUM = 81
STEPS = 50
SHIFT = 3.0
GUIDE = 6.0


class Sampler:

    def __init__(self, ckpt_dir, device_id=0, patch=True):
        from wan.configs import WAN_CONFIGS
        from wan.modules.model import WanModel
        from wan.modules.vae import WanVAE

        self.cfg = WAN_CONFIGS["t2v-1.3B"]
        self.device = torch.device(f"cuda:{device_id}")
        self.num_train_timesteps = self.cfg.num_train_timesteps
        self.param_dtype = self.cfg.param_dtype
        self.vae_stride = self.cfg.vae_stride
        self.patch_size = self.cfg.patch_size
        self.vae = WanVAE(vae_pth=os.path.join(ckpt_dir, self.cfg.vae_checkpoint), device=self.device)
        self.model = WanModel.from_pretrained(ckpt_dir)
        self.model.eval().requires_grad_(False)
        if patch:
            install_cross_attention_patch(self.model)
        self.model.to(self.device)

    @classmethod
    def from_wan_t2v(cls, w, patch=True):
        """Share the DiT and VAE of an already-built wan.WanT2V (for the equivalence check)."""
        self = cls.__new__(cls)
        self.cfg, self.device = w.config, w.device
        self.num_train_timesteps, self.param_dtype = w.num_train_timesteps, w.param_dtype
        self.vae_stride, self.patch_size = w.vae_stride, w.patch_size
        self.vae, self.model = w.vae, w.model
        if patch:
            install_cross_attention_patch(self.model)
        return self

    @torch.no_grad()
    def generate(self, context, context_null, seed, steps=STEPS, decode=True, timing=None):
        """context / context_null: lists with one [L, 4096] bf16 tensor on self.device."""
        from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

        F = FRAME_NUM
        size = SIZE
        target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1], size[0] // self.vae_stride[2])
        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) * target_shape[1])
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise = [torch.randn(*target_shape, dtype=torch.float32, device=self.device, generator=seed_g)]

        if timing is not None:
            torch.cuda.synchronize()
            t_start = time.perf_counter()
        with amp.autocast(dtype=self.param_dtype), torch.no_grad():
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps, shift=1, use_dynamic_shifting=False)
            sample_scheduler.set_timesteps(steps, device=self.device, shift=SHIFT)
            timesteps = sample_scheduler.timesteps
            latents = noise
            arg_c = {"context": context, "seq_len": seq_len}
            arg_null = {"context": context_null, "seq_len": seq_len}
            for i, t in enumerate(timesteps):
                timestep = torch.stack([t])
                CTRL.step = i
                CTRL.branch = "cond"
                noise_pred_cond = self.model(latents, t=timestep, **arg_c)[0]
                CTRL.branch = "uncond"
                noise_pred_uncond = self.model(latents, t=timestep, **arg_null)[0]
                noise_pred = noise_pred_uncond + GUIDE * (noise_pred_cond - noise_pred_uncond)
                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0), t, latents[0].unsqueeze(0),
                    return_dict=False, generator=seed_g)[0]
                latents = [temp_x0.squeeze(0)]
            x0 = latents
            if timing is not None:
                torch.cuda.synchronize()
                timing["denoise_s"] = time.perf_counter() - t_start
            video = self.vae.decode(x0)[0] if decode else None
        if timing is not None:
            torch.cuda.synchronize()
            timing["total_s"] = time.perf_counter() - t_start
        return video, x0[0]
