"""Self-Forcing helpers for part 2, step 3. Import from a staged Self-Forcing repo (cwd = repo root, branch tempo-L).

The clone's WanT2VCrossAttention (branch tempo-L) runs `tempo_attn(q, k, v, tempo_bias, current_start)` when
`tempo_bias` is set and the original flash_attention call when it is None. The path is fa2kv (fused_attention.py; chosen in step
2b): tempo_bias = (qb [21*1560] bf16, ke [512, 8] bf16). One static pair per process is installed on all 30 blocks; a
new prompt's values are copied into it (tensor identity is kept for CUDA graphs).
"""
import os
import sys

import torch

TEMPO = os.environ.get("TEMPO_ROOT", os.path.expanduser("~/tempo"))
sys.path.insert(0, os.path.join(TEMPO, "src"))

from tempo_ctrl.fused_attention import (fa2_kernel_head_dim, fa2kv_attention, fa2kv_tables, fa2kv_zero_tables, frame_table_L,  # noqa: E402,F401
                                        masked_attention, token_table, zero_table)

N_BLOCKS = 30


def build_pipeline(dev):
    """Exactly the ~/proj latency harness (j3b_run.py) setup: generator_ema, bf16, T5 via DynamicSwapInstaller."""
    from omegaconf import OmegaConf
    from pipeline import CausalInferencePipeline
    from demo_utils.memory import gpu, DynamicSwapInstaller
    from wan.modules.model import prebuild_sinusoid_cache
    cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"),
                          OmegaConf.load("configs/self_forcing_dmd.yaml"))
    pipe = CausalInferencePipeline(cfg, device=dev)
    sd = torch.load("checkpoints/self_forcing_dmd.pt", map_location="cpu")
    pipe.generator.load_state_dict(sd["generator_ema"])
    del sd
    pipe = pipe.to(dtype=torch.bfloat16)
    DynamicSwapInstaller.install_model(pipe.text_encoder, device=gpu)
    pipe.generator.to(device=gpu)
    pipe.vae.to(device=gpu)
    prebuild_sinusoid_cache(256, dev)
    return pipe


def install_tempo_bias(model, table, attn=fa2kv_attention):
    """Set (table tensor) or clear (None) the tempo bias on every cross-attention module of a CausalWanModel."""
    from wan.modules.model import WanT2VCrossAttention
    assert len(model.blocks) == N_BLOCKS, len(model.blocks)
    for blk in model.blocks:
        ca = blk.cross_attn
        assert type(ca) is WanT2VCrossAttention, type(ca)
        assert "current_start" in ca.forward.__code__.co_varnames, "clone is not on branch tempo-L"
        ca.tempo_bias = table                       # instance attributes: tempo_attn is not bound to the module
        ca.tempo_attn = attn if table is not None else None


def l_fa2kv_tables(row, tok, device, beta=2.0, gamma=2.0):
    """fa2kv (qb, ke) for L(beta, gamma) on a benchmark row, and the object slots."""
    from tempo_ctrl.tokens import object_token_indices
    obj_idx, n_real, _ = object_token_indices(tok, row["prompt"], row["temp_object"])
    return fa2kv_tables(row["mask"], obj_idx, beta, gamma, device), obj_idx


def l_frame_table(row, tok, beta=2.0, gamma=2.0):
    """[21, 512] fp32 L frame table for a benchmark row (object slots from tempo_ctrl.tokens)."""
    from tempo_ctrl.tokens import object_token_indices
    obj_idx, n_real, _ = object_token_indices(tok, row["prompt"], row["temp_object"])
    return frame_table_L(row["mask"], obj_idx, beta, gamma), obj_idx, n_real
