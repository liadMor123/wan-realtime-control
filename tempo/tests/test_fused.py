"""CPU tests for the fused-mask path (part 2, step 2): tables, absolute-frame indexing, casts, failure modes.

Run on the login node:  ~/tempo/venv/bin/python -m pytest -q ~/tempo/tests (TEMPO_ROOT overrides ~/tempo)
"""
import os
import sys

import pytest
import torch

TEMPO_ROOT = os.environ.get("TEMPO_ROOT", os.path.expanduser("~/tempo"))
sys.path.insert(0, os.path.join(TEMPO_ROOT, "src"))
from tempo_ctrl import benchmark  # noqa: E402
from tempo_ctrl.fused_attention import (HW, LK, N_LAT, frame_table_L, masked_attention, reference_fp64,  # noqa: E402
                                        token_table, zero_table)
from tempo_ctrl.tokens import load_tokenizer  # noqa: E402
from tempo_ctrl.cross_attention_arms import CTRL, Controller, _frame_bias_table, fused_mask_table  # noqa: E402

CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def rows_tok():
    return benchmark.load_one_object(), load_tokenizer(benchmark.WAN_ROOT, benchmark.CKPT)


def test_frame_table_matches_part1_bias_on_all_80_prompts(rows_tok):
    rows, tok = rows_tok
    for r in rows:
        benchmark.configure_controller({"arm": "L", "beta": 2.0, "gamma": 2.0, "seed": 42}, r, tok)
        want = _frame_bias_table(CTRL, LK, CPU)
        got = frame_table_L(CTRL.mask, CTRL.obj_idx, 2.0, 2.0)
        assert torch.equal(got, want.cpu()), r["prompt_id"]
        on = CTRL.mask > 0.5
        assert (got[on][:, CTRL.obj_idx] == 2).all() and (got[~on][:, CTRL.obj_idx] == -2).all()
        others = [i for i in range(LK) if i not in CTRL.obj_idx]
        assert (got[:, others] == 0).all()


def test_token_table_rows_follow_absolute_frames():
    ft = torch.arange(N_LAT * LK, dtype=torch.float32).view(N_LAT, LK) % 64 - 32   # distinct per frame, bf16-exact
    tt = token_table(ft, CPU)
    assert tt.shape == (N_LAT * HW, LK) and tt.dtype == torch.bfloat16 and tt.is_contiguous()
    for t in (0, 1, HW - 1, HW, 5 * HW + 7, N_LAT * HW - 1):
        assert torch.equal(tt[t].float(), ft[t // HW]), t
    with pytest.raises(ValueError):
        token_table(torch.full((N_LAT, LK), 0.1), CPU)                   # 0.1 is not exact in bf16
    assert torch.count_nonzero(zero_table(CPU)) == 0


def test_frame_table_rejects_bad_input():
    with pytest.raises(ValueError):
        frame_table_L([1] * 20, [3], 2, 2)
    with pytest.raises(ValueError):
        frame_table_L([0.5] * 21, [3], 2, 2)
    with pytest.raises(ValueError):
        frame_table_L([1] * 21, [], 2, 2)
    with pytest.raises(ValueError):
        frame_table_L([1] * 21, [512], 2, 2)


def _qkv(n_frames, heads=2, d=128, seed=0, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(1, n_frames * HW, heads, d, generator=g) * scale
    k = torch.randn(1, LK, heads, d, generator=g) * scale
    v = torch.randn(1, LK, heads, d, generator=g)
    return q, k, v


def test_masked_attention_matches_fp64_full_sequence():
    q, k, v = _qkv(N_LAT)
    mask = [0] * 5 + [1] * 16
    ft = frame_table_L(mask, [7, 8], 2.0, 2.0)
    out = masked_attention(q, k, v, token_table(ft, CPU), 0)
    assert out.shape == q.shape and out.dtype == q.dtype
    ref = reference_fp64(q, k, v, ft, 0)
    err = (out.double() - ref).abs()
    assert err.max().item() < 2e-2 and err.mean().item() < 2e-3, (err.max().item(), err.mean().item())
    # the bias must matter: without it the output differs by more than the bf16 error
    ref0 = reference_fp64(q, k, v, torch.zeros(N_LAT, LK), 0)
    assert (ref - ref0).abs().max().item() > 10 * err.max().item()


def test_three_frame_blocks_index_the_table_by_absolute_frame():
    """Self-Forcing feeds 3 frames per pass; block f0 must read table rows of frames f0..f0+2, not 0..2."""
    q, k, v = _qkv(N_LAT, seed=1, scale=2.0)
    ft = torch.randint(-4, 5, (N_LAT, LK), generator=torch.Generator().manual_seed(5)).float()  # frame-distinct
    tt = token_table(ft, CPU)
    for f0 in range(0, N_LAT, 3):
        qb = q[:, f0 * HW:(f0 + 3) * HW]
        out = masked_attention(qb, k, v, tt, f0 * HW)
        ref = reference_fp64(q, k, v, ft, 0)[:, f0 * HW:(f0 + 3) * HW]
        wrong = reference_fp64(qb, k, v, ft, 0)                  # the bug this test guards against: rows 0..2
        e = (out.double() - ref).abs().max().item()
        assert e < 3e-2, (f0, e)
        if f0:
            assert (out.double() - wrong).abs().max().item() > 10 * e, f0


def test_masked_attention_casts_like_flash_attention():
    q, k, v = _qkv(3, seed=2)
    tt = zero_table(CPU)
    o32 = masked_attention(q.float(), k.float(), v.float(), tt, 0)            # fp32 in -> cast to bf16 inside
    obf = masked_attention(q.bfloat16(), k.bfloat16(), v.bfloat16(), tt, 0)
    assert o32.dtype == torch.float32 and obf.dtype == torch.bfloat16
    assert torch.equal(o32, obf.float())                                       # output = bf16 result in q's dtype


def test_masked_attention_rejects_misaligned_rows():
    q, k, v = _qkv(3, seed=3)
    tt = zero_table(CPU)
    with pytest.raises(ValueError):
        masked_attention(q, k, v, tt, 7)                     # not a frame boundary
    with pytest.raises(ValueError):
        masked_attention(q, k, v, tt, 19 * HW)               # rows past frame 20
    with pytest.raises(ValueError):
        masked_attention(q, k[:, :500], v[:, :500], tt, 0)   # Lk mismatch
    with pytest.raises(TypeError):
        masked_attention(q, k, v, tt.float(), 0)             # table must be in the attention dtype


def test_controller_fused_table_selection(rows_tok):
    rows, tok = rows_tok
    r = rows[2]
    benchmark.configure_controller({"arm": "L", "beta": 2.0, "gamma": 2.0, "seed": 42, "path": "fused"}, r, tok)
    CTRL.step, CTRL.branch = 0, "cond"
    t_on = fused_mask_table(CTRL, LK, CPU)
    assert torch.equal(t_on, token_table(frame_table_L(CTRL.mask, CTRL.obj_idx, 2.0, 2.0), CPU))
    CTRL.branch = "uncond"
    assert torch.count_nonzero(fused_mask_table(CTRL, LK, CPU)) == 0
    CTRL.branch, CTRL.k_steps, CTRL.step = "cond", 10, 10
    assert torch.count_nonzero(fused_mask_table(CTRL, LK, CPU)) == 0
    benchmark.configure_controller({"arm": "B0", "seed": 42, "path": "fused"}, r, tok)
    CTRL.branch, CTRL.step = "cond", 0
    assert torch.count_nonzero(fused_mask_table(CTRL, LK, CPU)) == 0
    with pytest.raises(ValueError):
        benchmark.configure_controller({"arm": "U", "beta": 2.0, "seed": 42, "path": "fused"}, r, tok)
    c = Controller(path="fused", arm="S")
    with pytest.raises(NotImplementedError):
        fused_mask_table(c, LK, CPU)
    assert benchmark.run_tag({"arm": "L", "beta": 2.0, "gamma": 2.0, "seed": 42, "path": "fused"}) == "L_b2g2_fused_s42"
    assert benchmark.run_tag({"arm": "B0", "seed": 42, "path": "flash"}) == "B0_flash_s42"


# ------------------------------------------------------------------ fa2kv (key augmentation through FA2)
from tempo_ctrl.fused_attention import (KA_C, KA_DIMS, KA_REL_ERR, fa2kv_augment, fa2kv_tables, fa2kv_zero_tables,  # noqa: E402
                                        frame_scalar_L)


def test_fa2kv_constants_sum_to_inverse_fp32_scale():
    s32 = float(torch.tensor(128 ** -0.5, dtype=torch.float32))
    assert KA_REL_ERR < 1e-9 and abs(sum(KA_C) * s32 - 1) < 1e-9
    assert all(float(torch.tensor(c).bfloat16()) == c for c in KA_C)          # each constant exact in bf16


def test_fa2kv_augmented_logits_equal_biased_logits_fp64():
    """Emulate FA2 in fp64 on the augmented tensors: logits must be q.k*s + bias, for 3-frame blocks at offsets."""
    q, k, v = _qkv(N_LAT, seed=4)
    q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()
    mask = [0] * 8 + [1] * 13
    obj = [5, 6, 30]
    ft = frame_table_L(mask, obj, 2.0, 2.0)
    tabs = fa2kv_tables(mask, obj, 2.0, 2.0, CPU)
    s = 128 ** -0.5
    for f0 in (0, 3, 9, 18):
        qb = q[:, f0 * HW:(f0 + 3) * HW]
        qx, kx, vx = fa2kv_augment(qb, k, v, tabs, f0 * HW)
        assert qx.shape[-1] == 128 + KA_DIMS and qx.dtype == torch.bfloat16 and qx.is_contiguous()
        assert torch.equal(qx[..., :128], qb) and torch.equal(kx[..., :128], k) and torch.equal(vx[..., :128], v)
        assert torch.count_nonzero(vx[..., 128:]) == 0
        lg_aug = torch.einsum("bqhd,bkhd->bhqk", qx.double(), kx.double()) * s
        lg_ref = torch.einsum("bqhd,bkhd->bhqk", qb.double(), k.double()) * s + \
            ft[f0:f0 + 3].double().repeat_interleave(HW, 0)[None, None]
        assert (lg_aug - lg_ref).abs().max().item() < 1e-5, f0
        ref = reference_fp64(qb, k, v, ft, f0 * HW)
        out = torch.einsum("bhqk,bkhd->bqhd", torch.softmax(lg_aug, -1), vx.double())[..., :128]
        assert (out - ref).abs().max().item() < 1e-5, f0


def test_fa2kv_zero_tables_and_validation():
    tz = fa2kv_zero_tables(CPU)
    assert torch.count_nonzero(tz[0]) == 0 and torch.count_nonzero(tz[1]) == 0
    q, k, v = _qkv(3, seed=5)
    qx, kx, _ = fa2kv_augment(q, k, v, tz, 0)
    assert torch.count_nonzero(qx[..., 128:]) == 0 and torch.count_nonzero(kx[..., 128:]) == 0
    assert torch.equal(frame_scalar_L([1] * 10 + [0] * 11, 2.0, 2.0)[9:11], torch.tensor([2.0, -2.0]))
    with pytest.raises(TypeError):
        fa2kv_augment(q, k, v, tz, torch.tensor(0))
    with pytest.raises(ValueError):
        fa2kv_augment(q, k, v, tz, 5)
    with pytest.raises(ValueError):
        fa2kv_augment(q[..., :64], k[..., :64], v[..., :64], tz, 0)
    with pytest.raises(ValueError):
        fa2kv_tables([1] * 21, [], 2.0, 2.0, CPU)


def test_controller_fa2kv_selection(rows_tok):
    from tempo_ctrl.cross_attention_arms import fa2kv_tables_for_controller
    rows, tok = rows_tok
    benchmark.configure_controller({"arm": "L", "beta": 2.0, "gamma": 2.0, "seed": 42, "path": "fa2kv"}, rows[2], tok)
    CTRL.step, CTRL.branch = 0, "cond"
    qb, ke = fa2kv_tables_for_controller(CTRL, LK, CPU)
    assert torch.count_nonzero(ke[CTRL.obj_idx]) == 3 * len(CTRL.obj_idx) and qb.abs().max() == 2
    CTRL.branch = "uncond"
    qb0, ke0 = fa2kv_tables_for_controller(CTRL, LK, CPU)
    assert torch.count_nonzero(qb0) == 0 and torch.count_nonzero(ke0) == 0
    assert benchmark.run_tag({"arm": "L", "beta": 2.0, "gamma": 2.0, "seed": 42, "path": "fa2kv"}) == "L_b2g2_fa2kv_s42"
