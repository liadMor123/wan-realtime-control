"""
Fused split-KV merge + output-projection GEMM in Triton (J13 Step 1).

Computes  y = ( sum_s exp(lse_s - LSE) * O_s ).bf16() @ W_o^T + b_o
from the FlashAttention-2 split-KV partials O_s = out_accum[s, 0, h, q, :]
(fp32, per-split normalized) and lse_s = softmax_lse_accum[s, 0, h, q],
replacing FA2's separate combine kernel plus the cuBLAS out-projection. The
GEMM's K-loop runs over heads (BLOCK_K = d = 128): each K-step merges one
head's partials for the tile's rows, rounds to bf16 at the same point FA2's
combine does, and feeds the tile to tl.dot.

Library module (no CLI); used by fused_merge_outproj_test.py and
fused_merge_outproj_ncu_target.py. merge_partials_fp32 is the fp32 reference.
"""
import torch
import triton
import triton.language as tl

_CONFIGS = [  # smem per stage ~ NS*BM*128*4 (fp32 partials) + BN*128*2 (W); A100 limit 166 KB -> keep stages low
    triton.Config({"BM": 64, "BN": 256}, num_warps=8, num_stages=1),
    triton.Config({"BM": 64, "BN": 256}, num_warps=8, num_stages=2),
    triton.Config({"BM": 64, "BN": 128}, num_warps=4, num_stages=2),
    triton.Config({"BM": 32, "BN": 256}, num_warps=4, num_stages=2),
    triton.Config({"BM": 32, "BN": 512}, num_warps=8, num_stages=1),
    triton.Config({"BM": 32, "BN": 128}, num_warps=4, num_stages=2),
]


@triton.autotune(configs=_CONFIGS, key=["Q", "N", "NS"])
@triton.jit
def _fused_merge_outproj_kernel(oacc_ptr, lse_ptr, w_ptr, b_ptr, out_ptr,
                                Q, N,
                                s_oacc_s, s_oacc_h, s_oacc_q,
                                s_lse_s, s_lse_h,
                                s_w_n, s_out_q,
                                NS: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                                BM: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    n_n = tl.cdiv(N, BN)
    pid_m = pid // n_n          # N-blocks of one M-block are adjacent -> shared partial rows hit L2
    pid_n = pid % n_n
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rd = tl.arange(0, D)
    mask_m = rm < Q
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for h in range(H):
        # --- merge prologue: weights from the NS log-sum-exps of this head, these rows
        m = tl.load(lse_ptr + h * s_lse_h + rm, mask=mask_m, other=float("-inf"))
        for s in tl.static_range(1, NS):
            l = tl.load(lse_ptr + s * s_lse_s + h * s_lse_h + rm, mask=mask_m, other=float("-inf"))
            m = tl.maximum(m, l)
        z = tl.zeros((BM,), dtype=tl.float32)
        a = tl.zeros((BM, D), dtype=tl.float32)
        for s in tl.static_range(0, NS):
            l = tl.load(lse_ptr + s * s_lse_s + h * s_lse_h + rm, mask=mask_m, other=float("-inf"))
            w = tl.exp(l - m)                       # exp(-inf) = 0 for empty splits
            z += w
            o = tl.load(oacc_ptr + s * s_oacc_s + h * s_oacc_h + rm[:, None] * s_oacc_q + rd[None, :],
                        mask=mask_m[:, None], other=0.0)
            a += w[:, None] * o
        z = tl.where(z == 0.0, 1.0, z)
        a16 = (a / z[:, None]).to(tl.bfloat16)      # same rounding point as FA2's combine (bf16 O)
        # --- GEMM step: this head's 128 input features of W_o  (W is [N, K], K = h*D + d)
        wt = tl.load(w_ptr + rn[:, None] * s_w_n + (h * D + rd)[None, :])   # [BN, D] bf16
        acc += tl.dot(a16, tl.trans(wt))
    bias = tl.load(b_ptr + rn).to(tl.float32)
    acc += bias[None, :]
    tl.store(out_ptr + rm[:, None] * s_out_q + rn[None, :], acc.to(tl.bfloat16), mask=mask_m[:, None])


def fused_merge_outproj(out_accum: torch.Tensor, lse_accum: torch.Tensor,
                        weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """out_accum [NS, 1, H, Q, Dr] fp32, lse_accum [NS, 1, H, Q] fp32 (FA2 return_partials layout);
    weight [N, H*128] bf16 (nn.Linear), bias [N]. Returns [1, Q, N] bf16 = o(merged attention)."""
    NS, B, H, Q, Dr = out_accum.shape
    assert B == 1 and Dr >= 128 and lse_accum.shape == (NS, 1, H, Q), (out_accum.shape, lse_accum.shape)
    N, K = weight.shape
    assert K == H * 128 and weight.dtype == torch.bfloat16 and weight.stride(1) == 1
    assert out_accum.stride(4) == 1 and lse_accum.stride(3) == 1 and NS in (2, 4, 8)
    out = torch.empty((1, Q, N), dtype=torch.bfloat16, device=out_accum.device)
    grid = lambda META: (triton.cdiv(Q, META["BM"]) * triton.cdiv(N, META["BN"]),)
    _fused_merge_outproj_kernel[grid](
        out_accum, lse_accum, weight, bias, out, Q, N,
        out_accum.stride(0), out_accum.stride(2), out_accum.stride(3),
        lse_accum.stride(0), lse_accum.stride(2),
        weight.stride(0), out.stride(1),
        NS=NS, H=H, D=128)
    return out


def merge_partials_fp32(out_accum, lse_accum, d=128):
    """Reference merge in fp32: [1, Q, H*d]."""
    LSE = torch.logsumexp(lse_accum, dim=0)                       # [1,H,Q]
    w = torch.exp(lse_accum - LSE.unsqueeze(0))                    # [NS,1,H,Q]
    merged = (w.unsqueeze(-1) * out_accum[..., :d]).sum(0)         # [1,H,Q,d]
    return merged.permute(0, 2, 1, 3).reshape(1, out_accum.shape[3], -1)   # [1,Q,H*d]
