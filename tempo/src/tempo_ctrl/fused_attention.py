"""Arm L through fused attention kernels (part 2, steps 2-3): an additive-mask SDPA path and the fa2kv path.

Two implementations of the same per-frame logit bias, both without the explicit matmul-softmax-matmul of
cross_attention_arms.explicit_attention:
  * `masked_attention`  : torch SDPA with a per-query-token additive mask (the first step-2 candidate);
  * `fa2kv_attention`   : Wan's own FlashAttention-2 kernel, the bias carried in KA_DIMS = 8 extra head dims
                          ("fa2kv", chosen in the step-2b bake-off; see the section below).

L's bias depends only on (latent frame of the query, key slot), so it is a 21 x 512 frame table, expanded once to a
per-query-token table [21*1560, 512] (bf16; the values +beta, -gamma, 0 used here are exact in bf16). A forward pass
whose first query is absolute token `q_start` reads table rows [q_start, q_start + Lq):

  Wan2.1 (step 2)       : Lq = 21 * 1560, q_start = 0
  Self-Forcing (step 3) : Lq =  3 * 1560 per block, q_start = current_start (= start frame * 1560)

The table is built once per prompt, before any compile or CUDA-graph capture; `masked_attention` does no host sync.
B0 on this path is the same call with an all-zero table, so B0 and L differ only in the table's values.
"""
import torch
import torch.nn.functional as F

N_LAT, HW, LK = 21, 30 * 52, 512
_HALF = (torch.float16, torch.bfloat16)


def frame_table_L(mask, obj_idx, beta, gamma, lk=LK):
    """[21, lk] fp32: +beta on obj_idx on frames with mask 1, -gamma on frames with mask 0, 0 on every other slot.
    Same values as cross_attention_arms._frame_bias_table for arm L."""
    m = torch.as_tensor(mask, dtype=torch.float32)
    if m.shape != (N_LAT,) or not set(m.tolist()) <= {0.0, 1.0}:
        raise ValueError(f"mask must be 21 binary entries, got {m.tolist()}")
    obj_idx = list(obj_idx)
    if not obj_idx or len(set(obj_idx)) != len(obj_idx) or min(obj_idx) < 0 or max(obj_idx) >= lk:
        raise ValueError(f"bad object slots {obj_idx}")
    t = torch.zeros(N_LAT, lk, dtype=torch.float32)
    t[:, obj_idx] = (beta * m - gamma * (1 - m))[:, None]
    return t


def token_table(frame_table, device, hw=HW, dtype=torch.bfloat16):
    """Expand a [21, lk] frame table to the per-query-token table [21*hw, lk] on `device`. Fails if not exact in dtype."""
    ft = frame_table.to(torch.float32).cpu()
    if ft.dim() != 2 or ft.shape[0] != N_LAT:
        raise ValueError(f"frame table must be [21, lk], got {tuple(ft.shape)}")
    if not torch.equal(ft.to(dtype).to(torch.float32), ft):
        raise ValueError(f"frame table values are not exact in {dtype}")
    return ft.to(dtype).repeat_interleave(hw, dim=0).to(device).contiguous()


def zero_table(device, lk=LK, hw=HW, dtype=torch.bfloat16):
    return torch.zeros(N_LAT * hw, lk, dtype=dtype, device=device)


@torch.library.custom_op("tempo::masked_sdpa", mutates_args=())
def _masked_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """q [B, n, Lq, d], k/v [B, n, Lk, d] (views), mask [Lq, Lk] -> [B, Lq, n, d] contiguous.
    A custom op so torch.compile treats the kernel call as opaque: Inductor cannot re-layout or materialise the
    broadcast [1, 1, Lq, Lk] mask (reviewer finding, 2026-09-26). Eager and compiled run the same SDPA call."""
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask.view(1, 1, *mask.shape))
    return o.transpose(1, 2).contiguous()


@_masked_sdpa.register_fake
def _(q, k, v, mask):
    b, n, lq, d = q.shape
    return q.new_empty((b, lq, n, v.shape[-1]))


def masked_attention(q, k, v, table, q_start):
    """q [B, Lq, n, d]; k, v [B, Lk, n, d]; table [T, Lk] per-query-token additive logit mask; q_start = absolute
    token index of q's first row. Casts like wan.modules.attention.flash_attention (half inputs, output in q's dtype),
    scale 1/sqrt(d) (SDPA's default, = flash_attention's softmax_scale=None). Returns [B, Lq, n, d]."""
    b, lq, n, d = q.shape
    lk = k.shape[1]
    if not isinstance(q_start, int):
        raise TypeError(f"q_start must be a Python int (a tensor would sync the host), got {type(q_start)}")
    if table.dim() != 2 or table.shape[1] != lk:
        raise ValueError(f"table {tuple(table.shape)} does not match Lk={lk}")
    if q_start % HW or lq % HW or q_start < 0 or q_start + lq > table.shape[0]:
        raise ValueError(f"query rows [{q_start}, {q_start + lq}) are not whole frames inside the table "
                         f"({table.shape[0]} rows)")
    out_dtype = q.dtype

    def half(x):
        return x if x.dtype in _HALF else x.to(torch.bfloat16)

    vh = half(v)
    qh, kh = half(q).to(vh.dtype), half(k).to(vh.dtype)
    if table.dtype != vh.dtype:
        raise TypeError(f"table dtype {table.dtype} != attention dtype {vh.dtype}")
    o = _masked_sdpa(qh.transpose(1, 2), kh.transpose(1, 2), vh.transpose(1, 2), table[q_start:q_start + lq])
    return o.type(out_dtype)


def reference_fp64(q, k, v, frame_table, q_start):
    """fp64 reference for masked_attention on the bf16-rounded inputs: softmax(q k^T / sqrt(d) + bias) v.
    frame_table [21, Lk] (fp32) is indexed by each query's absolute latent frame. Returns [B, Lq, n, d] fp64."""
    b, lq, n, d = q.shape
    f0 = q_start // HW
    rows = frame_table[f0:f0 + lq // HW].to(device=q.device, dtype=torch.float64).repeat_interleave(HW, 0)
    out = []
    for i in range(b):
        qd = q[i].to(torch.bfloat16).double().transpose(0, 1)          # [n, Lq, d]
        kd = k[i].to(torch.bfloat16).double().transpose(0, 1)
        vd = v[i].to(torch.bfloat16).double().transpose(0, 1)
        lg = qd @ kd.transpose(1, 2) / d ** 0.5 + rows[None]
        out.append((torch.softmax(lg, -1) @ vd).transpose(0, 1))
        del qd, kd, vd, lg
    return torch.stack(out)


def sdpa_backend_from_kernels(names):
    """Name the SDPA backend from CUDA kernel names of a profiler trace."""
    s = " ".join(names).lower()
    if "fmha_cutlass" in s or "efficient_attention" in s or "mem_eff" in s:
        return "mem_efficient"
    if "cudnn" in s and ("sdpa" in s or "fmha" in s or "attention" in s):
        return "cudnn"
    if "flash" in s:
        return "flash"
    return "math_or_unknown"


# ---------------------------------------------------------------------------------------------------------------------
# fa2kv: L through Wan's own FA2 kernel by key augmentation (part 2 step 2b choice; pre-registered decision 14).
#
# L's bias is rank-1 per frame: b(f) on the object's key slots, 0 elsewhere. Append KA_DIMS = 8 head dims (d 128 -> 136):
#   q_ext[t] = (b(f(t)), b(f(t)), b(f(t)), 0, ..., 0)      k_ext[i] = (c0, c1, c2, 0, ..., 0) if i in O else 0
#   v_ext    = 0
# with bf16 constants c0 + c1 + c2 = 1 / fp32(128^-0.5) (to ~1e-11). FA2 (softmax_scale = 128^-0.5 passed explicitly)
# then computes (q.k + b(f) * sum(c)) * scale = q.k * scale + b(f) * (1 +- 1e-7) on object slots, and q.k * scale on all
# others. The products b * c are exact in the fp32 MMA accumulator. With b = 0 (B0) the extension adds exact zeros to
# every logit, but d = 136 runs FA2's hdim-160 kernel (other key-block size), so B0 on this path is not guaranteed to be
# bitwise equal to the d = 128 call; B0 is defined as fa2kv with zero tables.
# ---------------------------------------------------------------------------------------------------------------------
KA_DIMS, KA_USED, KA_HEAD = 8, 3, 128


def _head_augmentation_constants(d=KA_HEAD):
    t = 1.0 / float(torch.tensor(d ** -0.5, dtype=torch.float32))
    cs, r = [], t
    for _ in range(KA_USED):
        c = float(torch.tensor(r, dtype=torch.float64).to(torch.bfloat16))
        cs.append(c)
        r -= c
    return cs, abs(sum(cs) - t) / t


KA_C, KA_REL_ERR = _head_augmentation_constants()


def frame_scalar_L(mask, beta, gamma):
    """[21] fp32 b(f) = +beta on frames with mask 1, -gamma with mask 0."""
    m = torch.as_tensor(mask, dtype=torch.float32)
    if m.shape != (N_LAT,) or not set(m.tolist()) <= {0.0, 1.0}:
        raise ValueError(f"mask must be 21 binary entries, got {m.tolist()}")
    return beta * m - gamma * (1 - m)


def fa2kv_tables(mask, obj_idx, beta, gamma, device, lk=LK, hw=HW, dtype=torch.bfloat16):
    """(qb [21*hw] per-query-token b(f), ke [lk, KA_DIMS] key extension) for L(beta, gamma)."""
    b = frame_scalar_L(mask, beta, gamma)
    if not torch.equal(b.to(dtype).float(), b):
        raise ValueError(f"b(f) not exact in {dtype}")
    obj_idx = list(obj_idx)
    if not obj_idx or len(set(obj_idx)) != len(obj_idx) or min(obj_idx) < 0 or max(obj_idx) >= lk:
        raise ValueError(f"bad object slots {obj_idx}")
    ke = torch.zeros(lk, KA_DIMS, dtype=torch.float64)
    ke[obj_idx, :KA_USED] = torch.tensor(KA_C, dtype=torch.float64)
    return (b.to(dtype).repeat_interleave(hw).to(device).contiguous(), ke.to(dtype).to(device).contiguous())


def fa2kv_zero_tables(device, lk=LK, hw=HW, dtype=torch.bfloat16):
    return (torch.zeros(N_LAT * hw, dtype=dtype, device=device), torch.zeros(lk, KA_DIMS, dtype=dtype, device=device))


def fa2kv_augment(q, k, v, tables, q_start):
    """Extended (q, k, v) [B, L, n, d + KA_DIMS]; same casts as flash_attention. Used by fa2kv_attention and tests."""
    qb, ke = tables
    b, lq, n, d = q.shape
    lk = k.shape[1]
    if not isinstance(q_start, int):
        raise TypeError(f"q_start must be a Python int (a tensor would sync the host), got {type(q_start)}")
    if d != KA_HEAD:
        raise ValueError(f"key augmentation constants are for head dim {KA_HEAD}, got {d}")
    if ke.shape != (lk, KA_DIMS) or qb.dim() != 1:
        raise ValueError(f"tables {tuple(qb.shape)}, {tuple(ke.shape)} do not match Lk={lk}")
    if q_start % HW or lq % HW or q_start < 0 or q_start + lq > qb.shape[0]:
        raise ValueError(f"query rows [{q_start}, {q_start + lq}) are not whole frames inside the table "
                         f"({qb.shape[0]} rows)")

    def half(x):
        return x if x.dtype in _HALF else x.to(torch.bfloat16)

    vh = half(v)
    qh, kh = half(q).to(vh.dtype), half(k).to(vh.dtype)
    if qb.dtype != vh.dtype or ke.dtype != vh.dtype:
        raise TypeError(f"table dtypes {qb.dtype}, {ke.dtype} != attention dtype {vh.dtype}")
    z = vh.new_zeros(())
    qx = torch.cat([qh, qb[q_start:q_start + lq].view(1, lq, 1, 1).expand(b, lq, n, KA_USED),
                    z.expand(b, lq, n, KA_DIMS - KA_USED)], -1)
    kx = torch.cat([kh, ke.view(1, lk, 1, KA_DIMS).expand(b, lk, n, KA_DIMS)], -1)
    vx = torch.cat([vh, z.expand(b, lk, n, KA_DIMS)], -1)
    return qx, kx, vx


_FA2 = None


def fa2kv_attention(q, k, v, tables, q_start):
    """L via Wan's FA2 kernel (flash_attn_func, d = 136, softmax_scale = 128^-0.5). Returns [B, Lq, n, 128] in q's dtype."""
    global _FA2
    if _FA2 is None:
        from flash_attn import flash_attn_func
        _FA2 = flash_attn_func
    flash_attn_func = _FA2
    out_dtype = q.dtype
    qx, kx, vx = fa2kv_augment(q, k, v, tables, q_start)
    o = flash_attn_func(qx, kx, vx, softmax_scale=KA_HEAD ** -0.5)
    return o[..., :KA_HEAD].type(out_dtype)


def fa2_kernel_head_dim(names):
    """Head dim of the FA2 forward kernel in a list of CUDA kernel names (None if FA2's forward kernel is absent)."""
    import re
    dims = [int(m.group(1)) for n in names for m in [re.search(r"flash_fwd\w*kernel<Flash_fwd_kernel_traits<(\d+)", n)] if m]
    return max(dims) if dims else None
