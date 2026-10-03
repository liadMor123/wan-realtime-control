"""Explicit cross-attention for Wan2.1 T2V with forward-only, per-frame edits.

`install_cross_attention_patch(model)` swaps WanT2VCrossAttention.forward on every block for
`_forward`, which dispatches on the global controller CTRL:

  CTRL.path == "flash"    : Wan's original flash_attention call (reference)
  CTRL.path == "explicit" : matmul -> (+bias or +S shift) -> softmax -> matmul
  CTRL.path == "fused"    : SDPA with a per-token additive mask (fused_attention.py; arms B0 and L only). The edited
                            branch gets the L table, every other forward the zero table (same kernel).
  CTRL.path == "fa2kv"    : L through Wan's FA2 kernel by key augmentation (fused_attention.py; B0 and L; same table rule).

The explicit path is used for every arm including B0, so arms differ only in
the edit. Inputs are cast to bf16 exactly as flash_attention casts them, the
math runs in fp32, and the output is rounded to bf16 and returned in q's dtype,
again exactly as flash_attention returns it.

Arms (logits l[q,i] = q.k_i / sqrt(d), per head; f(q) = latent frame of query q):
  B0 : no edit
  U  : l[q,i] += beta                     for i in T (temporal-phrase tokens), all frames
  L  : l[q,i] += beta*m_f - gamma*(1-m_f) for i in O (object tokens). With K objects (two-object benchmark),
       object k adds beta*m_k,f - gamma*(1-m_k,f) on its own slots O_k; the O_k must be disjoint. Object 1 is
       (obj_idx, mask), objects 2..K are extra_objs (arm L on the explicit path only; K = 1 is unchanged).
  S  : per query, set s = sum_{i in O} a[q,i] to s* = s_lo + m_f*(s_hi - s_lo) via the
       closed-form shift delta = log[s*(1-s) / (s(1-s*))] added to l[q,i], i in O. delta is
       computed as logit(s*) - (lse(l_O) - lse(l_notO)), exact when s underflows fp32 or is
       near 1; the softmax is then taken once on the shifted logits.
  P  : l[q,i] = -inf for i in O on frames with m_f = 0 (object keys masked out)
"""
from dataclasses import dataclass, field
from typing import List, Optional

import torch

N_LAT, HW = 21, 30 * 52


@dataclass
class Controller:
    path: str = "explicit"          # "flash" | "explicit" | "fused" | "fa2kv"
    arm: str = "B0"                 # B0 | U | L | S | P
    beta: float = 0.0
    gamma: float = 0.0
    s_hi: float = 0.0
    s_lo: float = 0.0
    mask: Optional[torch.Tensor] = None      # [21] float in [0, 1]
    obj_idx: List[int] = field(default_factory=list)
    temporal_idx: List[int] = field(default_factory=list)
    extra_objs: list = field(default_factory=list)   # objects 2..K: [(slot list, [21] mask)], arm L only
    branches: str = "cond"          # "cond" | "both": which CFG branch gets the edit
    k_steps: Optional[int] = None   # edit only steps < k_steps (None = all)
    # runtime state, set by the sampler
    step: int = 0
    branch: str = "cond"
    # diagnostics: per (step, layer) mean share per latent frame of each index group
    # (e.g. obj / tmp / pad), cond branch, measured AFTER the edit (post-softmax)
    record_shares: bool = False
    share_groups: dict = field(default_factory=dict)
    shares: list = field(default_factory=list)
    _layer: int = 0
    _cache: dict = field(default_factory=dict)   # per-configuration device tensors

    def objects(self):
        """[(slots, [21] float32 mask)] for objects 1..K; raises if two objects share a key slot."""
        objs = [(self.obj_idx, self.mask)] + [(list(i), m) for i, m in self.extra_objs]
        seen = set()
        for idx, _ in objs:
            if seen & set(idx):
                raise ValueError(f"object token sets overlap: {[o[0] for o in objs]}")
            seen |= set(idx)
        return [(idx, m.to(dtype=torch.float32)) for idx, m in objs]

    def edit_active(self):
        if self.arm == "B0":
            return False
        if self.extra_objs and self.arm != "L":
            raise NotImplementedError(f"K > 1 objects are defined for arm L only, not {self.arm}")
        if self.branches != "cond":
            raise NotImplementedError(
                "edit in both CFG branches is undefined: the uncond context is the negative "
                "prompt, which has no object/temporal tokens at obj_idx/temporal_idx")
        if self.branch != "cond":
            return False
        if self.k_steps is not None and self.step >= self.k_steps:
            return False
        return True


CTRL = Controller()


def _cached_tensor(ctrl, key, device, build):
    k = (key, str(device))
    if k not in ctrl._cache:
        ctrl._cache[k] = build()
    return ctrl._cache[k]


def _frame_bias_table(ctrl, lk, device):
    """[21, lk] additive logit bias for arms U, L, P (None if the arm adds none). Built once."""
    if ctrl.arm not in ("U", "L", "P"):
        return None

    def build():
        m = ctrl.mask.to(dtype=torch.float32)                 # CPU
        bias = torch.zeros(N_LAT, lk, dtype=torch.float32)
        if ctrl.arm == "U":
            bias[:, ctrl.temporal_idx] = ctrl.beta
        elif ctrl.arm == "L":
            for idx, mk in ctrl.objects():                    # K = 1: the single-object table, unchanged
                bias[:, idx] = (ctrl.beta * mk - ctrl.gamma * (1 - mk))[:, None]
        else:
            off = (m < 0.5).nonzero().flatten()
            bias[off[:, None], torch.tensor(ctrl.obj_idx)[None, :]] = float("-inf")
        return bias.to(device)
    return _cached_tensor(ctrl, ("bias", lk), device, build)


def _edited_attention(qh, kh, vh, ctrl, edit):
    n, lq, d = qh.shape
    lk = kh.shape[1]
    logits = torch.matmul(qh, kh.transpose(1, 2)).mul_(d ** -0.5)   # [n, Lq, Lk]
    if edit:
        bias = _frame_bias_table(ctrl, lk, logits.device)
        if bias is not None:
            logits.view(n, N_LAT, HW, lk).add_(bias.view(1, N_LAT, 1, lk))
        elif ctrl.arm == "S":
            oi = _cached_tensor(ctrl, "obj", logits.device, lambda: torch.tensor(ctrl.obj_idx, device=logits.device))
            t = _cached_tensor(ctrl, "target", logits.device, lambda: (
                ctrl.s_lo + ctrl.mask.float() * (ctrl.s_hi - ctrl.s_lo)).to(logits.device).view(1, N_LAT, 1, 1))
            # delta = logit(s*) - logit(s), with logit(s) = lse(l_O) - lse(l_notO): exact for s
            # near 0 and near 1 (no log(1 - s) cancellation).
            lO = logits.index_select(-1, oi)                                  # [n, Lq, |O|]
            lse_O = torch.logsumexp(lO, -1, keepdim=True)
            logits.index_fill_(-1, oi, float("-inf"))
            lse_notO = torch.logsumexp(logits, -1, keepdim=True)
            logit_s = (lse_O - lse_notO).view(n, N_LAT, HW, 1)
            delta = torch.log(t) - torch.log1p(-t) - logit_s
            logits.index_copy_(-1, oi, (lO.view(n, N_LAT, HW, -1) + delta).view(n, lq, -1))
    p = torch.softmax(logits, dim=-1)
    del logits
    if ctrl.record_shares and ctrl.branch == "cond":
        rec = {}
        for name, idx in ctrl.share_groups.items():
            g = p.index_select(-1, torch.tensor(idx, device=p.device)).sum(-1)
            rec[name] = g.view(n, N_LAT, HW).mean(dim=(0, 2)).cpu()
        ctrl.shares.append((ctrl.step, ctrl._layer, rec))
    return torch.matmul(p, vh)                               # [n, Lq, d]


def fused_mask_table(ctrl, lk, device):
    """Per-query-token bf16 table for the fused path: L's table while the edit is active, else all zeros."""
    from .fused_attention import frame_table_L, token_table, zero_table
    if ctrl.arm not in ("B0", "L"):
        raise NotImplementedError(f"arm {ctrl.arm} is not defined on the fused path")
    if ctrl.extra_objs:
        raise NotImplementedError("K > 1 objects are implemented on the explicit path only")
    if ctrl.edit_active():
        return _cached_tensor(ctrl, ("tok_L", lk), device, lambda: token_table(
            frame_table_L(ctrl.mask, ctrl.obj_idx, ctrl.beta, ctrl.gamma, lk), device))
    return _cached_tensor(ctrl, ("tok_0", lk), device, lambda: zero_table(device, lk))


def fa2kv_tables_for_controller(ctrl, lk, device):
    """fa2kv tables: L's while the edit is active, else zeros (B0, uncond branch, steps >= k)."""
    from .fused_attention import fa2kv_tables, fa2kv_zero_tables
    if ctrl.arm not in ("B0", "L"):
        raise NotImplementedError(f"arm {ctrl.arm} is not defined on the fa2kv path")
    if ctrl.extra_objs:
        raise NotImplementedError("K > 1 objects are implemented on the explicit path only")
    if ctrl.edit_active():
        return _cached_tensor(ctrl, ("ka_L", lk), device, lambda: fa2kv_tables(
            ctrl.mask, ctrl.obj_idx, ctrl.beta, ctrl.gamma, device, lk))
    return _cached_tensor(ctrl, ("ka_0", lk), device, lambda: fa2kv_zero_tables(device, lk))


def explicit_attention(q, k, v, ctrl):
    """q [1, Lq, n, d]; k, v [1, Lk, n, d]. Returns [1, Lq, n, d] in q's dtype (bf16-rounded)."""
    out_dtype = q.dtype
    b, lq, n, d = q.shape
    assert b == 1 and lq == N_LAT * HW, (b, lq)
    qh = q[0].to(torch.bfloat16).transpose(0, 1).float()     # [n, Lq, d]
    kh = k[0].to(torch.bfloat16).transpose(0, 1).float()     # [n, Lk, d]
    vh = v[0].to(torch.bfloat16).transpose(0, 1).float()
    edit = ctrl.edit_active()
    # bf16 values are exact in TF32 and products accumulate in fp32, so TF32 changes
    # QK^T only through summation order; P@V under TF32 keeps 10 mantissa bits of P,
    # more than flash-attn's bf16 P. Scoped here so no other fp32 matmul is affected.
    # The DiT forward runs under bf16 autocast, which would cast these fp32 matmuls to bf16.
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        with torch.autocast("cuda", enabled=False):
            out = _edited_attention(qh, kh, vh, ctrl, edit)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
    return out.transpose(0, 1).unsqueeze(0).to(torch.bfloat16).to(out_dtype)


def install_cross_attention_patch(model):
    """Patch every block's cross-attention forward. Idempotent."""
    from wan.modules.attention import flash_attention
    from wan.modules.model import WanT2VCrossAttention

    from .fused_attention import fa2kv_attention, masked_attention

    def _forward(self, x, context, context_lens):
        b, n, d = x.size(0), self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        if CTRL.path == "flash":
            assert CTRL.arm == "B0", "the flash path carries no edit"
            x = flash_attention(q, k, v, k_lens=context_lens)
        elif CTRL.path == "fa2kv":
            assert context_lens is None, "WanModel passes context_lens=None; padding is attended"
            CTRL._layer = self._tempo_layer
            x = fa2kv_attention(q, k, v, fa2kv_tables_for_controller(CTRL, k.shape[1], q.device), 0)
        elif CTRL.path == "fused":
            assert context_lens is None, "WanModel passes context_lens=None; padding is attended"
            CTRL._layer = self._tempo_layer
            x = masked_attention(q, k, v, fused_mask_table(CTRL, k.shape[1], q.device), 0)
        else:
            assert context_lens is None, "WanModel passes context_lens=None; padding is attended"
            CTRL._layer = self._tempo_layer
            x = explicit_attention(q, k, v, CTRL)
        x = x.flatten(2)
        return self.o(x)

    for i, blk in enumerate(model.blocks):
        ca = blk.cross_attn
        assert type(ca) is WanT2VCrossAttention, type(ca)
        ca._tempo_layer = i
        ca.forward = _forward.__get__(ca)
    return model
