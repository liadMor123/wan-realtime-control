"""
ncu target for the J13 fused kernel (attribution only).

Inside an explicit CUDA profiler range, after warmup (including Triton
autotuning), launches once each: the UNFUSED pair -- flash_attn_with_kvcache
with num_splits = s (split kernel + combine) followed by F.linear -- and the
FUSED pair -- the same split kernel with return_partials=True followed by
fused_merge_outproj. Shapes are the real ones at chunk position $J13_POS
(default 6, K = 32760) with $J13_SPLITS splits (default 4). Requires the
patched flash_attn build that exposes return_partials.

Writes nothing; ncu collects the kernels in the range. Prints the max abs
difference between the two outputs as a sanity check.
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fused_merge_outproj import fused_merge_outproj  # noqa: E402
from flash_attn import flash_attn_with_kvcache  # noqa: E402

B, Q, H, D, N, KMAX = 1, 4680, 12, 128, 1536, 32760
pos = int(os.environ.get("J13_POS", "6"))
num_splits = int(os.environ.get("J13_SPLITS", "4"))
torch.manual_seed(0)
dev = "cuda"
q = torch.randn(B, Q, H, D, device=dev, dtype=torch.bfloat16)
k = torch.randn(B, KMAX, H, D, device=dev, dtype=torch.bfloat16)
v = torch.randn(B, KMAX, H, D, device=dev, dtype=torch.bfloat16)
W = (torch.randn(N, H * D, device=dev) * 0.02).to(torch.bfloat16)
bias = (torch.randn(N, device=dev) * 0.02).to(torch.bfloat16)
cache_seqlens = torch.full((B,), (pos + 1) * 4680, dtype=torch.int32, device=dev)

for _ in range(3):   # warm (incl. Triton autotune) outside the profiler range
    o = flash_attn_with_kvcache(q, k, v, cache_seqlens=cache_seqlens, num_splits=num_splits)
    y = F.linear(o.flatten(2), W, bias)
    _, _, lse_acc, o_acc = flash_attn_with_kvcache(q, k, v, cache_seqlens=cache_seqlens,
                                                   num_splits=num_splits, return_partials=True)
    y_fused = fused_merge_outproj(o_acc, lse_acc, W, bias)
torch.cuda.synchronize()

torch.cuda.cudart().cudaProfilerStart()
# unfused: split kernel + combine, then the cuBLAS/ATen GEMM with bias
o = flash_attn_with_kvcache(q, k, v, cache_seqlens=cache_seqlens, num_splits=num_splits)
y = F.linear(o.flatten(2), W, bias)
# fused: split kernel only (partials), then the Triton merge + out-projection
_, _, lse_acc, o_acc = flash_attn_with_kvcache(q, k, v, cache_seqlens=cache_seqlens,
                                               num_splits=num_splits, return_partials=True)
y_fused = fused_merge_outproj(o_acc, lse_acc, W, bias)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
print("ncu target done", float((y.float() - y_fused.float()).abs().max()))
