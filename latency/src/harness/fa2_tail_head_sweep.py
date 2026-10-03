#!/usr/bin/env python3
"""
FlashAttention-2 wave-quantization diagnostic: kernel time vs. head count.

Times the varlen FA2 forward at the real query length (Q = 4680, d = 128) for
H = 9..24 heads at positions 3 and 6 (K = 18720 / 32760). The CTA count is
ceil(Q/128) * H on 216 resident CTAs per wave (2 per SM x 108 SMs), so time
should step up exactly where ceil(37H/216) increments if the tail is wave
quantization. Measurement only -- no kernel is written.

Writes the file named by $J6_FA2_OUT (default j6_fa2sweep.json): one row per
(position, heads) with CTAs, waves and median ms over 25 timed iterations.
"""
import json, math, os, statistics as st, sys
import torch
sys.path.insert(0, os.getcwd())
from wan.modules.attention import flash_attention

dev = torch.device("cuda"); torch.set_grad_enabled(False)
Q, HD, TILE, CTA_PER_WAVE = 4680, 128, 128, 216   # 2 CTAs/SM x 108 SMs (ncu)
QT = math.ceil(Q / TILE)                           # 37 query tiles
rows = []
for pos, K in ((3, 18720), (6, 32760)):
    for H in range(9, 25):
        q = torch.randn([1, Q, H, HD], device=dev, dtype=torch.bfloat16) * 0.05
        k = torch.randn([1, K, H, HD], device=dev, dtype=torch.bfloat16) * 0.05
        v = torch.randn([1, K, H, HD], device=dev, dtype=torch.bfloat16) * 0.05
        for _ in range(5):
            flash_attention(q, k, v)
        torch.cuda.synchronize()
        ts = []
        for _ in range(25):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record(); flash_attention(q, k, v); e.record()
            torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
        ms = st.median(ts)
        cta = QT * H
        waves = cta / CTA_PER_WAVE
        rows.append({"position": pos, "K": K, "heads": H, "ctas": cta,
                     "waves_exact": round(waves, 3), "waves_ceil": math.ceil(waves),
                     "ms": round(ms, 4), "us_per_cta": round(1e3 * ms / cta, 4)})
        del q, k, v
        torch.cuda.empty_cache()

for pos in (3, 6):
    print(f"\n=== position {pos} (K={18720 if pos==3 else 32760}) ===")
    print(f"{'H':>4}{'CTAs':>7}{'waves':>8}{'ceil':>6}{'ms':>10}{'us/CTA':>10}   step")
    prev = None
    for r in [x for x in rows if x["position"] == pos]:
        mark = ""
        if prev is not None and r["waves_ceil"] != prev:
            mark = f"  <-- wave count {prev} -> {r['waves_ceil']}"
        prev = r["waves_ceil"]
        print(f"{r['heads']:>4}{r['ctas']:>7}{r['waves_exact']:>8.2f}{r['waves_ceil']:>6}"
              f"{r['ms']:>10.4f}{r['us_per_cta']:>10.4f}{mark}")

json.dump(rows, open(os.environ.get("J6_FA2_OUT", "j6_fa2sweep.json"), "w"), indent=2)
print("\n[fa2sweep] written")
