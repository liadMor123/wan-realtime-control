#!/usr/bin/env python3
"""
Position-dependent split-KV rule s(pos) from measured attention times (J10).

Times flash_attn_with_kvcache for num_splits s = 1..8 (and the FA2 heuristic)
and the varlen call at all seven chunk positions. From a linear fit of time vs
K at each s it extracts the K-independent combine cost combine(s) (a line
through s in {1, 2, 4}), recovers the unquantized floor from s = 1 and the
wave-quantization waste of 444 CTAs on 108 SMs, and predicts the attention
time for every (pos, s). The rule is argmin_s of the prediction per position;
it is validated against the measurements and summed over the seven positions.
Measurement plus closed-form rule; no kernel written.

Writes the file named by $J10_OUT (default j10_rule.json).
"""
import json, math, os, statistics as st, sys
import numpy as np, torch
sys.path.insert(0, os.getcwd())
from flash_attn import flash_attn_with_kvcache
from wan.modules.attention import flash_attention

dev = torch.device("cuda"); torch.set_grad_enabled(False)
Q, H, HD, SM, TILE = 4680, 12, 128, 108, 128
CTA = math.ceil(Q / TILE) * H                      # 444, from the J6 staircase
KMAX = 32760
kc = torch.randn([1, KMAX, H, HD], device=dev, dtype=torch.bfloat16) * 0.05
vc = torch.randn([1, KMAX, H, HD], device=dev, dtype=torch.bfloat16) * 0.05
q = torch.randn([1, Q, H, HD], device=dev, dtype=torch.bfloat16) * 0.05


def timeit(fn, n=25, warm=6):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(n):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    return st.median(ts)


def waves(s):
    return math.ceil(CTA * s / SM)


def waste(s):
    return (waves(s) - CTA * s / SM) / waves(s)


measured = {}
print("measured attention ms, s = 1..8 and the FA2 heuristic, positions 0..6")
print(f"{'pos':>4}{'K':>7}" + "".join(f"{('s='+str(s)):>9}" for s in range(1, 9)) + f"{'heur':>9}{'varlen':>9}")
for pos in range(7):
    K = (pos + 1) * 4680
    cs = torch.full((1,), K, dtype=torch.int32, device=dev)
    row = {}
    for s in list(range(1, 9)) + [0]:
        row[s] = timeit(lambda: flash_attn_with_kvcache(q, kc, vc, cache_seqlens=cs,
                                                        causal=False, num_splits=s))
    row["varlen"] = timeit(lambda: flash_attention(q, kc[:, :K], vc[:, :K]))
    measured[pos] = row
    print(f"{pos:>4}{K:>7}" + "".join(f"{row[s]:>9.4f}" for s in range(1, 9))
          + f"{row[0]:>9.4f}{row['varlen']:>9.4f}", flush=True)

# ---- combine(s): K-independent term, from a fit of ms vs K at each s --------
Ks = np.array([(p + 1) * 4680 for p in range(7)], float)
alpha, beta = {}, {}
for s in range(1, 9):
    y = np.array([measured[p][s] for p in range(7)])
    A = np.vstack([np.ones(7), Ks]).T
    (a_, b_), *_ = np.linalg.lstsq(A, y, rcond=None)
    alpha[s], beta[s] = float(a_), float(b_)
print("\nK-independent term alpha(s) [ms] and slope beta(s) [ms per token]:")
for s in range(1, 9):
    print(f"  s={s}: alpha={alpha[s]:.4f}  beta={beta[s]*1e6:.4f} us/1k-K  waves={waves(s):>2} waste={100*waste(s):5.2f}%")

# combine(s) as a line through s in {1,2,4}, per the brief
sA = np.array([1, 2, 4], float); yA = np.array([alpha[1], alpha[2], alpha[4]])
A = np.vstack([np.ones(3), sA]).T
(c0, c1), *_ = np.linalg.lstsq(A, yA, rcond=None)
combine = lambda s: c0 + c1 * s
print(f"\ncombine(s) = {c0:.4f} + {c1:.4f}*s  ms   (line through s in 1,2,4)")

# unquantized floor from s=1: measured = floor/(1-waste(1)) + combine(1)
floor = {p: (measured[p][1] - combine(1)) * (1 - waste(1)) for p in range(7)}
pred = lambda p, s: floor[p] / (1 - waste(s)) + combine(s)

print("\n=== rule: s(pos) = argmin over s in 1..8 of predicted attention time ===")
print(f"{'pos':>4}{'K':>7}{'s*':>4}{'pred ms':>10}{'measured@s*':>10}{'measured@s=4':>10}{'heur':>9}{'varlen':>9}{'best measured s':>12}")
rule = {}
for p in range(7):
    ps = {s: pred(p, s) for s in range(1, 9)}
    s_star = min(ps, key=ps.get)
    best_meas = min(range(1, 9), key=lambda s: measured[p][s])
    rule[p] = s_star
    print(f"{p:>4}{(p+1)*4680:>7}{s_star:>4}{ps[s_star]:>10.4f}{measured[p][s_star]:>10.4f}"
          f"{measured[p][4]:>10.4f}{measured[p][0]:>9.4f}{measured[p]['varlen']:>9.4f}{best_meas:>12}")

print("\n=== validation: predicted vs measured at positions 0, 3, 6 ===")
for p in (0, 3, 6):
    for s in (1, 2, 4, 8):
        e = 100 * (pred(p, s) - measured[p][s]) / measured[p][s]
        print(f"  pos {p} s={s}: predicted {pred(p,s):.4f} measured {measured[p][s]:.4f}  err {e:+.2f}%")

tot_rule = sum(measured[p][rule[p]] for p in range(7))
tot_s4 = sum(measured[p][4] for p in range(7))
tot_h = sum(measured[p][0] for p in range(7))
tot_v = sum(measured[p]["varlen"] for p in range(7))
print(f"\nsum over the 7 positions (attention ms/pass-equivalent):")
print(f"  varlen {tot_v:.3f} | FA2 heuristic {tot_h:.3f} | fixed s=4 {tot_s4:.3f} | rule {tot_rule:.3f}")
print(f"  rule vs fixed s=4: {100*(tot_rule-tot_s4)/tot_s4:+.2f}%   rule vs heuristic: {100*(tot_rule-tot_h)/tot_h:+.2f}%")

json.dump({"measured": {str(p): {str(k): v for k, v in r.items()} for p, r in measured.items()},
           "alpha": alpha, "beta": beta, "combine_line": [c0, c1],
           "waves": {s: waves(s) for s in range(1, 9)},
           "waste": {s: waste(s) for s in range(1, 9)},
           "floor": floor, "rule": rule,
           "totals": {"varlen": tot_v, "heuristic": tot_h, "s4": tot_s4, "rule": tot_rule}},
          open(os.environ.get("J10_OUT", "j10_rule.json"), "w"), indent=2)
print("\n[j10] written")
