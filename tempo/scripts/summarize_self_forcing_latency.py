#!/usr/bin/env python3
"""Q2 latency from the step-3 latency job: per chunk position, median chunk time of L vs unmodified Self-Forcing.

  summarize_self_forcing_latency.py results/step3/lat_<job>   -> <dir>/summary.json, <dir>/summary.md
Cost = mean over the 7 positions of (median L / median unmodified - 1), per mode; Q2's latency half uses `final`.
"""
import json
import os
import statistics as st
import sys


def load_one(d, name, mode):
    p = os.path.join(d, f"{name}.json")
    if not os.path.isfile(p):
        return None
    rep = json.load(open(p))
    if mode in rep.get("failures", {}):
        raise SystemExit(f"{name}: mode {mode} FAILED: {rep['failures'][mode]['error'][:300]}")
    return rep["modes"][mode], rep.get("tempo")


def load(d, name, mode):
    """One run, or the ABBA pair <name> + <name>_b pooled: medians over all measured videos of both processes."""
    a, b = load_one(d, name, mode), load_one(d, name + "_b", mode)
    if a is None or b is None:
        return a
    (ma, t), (mb, _) = a, b
    n = len(ma["per_position"])
    vids = ma["videos"] + mb["videos"]
    pp = {}
    for i in range(n):
        xs = [v["chunk_ms"][i] for v in vids]
        pp[str(i)] = {"median": st.median(xs), "spread_pct": 100 * (max(xs) - min(xs)) / st.median(xs),
                      "process_medians": [ma["per_position"][str(i)]["median"], mb["per_position"][str(i)]["median"]]}
    pk = [v["peak_reserved_gb"] for v in vids]
    return {"per_position": pp, "videos": vids, "fit": [ma["fit"], mb["fit"]],
            "peak_reserved_gb": {"median": st.median(pk)},
            "one_replay_per_pass": [ma.get("one_replay_per_pass"), mb.get("one_replay_per_pass")]}, t


def main():
    d = sys.argv[1]
    out, lines = {}, [f"# Step 3 latency ({d})", ""]
    for mode, a, b in (("final", "orig_final", "L_final"), ("eager-patched", "orig_eager", "L_eager")):
        A, B = load(d, a, mode), load(d, b, mode)
        if A is None or B is None:
            lines.append(f"{mode}: missing ({a}: {A is not None}, {b}: {B is not None})")
            continue
        (ma, ta), (mb, tb) = A, B
        assert ta in (None, {"tempo": "none"}) and tb["tempo"] == "L", (ta, tb)
        n = len(ma["per_position"])
        pos = []
        for i in range(n):
            ua, ub = ma["per_position"][str(i)]["median"], mb["per_position"][str(i)]["median"]
            pos.append({"pos": i, "orig_ms": ua, "L_ms": ub, "delta_ms": ub - ua, "cost": ub / ua - 1,
                        "orig_spread_pct": ma["per_position"][str(i)]["spread_pct"],
                        "L_spread_pct": mb["per_position"][str(i)]["spread_pct"]})
        cost = st.mean(p["cost"] for p in pos)
        out[mode] = {"positions": pos, "mean_cost": cost, "n_videos": len(ma["videos"]),
                     "orig_fit": ma["fit"], "L_fit": mb["fit"],
                     "peak_reserved_gb": {"orig": ma["peak_reserved_gb"]["median"], "L": mb["peak_reserved_gb"]["median"]},
                     "graphs": {"orig": ma.get("one_replay_per_pass"), "L": mb.get("one_replay_per_pass")}}
        lines += [f"## {mode}: mean cost {100 * cost:+.2f} % ({len(ma['videos'])} vs {len(mb['videos'])} measured videos; median per position)", "",
                  "| position | unmodified ms | L ms | Δ ms | cost | spread unmod / L |", "|---|---|---|---|---|---|"]
        for p in pos:
            lines.append(f"| {p['pos']} | {p['orig_ms']:.1f} | {p['L_ms']:.1f} | {p['delta_ms']:+.1f} | "
                         f"{100 * p['cost']:+.2f} % | {p['orig_spread_pct']:.1f} % / {p['L_spread_pct']:.1f} % |")
        lines += ["", f"peak reserved GB: unmodified {out[mode]['peak_reserved_gb']['orig']:.2f}, "
                      f"L {out[mode]['peak_reserved_gb']['L']:.2f}; one replay per pass: {out[mode]['graphs']}", ""]
    if "final" in out:
        out["Q2_latency_met"] = bool(out["final"]["mean_cost"] <= 0.02)
        lines.append(f"**Q2 latency (final mode, ≤ +2 %): {'met' if out['Q2_latency_met'] else 'MISSED'}**")
    json.dump(out, open(os.path.join(d, "summary.json"), "w"), indent=1)
    open(os.path.join(d, "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
