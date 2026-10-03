#!/usr/bin/env python3
"""Concatenate every per-video row of results/rows/*.jsonl (parts 1 and 2) into results/runs.jsonl, one row per line,
each augmented with "source_file" (path relative to ~/tempo); write results/runs_index.md with counts per
(phase, run_tag). Re-runnable: rebuilds both files from scratch, so rows appended later (e.g. phase 5) are picked up.

  collect_run_rows.py  -> results/runs.jsonl, results/runs_index.md
Phase 0 rows have no run_tag; the index shows "video=<v>" for them. Rows are copied unchanged otherwise
(no de-duplication; re-runs of the same video are kept and counted in the "repeats" column).
"""
import collections
import glob
import json
import os

TEMPO = os.environ.get("TEMPO_ROOT", os.path.expanduser("~/tempo"))
PART = {0: 1, 1: 1, 2: 1}                      # phases 0-2: part 1; 22, 23, 4, 5, ...: part 2


def main():
    files = sorted(glob.glob(os.path.join(TEMPO, "results", "rows", "*.jsonl")))
    out_p = os.path.join(TEMPO, "results", "runs.jsonl")
    groups = collections.defaultdict(list)
    n, bad = 0, []
    with open(out_p + ".tmp", "w") as out:
        for p in files:
            rel = os.path.relpath(p, TEMPO)
            for k, line in enumerate(open(p), 1):
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError as e:
                    bad.append(f"{rel}:{k}: {e}")
                    continue
                r["source_file"] = rel
                out.write(json.dumps(r) + "\n")
                n += 1
                tag = r.get("run_tag") or (f"video={r['video']}" if "video" in r else "(none)")
                groups[(r.get("phase"), tag)].append(r)
    os.replace(out_p + ".tmp", out_p)

    def phase_key(ph):
        return (PART.get(ph, 2), {22: 2, 23: 3}.get(ph, ph if ph is not None else -1))   # part 2 in step order

    md = ["# results/runs.jsonl index", "",
          f"{n} per-video rows from {len(files)} files in `results/rows/` (each row carries `source_file`). "
          "Part 1 = phases 0–2; part 2 = phases 22 (step 2), 23 (steps 3/3b/3c, Self-Forcing), 4 (step 4), "
          "5 (step 5a, two objects). Rebuild: `python scripts/collect_run_rows.py`.", "",
          "| part | phase | run_tag | rows | prompts | seeds | repeats | source files |",
          "|---|---|---|---|---|---|---|---|"]
    for (ph, tag) in sorted(groups, key=lambda g: (phase_key(g[0]), str(g[1]))):
        rs = groups[(ph, tag)]
        keys = collections.Counter((r.get("prompt_id"), r.get("seed")) for r in rs)
        rep = sum(c - 1 for c in keys.values())
        srcs = sorted({os.path.basename(r["source_file"]) for r in rs})
        src = ", ".join(srcs) if len(srcs) <= 3 else f"{len(srcs)} files ({srcs[0]} … {srcs[-1]})"
        md.append(f"| {PART.get(ph, 2)} | {ph} | {tag} | {len(rs)} | {len({r.get('prompt_id') for r in rs})} | "
                  f"{','.join(str(s) for s in sorted({r.get('seed') for r in rs}, key=str))} | {rep} | {src} |")
    per_phase = collections.Counter(ph for (ph, _), rs in groups.items() for _ in rs)
    md += ["", "Rows per phase: " + ", ".join(f"{ph}: {per_phase[ph]}" for ph in sorted(per_phase, key=phase_key)), "",
           "`repeats` = rows beyond the first for the same (prompt_id, seed) within the group (e.g. the flash B0 and "
           "explicit L reference videos re-timed in both step-2 jobs 162864 and 162925)."]
    if bad:
        md += ["", "Unparseable lines skipped:", ""] + [f"- {b}" for b in bad]
    open(os.path.join(TEMPO, "results", "runs_index.md"), "w").write("\n".join(md) + "\n")
    print(f"wrote {out_p} ({n} rows from {len(files)} files; {len(bad)} bad lines) and results/runs_index.md")


if __name__ == "__main__":
    main()
