"""Per-latent-frame masks for Wan2.1 (81 output frames @ 16 fps -> 21 latent frames).

Wan's causal VAE packs latent frame 0 = output frame 0 and latent frame j >= 1 =
output frames 4j-3 .. 4j (temporal stride 4). A latent frame is "on" for an
interval when the majority of its output frames fall inside it (ties -> off).
"""
import re

import numpy as np

FPS = 16
N_OUT = 81
N_LAT = 21
DURATION = N_OUT / FPS  # 5.0625 s

ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}


def latent_to_output_frames(j):
    if not 0 <= j < N_LAT:
        raise ValueError(f"latent frame {j} out of range")
    return [0] if j == 0 else list(range(4 * j - 3, 4 * j + 1))


def output_to_latent_frame(k):
    if not 0 <= k < N_OUT:
        raise ValueError(f"output frame {k} out of range")
    return 0 if k == 0 else (k + 3) // 4


def interval_to_mask(t0, t1, soft=False):
    """Mask for "object visible during [t0, t1) seconds" (t1 may be DURATION)."""
    frac = np.zeros(N_LAT)
    for j in range(N_LAT):
        ks = latent_to_output_frames(j)
        frac[j] = np.mean([t0 <= k / FPS < t1 for k in ks])
    return frac if soft else (frac > 0.5).astype(int)


def parse_temporal_phrase(prompt):
    """Return (t0, t1) seconds for the object's visibility, or raise ValueError.

    Rules (appearance = present from onset to the end of the video, as in the
    TempoControl one-object benchmark):
      "during the <ordinal> second"   -> onset at (n-1) s
      "during the last second"        -> onset at 4 s (the 5th, last full second)
      "in the first/second half"      -> [0, D/2) / [D/2, D)
      "after <n> seconds"             -> onset at n s
      "for the first <n> seconds"     -> [0, n)
    """
    p = prompt.lower()
    m = re.search(r"during the (first|second|third|fourth|fifth|last) second", p)
    if m:
        w = m.group(1)
        n = int(DURATION) if w == "last" else ORDINALS[w]
        return float(n - 1), DURATION
    m = re.search(r"in the (first|second) half", p)
    if m:
        return (0.0, DURATION / 2) if m.group(1) == "first" else (DURATION / 2, DURATION)
    m = re.search(r"after (\d+(?:\.\d+)?) seconds?", p)
    if m:
        return float(m.group(1)), DURATION
    m = re.search(r"for the first (\d+(?:\.\d+)?) seconds?", p)
    if m:
        return 0.0, float(m.group(1))
    raise ValueError(f"no temporal phrase recognised in: {prompt!r}")


def mask_from_prompt(prompt, soft=False):
    t0, t1 = parse_temporal_phrase(prompt)
    return interval_to_mask(t0, t1, soft=soft)


def parse_benchmark_mask(s):
    vals = [int(x) for x in s.strip().split()]
    if len(vals) != N_LAT or set(vals) - {0, 1}:
        raise ValueError(f"benchmark mask must be {N_LAT} values in {{0,1}}: {s!r}")
    return np.array(vals)


TEMPORAL_PHRASE_RE = re.compile(
    r"during the (?:first|second|third|fourth|fifth|last) second of the video"
    r"|in the (?:first|second) half(?: of the video)?"
    r"|after \d+(?:\.\d+)? seconds?"
    r"|for the first \d+(?:\.\d+)? seconds?", re.IGNORECASE)
