"""CPU preflight for scripts/self_forcing_prompt_switch_arms.py's Switcher (step 3c) on a mock Self-Forcing pipeline (dummy tensors).

The mock generator mimics the cache semantics that matter: each pass writes a K value into the self-attention cache
slot of its chunk (derived from its input frames and conditioning), and computes the cross-attention K from the
conditioning only when is_init is False. The mock inference loop issues the pipeline's 7 chunks x 5 passes.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from self_forcing_prompt_switch_arms import NFPB, PASSES, Switcher  # noqa: E402

FS = 4                                    # mock frame_seq_length
NB = 2                                    # mock blocks


class MockPipe:
    def __init__(self):
        self.frame_seq_length = FS
        self.kv_cache1 = [{"k": torch.full((1, 21 * FS), -1.0), "v": torch.full((1, 21 * FS), -1.0)} for _ in range(NB)]
        self.crossattn_cache = [{"k": torch.zeros(1), "v": torch.zeros(1), "is_init": False} for _ in range(NB)]
        self.calls = []

    def forward(self, noisy_image_or_video, conditional_dict, timestep, kv_cache, crossattn_cache, current_start):
        c = conditional_dict["prompt_embeds"]
        for blk in crossattn_cache:
            if not blk["is_init"]:
                blk["k"] = c.clone()
                blk["v"] = c.clone()
                blk["is_init"] = True
        s = slice(current_start, current_start + NFPB * FS)
        # block-causal read: everything before this chunk + this chunk
        ctx = kv_cache[0]["k"][0, :current_start].sum() if current_start else torch.tensor(0.0)
        val = noisy_image_or_video.mean() * 100 + crossattn_cache[0]["k"].item() + 0.001 * ctx
        for blk in kv_cache:
            blk["k"][0, s] = val
            blk["v"][0, s] = val
        self.calls.append({"start": current_start, "cond": c.item(), "t": int(timestep.max()),
                           "x": noisy_image_or_video.mean().item()})
        return None, noisy_image_or_video + 0.0


def cond(v):
    return {"prompt_embeds": torch.tensor([float(v)])}


def run(mode, switch, pre=1.0, full=2.0):
    p = MockPipe()
    sw = Switcher(p, p.forward, sync=lambda: None)
    sw.reset(mode, switch, cond(full))
    for ch in range(7):
        x = torch.full((1, 3), float(ch + 1))            # chunk ch's clean frames have mean ch + 1
        for k in range(PASSES):
            t = torch.zeros(1, 3, dtype=torch.long) if k == PASSES - 1 else torch.full((1, 3), 1000 - 250 * k)
            sw(noisy_image_or_video=x, conditional_dict=cond(pre), timestep=t, kv_cache=p.kv_cache1,
               crossattn_cache=p.crossattn_cache, current_start=ch * NFPB * FS)
    return p, sw


def slot(p, ch):
    return p.kv_cache1[0]["k"][0, ch * NFPB * FS].item()


def test_rf_keeps_self_cache_and_recaches_cross():
    p, sw = run("rf", 3)
    assert sw.n == 35 and len(sw.chunk_ms()) == 7 and sw.n_recache == 0 and len(p.calls) == 35
    assert [c["cond"] for c in p.calls] == [1.0] * 15 + [2.0] * 20
    full, _ = run("rf", None, pre=2.0)
    pre, _ = run("rf", None, pre=1.0)
    assert all(slot(p, ch) == slot(pre, ch) for ch in range(3))              # chunks 0-2 still under the PRE prompt
    assert all(slot(p, ch) != slot(full, ch) for ch in range(7))             # later chunks read the stale PRE context
    assert p.crossattn_cache[0]["k"].item() == 2.0


def test_ll_rebuilds_both_caches_in_chunk_order():
    p, sw = run("ll", 3)
    assert sw.n == 35 and sw.n_recache == 3 and len(p.calls) == 38
    rc = p.calls[15:18]                                    # the re-cache runs before the switch chunk's first pass
    assert [c["start"] for c in rc] == [0, NFPB * FS, 2 * NFPB * FS]
    assert [c["x"] for c in rc] == [1.0, 2.0, 3.0]         # the recorded clean frames of chunks 0, 1, 2
    assert all(c["cond"] == 2.0 and c["t"] == 0 for c in rc)
    assert [c["cond"] for c in p.calls[:15]] == [1.0] * 15 and all(c["cond"] == 2.0 for c in p.calls[18:])
    # the mock's clean frames do not depend on the prompt, so a correct re-cache leaves exactly the cache of a plain
    # full-prompt run (wrong order, positions or frames would change the context-dependent values)
    full, _ = run("rf", None, pre=2.0)
    assert torch.equal(p.kv_cache1[0]["k"], full.kv_cache1[0]["k"])
    assert sw.recache_ms is not None


def test_ll_identity_switch_equals_plain():
    p0, _ = run("rf", None, pre=2.0)
    p1, _ = run("ll", 3, pre=2.0, full=2.0)
    assert torch.equal(p0.kv_cache1[0]["k"], p1.kv_cache1[0]["k"])


def test_ll_switch_at_0_no_recache_passes():
    p, sw = run("ll", 0)
    assert sw.n_recache == 0 and len(p.calls) == 35 and all(c["cond"] == 2.0 for c in p.calls)


def test_ll_clean_frames_are_copies():
    p = MockPipe()
    sw = Switcher(p, p.forward, sync=lambda: None)
    sw.reset("ll", 2, cond(2.0))
    x = torch.full((1, 3), 1.0)
    for ch in range(2):
        for k in range(PASSES):
            sw(noisy_image_or_video=x, conditional_dict=cond(1.0), timestep=torch.zeros(1, 3, dtype=torch.long),
               kv_cache=p.kv_cache1, crossattn_cache=p.crossattn_cache, current_start=ch * NFPB * FS)
        x.fill_(9.0)                                        # a reused buffer overwritten in place
    assert [c[0].mean().item() for c in sw.clean] == [1.0, 9.0]


def test_wrong_switch_position_raises():
    p = MockPipe()
    sw = Switcher(p, p.forward, sync=lambda: None)
    sw.reset("rf", 1, cond(2.0))
    sw.n = 6                                               # pretend the switch chunk's first pass was missed
    with pytest.raises(RuntimeError):
        sw(noisy_image_or_video=torch.ones(1, 3), conditional_dict=cond(1.0), timestep=torch.ones(1, 3),
           kv_cache=p.kv_cache1, crossattn_cache=p.crossattn_cache, current_start=NFPB * FS)


def test_positional_call_raises():
    p = MockPipe()
    sw = Switcher(p, p.forward, sync=lambda: None)
    with pytest.raises(RuntimeError):
        sw(torch.ones(1))


def test_ll_wrong_order_would_be_detected():
    """Sanity of the mock itself: re-caching the chunks in reverse order gives a different cache."""
    p, sw = run("ll", 3)
    q = MockPipe()
    for j in (2, 1, 0):
        q.forward(noisy_image_or_video=torch.full((1, 3), float(j + 1)), conditional_dict=cond(2.0),
                  timestep=torch.zeros(1, 3, dtype=torch.long), kv_cache=q.kv_cache1, crossattn_cache=q.crossattn_cache,
                  current_start=j * NFPB * FS)
    assert slot(q, 2) != slot(p, 2)
