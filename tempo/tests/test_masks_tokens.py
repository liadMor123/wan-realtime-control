"""CPU tests: latent-frame mapping, free-form parser vs. the benchmark masks, token lookup.

Run on the login node:  ~/tempo/venv/bin/python -m pytest -q ~/tempo/tests (TEMPO_ROOT overrides ~/tempo)
"""
import os
import sys

import numpy as np
import pytest

TEMPO_ROOT = os.environ.get("TEMPO_ROOT", os.path.expanduser("~/tempo"))
sys.path.insert(0, os.path.join(TEMPO_ROOT, "src"))
from tempo_ctrl import benchmark, masks  # noqa: E402
from tempo_ctrl.tokens import load_tokenizer, object_token_indices, temporal_token_indices  # noqa: E402


def test_latent_frame_mapping_covers_81_frames_once():
    seen = [k for j in range(masks.N_LAT) for k in masks.latent_to_output_frames(j)]
    assert seen == list(range(masks.N_OUT))
    assert all(masks.output_to_latent_frame(k) == j
               for j in range(masks.N_LAT) for k in masks.latent_to_output_frames(j))
    assert masks.latent_to_output_frames(1) == [1, 2, 3, 4]
    assert masks.latent_to_output_frames(20) == [77, 78, 79, 80]


def test_parser_reproduces_every_benchmark_mask():
    rows = benchmark.load_one_object()
    assert len(rows) == 80
    for r in rows:
        assert np.array_equal(masks.mask_from_prompt(r["prompt"]), r["mask"]), r["prompt"]


def test_parser_free_form():
    half = masks.mask_from_prompt("a dog appears only in the second half of the video")
    assert half[:10].sum() == 0 and half[11:].all()
    first2 = masks.mask_from_prompt("a cat is visible for the first 2 seconds")
    assert first2[:8].all() and not first2[9:].any()
    with pytest.raises(ValueError):
        masks.mask_from_prompt("a dog runs in a park")


def _tempocontrol_indices(prompt, token, hf_tokenizer):
    """Verbatim logic of TempoControl generate_benchmark_videos.py:replace_token_with_token_idx."""
    enc = hf_tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=True)
    offsets = enc["offset_mapping"]
    pos, start = [], 0
    while True:
        s = prompt.find(token, start)
        if s == -1:
            break
        pos.append((s, s + len(token)))
        start = s + len(token)
    idx = set()
    for a, b in pos:
        for i, (s, e) in enumerate(offsets):
            if not (e <= a or s >= b):
                idx.add(i)
    return sorted(idx)


@pytest.fixture(scope="module")
def tok():
    return load_tokenizer(benchmark.WAN_ROOT, benchmark.CKPT)


def test_object_tokens_match_tempocontrol_and_decode_to_object(tok):
    for r in benchmark.load_one_object():
        idx, n_real, pieces = object_token_indices(tok, r["prompt"], r["temp_object"])
        assert idx == _tempocontrol_indices(r["prompt"], r["temp_object"], tok.tokenizer), r["prompt"]
        assert max(idx) < n_real <= 512
        decoded = tok.tokenizer.convert_tokens_to_string(pieces).strip()
        assert decoded == r["temp_object"], (decoded, r["temp_object"])


def test_temporal_tokens_cover_phrase_only(tok):
    for r in benchmark.load_one_object():
        idx, _, pieces = temporal_token_indices(tok, r["prompt"])
        text = tok.tokenizer.convert_tokens_to_string(pieces).strip()
        assert text.startswith("during the") and text.endswith("of the video"), text
        obj_idx, _, _ = object_token_indices(tok, r["prompt"], r["temp_object"])
        assert not set(idx) & set(obj_idx)
