"""CPU tests for the two-object benchmark (part 2, step 5; brief §5): K-object L, pair data, pair selection.

Run on the login node:  ~/tempo/venv/bin/python -m pytest -q ~/tempo/tests (TEMPO_ROOT overrides ~/tempo)
"""
import csv
import json
import os
import re
import sys

import numpy as np
import pytest
import torch

TEMPO_ROOT = os.environ.get("TEMPO_ROOT", os.path.expanduser("~/tempo"))
sys.path.insert(0, os.path.join(TEMPO_ROOT, "src"))
sys.path.insert(0, os.path.join(TEMPO_ROOT, "scripts"))
from tempo_ctrl import benchmark  # noqa: E402
from tempo_ctrl.fused_attention import LK, frame_table_L  # noqa: E402
from tempo_ctrl.tokens import load_tokenizer, span_token_indices  # noqa: E402
from tempo_ctrl.cross_attention_arms import CTRL, Controller, _frame_bias_table, fa2kv_tables_for_controller, fused_mask_table  # noqa: E402
from test_masks_tokens import _tempocontrol_indices  # noqa: E402

import select_scorable_two_object_pairs  # noqa: E402
import run_official_temporal_accuracy  # noqa: E402
from generate_benchmark_videos import parse_ids  # noqa: E402

CPU = torch.device("cpu")
L22 = {"arm": "L", "beta": 2.0, "gamma": 2.0, "seed": 42, "bench": "two_objects"}


@pytest.fixture(scope="module")
def tok():
    return load_tokenizer(benchmark.WAN_ROOT, benchmark.CKPT)


@pytest.fixture(scope="module")
def pairs():
    return benchmark.load_two_objects()


def _raw_csv():
    with open(benchmark.TWO_OBJECTS_CSV, newline="") as f:
        return list(csv.DictReader(f))


def _step5a_ids():
    return parse_ids(open(os.path.join(benchmark.TEMPO, "data", "step5a_ids.txt")).read().strip())


def _bias(ctrl):
    ctrl._cache = {}
    return _frame_bias_table(ctrl, LK, CPU)


def _pre_k_objects_L(ctrl, lk=LK):
    """Frozen copy of cross_attention_arms._frame_bias_table for arm L before the K-object change (single object)."""
    m = ctrl.mask.to(dtype=torch.float32)
    bias = torch.zeros(21, lk, dtype=torch.float32)
    bias[:, ctrl.obj_idx] = (ctrl.beta * m - ctrl.gamma * (1 - m))[:, None]
    return bias


# ------------------------------------------------------------------ pair data and selection
def test_pairs_come_from_the_file(pairs):
    raw = _raw_csv()
    assert len(pairs) == len(raw) == 82
    for i, (p, r) in enumerate(zip(pairs, raw)):
        assert p["prompt_id"] == i and p["prompt"] == r["prompt"] and p["original_prompt"] == r["original_prompt"]
        assert (p["temp_object"], p["static_object"]) == (r["temp_object"], r["static_object"])
        assert benchmark.row_video_name(p) == f"{r['original_prompt']}-0.mp4"
        # brief §5 facts: one timing pattern
        assert r["control_signal1"].split() == ["0"] * 10 + ["1"] * 11 and r["control_signal2"].split() == ["1"] * 21
    assert len({p["original_prompt"] for p in pairs}) == 82
    assert benchmark.row_video_name(benchmark.load_one_object()[0]) == benchmark.video_name(benchmark.load_one_object()[0]["prompt"])


def test_excluded_pairs_are_exactly_the_non_coco_names(pairs):
    names = select_scorable_two_object_pairs.yolo_names()
    assert len(names) == 80 and {"person", "tv", "cell phone", "hair drier", "sports ball"} <= names
    assert tuple(select_scorable_two_object_pairs.unscorable(pairs, names)) == select_scorable_two_object_pairs.UNSCORABLE
    assert len(pairs) - len(select_scorable_two_object_pairs.UNSCORABLE) == 65


def test_step5a_ids_file_is_first_20_scorable(pairs):
    ids = _step5a_ids()
    assert ids == select_scorable_two_object_pairs.select_first_scorable(pairs, select_scorable_two_object_pairs.yolo_names())
    assert ids == [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 18, 19, 20, 21, 22, 23, 24, 25, 26]
    assert not set(ids) & set(select_scorable_two_object_pairs.UNSCORABLE)


def test_shards_partition_the_20_pairs():
    ids = _step5a_ids()
    shards = [ids[si::4] for si in range(4)]            # generate_benchmark_videos.py: parse_ids(ids)[si::sn]
    assert [len(s) for s in shards] == [5] * 4 and sorted(sum(shards, [])) == ids


def test_run_tags_are_new_and_old_tags_unchanged():
    import json
    new = [benchmark.run_tag(r) for r in json.load(open(os.path.join(benchmark.TEMPO, "runs", "step5a.json")))]
    assert new == ["B0_2obj_s42", "L_b2g2_2obj_s42"]
    old = [benchmark.run_tag(r) for r in json.load(open(os.path.join(benchmark.TEMPO, "runs", "step4.json")))]
    assert old == ["B0_s42", "L_b2g2_s42"]
    import glob
    others = {benchmark.run_tag(r) for f in glob.glob(os.path.join(benchmark.TEMPO, "runs", "*.json"))
              if not f.endswith("step5a.json") for r in json.load(open(f))}
    assert not set(new) & others


def test_metric_subset_csv_keeps_rows_and_names_videos_by_original_prompt(tmp_path):
    ids = _step5a_ids()
    csv_path, sub_name, res_name, col = run_official_temporal_accuracy.BENCH["two-object"]
    assert csv_path == benchmark.TWO_OBJECTS_CSV and col == "original_prompt"
    assert res_name == "temporal_accuracy_two_objects.json"
    rows = run_official_temporal_accuracy.write_subset_csv(ids, str(tmp_path / sub_name), csv_path)
    raw = _raw_csv()
    with open(tmp_path / sub_name, newline="") as f:
        assert list(csv.DictReader(f)) == [raw[i] for i in ids] == rows


# ------------------------------------------------------------------ token sets
def test_token_sets_decode_to_object_words_and_do_not_overlap(tok, pairs):
    names = select_scorable_two_object_pairs.yolo_names()
    scorable = [p for p in pairs if p["prompt_id"] not in select_scorable_two_object_pairs.UNSCORABLE]
    assert len(scorable) == 65
    for p in scorable:
        info = benchmark.configure_controller(L22, p, tok)
        for obj, idx in ((p["temp_object"], info["obj_idx"]), (p["static_object"], info["static_idx"])):
            spans = [m.span() for m in re.finditer(r"\b" + re.escape(obj) + r"\b", p["prompt"])]
            assert spans, (p["prompt_id"], obj)
            union = set()
            for sp in spans:                                     # every occurrence decodes back to the object word
                oi, n_real, enc = span_token_indices(tok, p["prompt"], [sp])
                pieces = [tok.tokenizer.convert_ids_to_tokens(enc["input_ids"][i]) for i in oi]
                assert tok.tokenizer.convert_tokens_to_string(pieces).strip() == obj, (p["prompt_id"], obj, pieces)
                union |= set(oi)
            assert sorted(union) == idx and max(idx) < n_real <= 512
            assert idx == _tempocontrol_indices(p["prompt"], obj, tok.tokenizer), (p["prompt_id"], obj)
            assert str(obj).strip().lower() in names
        assert not set(info["obj_idx"]) & set(info["static_idx"]), p["prompt_id"]
        assert not set(info["tmp_idx"]) & (set(info["obj_idx"]) | set(info["static_idx"])), p["prompt_id"]
        assert CTRL.extra_objs and CTRL.extra_objs[0][0] == info["static_idx"]


def test_overlapping_object_sets_fail_loudly(tok, pairs):
    p = dict(pairs[0], static_object=pairs[0]["temp_object"])
    with pytest.raises(ValueError, match="overlap"):
        benchmark.configure_controller(L22, p, tok)
    c = Controller(arm="L", beta=2.0, gamma=2.0, mask=torch.ones(21), obj_idx=[5, 6],
                   extra_objs=[([6, 7], torch.ones(21))])
    with pytest.raises(ValueError, match="overlap"):
        _bias(c)


# ------------------------------------------------------------------ masks and the K-object table
def test_masks_match_the_file(tok, pairs):
    raw = _raw_csv()
    for p in pairs:
        if p["prompt_id"] in select_scorable_two_object_pairs.UNSCORABLE:
            continue
        r = raw[p["prompt_id"]]
        m1 = torch.tensor([float(x) for x in r["control_signal1"].split()])
        m2 = torch.tensor([float(x) for x in r["control_signal2"].split()])
        info = benchmark.configure_controller(L22, p, tok)
        assert torch.equal(CTRL.mask, m1) and torch.equal(CTRL.extra_objs[0][1], m2)
        b = _bias(CTRL)
        t, s = info["obj_idx"], info["static_idx"]
        assert torch.equal(b[:, t], (2 * m1 - 2 * (1 - m1))[:, None].expand(21, len(t)))
        assert torch.equal(b[:, s], (2 * m2 - 2 * (1 - m2))[:, None].expand(21, len(s)))
        assert (b[:, s] == 2).all()                              # static object: +2 on every frame
        others = [i for i in range(LK) if i not in t + s]
        assert (b[:, others] == 0).all()
        assert torch.equal(b, frame_table_L(m1, t, 2.0, 2.0) + frame_table_L(m2, s, 2.0, 2.0))


def test_k1_table_is_byte_identical_to_single_object_L(tok):
    rows = benchmark.load_one_object()
    for r in rows[::7] + rows[1::20]:                            # several one-object prompts, every timing
        for beta, gamma in ((2.0, 2.0), (4.0, 1.0), (0.0, 0.0)):
            benchmark.configure_controller({"arm": "L", "beta": beta, "gamma": gamma, "seed": 42}, r, tok)
            assert CTRL.extra_objs == []
            got = _bias(CTRL).numpy().tobytes()
            assert got == _pre_k_objects_L(CTRL).numpy().tobytes(), r["prompt_id"]
            assert got == frame_table_L(CTRL.mask, CTRL.obj_idx, beta, gamma).numpy().tobytes(), r["prompt_id"]


def test_k1_on_a_pair_equals_the_temporal_object_alone(tok, pairs):
    for i in _step5a_ids():
        benchmark.configure_controller(L22, pairs[i], tok)
        CTRL.extra_objs = []
        assert _bias(CTRL).numpy().tobytes() == frame_table_L(CTRL.mask, CTRL.obj_idx, 2.0, 2.0).numpy().tobytes()


def test_k2_is_explicit_path_and_arm_L_only(tok, pairs):
    p = pairs[0]
    for arm in ("U", "S", "P"):
        with pytest.raises(ValueError):
            benchmark.configure_controller(dict(L22, arm=arm), p, tok)
    for path in ("fused", "fa2kv", "flash"):
        with pytest.raises(ValueError):
            benchmark.configure_controller(dict(L22, path=path), p, tok)
    benchmark.configure_controller(L22, p, tok)
    CTRL.branch, CTRL.step = "cond", 0
    with pytest.raises(NotImplementedError):
        fused_mask_table(CTRL, LK, CPU)
    with pytest.raises(NotImplementedError):
        fa2kv_tables_for_controller(CTRL, LK, CPU)
    CTRL.arm = "U"
    with pytest.raises(NotImplementedError):
        CTRL.edit_active()
    with pytest.raises(ValueError, match="does not match"):      # bench of the run and of the row must agree
        benchmark.configure_controller({"arm": "L", "beta": 2.0, "gamma": 2.0, "seed": 42}, p, tok)
    with pytest.raises(ValueError, match="does not match"):
        benchmark.configure_controller(L22, benchmark.load_one_object()[0], tok)


def test_b0_pair_has_no_edit(tok, pairs):
    benchmark.configure_controller(dict(L22, arm="B0"), pairs[0], tok)
    CTRL.branch, CTRL.step = "cond", 0
    assert not CTRL.edit_active() and _frame_bias_table(CTRL, LK, CPU) is None
    assert np.array_equal(np.asarray(pairs[0]["static_mask"]), np.ones(21))


def test_all_5b_pairs_configure_for_both_arms(tok, pairs):
    """Every 5b pair (incl. the 17 unscorable ones) must configure for both step-5a runs without error."""
    ids = []
    for part in open(os.path.join(benchmark.TEMPO, "data", "step5b_ids.txt")).read().strip().split(","):
        a, _, b = part.partition("-")
        ids += list(range(int(a), int(b) + 1)) if b else [int(a)]
    assert len(ids) == 62
    runs = json.load(open(os.path.join(benchmark.TEMPO, "runs", "step5a.json")))
    by_id = {p["prompt_id"]: p for p in pairs}
    for i in ids:
        for r in runs:
            benchmark.configure_controller(r, by_id[i], tok)
