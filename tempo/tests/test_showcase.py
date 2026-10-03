"""CPU tests for the showcase prompts (phase 25): TempoControl's project-page prompts, not in the benchmark, generated
with the step-4 / step-5a protocols (slurm/showcase_generate.sbatch) and scored by slurm/showcase_score.sbatch.

Run on the login node:  ~/tempo/venv/bin/python -m pytest -q ~/tempo/tests (TEMPO_ROOT overrides ~/tempo)
"""
import csv
import glob
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
from tempo_ctrl import benchmark, masks  # noqa: E402
from tempo_ctrl.tokens import load_tokenizer, span_token_indices  # noqa: E402
from tempo_ctrl.cross_attention_arms import CTRL  # noqa: E402

import run_official_temporal_accuracy  # noqa: E402
from generate_benchmark_videos import check_prompt_set, parse_ids  # noqa: E402

T = benchmark.TEMPO

# verbatim from TempoControl's project page (missing articles in 1 and 2 are the page's own)
WEBSITE = {
    0: "An empty scene. Suddenly, during the last second of the video, a dog appears out of nowhere, drawing all "
       "attention.",
    1: "The video begins with a serene view centered on the cat, with no sign of the dog. Suddenly, in the second "
       "half, dog unexpectedly appears, altering the dynamic of the scene.",
    2: "The video begins with a serene view centered on bird, with no sign of cat. Suddenly, in the second half, cat "
       "unexpectedly appears, altering the dynamic of the scene.",
    3: "The video begins with a serene view centered on the wine glass, with no sign of the chair. Suddenly, in the "
       "second half, the chair unexpectedly appears, altering the dynamic of the scene.",
    4: "The video begins with a serene view centered on the bicycle, with no sign of the truck. Suddenly, in the "
       "second half, the truck unexpectedly appears, altering the dynamic of the scene.",
    5: "The video begins with a serene view centered on the sheep, with no sign of the horse. Suddenly, in the "
       "second half, the horse unexpectedly appears, altering the dynamic of the scene.",
}
# id -> (bench, temp_object, static_object, source ids, original_prompt)
SPEC = {
    0: ("one_object", "dog", "", list(range(60, 80)), ""),
    1: ("two_objects", "dog", "cat", [1], "a cat and a dog (website)"),
    2: ("two_objects", "cat", "bird", [0], "a bird and a cat (website)"),
    3: ("two_objects", "chair", "wine glass", [55], "a wine glass and a chair (website)"),
    4: ("two_objects", "truck", "bicycle", [50], "a bicycle and a truck (website)"),
    5: ("two_objects", "horse", "sheep", [3], "a sheep and a horse (website)"),
}
# umT5 slots (reported in the showcase notes; pinned so a tokenizer change is noticed)
TOKENS = {0: ([17], None), 1: ([19, 30], [12]), 2: ([17, 28], [11]), 3: ([20, 32], [12, 13]),
          4: ([19, 31], [12]), 5: ([20, 32], [12, 13])}
TAGS = {"one_object": ["showcase_B0_s42", "showcase_L_b2g2_s42"],
        "two_objects": ["showcase_B0_2obj_s42", "showcase_L_b2g2_2obj_s42"]}
RUNS = {"one_object": "showcase.json", "two_objects": "showcase_2obj.json"}


@pytest.fixture(scope="module")
def tok():
    return load_tokenizer(benchmark.WAN_ROOT, benchmark.CKPT)


@pytest.fixture(scope="module")
def show():
    return {**benchmark.load_showcase("one_object"), **benchmark.load_showcase("two_objects")}


def _raw():
    with open(benchmark.SHOWCASE_CSV, newline="") as f:
        return list(csv.DictReader(f))


def _bench_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _runs(b):
    return json.load(open(os.path.join(T, "runs", RUNS[b])))


# ------------------------------------------------------------------ data
def test_prompts_are_verbatim_and_not_in_the_benchmark(show):
    raw = _raw()
    assert [int(r["id"]) for r in raw] == list(WEBSITE)
    bench_prompts = {r["prompt"] for p in (benchmark.ONE_OBJECT_CSV, benchmark.TWO_OBJECTS_CSV) for r in _bench_csv(p)}
    for r in raw:
        i = int(r["id"])
        assert r["prompt"] == WEBSITE[i] == show[i]["prompt"]
        assert r["prompt"] not in bench_prompts
        b, t, s, src, orig = SPEC[i]
        assert (r["bench"], r["temp_object"], r["static_object"], parse_ids(r["source_ids"]), r["original_prompt"]) \
            == (b, t, s, src, orig)
        assert show[i]["source_ids"] == src and show[i]["temp_object"] == t
        assert show[i].get("static_object", "") == s and show[i].get("original_prompt", "") == orig


def test_one_object_mask_is_the_benchmark_last_second_mask(show):
    rows = _bench_csv(benchmark.ONE_OBJECT_CSV)
    last = [i for i, r in enumerate(rows) if "during the last second" in r["prompt"]]
    assert last == list(range(60, 80))
    assert len({rows[i]["control_signal1"] for i in last}) == 1           # all 20 last-second rows share the mask
    assert show[0]["control_signal1"] == rows[60]["control_signal1"] == " ".join(["0"] * 17 + ["1"] * 4)
    assert np.array_equal(show[0]["mask"], masks.mask_from_prompt(show[0]["prompt"]))
    assert "original_prompt" not in show[0] and "static_object" not in show[0]


def test_two_object_masks_are_the_source_pairs(show):
    rows = _bench_csv(benchmark.TWO_OBJECTS_CSV)
    assert len({(r["control_signal1"], r["control_signal2"]) for r in rows}) == 1   # all 82 pairs share the masks
    for i in range(1, 6):
        src = rows[SPEC[i][3][0]]
        assert (show[i]["control_signal1"], show[i]["control_signal2"]) == (src["control_signal1"], src["control_signal2"])
        assert np.array_equal(show[i]["mask"], [0] * 10 + [1] * 11) and np.array_equal(show[i]["static_mask"], [1] * 21)


def test_loader_rejects_a_mask_that_differs_from_the_source(tmp_path, monkeypatch):
    raw = _raw()
    raw[2]["control_signal1"] = " ".join(["0"] * 11 + ["1"] * 10)
    p = tmp_path / "s.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(raw[0]))
        w.writeheader()
        w.writerows(raw)
    monkeypatch.setattr(benchmark, "SHOWCASE_CSV", str(p))
    with pytest.raises(ValueError, match="control_signal1"):
        benchmark.load_showcase("two_objects")


# ------------------------------------------------------------------ token sets
def test_token_sets_decode_to_the_object_words_and_do_not_overlap(tok, show):
    for i, row in show.items():
        b = SPEC[i][0]
        for run in _runs(b):
            info = benchmark.configure_controller(run, row, tok)
            objs = [(row["temp_object"], info["obj_idx"])]
            if b == "two_objects":
                objs.append((row["static_object"], info["static_idx"]))
                assert CTRL.extra_objs and CTRL.extra_objs[0][0] == info["static_idx"]
                assert torch.equal(CTRL.extra_objs[0][1], torch.ones(21))
            else:
                assert CTRL.extra_objs == []
            assert torch.equal(CTRL.mask, torch.tensor(row["mask"], dtype=torch.float32))
            for obj, idx in objs:
                spans = [m.span() for m in re.finditer(r"\b" + re.escape(obj) + r"\b", row["prompt"])]
                assert spans
                union = set()
                for sp in spans:                                     # every occurrence decodes back to the word
                    oi, n_real, enc = span_token_indices(tok, row["prompt"], [sp])
                    pieces = [tok.tokenizer.convert_ids_to_tokens(enc["input_ids"][k]) for k in oi]
                    assert tok.tokenizer.convert_tokens_to_string(pieces).strip() == obj, (i, obj, pieces)
                    union |= set(oi)
                assert sorted(union) == idx and max(idx) < n_real <= 512
            assert info["obj_idx"] == TOKENS[i][0] and info.get("static_idx") == TOKENS[i][1]
            assert len(info["obj_idx"]) == (1 if b == "one_object" else 2)   # two-object: both mentions of temp obj
            sets = [set(info["obj_idx"]), set(info.get("static_idx") or []), set(info["tmp_idx"])]
            assert not (sets[0] & sets[1]) and not (sets[2] & (sets[0] | sets[1])), i
            assert info["tmp_tokens"][-1] in ("▁video", "▁half")


# ------------------------------------------------------------------ names, tags, protocol
def test_video_names_are_unique_and_new(show):
    names = [benchmark.row_video_name(r) for r in show.values()]
    assert len(set(names)) == len(names) == 6
    taken = {benchmark.row_video_name(r) for b in ("one_object", "two_objects") for r in benchmark.load_benchmark(b)}
    assert not set(names) & taken
    assert names[0] == f"{WEBSITE[0]}-0.mp4"                    # official one-object metric: f"{prompt}-0.mp4"
    assert all(n.endswith(" (website)-0.mp4") and "/" not in n for n in names[1:])


def test_runs_are_the_step4_and_step5a_protocols_with_new_tags():
    for b, ref in (("one_object", "step4.json"), ("two_objects", "step5a.json")):
        runs = _runs(b)
        assert [{k: v for k, v in r.items() if k != "set"} for r in runs] == json.load(open(os.path.join(T, "runs", ref)))
        assert all(r["set"] == "showcase" for r in runs)
        assert [benchmark.run_tag(r) for r in runs] == TAGS[b]
    new = TAGS["one_object"] + TAGS["two_objects"]
    others = {benchmark.run_tag(r) for f in glob.glob(os.path.join(T, "runs", "*.json"))
              if os.path.basename(f) not in RUNS.values() for r in json.load(open(f))}
    assert not set(new) & others
    # without "set" the tags are unchanged
    assert benchmark.run_tag({"arm": "L", "beta": 2.0, "gamma": 2.0, "seed": 42}) == "L_b2g2_s42"


def test_prompt_set_and_runs_must_agree():
    check_prompt_set(_runs("one_object"), "showcase")
    check_prompt_set(json.load(open(os.path.join(T, "runs", "step5a.json"))), None)
    with pytest.raises(ValueError):
        check_prompt_set(json.load(open(os.path.join(T, "runs", "step4.json"))), "showcase")
    with pytest.raises(ValueError):
        check_prompt_set(_runs("two_objects"), None)


def test_showcase_ids_index_their_own_benchmark_only(show):
    assert sorted(benchmark.load_showcase("one_object")) == [0]
    assert sorted(benchmark.load_showcase("two_objects")) == [1, 2, 3, 4, 5]
    with pytest.raises(ValueError, match="does not match"):
        benchmark.configure_controller(_runs("one_object")[1], show[1], None)


# ------------------------------------------------------------------ official metric CSV
def test_metric_subset_csv_has_the_official_columns(tmp_path, show):
    for mb, b, ids in (("one-object", "one_object", [0]), ("two-object", "two_objects", [1, 2, 3, 4, 5])):
        csv_path, sub_name, _, col = run_official_temporal_accuracy.BENCH[mb]
        p = tmp_path / sub_name
        rows = run_official_temporal_accuracy.write_showcase_subset_csv(ids, str(p), mb)
        with open(csv_path, newline="") as f:
            official = csv.DictReader(f).fieldnames
        with open(p, newline="") as f:
            rd = csv.DictReader(f)
            got = list(rd)
            assert rd.fieldnames == official == run_official_temporal_accuracy.OFFICIAL_COLS[mb]
        assert got == rows and len(got) == len(ids)
        for i, r in zip(ids, got):
            assert f"{r[col]}-0.mp4" == benchmark.row_video_name(show[i])          # the metric finds our video
            assert r["prompt"] == WEBSITE[i] and r["temp_object"] == SPEC[i][1]
            assert r["control_signal1"] == show[i]["control_signal1"]
            if b == "two_objects":
                assert r["static_object"] == SPEC[i][2] and r["control_signal2"] == show[i]["control_signal2"]
                assert r["original_prompt"] == f"a {r['Object 1']} and a {r['Object 2']} (website)"
        with pytest.raises(ValueError):
            run_official_temporal_accuracy.write_showcase_subset_csv([1 if b == "one_object" else 0], str(p), mb)


# ------------------------------------------------------------------ jobs
def _sbatch(name):
    return open(os.path.join(T, "slurm", name)).read()


def test_generation_job():
    s = _sbatch("showcase_generate.sbatch")
    assert re.search(r"^#SBATCH --qos=12h_4g$", s, re.M) and re.search(r"^#SBATCH --gres=gpu:1$", s, re.M)
    assert re.search(r"^#SBATCH --array=0-1$", s, re.M) and re.search(r"^#SBATCH --time=01:00:00$", s, re.M)
    assert "A100-SXM4-40GB" in s and "torch.cuda.is_available()" in s
    assert s.count("--prompt-set showcase") == 2 and s.count("--phase 25") == 2
    assert "--runs runs/showcase_2obj.json --ids 1-5" in s and "--shard ${SLURM_ARRAY_TASK_ID}/2" in s
    assert "--runs runs/showcase.json --ids 0" in s
    shards = [[1, 2, 3, 4, 5][si::2] for si in range(2)]
    n = [2 * len(shards[0]), 2 * len(shards[1]) + 2]             # shard 1 also runs the one-object prompt
    assert n == [6, 6] and sum(n) == 12


def test_scoring_job():
    s = _sbatch("showcase_score.sbatch")
    assert re.search(r"^#SBATCH --qos=2h_2g$", s, re.M) and re.search(r"^#SBATCH --gres=gpu:1$", s, re.M)
    assert "A100-SXM4-40GB" in s and "torch.cuda.is_available()" in s and "NVIDIA_TF32_OVERRIDE=0" in s
    assert s.count("--require-cuda") >= 4 and "OUT=phase25_gpu" in s
    for t in TAGS["one_object"] + TAGS["two_objects"]:
        assert t in s
