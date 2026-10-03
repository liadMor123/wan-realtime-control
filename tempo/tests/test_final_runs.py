"""CPU tests for the final runs: step 2's accuracy check (step2acc, phase 24) and step 5b (all 82 two-object pairs),
and the GPU scoring job. Id lists, run configs, job files, the metric device check.

Run on the login node:  ~/tempo/venv/bin/python -m pytest -q ~/tempo/tests (TEMPO_ROOT overrides ~/tempo)
"""
import glob
import json
import os
import re
import sys

import pytest

TEMPO_ROOT = os.environ.get("TEMPO_ROOT", os.path.expanduser("~/tempo"))
sys.path.insert(0, os.path.join(TEMPO_ROOT, "src"))
sys.path.insert(0, os.path.join(TEMPO_ROOT, "scripts"))
from tempo_ctrl import benchmark  # noqa: E402
from tempo_ctrl.tokens import load_tokenizer  # noqa: E402
from tempo_ctrl.cross_attention_arms import CTRL  # noqa: E402

import select_scorable_two_object_pairs  # noqa: E402
import run_official_temporal_accuracy  # noqa: E402
from generate_benchmark_videos import parse_ids  # noqa: E402

T = benchmark.TEMPO


def _ids(name):
    return parse_ids(open(os.path.join(T, "data", name)).read().strip())


def _sbatch(name):
    return open(os.path.join(T, "slurm", name)).read()


def _shards(ids, n=4):
    return [ids[si::n] for si in range(n)]                     # generate_benchmark_videos.py: parse_ids(ids)[si::sn]


# ------------------------------------------------------------------ step 5b ids
def test_step5b_ids_are_the_82_minus_step5a_in_file_order():
    a, b = _ids("step5a_ids.txt"), _ids("step5b_ids.txt")
    assert len(b) == 62 and len(set(b)) == 62
    assert not set(a) & set(b)
    assert sorted(a + b) == list(range(82))
    assert b == [i for i in range(82) if i not in set(a)]      # file order
    assert b == sorted(b)
    pairs = benchmark.load_two_objects()
    assert len(pairs) == 82


def test_step5b_contains_all_17_unscorable_and_45_scorable():
    b = _ids("step5b_ids.txt")
    assert set(select_scorable_two_object_pairs.UNSCORABLE) <= set(b)
    assert len([i for i in b if i not in select_scorable_two_object_pairs.UNSCORABLE]) == 45
    assert len([i for i in range(82) if i not in select_scorable_two_object_pairs.UNSCORABLE]) == 65


def test_step5b_shards_are_balanced_and_partition_the_62():
    b = _ids("step5b_ids.txt")
    sh = _shards(b)
    assert [len(s) for s in sh] == [16, 16, 15, 15]
    assert sorted(sum(sh, [])) == b


def test_step5b_job_reuses_5a_protocol():
    s = _sbatch("two_objects_remaining_62_generate.sbatch")
    assert "--runs runs/step5a.json" in s and 'cat data/step5b_ids.txt' in s and "--phase 5" in s
    assert re.search(r"^#SBATCH --qos=12h_4g$", s, re.M) and re.search(r"^#SBATCH --gres=gpu:1$", s, re.M)
    assert re.search(r"^#SBATCH --array=0-3$", s, re.M) and re.search(r"^#SBATCH --time=02:40:00$", s, re.M)
    assert "A100-SXM4-40GB" in s and "torch.cuda.is_available()" in s
    assert "TEMPO_TEXT_CACHE_TAG=5b_${SLURM_ARRAY_TASK_ID}" in s  # own per-shard T5 cache, 5a's files untouched
    runs = json.load(open(os.path.join(T, "runs", "step5a.json")))
    assert [benchmark.run_tag(r) for r in runs] == ["B0_2obj_s42", "L_b2g2_2obj_s42"]


# ------------------------------------------------------------------ step 2 accuracy check
def test_step2acc_runs_are_flash_B0_and_fa2kv_L():
    runs = json.load(open(os.path.join(T, "runs", "step2acc.json")))
    assert runs == [{"arm": "B0", "seed": 42, "path": "flash"},
                    {"arm": "L", "beta": 2.0, "gamma": 2.0, "seed": 42, "path": "fa2kv"}]
    tags = [benchmark.run_tag(r) for r in runs]
    assert tags == ["B0_flash_s42", "L_b2g2_fa2kv_s42"]
    others = {benchmark.run_tag(r) for f in glob.glob(os.path.join(T, "runs", "*.json"))
              if not f.endswith("step2acc.json") for r in json.load(open(f))}
    assert not set(tags) & others


def test_step2acc_configures_both_paths_on_every_heldout_prompt():
    tok = load_tokenizer(benchmark.WAN_ROOT, benchmark.CKPT)
    rows = benchmark.load_one_object()
    runs = json.load(open(os.path.join(T, "runs", "step2acc.json")))
    for i in _ids("heldout_ids.txt"):
        benchmark.configure_controller(runs[0], rows[i], tok)
        assert (CTRL.path, CTRL.arm) == ("flash", "B0") and not CTRL.edit_active()
        benchmark.configure_controller(runs[1], rows[i], tok)
        CTRL.step, CTRL.branch = 0, "cond"
        assert (CTRL.path, CTRL.arm, CTRL.beta, CTRL.gamma) == ("fa2kv", "L", 2.0, 2.0) and CTRL.edit_active()
        CTRL.branch = "uncond"
        assert not CTRL.edit_active()


def test_step2acc_uses_exactly_the_heldout_ids():
    h = _ids("heldout_ids.txt")
    assert h == list(range(2, 7)) + list(range(22, 27)) + list(range(42, 47)) + list(range(62, 67))
    assert not set(h) & set(_ids("pilot_ids.txt"))
    s = _sbatch("fa2kv_accuracy_generate.sbatch")
    assert '--ids "$(cat data/heldout_ids.txt)"' in s and "--runs runs/step2acc.json" in s and "--phase 24" in s
    sh = _shards(h)
    assert [len(x) for x in sh] == [5] * 4 and sorted(sum(sh, [])) == h
    assert re.search(r"^#SBATCH --qos=12h_4g$", s, re.M) and re.search(r"^#SBATCH --gres=gpu:1$", s, re.M)
    assert re.search(r"^#SBATCH --array=0-3$", s, re.M)
    assert "A100-SXM4-40GB" in s and "torch.cuda.is_available()" in s


def test_phase24_rows_do_not_mix_with_phase2():
    import fnmatch
    # paired_accuracy_analysis.load_phase_rows globs phase<N>_*.jsonl
    assert not fnmatch.fnmatch("phase24_1_0of4.jsonl", "phase2_*.jsonl")
    assert not fnmatch.fnmatch("phase2_1_0of4.jsonl", "phase24_*.jsonl")


# ------------------------------------------------------------------ GPU scoring
def test_score_gpu_job():
    s = _sbatch("score_gpu.sbatch")
    assert re.search(r"^#SBATCH --qos=2h_2g$", s, re.M) and re.search(r"^#SBATCH --gres=gpu:1$", s, re.M)
    assert "A100-SXM4-40GB" in s and "torch.cuda.is_available()" in s
    assert s.count("--require-cuda") >= 4                     # preflight, metric, CLIP, CLIP re-score
    assert "NVIDIA_TF32_OVERRIDE=0" in s
    assert "phase24|*_gpu" in s                               # never the CPU-scored dirs


def test_device_check_accepts_cuda_and_rejects_cpu():
    ok = {"cuda_available": True, "setup_model": [{"predictor_device": "cuda:0", "param_device": "cuda:0"}],
          "forward_calls": 21, "forward_input_devices": {"cuda:0": 21}, "forward_param_devices": {"cuda:0": 21}}
    assert run_official_temporal_accuracy.check_device_log(ok, 20)
    for bad in ({**ok, "cuda_available": False},
                {**ok, "setup_model": [{"predictor_device": "cpu", "param_device": "cpu"}]},
                {**ok, "forward_input_devices": {"cuda:0": 20, "cpu": 1}},
                {**ok, "setup_model": []},
                {**ok, "forward_calls": 3}):
        with pytest.raises(RuntimeError):
            run_official_temporal_accuracy.check_device_log(bad, 20)


def test_device_wrapper_compiles_and_only_wraps():
    code = run_official_temporal_accuracy.DEVICE_WRAPPER
    compile(code, "<wrapper>", "exec")
    assert "runpy.run_path(sys.argv[0], run_name=\"__main__\")" in code
    assert "return _setup(self" not in code                    # the original is called, its result returned
    assert "return r" in code and "return _fwd(self, im, *a, **k)" in code
