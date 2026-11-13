import os

import pytest
import torch

from cspace.archive import Archive, RecordInvalid
from cspace.evaluator.evaluate import evaluate_bundle
from cspace.evaluator.train import Budget, lr_at, resolve_recipe

from conftest import write_bundle

BUDGET = dict(horizon_flops=4e9, checkpoint_fracs=[0.25, 1.0], eval_tokens=4096)


def _eval(bundle, task, **kw):
    kw.setdefault("budget", BUDGET)
    return evaluate_bundle(bundle, task, device="cpu", enforce_lock=False, log=lambda m: None, **kw)


def test_training_learns_and_records(tiny_baseline, tiny_task):
    r = _eval(tiny_baseline, tiny_task)
    assert r["status"] == "ok", r["error"]
    cks = r["checkpoints"]
    assert [c["frac"] for c in cks] == [0.25, 1.0]
    assert cks[-1]["dev_loss"] < cks[0]["dev_loss"] < 4.16  # below uniform ln(64)
    assert cks[-1]["flops"] >= BUDGET["horizon_flops"]
    assert r["costs"]["train_flops"] == cks[-1]["flops"]
    assert r["final"]["split"] == "dev" and r["final"]["loss"] == cks[-1]["dev_loss"]
    assert r["params"]["total"] > r["params"]["nonembedding"] > 0


def test_smoke_run_is_prefix_of_full_run(tiny_baseline, tiny_task):
    full = _eval(tiny_baseline, tiny_task)
    smoke = _eval(tiny_baseline, tiny_task, budget={**BUDGET, "checkpoint_fracs": [0.25]})
    assert smoke["status"] == "ok"
    assert smoke["checkpoints"][0]["dev_loss"] == pytest.approx(full["checkpoints"][0]["dev_loss"], abs=1e-6)
    assert smoke["costs"]["train_flops"] < full["costs"]["train_flops"]


def test_runs_are_deterministic(tiny_baseline, tiny_task):
    a, b = _eval(tiny_baseline, tiny_task), _eval(tiny_baseline, tiny_task)
    assert [c["dev_loss"] for c in a["checkpoints"]] == [c["dev_loss"] for c in b["checkpoints"]]


@pytest.mark.skipif(os.environ.get("CSPACE_GPU_TESTS") != "1" or not torch.cuda.is_available(),
                    reason="GPU tests are opt-in: set CSPACE_GPU_TESTS=1")
def test_gpu_runs_are_bitwise_deterministic(tiny_baseline, tiny_task):
    # Regression: the default memory-efficient attention backward is nondeterministic on CUDA.
    kw = dict(budget=BUDGET, device="cuda", enforce_lock=False, log=lambda m: None)
    a, b = evaluate_bundle(tiny_baseline, tiny_task, **kw), evaluate_bundle(tiny_baseline, tiny_task, **kw)
    assert a["status"] == b["status"] == "ok", (a["error"], b["error"])
    assert a["attention_backend"] in ("cudnn", "math")
    assert [c["dev_loss"] for c in a["checkpoints"]] == [c["dev_loss"] for c in b["checkpoints"]]


def test_different_init_seed_changes_result(tiny_baseline, tiny_task):
    a = _eval(tiny_baseline, tiny_task)
    b = _eval(tiny_baseline, tiny_task, seeds={"init": 1, "data": 0})
    assert a["checkpoints"][-1]["dev_loss"] != b["checkpoints"][-1]["dev_loss"]


def test_confirm_split_is_sealed(tiny_baseline, tiny_task):
    assert _eval(tiny_baseline, tiny_task, split="confirm")["status"] == "refused"
    r = _eval(tiny_baseline, tiny_task, split="confirm", allow_sealed=True)
    assert r["status"] == "ok" and r["final"]["split"] == "confirm"


def test_invalid_candidates_do_not_train(tmp_path, tiny_task):
    leaky = write_bundle(tmp_path / "leaky", "import os\ndef build_model(v, t):\n    return None\n")
    r = _eval(leaky, tiny_task)
    assert r["status"] == "invalid" and r["error"] == "lint"

    bad_cfg = write_bundle(tmp_path / "cfg", "import torch\ndef build_model(v, t):\n    return None\n",
                           {"lr": 5.0})
    assert _eval(bad_cfg, tiny_task)["status"] == "invalid"

    broken = write_bundle(tmp_path / "broken", "import torch\ndef build_model(v, t):\n    return 3\n")
    r = _eval(broken, tiny_task)
    assert r["status"] == "invalid" and "build" in r["error"]


def test_recipe_bounds():
    assert resolve_recipe({})["lr"] == pytest.approx(2e-3)
    with pytest.raises(ValueError):
        resolve_recipe({"lr": 1.0})
    with pytest.raises(ValueError):
        resolve_recipe({"batch_size": 64})


def test_schedule_and_budget_validation():
    recipe = resolve_recipe({"warmup_frac": 0.1, "min_lr_frac": 0.1})
    assert lr_at(0, 100, recipe) < lr_at(9, 100, recipe) == pytest.approx(recipe["lr"])
    assert lr_at(100, 100, recipe) == pytest.approx(recipe["lr"] * 0.1)
    with pytest.raises(ValueError):
        Budget(horizon_flops=1e9, checkpoint_fracs=(0.5, 0.25))


def test_archive_roundtrip(tmp_path, tiny_baseline, tiny_task):
    with Archive(tmp_path / "archive") as archive:
        seed, created = archive.add_candidate(tiny_baseline, action_mode="seed")
        assert created
        again, created = archive.add_candidate(tiny_baseline, action_mode="seed")
        assert not created and again["id"] == seed["id"]

        out = archive.materialize(seed["bundle_hash"], tmp_path / "materialized")
        assert (out / "model.py").read_bytes() == (tiny_baseline / "model.py").read_bytes()

        result = _eval(out, tiny_task)
        exp = archive.add_experiment(result, seed["id"], purpose="test")
        assert archive.get_experiment(exp["id"])["checkpoints"] == result["checkpoints"]
        units = {(r["ledger"], r["unit"]) for r in archive.ledger_summary()}
        assert ("train_compute", "flops") in units

        with pytest.raises(RecordInvalid):
            archive.add_experiment({**result, "status": "bogus"}, seed["id"])
        child_dir = write_bundle(tmp_path / "child", (tiny_baseline / "model.py").read_text() + "\n# variant\n")
        with pytest.raises(RecordInvalid):
            archive.add_candidate(child_dir, action_mode="challenge", research_parent_id="c_0000000000000000")
        child, _ = archive.add_candidate(child_dir, action_mode="challenge", research_parent_id=seed["id"],
                                         proposal=dict(commitment="fixed depth", change="c", mechanism="m",
                                                       prediction="p", control="k", test="t", scope="s"))
        assert child["research_parent_id"] == seed["id"]
