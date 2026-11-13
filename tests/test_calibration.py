"""CPU-only tests: calibration cohort, mechanism probes, and pacing."""

import json
import math
import time

import pytest

from cspace.agents.campaign import Campaign
from cspace.calibration import Cohort, CohortError, screen_decision, spearman, usefulness_label
from cspace.evaluator.evaluate import evaluate_bundle
from cspace.evaluator.pacing import Pacer

from conftest import tiny_baseline_source, write_bundle

BUDGET = dict(horizon_flops=4e9, checkpoint_fracs=[0.25, 1.0], eval_tokens=4096)
BOX = dict(id="b", version=0, parent_version=None,
           context=dict(fixed_requirements=["next-token prediction"], editable_priors=[], permitted_constructions=[]),
           nodes=[dict(id="a", type="assumption", statement="s", scope="x", sources=["src"], status="established",
                       active=True, supports=[])])


# ---- statistics and rules ---------------------------------------------------
def test_spearman():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 2, 2, 3], [1, 2, 3, 4]) == pytest.approx(0.9486833)  # average ranks for ties
    assert spearman([1, 2], [1, 2]) is None and spearman([1, 1, 1], [1, 2, 3]) is None
    assert spearman([1, 2, None, 4], [1, 2, 3, 4]) == pytest.approx(1.0)


def test_screening_and_usefulness_rules():
    rule = dict(advance_max_delta=0.0, reject_min_delta=0.05, reject_requires_no_closing=True)
    assert screen_decision([0.02, -0.01], "ok", rule)[0] == "advance"
    assert screen_decision([0.05, 0.08], "ok", rule)[0] == "reject"
    assert screen_decision([0.20, 0.08], "ok", rule)[0] == "hold"     # big gap but closing
    assert screen_decision([0.01, 0.02], "ok", rule)[0] == "hold"
    assert screen_decision([], "diverged", rule)[0] == "reject"
    u = dict(delta=0.02, uncertain_band=0.01)
    assert usefulness_label(-0.03, "ok", u) == "useful"
    assert usefulness_label(-0.015, "ok", u) == "uncertain"
    assert usefulness_label(0.0, "ok", u) == "not_useful"
    assert usefulness_label(None, "diverged", u) == "failed"


# ---- pacing and probes --------------------------------------------------------
def test_pacer_on_cpu():
    p = Pacer("cpu", throttle=0.0, temp_limit_c=80)
    assert p.temp_limit is None and not p.active  # temperature limits only apply to CUDA devices
    for step in range(100):
        p.step_start()
        p.step_end(step)
    assert p.paused_seconds == 0.0
    t = Pacer("cpu", throttle=0.5)
    t.step_start()
    time.sleep(0.02)
    t.step_end(1)
    assert t.paused_seconds >= 0.015
    with pytest.raises(ValueError):
        Pacer("cpu", throttle=1.0)


def test_throttle_does_not_change_results_and_probes_are_recorded(tiny_baseline, tiny_task):
    kw = dict(budget=BUDGET, device="cpu", enforce_lock=False, log=lambda m: None)
    a = evaluate_bundle(tiny_baseline, tiny_task, **kw)
    b = evaluate_bundle(tiny_baseline, tiny_task, throttle=0.3, probes=False, **kw)
    assert a["status"] == b["status"] == "ok"
    assert [c["dev_loss"] for c in a["checkpoints"]] == [c["dev_loss"] for c in b["checkpoints"]]
    assert b["costs"]["pacing"]["paused_seconds"] > 0 and b["probes"] is None

    probes = a["probes"]
    assert [c["delay"] for c in probes["copy"]] == [8, 16]  # delays capped at half the context (T=32)
    for c in probes["copy"]:
        assert all(math.isfinite(c[k]) for k in ("first_half_loss", "second_half_loss", "copy_gain"))
        assert 0 <= c["copy_accuracy"] <= 1
    assert [p["positions"] for p in probes["position_loss"]] == ["0-15", "16-31"]
    assert all(math.isfinite(p["loss"]) for p in probes["position_loss"])


# ---- calibration cohort --------------------------------------------------------
def _variant(tmp_path, name, d_model=32, n_layers=2, comment=""):
    src = tiny_baseline_source().replace("d_model=32, n_layers=2", f"d_model={d_model}, n_layers={n_layers}")
    return write_bundle(tmp_path / name, src + f"\n# {name}{comment}\n", {"lr": 0.003, "warmup_frac": 0.05})


@pytest.fixture
def source_campaign(tmp_path, tiny_task):
    box = tmp_path / "box.json"
    box.write_text(json.dumps(BOX))
    camp = Campaign.create("src", root=tmp_path / "campaigns", box=str(box), task=str(tiny_task),
                           horizon_flops=4e9, smoke_fracs=[0.25, 0.5], eval_tokens=4096, device="cpu",
                           eval_time_limit_s=600)
    leaky = write_bundle(tmp_path / "leaky", tiny_baseline_source().replace("is_causal=True", "is_causal=False"))
    plan = [  # (bundle, change_type, run a real smoke test?)
        (_variant(tmp_path, "ord1"), "parameter_change", True),
        (leaky, "formulation_change", True),               # engineering-invalid: must be skipped
        (_variant(tmp_path, "ref1", d_model=48), "formulation_change", False),
        (_variant(tmp_path, "ord2", n_layers=1), "mechanism_change", False),
        (_variant(tmp_path, "ref2", n_layers=3), "formulation_change", False),
        (_variant(tmp_path, "ord3", d_model=16), "parameter_change", False),
    ]
    ids = []
    for i, (bundle, change_type, real) in enumerate(plan):
        rec, _ = camp.archive.add_candidate(bundle, action_mode="challenge", research_parent_id=camp.seed_id)
        camp.record_candidate(rec["id"], f"j{i}", 1, None, change_type, None, None)
        if real:
            exp = camp.evaluate(rec["id"], "smoke", log=lambda m: None)
        else:  # outcome-blind enrollment reads only checks.passed; give a deliberately absurd loss
            exp = camp.archive.add_experiment(
                {"status": "ok", "split": "dev", "checks": {"passed": True}, "evaluator_version": "fake",
                 "task_hash": camp.task.task_hash, "budget": camp.budget, "seeds": {"init": 0, "data": 0},
                 "checkpoints": [{"frac": 0.5, "dev_loss": 99.0, "flops": 2e9, "step": 1, "tokens": 1, "train_seconds": 0}], "costs": {}, "started": 0, "finished": 0},
                rec["id"], purpose="smoke")
        camp.record_smoke(rec["id"], exp)
        ids.append(rec["id"])
    camp.close()
    return tmp_path / "campaigns", ids


def test_cohort_end_to_end(tmp_path, tiny_task, tiny_baseline, source_campaign):
    root, ids = source_campaign
    control = _variant(tmp_path, "ctl_wide", d_model=40)
    coh = Cohort.create("c", root=tmp_path / "cohorts", sources=["src"], campaigns_root=str(root),
                        baseline=str(tiny_baseline), controls=[str(control)], cap=4, reform_quota=0.5,
                        task=str(tiny_task), horizon_flops=4e9, smoke_fracs=[0.25, 0.5], long_fracs=[0.5, 1.0],
                        eval_tokens=4096, device="cpu", eval_time_limit_s=600)
    try:
        with pytest.raises(CohortError):
            coh.run_phase_runs("long")  # screening must be frozen first

        members = coh.enroll(log=lambda m: None)
        agents = [m for m in members if m["provenance"] == "agent"]
        assert [m["source_candidate_id"] for m in agents] == [ids[0], ids[2], ids[3], ids[4]]  # creation order
        assert sorted(m["stratum"] for m in agents) == ["ordinary", "ordinary", "reformulation", "reformulation"]
        enrollment = json.loads((coh.dir / "enrollment.json").read_text())
        assert enrollment["skipped_engineering_invalid"] == [ids[1]]
        assert {m["provenance"] for m in members} == {"agent", "control", "baseline"}

        coh.run(until="screen", log=lambda m: None)
        # identical prior smoke run in the source campaign is reused; others are run fresh
        reused = {r["member_id"]: r["reused_from"] for r in coh.db.execute("SELECT * FROM runs WHERE phase='smoke'")}
        assert reused["m01"] and reused["m01"].startswith("src/") and reused["m02"] is None
        screening = json.loads((coh.dir / "screening.json").read_text())
        assert len(screening["decisions"]) == 5 and all(d["decision"] in ("advance", "hold", "reject")
                                                       for d in screening["decisions"])
        assert not any(r["phase"] == "long" for r in coh.db.execute("SELECT phase FROM runs"))

        coh.run(log=lambda m: None)
        rep = json.loads((coh.dir / "report.json").read_text())
        assert rep["agents"]["n"] == 4 and rep["controls"]["n"] == 1
        assert set(rep["agents_by_stratum"]) == {"reformulation", "ordinary"}
        for r in rep["rows"]:
            assert r["long_status"] == "ok" and r["long_delta_final"] is not None
            assert r["usefulness"] in ("useful", "uncertain", "not_useful")
            assert r["long_probes"]["copy"]
        # long runs use fresh seeds and every member was trained long regardless of its decision
        long_exps = [coh.run_of(m["member_id"], "long")[0] for m in coh.members()]
        assert all(e["seeds"] == {"init": 1, "data": 1} for e in long_exps)
        mu = rep["agents"]["missed_useful_fraction"]
        assert mu["denominator"] == rep["agents"]["usefulness"]["useful"]
        assert (coh.dir / "report.md").read_text().startswith("# Calibration cohort")
        assert (coh.dir / "members.csv").is_file()
        assert "report=done" in coh.status()
    finally:
        coh.close()

    # the spec is frozen
    spec_path = tmp_path / "cohorts" / "c" / "cohort.json"
    stored = json.loads(spec_path.read_text())
    stored["spec"]["cap"] = 99
    spec_path.write_text(json.dumps(stored))
    with pytest.raises(CohortError):
        Cohort("c", root=tmp_path / "cohorts")
