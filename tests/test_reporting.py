"""CPU-only tests: executors, parallel waves, campaign reports, Stage B comparison, confirmation."""

import json
import threading
import time

import pytest

from cspace.agents.campaign import Campaign
from cspace.agents.llm import ScriptedLLM
from cspace.confirm import Confirmation, paired_interval, t_quantile
from cspace.executor import (Executor, get_executor, normalize_spec, pack_request, register_executor,
                             unpack_request)
from cspace.reporting import CampaignReport, compare

from conftest import make_tiny_task
from test_agents import BOX, make_script

BUDGET = dict(horizon_flops=4e9, checkpoint_fracs=[0.5, 1.0], eval_tokens=4096)


# ---- executors ----------------------------------------------------------------
def _request(bundle, task):
    return dict(bundle_dir=str(bundle), task_dir=str(task), split="dev", budget=BUDGET, seeds={"init": 0, "data": 0},
                device="cpu", time_limit_s=600, max_params=60_000_000)


def test_pack_roundtrip(tmp_path, tiny_baseline, tiny_task):
    packed = json.loads(json.dumps(pack_request(_request(tiny_baseline, tiny_task))))
    assert "bundle_dir" not in packed["request"] and "task_dir" not in packed["request"]
    req = unpack_request(packed, tiny_task.parent, tmp_path / "remote")
    assert (tmp_path / "remote" / "bundle" / "model.py").read_bytes() == (tiny_baseline / "model.py").read_bytes()
    assert req["task_dir"] == str(tiny_task)
    packed["bundle_files"]["model.py"] = packed["bundle_files"]["train_config.json"]
    with pytest.raises(ValueError):
        unpack_request(packed, tiny_task.parent, tmp_path / "remote2")
    with pytest.raises(FileNotFoundError):
        unpack_request(pack_request(_request(tiny_baseline, tiny_task)), tmp_path / "nowhere", tmp_path / "r3")


def test_loopback_matches_local(tiny_baseline, tiny_task):
    local = get_executor("local").run(_request(tiny_baseline, tiny_task), timeout_s=900)
    loop = get_executor({"kind": "loopback", "task_root": str(tiny_task.parent)}).run(
        _request(tiny_baseline, tiny_task), timeout_s=900)
    assert local["status"] == loop["status"] == "ok"
    assert [c["dev_loss"] for c in local["checkpoints"]] == [c["dev_loss"] for c in loop["checkpoints"]]
    assert local["bundle_hash"] == loop["bundle_hash"]
    assert loop["executor"]["kind"] == "loopback" and "torch" in loop["host"]


def test_executor_slots_bound_concurrency():
    state = {"now": 0, "max": 0}
    lock = threading.Lock()

    @register_executor
    class Sleepy(Executor):
        kind = "sleepy_test"

        def _execute(self, request, timeout_s):
            with lock:
                state["now"] += 1
                state["max"] = max(state["max"], state["now"])
            time.sleep(0.05)
            with lock:
                state["now"] -= 1
            return {"status": "ok", "i": request["i"]}

    ex = get_executor({"kind": "sleepy_test", "slots": 2})
    assert get_executor({"kind": "sleepy_test", "slots": 2}) is ex  # shared per spec
    out = {}
    ex.run_all([(i, {"i": i}) for i in range(6)], lambda key, result: out.update({key: result["i"]}))
    assert out == {i: i for i in range(6)} and state["max"] == 2
    with pytest.raises(ValueError):
        normalize_spec({"kind": "no_such_backend"})


def test_t_interval():
    assert t_quantile(0.975, 2) == pytest.approx(4.303, abs=1e-3)
    assert t_quantile(0.975, 10) == pytest.approx(2.228, abs=1e-3)
    iv = paired_interval([-0.05, -0.04, -0.06], 0.05)
    assert iv["mean"] == pytest.approx(-0.05) and iv["upper"] == pytest.approx(-0.02516, abs=1e-4)
    assert paired_interval([-0.05], 0.05)["upper"] is None


# ---- campaigns, reports, comparison, confirmation -------------------------------
@pytest.fixture(scope="module")
def study(tmp_path_factory):
    root = tmp_path_factory.mktemp("study")
    task = make_tiny_task(root / "task")
    box = root / "box.json"
    box.write_text(json.dumps(BOX))
    common = dict(root=root / "campaigns", box=str(box), task=str(task), horizon_flops=4e9, smoke_fracs=[0.5, 1.0],
                  eval_tokens=4096, max_turns=10, eval_time_limit_s=600, device="cpu")
    camps = {}
    for cond in ("dag", "flat"):
        camp = Campaign.create(cond, condition=cond, parallel_jobs=4, executor={"kind": "local", "slots": 2},
                               **common)
        camp.run(1, ScriptedLLM(make_script()), log=lambda m: None)
        camps[cond] = camp.name
        camp.close()
    # an unmatched flat arm (different horizon) for the mismatch check
    Campaign.create("flat_other", condition="flat", **{**common, "horizon_flops": 5e9}).close()
    return {"root": root, "campaigns": root / "campaigns", "task": task}


def test_parallel_wave_completed(study):
    camp = Campaign("dag", root=study["campaigns"])
    try:
        jobs = camp.jobs(1)
        assert [j["plan"]["mode"] for j in jobs] == ["challenge", "challenge", "challenge", "reconstruction"]
        assert all(j["status"] == "completed" for j in jobs), [(j["plan"]["id"], j["status"], j["error"]) for j in jobs]
        assert camp._meta_get("next_wave") == "2"
        exps = camp.archive.list_experiments()
        assert all(e.get("executor", {}).get("kind") == "local" for e in exps)
    finally:
        camp.close()


def test_campaign_report(study):
    camp = Campaign("dag", root=study["campaigns"])
    try:
        rep = CampaignReport(camp).write()
    finally:
        camp.close()
    d = rep["dir"]
    for f in ("ledger.csv", "ledger.json", "jobs.json", "cases.md", "cases_neutral.md", "summary.json", "summary.md"):
        assert (d / f).is_file(), f
    ledger = rep["ledger"]
    # the reconstruction's leaky first attempt is an engineering failure, then repaired
    recon = [r for r in ledger if r["mode"] == "reconstruction" and r["role"] != "ablation"]
    assert [r["role"] for r in recon] == ["attempt", "final"]
    assert recon[0]["failure_kind"].startswith("checks:") and recon[1]["status"] == "ok"
    assert any(r["role"] == "ablation" and r["delta_common"] is not None for r in ledger)
    s = rep["summary"]
    assert s["final_valid"] == 4 and s["repairs_used"] == 1 and s["attempt_failures"]
    assert s["dependency_use"]["with_retrieval_before_results"] >= 1  # the challenge queried the box first
    assert s["costs"]["train_flops"] > 0 and s["costs"]["train_tokens_processed"] > 0
    full, neutral = (d / "cases.md").read_text(), (d / "cases_neutral.md").read_text()
    assert "dependency retrievals before results" in full and "dependency retrievals" not in neutral
    assert "**Outcomes:**" in neutral


def test_study_compare(study):
    with pytest.raises(ValueError, match="not matched"):
        compare("bad", [("dag", "flat_other")], campaigns_root=study["campaigns"], out_root=study["root"] / "studies",
                log=lambda m: None)
    r = compare("s1", [("dag", "flat")], campaigns_root=study["campaigns"], out_root=study["root"] / "studies",
                log=lambda m: None)
    out = study["root"] / "studies" / "s1"
    assert (out / "comparison.md").read_text().startswith("# Stage B comparison")
    assert r["pairs"][0]["match_problems"] == []
    labels = [m["metric"] for m in r["pairs"][0]["metrics"]]
    assert "best delta at common checkpoint" in labels and "inference USD" in labels
    key = json.loads((out / "blinded" / "key.json").read_text())["cases"]
    assert len(key) == 8 and {v["arm"] for v in key.values()} == {"dag", "flat"}
    blinded = (out / "blinded" / "cases.md").read_text()
    assert "dag" not in blinded.lower() and "flat" not in blinded.lower()  # no arm names in the cards
    assert (out / "blinded" / "judging.csv").read_text().startswith("case_id,category")


def test_confirmation(study):
    camp = Campaign("dag", root=study["campaigns"])
    cand = camp.jobs(1)[0]["final_candidate_id"]
    n_before = len(camp.archive.list_experiments())
    camp.close()
    with pytest.raises(ValueError):
        Confirmation.create("bad", root=study["root"] / "conf", campaign="dag", campaigns_root=str(study["campaigns"]),
                            candidates=["c_0000000000000000"])
    conf = Confirmation.create("c1", root=study["root"] / "conf", campaign="dag",
                               campaigns_root=str(study["campaigns"]), candidates=[cand, "incumbent"],
                               split="confirm", seeds=[1, 2], device="cpu", eval_time_limit_s=600,
                               executor={"kind": "local", "slots": 2})
    try:
        rep = conf.run(log=lambda m: None)
        assert all(conf.run_of(m, s)["final"]["split"] == "confirm" for m in conf.members() for s in (1, 2))
        row = rep["candidates"][0]
        assert row["interval"]["n"] == 2 and row["verdict"] in ("meets threshold", "worse than baseline",
                                                               "not established")
        assert "Bonferroni" in rep["joint_procedure"]
    finally:
        conf.close()
    camp = Campaign("dag", root=study["campaigns"])
    try:  # sealed results never enter the campaign archive that agents read
        exps = camp.archive.list_experiments()
        assert len(exps) == n_before and all(e["split"] == "dev" for e in exps)
    finally:
        camp.close()
