"""Offline end-to-end test of the research loop with a scripted model."""

import json

import pytest

from cspace.agents.campaign import Campaign
from cspace.agents.llm import ScriptedLLM
from cspace.agents.space_ops import PatchError, apply_patch, render_dag, render_flat

from conftest import tiny_baseline_source


def node(nid, supports=(), active=True, ntype="assumption"):
    return dict(id=nid, type=ntype, statement=f"statement {nid}", scope="small causal LMs", sources=["src"],
                status="established", active=active, supports=[list(g) for g in supports])


BOX = dict(id="test-box", version=0, parent_version=None,
           context=dict(fixed_requirements=["next-token prediction"], editable_priors=["prior"],
                        permitted_constructions=["torch modules"]),
           nodes=[node("fixed_depth"), node("kv_memory"), node("residual"),
                  node("decoder", supports=[["fixed_depth", "kv_memory", "residual"]], ntype="method"),
                  node("ut", active=False, ntype="method")])

LEAKY = tiny_baseline_source().replace("is_causal=True", "is_causal=False")


def make_script():
    state = {"n": 0}

    def script(messages, tools):
        last = messages[-1]
        text = last.get("content") or ""
        names = {t["function"]["name"] for t in tools}
        brief = messages[1]["content"]
        state["n"] += 1
        uniq = f"\n# variant {state['n']}\n"
        if last["role"] == "user" and "query_box" in names and "MODE: CHALLENGE" in brief:
            return "query_box", {"node_id": "fixed_depth"}
        if last["role"] == "user" or (last["role"] == "tool" and text.startswith("{")):
            if last["role"] == "user":
                return "list_archive", {}
        if "IMPLEMENTATION FAILURE" in text or text.startswith("ERROR: this exact bundle"):
            return "submit_proposal", proposal(tiny_baseline_source() + uniq, patch="MODE: RECON" not in brief)
        if "SMOKE TEST RESULT" in text or "Repairs exhausted" in text:
            return "record_reflection", {
                "observations": "loss recorded", "interpretation": "no clear effect", "evidence_class": "inconclusive",
                "next_question": "does depth recurrence help later?",
                "followup": {"change": "longer run", "prediction": "gap closes", "test": "compare at 0.5"}}
        # first submission: the reconstruction job submits a leaky model to exercise repair
        if "MODE: RECONSTRUCTION" in brief and not any("IMPLEMENTATION FAILURE" in (m.get("content") or "")
                                                       for m in messages):
            return "submit_proposal", proposal(LEAKY + uniq, patch=False, recon=True)
        return "submit_proposal", proposal(tiny_baseline_source() + uniq, patch="MODE: RECONSTRUCTION" not in brief,
                                           ablation="MODE: CHALLENGE" in brief)

    return script


def proposal(code, patch=True, recon=False, ablation=False):
    args = {"proposal": {k: f"{k} text" for k in ("commitment", "change", "mechanism", "prediction", "control",
                                                    "test", "scope")},
            "change_type": "uncertain", "model_py": code, "train_config": {"lr": 0.003}}
    if patch:
        args["space_patch"] = [{"op": "suspend", "node_id": "fixed_depth"},
                               {"op": "add_node", "node": node("recurrent_depth", ntype="method")}]
    if recon:
        args["new_formulation"] = {"context": BOX["context"], "nodes": [node("state"), node("model", [["state"]])]}
    if ablation:
        args["ablation_model_py"] = code + "\n# ablation\n"
    return args


@pytest.fixture
def campaign(tmp_path, tiny_task):
    box_path = tmp_path / "box.json"
    box_path.write_text(json.dumps(BOX))
    camp = Campaign.create("t", root=tmp_path / "campaigns", condition="dag", box=str(box_path), task=str(tiny_task),
                           horizon_flops=4e9, smoke_fracs=[0.5, 1.0], eval_tokens=4096, max_turns=10,
                           eval_time_limit_s=600, device="cpu")
    yield camp
    camp.close()


def test_patch_semantics():
    new, report = apply_patch(BOX, [{"op": "suspend", "node_id": "fixed_depth"}])
    assert new["version"] == 1 and report["needs_review"] == ["decoder"] and report["unsupported"] == ["decoder"]
    with pytest.raises(PatchError):
        apply_patch(BOX, [{"op": "set_supports", "node_id": "fixed_depth", "supports": [["decoder"]]}])  # cycle
    with pytest.raises(PatchError):
        apply_patch(BOX, [{"op": "suspend", "node_id": "nope"}])
    assert "[decoder]" in render_dag(BOX) and "This relies on fixed_depth together with kv_memory" in render_flat(BOX)


def test_wave_end_to_end(campaign):
    log = []
    campaign.run(1, ScriptedLLM(make_script()), log=log.append)
    jobs = campaign.jobs(1)
    assert [j["plan"]["mode"] for j in jobs] == ["challenge", "challenge", "challenge", "reconstruction"]
    assert all(j["status"] == "completed" for j in jobs), [(j["plan"]["id"], j["status"], j["error"]) for j in jobs]
    seed = campaign.seed_id
    assert jobs[0]["plan"]["research_parent_id"] == seed and jobs[1]["plan"]["research_parent_id"] == seed

    # challenges patched the formulation; the query happened before submission
    ch = jobs[1]
    assert ch["space_report"]["needs_review"] == ["decoder"]
    assert ch["retrievals"][0] == {"turn": 1, "tool": "query_box", "target": "fixed_depth", "phase": "propose"}
    space, _ = campaign.formulation_of(ch["final_candidate_id"])
    assert any(n["id"] == "recurrent_depth" for n in space["nodes"])
    assert ch["ablation_candidate_id"] and "ablation" in ch["summary"]

    # reconstruction: leaky first attempt -> repair -> valid; no research parent, clean code base
    rc = jobs[3]
    assert len(rc["candidates"]) == 2
    first, final = (campaign.archive.get_candidate(c) for c in rc["candidates"])
    assert first["research_parent_id"] is None and first["code_base_id"] is None
    assert final["code_base_id"] == first["id"]
    assert campaign.meta(first["id"])["smoke_status"] == "invalid"
    assert rc["summary"]["checkpoints"][0]["baseline_dev_loss"] > 0

    # costs and transcripts
    units = {(r["ledger"], r["unit"]) for r in campaign.archive.ledger_summary()}
    assert ("inference", "input_tokens") in units and ("train_compute", "flops") in units
    assert json.loads(campaign.archive.get_bytes(rc["transcript"]))[0]["role"] == "system"

    # next wave: a queued follow-up becomes the continuation, consuming its origin's allowance
    plans = campaign.plan_wave(2)
    assert plans[0].mode == "continuation" and plans[0].followup["change"] == "longer run"
    origin = plans[0].origin_id
    allowance = campaign.db.execute("SELECT allowance FROM origins WHERE id = ?", (origin,)).fetchone()[0]
    assert allowance == campaign.config["continuation_allowance"] - 1
    # slot 2 picks the least recently expanded valid state: no valid state was older than wave 1,
    # so the pick is a wave-1 tie broken by id, and its clock now reads wave 2
    states = {s["candidate_id"]: s for s in campaign.valid_states()}
    assert all(s["last_expanded_wave"] >= 1 for s in states.values())
    assert states[plans[2].research_parent_id]["last_expanded_wave"] == 2
    assert plans[3].mode == "reconstruction" and plans[3].research_parent_id is None
