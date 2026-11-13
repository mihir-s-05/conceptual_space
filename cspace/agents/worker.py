"""One research job: a tool-using conversation with a worker model.

Flow: information tools (read from the wave's frozen snapshot) -> submit_proposal
(commitment recorded before any result) -> engineering checks + smoke test
(bounded repairs on implementation failure) -> record_reflection.
"""

import json
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import jsonschema

from ..evaluator.train import RECIPE_BOUNDS
from . import prompts
from .campaign import JobRecord
from .llm import LLMError
from .space_ops import PATCH_SCHEMA, PatchError, apply_patch, query, render_dag
from ..archive.store import RecordInvalid
from ..box import validate_space

CHANGE_TYPES = ["ordinary_tuning", "component_switch", "formulation_change", "uncertain"]
EVIDENCE_CLASSES = ["supported", "contradicted", "inconclusive"]
FAILED = ("invalid", "error", "oom", "diverged", "timeout", "overheat", "refused")

PROPOSAL = {
    "type": "object",
    "required": ["commitment", "change", "mechanism", "prediction", "control", "test", "scope"],
    "properties": {
        "commitment": {"type": "string", "description": "The commitment being questioned (node ids or register entries, plus prose)."},
        "change": {"type": "string", "description": "What the new formulation does differently."},
        "mechanism": {"type": "string", "description": "Why it should matter."},
        "prediction": {"type": "string", "description": "A discriminating, checkable prediction for the smoke test."},
        "control": {"type": "string", "description": "What the comparison is (baseline control, and ablation if submitted)."},
        "test": {"type": "string", "description": "What the smoke test measures and how you will read it."},
        "scope": {"type": "string", "description": "The regime the claim is about."},
    },
}
FOLLOWUP = {
    "type": "object",
    "required": ["change", "prediction", "test"],
    "properties": {"change": {"type": "string"}, "prediction": {"type": "string"}, "test": {"type": "string"}},
}
NEW_FORMULATION = {
    "type": "object",
    "description": "Optional new dependency graph built from scratch: {context: {fixed_requirements, editable_priors, "
                   "permitted_constructions}, nodes: [{id, type, statement, scope, sources, status, active, supports}]}. "
                   "type in definition|assumption|theorem|empirical_claim|method; status in established|supported|"
                   "conditional|disputed|unresolved|suspended.",
    "required": ["context", "nodes"],
    "properties": {"context": {"type": "object"}, "nodes": {"type": "array", "items": {"type": "object"}}},
}


def _fn(name, description, parameters):
    return {"type": "function", "function": {"name": name, "description": description, "parameters": parameters}}


class Worker:
    def __init__(self, campaign, llm, plan, log=print):
        self.c = campaign
        self.llm = llm
        self.plan = plan
        self.log = log
        self.cfg = campaign.config
        self.cond = self.cfg["condition"]
        self.job = JobRecord(plan=asdict(plan))
        self.snapshot = campaign.get_snapshot(plan.evidence_snapshot_id)
        self.visible = {e["candidate_id"]: e for e in self.snapshot["entries"] if e.get("candidate_id")}
        if plan.mode == "reconstruction":
            self.base_space, self.base_register = None, None
        else:
            self.base_space, self.base_register = campaign.formulation_of(plan.research_parent_id)
        self.phase = "propose"
        self.repairs_left = int(self.cfg["max_repairs"])
        self.donors = []
        self.turn = 0
        self.finished = False
        self.last_candidate = None
        self.candidate_space_sid = None

    # ---- tool definitions --------------------------------------------------
    def tool_specs(self):
        tools = [
            _fn("list_archive", "List archived candidates and their smoke-test observations (frozen at wave start).",
                {"type": "object", "properties": {}}),
            _fn("read_candidate", "Read an archived candidate: code, train config, proposal, reflection, observations. "
                "Logged; candidates other than your parent are credited as donors.",
                {"type": "object", "required": ["candidate_id"], "properties": {"candidate_id": {"type": "string"}}}),
            _fn("read_formulation", "Read the reference formulation of the starting model"
                + (" (logged retrieval)." if self.plan.mode == "reconstruction" else "."),
                {"type": "object", "properties": {}}),
        ]
        if self.cond == "dag":
            tools.append(_fn("query_box", "Dependency query on the formulation: a node, its direct dependents, all "
                             "descendants, and which nodes would lose all support if it were suspended.",
                             {"type": "object", "required": ["node_id"], "properties": {"node_id": {"type": "string"}}}))
        submit = {
            "type": "object", "required": ["proposal", "change_type", "model_py"],
            "properties": {
                "proposal": PROPOSAL,
                "change_type": {"type": "string", "enum": CHANGE_TYPES},
                "model_py": {"type": "string", "description": "Complete source of model.py."},
                "train_config": {"type": "object", "description": "Optional recipe overrides.",
                                 "properties": {k: {"type": "number"} for k in RECIPE_BOUNDS}},
                "ablation_model_py": {"type": "string", "description": "Optional model.py identical except the "
                                      "proposed mechanism is disabled; trained identically as the mechanism control."},
                "notes": {"type": "string"},
            },
        }
        reflect = {
            "type": "object", "required": ["observations", "interpretation", "evidence_class", "next_question"],
            "properties": {
                "observations": {"type": "string", "description": "What was measured, without interpretation."},
                "interpretation": {"type": "string"},
                "evidence_class": {"type": "string", "enum": EVIDENCE_CLASSES},
                "next_question": {"type": "string"},
                "followup": {**FOLLOWUP, "description": "Optional: one specific, cheap, informative next test, "
                             "precommitted (change, prediction, test). May be developed by a later continuation job."},
            },
        }
        if self.cond == "dag":
            if self.plan.mode == "reconstruction":
                submit["properties"]["new_formulation"] = NEW_FORMULATION
            else:
                submit["properties"]["space_patch"] = PATCH_SCHEMA
            reflect["properties"]["space_patch"] = {**PATCH_SCHEMA, "description": "Optional post-result revision of "
                                                    "your branch formulation. " + PATCH_SCHEMA["description"]}
        else:
            submit["properties"]["register_revision"] = {"type": "string", "description": "Optional complete revised register for your branch."}
            reflect["properties"]["register_revision"] = {"type": "string", "description": "Optional post-result revision of your branch register."}
        self.schemas = {"submit_proposal": submit, "record_reflection": reflect}
        tools.append(_fn("submit_proposal", "Commit your proposal and implementation; runs checks and the smoke test.", submit))
        tools.append(_fn("record_reflection", "After results: record observations, interpretation, evidence class, "
                         "next question, and optional follow-up. Ends the job.", reflect))
        return tools

    # ---- context blocks ----------------------------------------------------
    def _parent_block(self):
        pid = self.plan.research_parent_id
        if pid is None:
            return None
        cand = self.c.archive.get_candidate(pid)
        files = {name: self.c.archive.get_bytes(h).decode("utf-8") for name, h in cand["files"].items()}
        entry = self.visible.get(pid, {})
        out = [f"PARENT STATE {pid} ({cand['action_mode']})",
               "observations: " + json.dumps(entry.get("observations"), indent=1)]
        if entry.get("proposal"):
            out.append("its proposal: " + json.dumps(entry["proposal"], indent=1))
        if entry.get("interpretation"):
            out.append(f"its interpretation [{entry.get('evidence_class')}]: {entry['interpretation']}")
        if entry.get("next_question"):
            out.append(f"its open question: {entry['next_question']}")
        out.append("model.py:\n```python\n" + files["model.py"] + "\n```")
        if "train_config.json" in files:
            out.append("train_config.json: " + files["train_config.json"])
        return "\n".join(out)

    def _formulation_block(self):
        if self.plan.mode == "reconstruction":
            return None
        if self.cond == "dag" and self.base_space:
            return "CURRENT FORMULATION (your parent's branch)\n" + render_dag(self.base_space)
        if self.cond == "flat" and self.base_register:
            return "CURRENT FORMULATION (your parent's branch)\n" + self.base_register
        return None

    # ---- main loop ---------------------------------------------------------
    def run(self) -> JobRecord:
        self.job.started = time.time()
        system = prompts.system_prompt(self.cond, self.c.task, self.c.budget, int(self.cfg["max_params"]))
        brief = prompts.job_brief(self.plan.mode,
                                  self.c.render_bulletin(self.snapshot, self.plan.mode == "reconstruction"),
                                  self._parent_block(), self._formulation_block(), self.plan.followup, self.plan.wave)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": brief}]
        tools = self.tool_specs()
        usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0, "reasoning_tokens": 0,
                 "seconds": 0.0}
        try:
            while self.turn < int(self.cfg["max_turns"]) and not self.finished:
                self.turn += 1
                reply = self.llm.chat(messages, tools)
                self.c.log_usage(reply.usage, self.plan.id)
                usage["calls"] += 1
                for k in ("prompt_tokens", "completion_tokens", "cached_tokens", "reasoning_tokens", "seconds"):
                    usage[k] += reply.usage.get(k, 0) or 0
                messages.append(reply.raw_message)
                if not reply.tool_calls:
                    messages.append({"role": "user", "content": "Respond with a tool call."})
                    continue
                for call in reply.tool_calls:
                    out = self._handle(call)
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": out})
            if not self.finished:
                self.job.error = f"stopped after {self.turn} turns in phase {self.phase}"
        except LLMError as e:
            self.job.error = f"LLM error: {e}"
        if self.job.status == "planned":
            self.job.status = "agent_failure"
        self.job.usage = usage
        self.job.finished = time.time()
        self.job.transcript = self.c.archive.put_bytes(json.dumps(messages).encode("utf-8"))
        return self.job

    def _handle(self, call):
        try:
            args = json.loads(call.arguments or "{}")
        except json.JSONDecodeError as e:
            return f"ERROR: arguments are not valid JSON ({e})."
        if not isinstance(args, dict):
            return "ERROR: arguments must be a JSON object."
        handler = {"list_archive": self._list_archive, "read_candidate": self._read_candidate,
                   "read_formulation": self._read_formulation, "query_box": self._query_box,
                   "submit_proposal": self._submit, "record_reflection": self._reflect}.get(call.name)
        if handler is None or (call.name == "query_box" and self.cond != "dag"):
            return f"ERROR: unknown tool {call.name!r}."
        if call.name in self.schemas:
            errors = sorted(jsonschema.Draft202012Validator(self.schemas[call.name]).iter_errors(args), key=str)
            if errors:
                return "ERROR: invalid arguments: " + "; ".join(e.message for e in errors[:5])
        try:
            return handler(args)
        except Exception as e:  # keep the job alive; report the problem to the agent and the log
            self.log(f"    tool {call.name} raised {type(e).__name__}: {e}")
            return f"ERROR: {type(e).__name__}: {e}"

    # ---- information tools -------------------------------------------------
    def _retrieval(self, tool, target):
        self.job.retrievals.append({"turn": self.turn, "tool": tool, "target": target, "phase": self.phase})

    def _list_archive(self, args):
        self._retrieval("list_archive", None)
        return self.c.render_bulletin(self.snapshot, observations_only=self.plan.mode == "reconstruction")

    def _read_candidate(self, args):
        cid = args["candidate_id"]
        if cid not in self.visible:
            return f"ERROR: {cid} is not in this wave's snapshot. Visible: {sorted(self.visible)}"
        self._retrieval("read_candidate", cid)
        if cid != self.plan.research_parent_id and cid not in self.donors:
            self.donors.append(cid)
        cand = self.c.archive.get_candidate(cid)
        files = {name: self.c.archive.get_bytes(h).decode("utf-8") for name, h in cand["files"].items()}
        return json.dumps({"candidate_id": cid, "files": files, **{k: v for k, v in self.visible[cid].items()
                                                                     if k != "candidate_id"}}, indent=1)

    def _reference_space(self):
        return self.c.get_space(self.c._meta_get("reference_space_id"))

    def _read_formulation(self, args):
        self._retrieval("read_formulation", "reference")
        ref = self._reference_space()
        if self.cond == "dag":
            return render_dag(ref)
        from .space_ops import render_flat

        return render_flat(ref)

    def _query_box(self, args):
        space = self.base_space
        if space is None:  # reconstruction: querying the reference graph is a retrieval
            space = self._reference_space()
            self._retrieval("query_box(reference)", args["node_id"])
        else:
            self._retrieval("query_box", args["node_id"])
        try:
            return json.dumps(query(space, args["node_id"]), indent=1)
        except PatchError as e:
            return f"ERROR: {e}"

    # ---- submission --------------------------------------------------------
    def _formulation_for_submission(self, args):
        """Return (space_sid, register_text, report) for the candidate, or raise PatchError."""
        if self.cond == "flat":
            return None, args.get("register_revision") or self.base_register, None
        if self.plan.mode == "reconstruction":
            nf = args.get("new_formulation")
            if not nf:
                return None, None, None
            space = {"id": f"{self.c.name}/{self.plan.id}", "version": 0, "parent_version": None,
                     "context": nf["context"], "nodes": nf["nodes"]}
            try:
                validate_space(space)
            except RecordInvalid as e:
                raise PatchError(f"new_formulation invalid: {e}") from None
            return self.c.put_space(space), None, None
        patch = args.get("space_patch") or []
        if not patch:
            return (self.c.put_space(self.base_space) if self.base_space else None), None, None
        space, report = apply_patch(self.base_space, patch)
        return self.c.put_space(space), None, report

    def _write_bundle(self, directory, model_py, train_config, notes):
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        (d / "model.py").write_text(model_py, encoding="utf-8")
        if train_config:
            (d / "train_config.json").write_text(json.dumps(train_config, indent=1), encoding="utf-8")
        if notes:
            (d / "NOTES.md").write_text(notes, encoding="utf-8")
        return d

    def _submit(self, args):
        if self.phase != "propose":
            return "ERROR: you already have results; call record_reflection."
        try:
            sid, register, report = self._formulation_for_submission(args)
        except PatchError as e:
            return f"ERROR: formulation change rejected: {e}. Fix space_patch/new_formulation and resubmit."
        p = args["proposal"]
        notes = (args.get("notes") or "") + "\n\n## Proposal\n" + json.dumps(p, indent=1)
        parent = self.plan.research_parent_id
        code_base = self.last_candidate or parent
        with tempfile.TemporaryDirectory(prefix="cspace_sub_") as tmp:
            bundle = self._write_bundle(Path(tmp) / "main", args["model_py"], args.get("train_config"), notes)
            rec, created = self.c.archive.add_candidate(
                bundle, action_mode=self.plan.mode, research_parent_id=parent, code_base_id=code_base,
                donor_ids=[d for d in self.donors if self.c.archive.get_candidate(d)],
                origin_id=self.plan.origin_id, evidence_snapshot_id=self.plan.evidence_snapshot_id,
                proposal=p, notes=f"job {self.plan.id}" + (f", repair of {self.last_candidate}" if self.last_candidate else ""))
            if not created:
                return (f"ERROR: this exact bundle already exists as {rec['id']}. Submit a changed implementation "
                        "(this did not use a repair).")
            ablation = None
            if args.get("ablation_model_py"):
                abl_dir = self._write_bundle(Path(tmp) / "ablation", args["ablation_model_py"], args.get("train_config"),
                                             f"Ablation control for {rec['id']} (job {self.plan.id}).")
                ablation, _ = self.c.archive.add_candidate(abl_dir, action_mode="control", research_parent_id=rec["id"],
                                                           code_base_id=rec["id"], origin_id=self.plan.origin_id,
                                                           evidence_snapshot_id=self.plan.evidence_snapshot_id,
                                                           notes=f"ablation for {rec['id']}")
        cid = rec["id"]
        self.c.record_candidate(cid, self.plan.id, self.plan.wave, self.plan.origin_id, args["change_type"], sid, register)
        self.job.candidates.append(cid)
        self.job.proposal, self.job.change_type = p, args["change_type"]
        self.job.space_report = report
        self.candidate_space_sid = sid
        self.last_candidate = cid
        self.log(f"    submitted {cid} ({args['change_type']}): {p['change'][:100]}")

        result = self.c.evaluate(cid, purpose=f"smoke {self.plan.id}", log=self.log)
        self.c.record_smoke(cid, result)
        self.job.experiments[cid] = result["id"]
        self.job.final_candidate_id = cid
        if result["status"] in FAILED:
            if self.repairs_left > 0:
                self.repairs_left -= 1
                return prompts.repair_message(result.get("error") or result["status"], result.get("checks"),
                                              result.get("log_tail", ""), self.repairs_left)
            self.job.status = "implementation_failure"
            self.job.summary = self.c.summarize(result)
            self.phase = "reflect"
            return ("Repairs exhausted; the implementation failed.\n" + prompts.result_message(self.job.summary)
                    + "\nYour evidence_class is recorded as an implementation failure regardless of what you choose.")
        ab_result = None
        if ablation is not None:
            self.c.record_candidate(ablation["id"], self.plan.id, self.plan.wave, self.plan.origin_id, "ablation", sid, register)
            ab_result = self.c.evaluate(ablation["id"], purpose=f"ablation {self.plan.id}", log=self.log)
            self.c.record_smoke(ablation["id"], ab_result)
            self.job.experiments[ablation["id"]] = ab_result["id"]
            self.job.ablation_candidate_id = ablation["id"]
        self.job.summary = self.c.summarize(result, ab_result)
        self.job.status = "completed"
        self.phase = "reflect"
        return prompts.result_message(self.job.summary)

    # ---- reflection --------------------------------------------------------
    def _reflect(self, args):
        if self.phase != "reflect":
            return "ERROR: submit_proposal first; reflection comes after results."
        cid = self.job.final_candidate_id
        refl = {k: args.get(k) for k in ("observations", "interpretation", "evidence_class", "next_question", "followup")}
        if self.job.status == "implementation_failure":
            refl["agent_evidence_class"] = refl["evidence_class"]
            refl["evidence_class"] = "implementation_failure"
        if self.cond == "dag" and args.get("space_patch"):
            base = self.c.get_space(self.candidate_space_sid)
            if base is None:
                return "ERROR: your candidate has no formulation to revise; omit space_patch."
            try:
                revised, report = apply_patch(base, args["space_patch"])
            except PatchError as e:
                return f"ERROR: space_patch rejected: {e}"
            self.c.record_revision(cid, space_sid=self.c.put_space(revised))
            refl["revision_report"] = report
        if self.cond == "flat" and args.get("register_revision"):
            self.c.record_revision(cid, register_text=args["register_revision"])
        refl["followup_enqueued"] = False
        if args.get("followup") and self.job.status == "completed":
            refl["followup_enqueued"] = self.c.enqueue_followup(self.plan.origin_id, cid, self.plan.id,
                                                                self.plan.wave, args["followup"])
        self.job.reflection = refl
        self.finished = True
        return "Recorded. The job is complete."
