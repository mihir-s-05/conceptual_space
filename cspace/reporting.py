"""Campaign reports and the Stage B DAG-vs-flat comparison.

campaign report  (archive/campaigns/<name>/report/)
    ledger.csv / ledger.json  every submitted candidate: provenance, attempt, change
                              category, outcome vs the baseline at each checkpoint,
                              failure kind, probes, costs
    jobs.json                 one row per job: repairs, retrievals before results,
                              formulation edits, evidence class, usage
    cases.md                  full case cards (incl. formulation details)
    cases_neutral.md          condition-neutral cards: commitment, mechanism,
                              implementation evidence, outcomes, scope; nothing
                              about how the formulation was represented
    summary.json / summary.md campaign-level metrics

study compare    (archive/studies/<name>/)
    comparison.json / .md     matched DAG/flat pairs: settings check, per-arm metrics,
                              per-pair differences, sign counts across pairs
    blinded/cases.md          neutral cards from all arms, shuffled, case ids only
    blinded/judging.csv       researcher judging sheet (category, mechanism evidence,
                              promising, notes)
    blinded/key.json          case id -> campaign/arm/job (open after judging)

Categories follow the study design: ordinary exploration, known-space switch,
substantive reformulation, uncertain. The agent's own label (submitted with the
proposal, before results) is reported; researcher judgments go in judging.csv.
"""

import csv
import json
import random
import re
import time
from pathlib import Path

from .paths import ARCHIVE_DIR

STUDIES_DIR = ARCHIVE_DIR / "studies"
CATEGORY = {"ordinary_tuning": "ordinary exploration", "component_switch": "known-space switch",
            "formulation_change": "substantive reformulation", "uncertain": "uncertain"}
CATEGORIES = list(CATEGORY.values())
DEPENDENCY_TOOLS = ("read_formulation", "query_box", "query_box(reference)")
MATCH_KEYS = ("task", "box_hash", "model", "reasoning_effort", "horizon_flops", "smoke_fracs", "eval_tokens", "seeds",
              "max_turns", "max_repairs", "continuation_allowance", "max_params", "eval_time_limit_s")


# ---------------------------------------------------------------------------
def failure_kind(exp):
    if exp is None:
        return "missing"
    status = exp.get("status")
    if status == "ok":
        return None
    err = exp.get("error") or ""
    if status == "invalid":
        if err == "lint":
            return "lint"
        m = re.match(r"failed checks: \[(.*)\]", err)
        if m:
            return "checks:" + ",".join(s.strip(" '\"") for s in m.group(1).split(",") if s.strip())
        if err.startswith("import failed"):
            return "import"
        if err.startswith("train_config"):
            return "train_config"
        return "invalid"
    return status


def _checkpoint_map(exp):
    return {round(c["frac"], 6): c for c in (exp or {}).get("checkpoints") or []}


def _probe_summary(probes):
    """Two scalars per run: copy gain at the longest delay, loss in the last position bucket."""
    if not probes or "copy" not in probes:
        return None
    copy = max(probes["copy"], key=lambda c: c["delay"]) if probes["copy"] else None
    pos = probes.get("position_loss") or []
    return {"copy_delay": copy["delay"] if copy else None, "copy_gain": copy["copy_gain"] if copy else None,
            "copy_accuracy": copy["copy_accuracy"] if copy else None,
            "late_position_loss": pos[-1]["loss"] if pos else None, "late_positions": pos[-1]["positions"] if pos else None}


def _sub(a, b):
    return None if a is None or b is None else round(a - b, 4)


class CampaignReport:
    def __init__(self, camp):
        self.c = camp
        self.cfg = camp.config
        self.condition = self.cfg["condition"]
        self.control = camp.control(log=lambda m: None)
        self.base = _checkpoint_map(self.control)
        self.common = round(max(self.cfg["smoke_fracs"]), 6)   # common training-compute checkpoint
        self.base_probes = _probe_summary(self.control.get("probes"))
        self.jobs = [j for j in camp.jobs() if j["status"] != "planned"]

    # ---- rows ----------------------------------------------------------------
    def _formulation_edited(self, cid, parent):
        m = self.c.meta(cid) or {}
        if parent is None:
            return bool(m.get("space_id") or m.get("register_text"))
        pspace, preg = self.c.formulation_of(parent)
        if self.condition == "dag":
            from .agents.space_ops import space_id

            return m.get("space_id") is not None and (pspace is None or m["space_id"] != space_id(pspace))
        return (m.get("register_text") or None) != (preg or None)

    def candidate_row(self, job, cid, role, attempt):
        exp = self.c.archive.get_experiment(job["experiments"].get(cid)) if job["experiments"].get(cid) else None
        cks = _checkpoint_map(exp)
        deltas = {f"{f:g}": _sub(c["dev_loss"], self.base[f]["dev_loss"]) for f, c in cks.items() if f in self.base}
        costs = (exp or {}).get("costs") or {}
        last = (exp or {}).get("checkpoints", [{}])[-1] if (exp or {}).get("checkpoints") else {}
        probes = _probe_summary((exp or {}).get("probes"))
        bp = self.base_probes or {}
        p = job["plan"]
        return {
            "campaign": self.c.name, "condition": self.condition, "job_id": p["id"], "wave": p["wave"],
            "slot": p["slot"], "mode": p["mode"], "origin_id": p["origin_id"], "research_parent_id": p["research_parent_id"],
            "candidate_id": cid, "role": role, "attempt": attempt,
            "change_type": job.get("change_type") if role != "ablation" else "ablation",
            "category": CATEGORY.get(job.get("change_type")) if role != "ablation" else "ablation",
            "status": (exp or {}).get("status", "missing"), "failure_kind": failure_kind(exp),
            "params_total": ((exp or {}).get("params") or {}).get("total"),
            "gflops_per_step": round(costs["flops_history"][0][1] / 1e9, 2) if costs.get("flops_history") else None,
            "delta_common": deltas.get(f"{self.common:g}"), "deltas": deltas,
            "dev_loss_common": cks[self.common]["dev_loss"] if self.common in cks else None,
            "train_tokens": last.get("tokens"), "train_flops": costs.get("train_flops"),
            "train_seconds": costs.get("train_seconds"), "peak_mem_bytes": costs.get("peak_mem_bytes"),
            "device": (exp or {}).get("device"),
            "copy_gain_delta": _sub((probes or {}).get("copy_gain"), bp.get("copy_gain")),
            "late_position_loss_delta": _sub((probes or {}).get("late_position_loss"), bp.get("late_position_loss")),
            "experiment_id": (exp or {}).get("id"),
        }

    def ledger(self):
        rows = []
        for job in self.jobs:
            cands = job["candidates"]
            for i, cid in enumerate(cands, 1):
                role = "final" if cid == job["final_candidate_id"] else "attempt"
                rows.append(self.candidate_row(job, cid, role, i))
            if job.get("ablation_candidate_id"):
                rows.append(self.candidate_row(job, job["ablation_candidate_id"], "ablation", None))
        return rows

    def job_rows(self, ledger):
        by_job = {}
        for r in ledger:
            by_job.setdefault(r["job_id"], []).append(r)
        out = []
        for job in self.jobs:
            p = job["plan"]
            rows = by_job.get(p["id"], [])
            final = next((r for r in rows if r["role"] == "final"), None)
            abl = next((r for r in rows if r["role"] == "ablation"), None)
            dep = [r for r in job["retrievals"] if r["tool"] in DEPENDENCY_TOOLS and r.get("phase") == "propose"]
            refl = job.get("reflection") or {}
            rep = job.get("space_report") or {}
            out.append({
                "campaign": self.c.name, "condition": self.condition, "job_id": p["id"], "wave": p["wave"],
                "slot": p["slot"], "mode": p["mode"], "status": job["status"],
                "change_type": job.get("change_type"), "category": CATEGORY.get(job.get("change_type")),
                "final_candidate_id": job.get("final_candidate_id"), "attempts": len(job["candidates"]),
                "repairs_used": max(0, len(job["candidates"]) - 1),
                "final_status": final["status"] if final else None,
                "final_failure_kind": final["failure_kind"] if final else None,
                "delta_common": final["delta_common"] if final else None,
                "ablation_delta_common": abl["delta_common"] if abl else None,
                "copy_gain_delta": final["copy_gain_delta"] if final else None,
                "late_position_loss_delta": final["late_position_loss_delta"] if final else None,
                "dependency_retrievals_before_results": len(dep),
                "dependency_targets": sorted({str(r["target"]) for r in dep}),
                "formulation_edited": bool(job.get("final_candidate_id")) and self._formulation_edited(
                    job["final_candidate_id"], p["research_parent_id"]),
                "nodes_suspended": len(rep.get("suspended") or []),
                "evidence_class": refl.get("evidence_class"), "followup_enqueued": refl.get("followup_enqueued"),
                "llm_calls": job["usage"].get("calls"), "prompt_tokens": job["usage"].get("prompt_tokens"),
                "cached_tokens": job["usage"].get("cached_tokens"),
                "completion_tokens": job["usage"].get("completion_tokens"),
                "reasoning_tokens": job["usage"].get("reasoning_tokens"),
                "llm_seconds": job["usage"].get("seconds"),
                "wall_seconds": (job["finished"] - job["started"]) if job.get("finished") and job.get("started") else None,
                "error": job.get("error"),
            })
        return out

    # ---- summary ---------------------------------------------------------------
    def summary(self, ledger, jobs, delta=0.02):
        attempts = [r for r in ledger if r["role"] != "ablation"]
        finals = [j for j in jobs if j["final_candidate_id"]]
        valid = [j for j in finals if j["final_status"] == "ok"]
        deltas = [j["delta_common"] for j in valid if j["delta_common"] is not None]
        ledg = {(r["ledger"], r["unit"]): r["total"] for r in self.c.archive.ledger_summary()}
        fail_kinds = {}
        for r in attempts:
            if r["failure_kind"]:
                fail_kinds[r["failure_kind"]] = fail_kinds.get(r["failure_kind"], 0) + 1
        challenges = [j for j in jobs if j["mode"] == "challenge"]
        best = min(valid, key=lambda j: j["delta_common"], default=None) if deltas else None
        return {
            "campaign": self.c.name, "condition": self.condition, "model": self.cfg["model"],
            "waves_completed": int(self.c._meta_get("next_wave", "1")) - 1,
            "common_checkpoint_frac": self.common,
            "baseline_dev_loss_common": self.base[self.common]["dev_loss"] if self.common in self.base else None,
            "jobs": {s: sum(j["status"] == s for j in jobs)
                     for s in ("completed", "implementation_failure", "agent_failure")},
            "jobs_by_mode": {m: sum(j["mode"] == m for j in jobs) for m in ("challenge", "continuation", "reconstruction")},
            "candidates_submitted": len(attempts), "ablations": sum(r["role"] == "ablation" for r in ledger),
            "attempt_failures": fail_kinds, "attempts_valid": sum(r["status"] == "ok" for r in attempts),
            "repairs_used": sum(j["repairs_used"] for j in jobs),
            "final_valid": len(valid), "final_total": len(finals),
            "categories": {c: sum(j["category"] == c for j in finals) for c in CATEGORIES},
            "beats_baseline": sum(d < 0 for d in deltas), "beats_baseline_by_delta": sum(d <= -delta for d in deltas),
            "usefulness_delta": delta,
            "best": {"job_id": best["job_id"], "candidate_id": best["final_candidate_id"],
                     "delta_common": best["delta_common"], "category": best["category"]} if best else None,
            "median_delta_common": sorted(deltas)[len(deltas) // 2] if deltas else None,
            "evidence_classes": {k: sum(j["evidence_class"] == k for j in jobs)
                                 for k in ("supported", "contradicted", "inconclusive", "implementation_failure")},
            "dependency_use": {
                "challenge_jobs": len(challenges),
                "with_retrieval_before_results": sum(j["dependency_retrievals_before_results"] > 0 for j in challenges),
                "retrievals_before_results": sum(j["dependency_retrievals_before_results"] for j in jobs),
                "formulation_edited": sum(bool(j["formulation_edited"]) for j in jobs),
                "nodes_suspended": sum(j["nodes_suspended"] for j in jobs),
            },
            "mechanism_evidence": {
                "with_ablation": sum(j["ablation_delta_common"] is not None for j in jobs),
                "copy_gain_improved": sum((j["copy_gain_delta"] or 0) > 0.05 for j in valid),
                "late_position_loss_improved": sum((j["late_position_loss_delta"] or 0) < -0.01 for j in valid),
            },
            "costs": {
                "input_tokens": ledg.get(("inference", "input_tokens"), 0),
                "cached_input_tokens": ledg.get(("inference", "cached_input_tokens"), 0),
                "output_tokens": ledg.get(("inference", "output_tokens"), 0),
                "reasoning_tokens": sum(j["reasoning_tokens"] or 0 for j in jobs),
                "inference_usd": ledg.get(("inference", "usd")),
                "llm_latency_seconds": ledg.get(("inference", "latency_seconds"), 0),
                "train_flops": ledg.get(("train_compute", "flops"), 0),
                "train_device_seconds": ledg.get(("train_compute", "device_seconds"), 0),
                "eval_device_seconds": ledg.get(("eval_compute", "device_seconds"), 0),
                "train_tokens_processed": sum(r["train_tokens"] or 0 for r in ledger),
                "peak_mem_bytes": max((r["peak_mem_bytes"] or 0 for r in ledger), default=0) or None,
                "devices": sorted({r["device"] for r in ledger if r["device"]}),
            },
        }

    # ---- case cards ------------------------------------------------------------
    def case_card(self, job, jrow, neutral):
        p, prop = job["plan"], job.get("proposal") or {}
        refl = job.get("reflection") or {}
        f = lambda v: "-" if v is None else f"{v:+.4f}"
        lines = []
        if not neutral:
            lines.append(f"### {self.c.name} / {p['id']} ({p['mode']}, wave {p['wave']}, slot {p['slot']})")
            lines.append(f"- agent label: {jrow['category'] or '-'} ({jrow['change_type'] or '-'}); "
                         f"status {job['status']}; candidate {job.get('final_candidate_id') or '-'}")
        if not prop:
            lines.append("- no proposal was submitted" + (f" ({job.get('error')})" if job.get("error") else ""))
            return "\n".join(lines)
        lines += [f"- **Commitment questioned:** {prop.get('commitment', '').strip()}",
                  f"- **Change:** {prop.get('change', '').strip()}",
                  f"- **Proposed mechanism:** {prop.get('mechanism', '').strip()}",
                  f"- **Prediction:** {prop.get('prediction', '').strip()}",
                  f"- **Scope:** {prop.get('scope', '').strip()}"]
        s = job.get("summary") or {}
        impl = [f"{jrow['attempts']} attempt(s), {jrow['repairs_used']} repair(s)",
                f"final run {jrow['final_status'] or '-'}" + (f" ({jrow['final_failure_kind']})"
                                                               if jrow["final_failure_kind"] else "")]
        if s.get("params_total"):
            impl.append(f"{s['params_total'] / 1e6:.2f}M params, {s.get('gflops_per_train_step')} GFLOP/step "
                        f"(baseline {self._baseline_gflops()})")
        lines.append("- **Implementation evidence:** " + "; ".join(impl))
        outs = [f"dev-loss delta vs baseline at {self.common:.0%} of the horizon: {f(jrow['delta_common'])}"]
        if jrow["ablation_delta_common"] is not None:
            outs.append(f"ablation {f(jrow['ablation_delta_common'])}")
        if jrow["copy_gain_delta"] is not None:
            outs.append(f"copy-probe gain vs baseline {f(jrow['copy_gain_delta'])}")
        if jrow["late_position_loss_delta"] is not None:
            outs.append(f"late-position loss vs baseline {f(jrow['late_position_loss_delta'])}")
        lines.append("- **Outcomes:** " + "; ".join(outs))
        if refl.get("interpretation"):
            lines.append(f"- **Agent interpretation** ({refl.get('evidence_class')}): {refl['interpretation'].strip()}")
        if not neutral:
            lines.append(f"- dependency retrievals before results: {jrow['dependency_retrievals_before_results']}"
                         + (f" ({', '.join(jrow['dependency_targets'][:8])})" if jrow["dependency_targets"] else "")
                         + f"; formulation edited: {jrow['formulation_edited']}; nodes suspended: {jrow['nodes_suspended']}")
        return "\n".join(lines)

    def _baseline_gflops(self):
        fh = (self.control.get("costs") or {}).get("flops_history")
        return round(fh[0][1] / 1e9, 2) if fh else "?"

    def representation_terms(self):
        """Strings that would reveal the arm in a neutral card: box node ids and DAG tool vocabulary."""
        from .box import load_space

        try:
            nodes = [n["id"] for n in load_space(self.cfg["box"])["nodes"]]
        except Exception:
            nodes = []
        return [n for n in nodes if len(n) > 3] + ["query_box", "space_patch", "DAG", "dependency graph",
                                                   "register_revision", "suspend"]

    # ---- write -----------------------------------------------------------------
    def write(self, out_dir=None, delta=0.02):
        out = Path(out_dir or self.c.dir / "report")
        out.mkdir(parents=True, exist_ok=True)
        ledger = self.ledger()
        jobs = self.job_rows(ledger)
        summary = self.summary(ledger, jobs, delta=delta)
        _write_json(out / "ledger.json", ledger)
        _write_csv(out / "ledger.csv", ledger, exclude=("deltas",))
        _write_json(out / "jobs.json", jobs)
        _write_json(out / "summary.json", summary)
        jmap = {j["job_id"]: j for j in jobs}
        cards = [self.case_card(job, jmap[job["plan"]["id"]], neutral=False) for job in self.jobs]
        (out / "cases.md").write_text(f"# Case cards: campaign `{self.c.name}` ({self.condition})\n\n"
                                      + "\n\n".join(cards) + "\n", encoding="utf-8")
        terms = self.representation_terms()
        neutral = []
        for job in self.jobs:
            card = self.case_card(job, jmap[job["plan"]["id"]], neutral=True)
            flagged = sorted({t for t in terms if t.lower() in card.lower()})
            neutral.append({"job_id": job["plan"]["id"], "card": card, "flagged_terms": flagged})
        (out / "cases_neutral.md").write_text(
            "# Condition-neutral case cards\n\nCards omit how the formulation was represented. "
            "`flagged` lists terms in the agent's own text that may reveal the condition; review before blinded use.\n\n"
            + "\n\n".join(f"### {n['job_id']}" + (f"  (flagged: {', '.join(n['flagged_terms'])})" if n["flagged_terms"]
                                                  else "") + "\n" + n["card"] for n in neutral) + "\n",
            encoding="utf-8")
        _write_json(out / "cases_neutral.json", neutral)
        (out / "summary.md").write_text(summary_markdown(summary), encoding="utf-8")
        return {"dir": out, "summary": summary, "ledger": ledger, "jobs": jobs, "neutral": neutral}


def summary_markdown(s):
    c = s["costs"]
    d = s["dependency_use"]
    best = s["best"]
    lines = [f"# Campaign `{s['campaign']}` ({s['condition']}, {s['model']})", "",
             f"- waves completed: {s['waves_completed']}; jobs {s['jobs']}; by mode {s['jobs_by_mode']}",
             f"- candidates submitted {s['candidates_submitted']} ({s['attempts_valid']} valid), ablations "
             f"{s['ablations']}, repairs used {s['repairs_used']}",
             f"- attempt failures: {s['attempt_failures'] or 'none'}",
             f"- final candidates valid: {s['final_valid']}/{s['final_total']}; agent categories {s['categories']}",
             f"- at {s['common_checkpoint_frac']:.0%} of the horizon (baseline {s['baseline_dev_loss_common']}): "
             f"{s['beats_baseline']} beat the baseline, {s['beats_baseline_by_delta']} by at least "
             f"{s['usefulness_delta']} nats; median delta {s['median_delta_common']}",
             f"- best: {best['candidate_id']} ({best['job_id']}, {best['category']}) delta {best['delta_common']:+.4f}"
             if best else "- best: none valid",
             f"- evidence classes: {s['evidence_classes']}",
             f"- dependency use: {d['with_retrieval_before_results']}/{d['challenge_jobs']} challenge jobs retrieved "
             f"dependencies before results ({d['retrievals_before_results']} retrievals); formulation edited in "
             f"{d['formulation_edited']} jobs; {d['nodes_suspended']} nodes suspended",
             f"- mechanism evidence: {s['mechanism_evidence']}",
             f"- inference: {c['input_tokens']:,.0f} input + {c['cached_input_tokens']:,.0f} cached + "
             f"{c['output_tokens']:,.0f} output tokens ({c['reasoning_tokens']:,} reasoning)"
             + (f", ${c['inference_usd']:.2f}" if c["inference_usd"] else "")
             + f", {c['llm_latency_seconds']:.0f}s model latency",
             f"- training: {c['train_flops']:.3g} FLOPs, {c['train_tokens_processed']:,} tokens, "
             f"{c['train_device_seconds'] / 3600:.2f} device-hours on {', '.join(c['devices']) or '-'}"
             + (f", peak memory {c['peak_mem_bytes'] / 2**30:.2f} GiB" if c["peak_mem_bytes"] else ""), ""]
    return "\n".join(lines)


def _write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")


def _write_csv(path, rows, exclude=()):
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    cols = [k for k in rows[0] if k not in exclude]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items() if k in cols})


def report_campaign(name, root=None, delta=0.02):
    from .agents.campaign import Campaign

    camp = Campaign(name, root=root)
    try:
        return CampaignReport(camp).write(delta=delta)
    finally:
        camp.close()


# ---------------------------------------------------------------------------
# Stage B: matched DAG/flat pairs
PAIR_METRICS = [  # (label, getter, better: "lower" | "higher" | None, format)
    ("final candidates valid", lambda s: s["final_valid"], "higher", "count"),
    ("implementation failures (attempts)", lambda s: sum(s["attempt_failures"].values()), "lower", "count"),
    ("repairs used", lambda s: s["repairs_used"], None, "count"),
    ("beat baseline", lambda s: s["beats_baseline"], "higher", "count"),
    ("beat baseline by delta", lambda s: s["beats_baseline_by_delta"], "higher", "count"),
    ("best delta at common checkpoint", lambda s: (s["best"] or {}).get("delta_common"), "lower", "loss"),
    ("median delta at common checkpoint", lambda s: s["median_delta_common"], "lower", "loss"),
    ("substantive reformulations (agent label)", lambda s: s["categories"]["substantive reformulation"], None, "count"),
    ("known-space switches (agent label)", lambda s: s["categories"]["known-space switch"], None, "count"),
    ("challenge jobs using dependencies before results", lambda s: s["dependency_use"]["with_retrieval_before_results"], None, "count"),
    ("jobs with formulation edits", lambda s: s["dependency_use"]["formulation_edited"], None, "count"),
    ("supported findings (agent)", lambda s: s["evidence_classes"]["supported"], None, "count"),
    ("input tokens (uncached)", lambda s: s["costs"]["input_tokens"], "lower", "count"),
    ("cached input tokens", lambda s: s["costs"]["cached_input_tokens"], None, "count"),
    ("output tokens", lambda s: s["costs"]["output_tokens"], "lower", "count"),
    ("inference USD", lambda s: s["costs"]["inference_usd"], "lower", "usd"),
    ("training FLOPs", lambda s: s["costs"]["train_flops"], None, "sci"),
    ("training device-hours", lambda s: s["costs"]["train_device_seconds"] / 3600, None, "hours"),
]


def check_match(dag_cfg, flat_cfg):
    problems = []
    if dag_cfg["condition"] != "dag" or flat_cfg["condition"] != "flat":
        problems.append(f"conditions are {dag_cfg['condition']}/{flat_cfg['condition']}, expected dag/flat")
    for k in MATCH_KEYS:
        if dag_cfg.get(k) != flat_cfg.get(k):
            problems.append(f"{k}: {dag_cfg.get(k)!r} vs {flat_cfg.get(k)!r}")
    return problems


def compare(study, pairs, campaigns_root=None, out_root=None, delta=0.02, seed=0, allow_mismatch=False, log=print):
    """pairs: [(dag_campaign_name, flat_campaign_name), ...]"""
    from .agents.campaign import Campaign

    out = Path(out_root or STUDIES_DIR) / study
    out.mkdir(parents=True, exist_ok=True)
    pair_results, all_cards, evaluators = [], [], set()
    for i, (dag_name, flat_name) in enumerate(pairs, 1):
        arms = {}
        for arm, name in (("dag", dag_name), ("flat", flat_name)):
            camp = Campaign(name, root=campaigns_root)
            try:
                rep = CampaignReport(camp).write(delta=delta)
                arms[arm] = {"name": name, "config": camp.config, "summary": rep["summary"]}
                evaluators |= {camp.archive.get_experiment(r["experiment_id"])["evaluator_version"]
                               for r in rep["ledger"] if r["experiment_id"]}
                for n in rep["neutral"]:
                    all_cards.append({"pair": i, "arm": arm, "campaign": name, **n})
            finally:
                camp.close()
        problems = check_match(arms["dag"]["config"], arms["flat"]["config"])
        ws = (arms["dag"]["summary"]["waves_completed"], arms["flat"]["summary"]["waves_completed"])
        if ws[0] != ws[1]:
            problems.append(f"waves completed: {ws[0]} vs {ws[1]}")
        if problems and not allow_mismatch:
            raise ValueError(f"pair {i} ({dag_name}, {flat_name}) is not matched: " + "; ".join(problems))
        metrics = []
        for label, get, better, kind in PAIR_METRICS:
            a, b = get(arms["dag"]["summary"]), get(arms["flat"]["summary"])
            diff = None if a is None or b is None else a - b
            favors = None
            if better and diff is not None and diff != 0:
                favors = "dag" if (diff < 0) == (better == "lower") else "flat"
            metrics.append({"metric": label, "dag": a, "flat": b, "dag_minus_flat": diff, "favors": favors,
                            "format": kind})
        pair_results.append({"pair": i, "dag": dag_name, "flat": flat_name, "match_problems": problems,
                             "metrics": metrics})
    if len(evaluators) > 1:
        msg = f"campaigns used {len(evaluators)} evaluator versions: {sorted(evaluators)}"
        if not allow_mismatch:
            raise ValueError(msg)
        log("warning: " + msg)

    sign_counts = []
    for k, (label, *_rest) in enumerate(PAIR_METRICS):
        fav = [p["metrics"][k]["favors"] for p in pair_results]
        sign_counts.append({"metric": label, "dag": fav.count("dag"), "flat": fav.count("flat"),
                            "tie_or_undirected": len(fav) - fav.count("dag") - fav.count("flat")})
    result = {
        "study": study, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "pairs": pair_results,
        "sign_counts": sign_counts, "usefulness_delta": delta, "evaluator_versions": sorted(evaluators),
        "caveats": [
            "Exploratory pilot: a few matched pairs support a descriptive comparison, not a rare-success-rate claim.",
            "A DAG-flat tie is inconclusive about graph benefit and does not refute the shared core approach; a DAG "
            "loss advantage alone does not demonstrate transformational creativity.",
            "Neither arm is compared with ordinary autoresearch; superiority over it needs an additional arm.",
            "Change categories here are the agents' pre-result labels; researcher judgments come from the blinded "
            "judging sheet. Judging was blinded only if key.json stayed closed until judging was complete.",
            "Graph queries, serialization and maintenance are charged to the DAG arm; one-time human preparation is "
            "reported separately.",
        ],
    }
    _write_json(out / "comparison.json", result)
    (out / "comparison.md").write_text(comparison_markdown(result), encoding="utf-8")
    _write_blinded(out / "blinded", all_cards, seed)
    log(f"wrote {out / 'comparison.md'} and {out / 'blinded'}")
    return result


def _write_blinded(d, cards, seed):
    d.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    cards = [c for c in cards if "no proposal was submitted" not in c["card"]]
    rng.shuffle(cards)
    key, body, sheet = {}, [], []
    for k, c in enumerate(cards, 1):
        cid = f"case-{k:03d}"
        key[cid] = {"pair": c["pair"], "arm": c["arm"], "campaign": c["campaign"], "job_id": c["job_id"],
                    "flagged_terms": c["flagged_terms"]}
        body.append(f"## {cid}\n{c['card']}")
        sheet.append({"case_id": cid, "category": "", "mechanism_evidence": "", "promising": "", "notes": ""})
    flagged = sum(bool(c["flagged_terms"]) for c in cards)
    (d / "cases.md").write_text(
        "# Blinded case cards\n\nJudge each case in judging.csv. category: ordinary exploration | known-space switch | "
        "substantive reformulation | uncertain. mechanism_evidence: none | weak | clear. promising: yes | no | unsure.\n"
        + (f"\n{flagged} card(s) contain terms that may reveal the condition (listed in key.json); "
           "edit or exclude them before judging if blinding matters.\n" if flagged else "")
        + "\n" + "\n\n".join(body) + "\n", encoding="utf-8")
    _write_csv(d / "judging.csv", sheet) if sheet else (d / "judging.csv").write_text(
        "case_id,category,mechanism_evidence,promising,notes\n", encoding="utf-8")
    _write_json(d / "key.json", {"note": "Do not open until judging is complete.", "seed": seed, "cases": key})


def comparison_markdown(r):
    def fmt(v, kind, signed=False):
        if v is None:
            return "-"
        sign = "+" if signed else ""
        return {"loss": f"{v:{sign}.4f}", "count": f"{v:{sign},.0f}", "usd": f"{v:{sign},.2f}",
                "sci": f"{v:{sign}.3g}", "hours": f"{v:{sign}.3f}"}[kind]

    lines = [f"# Stage B comparison `{r['study']}`", "",
             f"{len(r['pairs'])} matched pair(s). Evaluator: {', '.join(v[:12] for v in r['evaluator_versions'])}.", ""]
    for p in r["pairs"]:
        lines += [f"## Pair {p['pair']}: `{p['dag']}` (DAG) vs `{p['flat']}` (flat)", ""]
        if p["match_problems"]:
            lines += ["**Match problems (allowed by --allow-mismatch):** " + "; ".join(p["match_problems"]), ""]
        lines += ["| metric | DAG | flat | DAG - flat | favors |", "| --- | --- | --- | --- | --- |"]
        lines += [f"| {m['metric']} | {fmt(m['dag'], m['format'])} | {fmt(m['flat'], m['format'])} | "
                  f"{fmt(m['dag_minus_flat'], m['format'], signed=True)} | {m['favors'] or '-'} |" for m in p["metrics"]]
        lines.append("")
    lines += ["## Across pairs (count of pairs favoring each arm)", "", "| metric | DAG | flat | tie / undirected |",
              "| --- | --- | --- | --- |"]
    lines += [f"| {s['metric']} | {s['dag']} | {s['flat']} | {s['tie_or_undirected']} |" for s in r["sign_counts"]]
    lines += ["", "## Caveats", ""] + [f"- {c}" for c in r["caveats"]] + [""]
    return "\n".join(lines)
