"""Stage A calibration cohort: do cheap smoke tests predict longer-run outcomes?

Phases (each idempotent, each requiring the previous one):

  enroll  Pick up to `cap` agent candidates from source campaigns by a fixed,
          outcome-blind rule: creation order, engineering-valid only, about
          `reform_quota` of slots for reformulations (the agent's pre-result
          change_type, or researcher overrides). Only `checks.passed` is read from
          source experiments, never a loss. Known controls and the baseline are
          added and labelled separately.
  smoke   Every member and the baseline get the smoke test (checkpoints at
          `smoke_fracs` of the horizon, smoke seeds). Identical prior smoke runs in
          the source campaign are reused.
  screen  The frozen screening rule turns smoke deltas into advance / hold / reject.
          Decisions are written and hashed before any long run exists.
  long    Every member and the baseline get a longer run with fresh seeds, whatever
          its screening decision (no selection on the short score).
  report  Paired smoke-vs-long results, rank correlations overall and by stratum,
          screening errors with explicit denominators, learning-curve crossings.

A cohort lives in archive/cohorts/<name>/ with its own archive, separate from every
campaign, so agents never see long-run outcomes.
"""

import csv
import json
import math
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path

from .archive import Archive
from .frozen import apply_overrides, read_frozen, spec_hash, write_frozen
from .paths import ARCHIVE_DIR, SEEDS_DIR
from .results import deltas_vs, first_line

COHORTS_DIR = ARCHIVE_DIR / "cohorts"
CONTROLS_DIR = SEEDS_DIR / "controls"
PHASES = ("enroll", "smoke", "screen", "long", "report")
AGENT_MODES = ("challenge", "continuation", "reconstruction")

DEFAULT_SPEC = dict(
    sources=[],                       # campaign names supplying agent candidates
    campaigns_root=None,              # optional override of the campaigns directory
    baseline=str(SEEDS_DIR / "baseline_decoder"),
    controls=["recurrent_depth", "depth_weighted", "wide_shallow"],  # names under seeds/controls, or paths
    cap=10,                           # agent candidates enrolled
    reform_quota=0.5,                 # share of agent slots reserved for reformulations
    stratum_overrides={},             # source candidate id -> "reformulation" | "ordinary" (set before results)
    task=None,
    horizon_flops=5.5e15,             # the planned full run; schedules always span it
    smoke_fracs=[0.05, 0.1],          # b1, b2
    long_fracs=[0.05, 0.1, 0.25, 0.5, 1.0],
    eval_tokens=1 << 19,
    smoke_seeds={"init": 0, "data": 0},
    long_seeds={"init": 1, "data": 1},  # fresh seeds: the long run is independent, not a continuation
    screening=dict(advance_max_delta=0.0, reject_min_delta=0.05, reject_requires_no_closing=True),
    usefulness=dict(delta=0.02, uncertain_band=0.01),
    device="auto",
    throttle=0.0,
    temp_limit_c=None,
    eval_time_limit_s=7200.0,
    max_params=60_000_000,
    executor={"kind": "local", "slots": 1},  # slots > 1 runs members concurrently
)

_SQL = """
CREATE TABLE IF NOT EXISTS cohort_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS members (
    member_id TEXT PRIMARY KEY, candidate_id TEXT NOT NULL, provenance TEXT NOT NULL, stratum TEXT,
    source TEXT, source_candidate_id TEXT, change_type TEXT, enroll_order INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS runs (
    member_id TEXT NOT NULL, phase TEXT NOT NULL, experiment_id TEXT NOT NULL, reused_from TEXT,
    PRIMARY KEY (member_id, phase));
CREATE TABLE IF NOT EXISTS decisions (
    member_id TEXT PRIMARY KEY, decision TEXT NOT NULL, reason TEXT NOT NULL, smoke_deltas TEXT NOT NULL);
"""


class CohortError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# statistics and rules
def _ranks(values):
    """1-based ranks, ties get their average rank."""
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in order[i:j + 1]:
            ranks[k] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(x, y):
    """Spearman rank correlation over pairs where both values are finite; None if undefined."""
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None and math.isfinite(a) and math.isfinite(b)]
    if len(pairs) < 3:
        return None
    rx, ry = _ranks([a for a, _ in pairs]), _ranks([b for _, b in pairs])
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx, syy = sum((a - mx) ** 2 for a in rx), sum((b - my) ** 2 for b in ry)
    return sxy / math.sqrt(sxx * syy) if sxx and syy else None


def screen_decision(deltas, status, rule):
    """advance / hold / reject from smoke deltas (candidate minus baseline) at b1, b2."""
    if status != "ok" or not deltas:
        return "reject", f"smoke run did not complete (status {status})"
    d1, d2 = deltas[0], deltas[-1]
    if d2 <= rule["advance_max_delta"]:
        return "advance", f"delta at b2 {d2:+.4f} <= {rule['advance_max_delta']:+.4f}"
    closing = d2 < d1
    needs_no_closing = rule.get("reject_requires_no_closing", True)
    if d2 >= rule["reject_min_delta"] and not (needs_no_closing and closing):
        return "reject", (f"delta at b2 {d2:+.4f} >= {rule['reject_min_delta']:+.4f}"
                          + (", gap not closing" if needs_no_closing else ""))
    return "hold", f"delta at b2 {d2:+.4f}" + (" (gap closing from b1)" if closing else "")


def usefulness_label(delta, status, rule):
    if status != "ok" or delta is None:
        return "failed"
    if delta <= -rule["delta"]:
        return "useful"
    if delta >= -rule["delta"] + rule["uncertain_band"]:
        return "not_useful"
    return "uncertain"


def _fraction(numerator, denominator):
    return {"numerator": len(numerator), "denominator": len(denominator),
            "value": len(numerator) / len(denominator) if denominator else None}


def cohort_stats(rows):
    useful = [r for r in rows if r["usefulness"] == "useful"]
    rejected = [r for r in rows if r["decision"] == "reject"]
    missed = [r for r in useful if r["decision"] == "reject"]
    final = [r["long_delta_final"] for r in rows]
    count = lambda key, values: {v: sum(r[key] == v for r in rows) for v in values}
    return {
        "n": len(rows),
        "spearman_smoke_b2_vs_long_final": spearman([r["smoke_delta_b2"] for r in rows], final),
        "spearman_smoke_b1_vs_long_final": spearman([r["smoke_delta_b1"] for r in rows], final),
        "missed_useful_fraction": _fraction(missed, useful),
        "useful_among_rejects_fraction": _fraction(missed, rejected),
        "decisions": count("decision", ("advance", "hold", "reject")),
        "usefulness": count("usefulness", ("useful", "uncertain", "not_useful", "failed")),
        "crossings": count("crossing", ("late_bloomer", "early_fader")),
    }


# ---------------------------------------------------------------------------
class Cohort:
    def __init__(self, name, root=None):
        self.name = name
        self.dir = Path(root or COHORTS_DIR) / name
        if not (self.dir / "cohort.json").is_file():
            raise FileNotFoundError(f"cohort {name!r} not found in {self.dir.parent}; run `cohort init` first")
        self.spec, self.spec_hash = read_frozen(self.dir / "cohort.json")
        self.archive = Archive(self.dir / "archive")
        self.db = self.archive.db
        self.db.executescript(_SQL)
        from .task import load_task

        self.task = load_task(self.spec["task"])

    @classmethod
    def create(cls, name, root=None, **overrides):
        from .task import default_task_dir

        spec = apply_overrides(DEFAULT_SPEC, overrides)
        spec["task"] = spec["task"] or str(default_task_dir())
        if not spec["sources"] and not spec["controls"]:
            raise ValueError("a cohort needs at least one source campaign or control")
        b1, b2 = spec["smoke_fracs"] if len(spec["smoke_fracs"]) == 2 else (0, 0)
        if not 0 < b1 < b2 <= 1:
            raise ValueError("smoke_fracs must be two increasing fractions (b1, b2)")
        if max(spec["long_fracs"]) != 1.0:
            raise ValueError("long_fracs must end at 1.0 (the full horizon)")
        spec["controls"] = [str(_control_path(c)) for c in spec["controls"]]
        spec["created"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        write_frozen(Path(root or COHORTS_DIR) / name / "cohort.json", spec)
        return cls(name, root)

    def close(self):
        self.archive.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _campaign(self, name):
        from .agents.campaign import Campaign

        return Campaign(name, root=self.spec.get("campaigns_root"))

    # ---- state ---------------------------------------------------------------
    def _get(self, key):
        row = self.db.execute("SELECT value FROM cohort_meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def _set(self, key, value):
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO cohort_meta VALUES (?, ?)", (key, str(value)))

    def phase_done(self, phase):
        return self._get(f"done_{phase}") is not None

    def members(self, provenance=None):
        q = "SELECT * FROM members" + (" WHERE provenance = ?" if provenance else "") + " ORDER BY enroll_order"
        return [dict(r) for r in self.db.execute(q, (provenance,) if provenance else ()).fetchall()]

    def run_of(self, member_id, phase):
        """(experiment, reused_from) of a member's run in a phase, or (None, None)."""
        row = self.db.execute("SELECT * FROM runs WHERE member_id = ? AND phase = ?", (member_id, phase)).fetchone()
        return (self.archive.get_experiment(row["experiment_id"]), row["reused_from"]) if row else (None, None)

    def decision_of(self, member_id):
        row = self.db.execute("SELECT * FROM decisions WHERE member_id = ?", (member_id,)).fetchone()
        return dict(row) if row else None

    def _deltas(self, member_id, phase):
        """(experiment, [(frac, candidate minus baseline dev loss)]) for a member's run."""
        exp, _ = self.run_of(member_id, phase)
        base, _ = self.run_of("baseline", phase)
        return exp, (deltas_vs(exp, base) if exp and base else [])

    # ---- enroll --------------------------------------------------------------
    def _eligible(self, camp):
        """Agent candidates of one campaign in creation order (wave, job, submission), with strata.
        Reads provenance, change_type and checks.passed only: outcome-blind."""
        rows = camp.db.execute(
            "SELECT c.id, c.action_mode, cm.change_type, cm.smoke_experiment_id FROM candidates c "
            "JOIN candidate_meta cm ON cm.candidate_id = c.id "
            "ORDER BY cm.creation_wave, cm.job_id, c.created, c.id").fetchall()
        eligible, invalid = [], []
        for r in rows:
            if r["action_mode"] not in AGENT_MODES:
                continue
            exp = camp.archive.get_experiment(r["smoke_experiment_id"]) if r["smoke_experiment_id"] else None
            if not (exp and (exp.get("checks") or {}).get("passed")):
                invalid.append(r["id"])
                continue
            stratum = self.spec["stratum_overrides"].get(r["id"]) or (
                "reformulation" if r["change_type"] == "formulation_change" else "ordinary")
            eligible.append(dict(source=camp.name, source_candidate_id=r["id"], change_type=r["change_type"],
                                 stratum=stratum, bundle_hash=camp.archive.get_candidate(r["id"])["bundle_hash"]))
        return eligible, invalid

    def enroll(self, log=print):
        if self.phase_done("enroll"):
            return self.members()
        with ExitStack() as stack:
            camps = {s: stack.enter_context(self._campaign(s)) for s in self.spec["sources"]}
            pool, invalid, seen = [], [], set()
            for camp in camps.values():
                eligible, bad = self._eligible(camp)
                invalid += bad
                for c in eligible:  # exact duplicates across sources count once
                    if c["bundle_hash"] not in seen:
                        seen.add(c["bundle_hash"])
                        pool.append(c)
            cap = int(self.spec["cap"])
            n_ref = round(cap * float(self.spec["reform_quota"]))
            ref = [c for c in pool if c["stratum"] == "reformulation"]
            ordinary = [c for c in pool if c["stratum"] != "reformulation"]
            chosen = ref[:n_ref] + ordinary[:cap - n_ref]
            chosen += [c for c in pool if c not in chosen][:cap - len(chosen)]  # fill a short stratum
            chosen.sort(key=pool.index)

            with tempfile.TemporaryDirectory(prefix="cspace_cohort_") as tmp:
                for i, c in enumerate(chosen, 1):
                    bundle = camps[c["source"]].archive.materialize(c["bundle_hash"], Path(tmp) / str(i))
                    rec, _ = self.archive.add_candidate(
                        bundle, action_mode="manual", notes=f"from campaign {c['source']} ({c['source_candidate_id']})")
                    self._add_member(f"m{i:02d}", rec["id"], "agent", i, c["stratum"], c["source"],
                                     c["source_candidate_id"], c["change_type"])
        order = len(chosen)
        for path in self.spec["controls"]:
            order += 1
            rec, _ = self.archive.add_candidate(path, action_mode="control", notes=f"known control {Path(path).name}")
            self._add_member(f"k_{Path(path).name}", rec["id"], "control", order, source="controls")
        rec, _ = self.archive.add_candidate(self.spec["baseline"], action_mode="seed", notes="baseline")
        self._add_member("baseline", rec["id"], "baseline", 0, source="seed")

        filled = len(chosen) - min(len(ref), n_ref) - min(len(ordinary), cap - n_ref)
        record = {
            "rule": "creation order (wave, job, submission); engineering-valid (checks.passed) only; stratified by "
                    "pre-result change_type or researcher override; exact duplicates once; losses never read",
            "cap": cap, "reformulation_slots": n_ref,
            "available": {"reformulation": len(ref), "ordinary": len(ordinary)},
            "shortfall_filled_from_other_stratum": max(0, filled),
            "skipped_engineering_invalid": invalid,
            "members": self.members(),
        }
        (self.dir / "enrollment.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
        self._set("enrollment_hash", spec_hash(record))
        self._set("done_enroll", time.time())
        n_reform = sum(c["stratum"] == "reformulation" for c in chosen)
        log(f"enrolled {len(chosen)} agent candidates ({n_reform} reformulation), {len(self.spec['controls'])} "
            f"controls, plus the baseline; {len(invalid)} engineering-invalid skipped")
        if len(chosen) < cap:
            log(f"  only {len(chosen)} eligible candidates were available for {cap} slots")
        return self.members()

    def _add_member(self, member_id, candidate_id, provenance, order, stratum=None, source=None,
                    source_candidate_id=None, change_type=None):
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO members VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                            (member_id, candidate_id, provenance, stratum, source, source_candidate_id,
                             change_type, order))

    # ---- smoke and long runs ---------------------------------------------------
    def _budget(self, phase):
        fracs = self.spec["smoke_fracs"] if phase == "smoke" else self.spec["long_fracs"]
        return dict(horizon_flops=self.spec["horizon_flops"], checkpoint_fracs=list(fracs),
                    eval_tokens=self.spec["eval_tokens"])

    def _seeds(self, phase):
        return dict(self.spec["smoke_seeds"] if phase == "smoke" else self.spec["long_seeds"])

    def _reusable_smoke(self, member):
        """(experiment, reference) of a finished smoke run with identical settings in the source campaign."""
        if member["provenance"] != "agent":
            return None
        from .evaluator.version import evaluator_version

        wanted = dict(status="ok", budget=self._budget("smoke"), seeds=self._seeds("smoke"),
                      evaluator_version=evaluator_version(), task_hash=self.task.task_hash)
        with self._campaign(member["source"]) as camp:
            for e in camp.archive.list_experiments(member["source_candidate_id"]):
                if all(e.get(k) == v for k, v in wanted.items()):
                    return e, f"{member['source']}/{e['id']}"
        return None

    def run_phase_runs(self, phase, log=print):
        from .executor import build_request, get_executor

        if phase == "long" and not self.phase_done("screen"):
            raise CohortError("screening decisions must be frozen before any long run (run `screen` first)")
        members = {m["member_id"]: m for m in self.members() if self.run_of(m["member_id"], phase)[0] is None}
        with tempfile.TemporaryDirectory(prefix="cspace_bundle_") as tmp:
            jobs = []
            for mid, m in members.items():
                reused = self._reusable_smoke(m) if phase == "smoke" else None
                if reused:
                    exp, ref = reused
                    self._record_run(m, phase, {k: v for k, v in exp.items()
                                                if k not in ("id", "candidate_id", "replay_of", "purpose")}, ref)
                    log(f"  {mid}: reused smoke run {ref}")
                    continue
                cand = self.archive.get_candidate(m["candidate_id"])
                bundle = self.archive.materialize(cand["bundle_hash"], Path(tmp) / mid)
                jobs.append((mid, build_request(bundle, self.spec["task"], self.spec, self._budget(phase),
                                                self._seeds(phase))))
                log(f"  {mid} ({m['provenance']}): {phase} run queued")

            def record(mid, result):
                rec = self._record_run(members[mid], phase, result)
                last = (rec.get("checkpoints") or [{}])[-1]
                log(f"  {mid}: {rec['status']}" + (f", dev loss {last['dev_loss']:.4f}" if "dev_loss" in last else "")
                    + (f" ({first_line(rec.get('error'))})" if rec["status"] != "ok" else ""))

            get_executor(self.spec["executor"]).run_all(jobs, record)
        self._set(f"done_{phase}", time.time())

    def _record_run(self, member, phase, result, reused_from=None):
        rec = self.archive.add_experiment(result, member["candidate_id"],
                                          purpose=phase + (f" (reused {reused_from})" if reused_from else ""))
        with self.db:
            self.db.execute("INSERT INTO runs VALUES (?, ?, ?, ?)", (member["member_id"], phase, rec["id"], reused_from))
        return rec

    # ---- screen ----------------------------------------------------------------
    def screen(self, log=print):
        if self.phase_done("screen"):
            return
        if not self.phase_done("smoke"):
            raise CohortError("run the smoke phase first")
        base, _ = self.run_of("baseline", "smoke")
        if base is None or base["status"] != "ok":
            raise CohortError("the baseline smoke run is missing or failed; cannot screen")
        decisions = []
        for m in self.members():
            if m["provenance"] == "baseline":
                continue
            exp, deltas = self._deltas(m["member_id"], "smoke")
            decision, reason = screen_decision([d for _, d in deltas], exp["status"], self.spec["screening"])
            decisions.append({"member_id": m["member_id"], "decision": decision, "reason": reason,
                              "smoke_deltas": deltas})
            with self.db:
                self.db.execute("INSERT OR REPLACE INTO decisions VALUES (?, ?, ?, ?)",
                                (m["member_id"], decision, reason, json.dumps(deltas)))
        record = {"rule": self.spec["screening"], "spec_hash": self.spec_hash,
                  "enrollment_hash": self._get("enrollment_hash"), "decisions": decisions,
                  "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        (self.dir / "screening.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
        self._set("screening_hash", spec_hash(record))
        self._set("done_screen", time.time())
        log("screening frozen: " + str({k: sum(d["decision"] == k for d in decisions)
                                        for k in ("advance", "hold", "reject")}))

    # ---- report ----------------------------------------------------------------
    def _member_row(self, m):
        s_exp, s_d = self._deltas(m["member_id"], "smoke")
        l_exp, l_d = self._deltas(m["member_id"], "long")
        decision = self.decision_of(m["member_id"])
        smoke_b2 = s_d[-1][1] if s_d else None
        final = l_d[-1][1] if l_d and l_d[-1][0] == 1.0 else None
        crossing = None
        if smoke_b2 is not None and final is not None:
            crossing = "late_bloomer" if smoke_b2 > 0 > final else "early_fader" if smoke_b2 < 0 < final else None
        return {
            **{k: m[k] for k in ("member_id", "provenance", "stratum", "change_type", "source",
                                 "source_candidate_id", "candidate_id")},
            "smoke_status": s_exp["status"] if s_exp else None,
            "smoke_delta_b1": s_d[0][1] if s_d else None, "smoke_delta_b2": smoke_b2,
            "decision": decision["decision"], "decision_reason": decision["reason"],
            "long_status": l_exp["status"] if l_exp else None,
            "long_deltas": {f"{f:g}": round(d, 4) for f, d in l_d},
            "long_delta_final": final,
            "usefulness": usefulness_label(final, l_exp["status"] if l_exp else "missing", self.spec["usefulness"]),
            "crossing": crossing,
            "smoke_train_flops": ((s_exp or {}).get("costs") or {}).get("train_flops"),
            "long_train_flops": ((l_exp or {}).get("costs") or {}).get("train_flops"),
            "long_probes": (l_exp or {}).get("probes"),
        }

    def report(self, log=print) -> dict:
        if not self.phase_done("long"):
            raise CohortError("run the long phase first")
        rows = [self._member_row(m) for m in self.members() if m["provenance"] != "baseline"]
        agents = [r for r in rows if r["provenance"] == "agent"]
        base_s, _ = self.run_of("baseline", "smoke")
        base_l, _ = self.run_of("baseline", "long")
        report = {
            "cohort": self.name, "spec_hash": self.spec_hash,
            "enrollment_hash": self._get("enrollment_hash"), "screening_hash": self._get("screening_hash"),
            "baseline": {"smoke": [(c["frac"], c["dev_loss"]) for c in base_s["checkpoints"]],
                         "long": [(c["frac"], c["dev_loss"]) for c in base_l["checkpoints"]],
                         "long_probes": base_l.get("probes")},
            "agents": cohort_stats(agents),
            "agents_by_stratum": {s: cohort_stats([r for r in agents if r["stratum"] == s])
                                  for s in ("reformulation", "ordinary")},
            "controls": cohort_stats([r for r in rows if r["provenance"] == "control"]),
            "rows": rows,
            "rules": {"screening": self.spec["screening"], "usefulness": self.spec["usefulness"]},
            "caveats": [
                "Descriptive cohort statistics, not population estimates: n is small, and candidates share "
                "ancestry and controls, so they are not independent samples.",
                "Usefulness uses one long-run seed against the baseline at the same seeds (a point estimate); "
                "repeat promising or ambiguous members with more seeds (`confirm`) before relying on them.",
                "Controls are reported separately from agent-generated candidates.",
            ],
        }
        (self.dir / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
        self._write_csv(rows)
        (self.dir / "report.md").write_text(self._markdown(report), encoding="utf-8")
        plot = self._plot(rows)
        self._set("done_report", time.time())
        log(f"wrote {self.dir / 'report.md'}" + (f" and {plot}" if plot else ""))
        return report

    def _write_csv(self, rows):
        cols = ["member_id", "provenance", "stratum", "change_type", "smoke_status", "smoke_delta_b1",
                "smoke_delta_b2", "decision", "long_status", "long_delta_final", "usefulness", "crossing",
                "source", "source_candidate_id", "candidate_id"]
        with open(self.dir / "members.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    def _markdown(self, rep):
        f = lambda v: "-" if v is None else f"{v:+.4f}"
        r3 = lambda v: "-" if v is None else f"{v:.3f}"
        b1, b2 = self.spec["smoke_fracs"]

        def table(rows):
            out = ["| member | stratum | smoke d@b1 | smoke d@b2 | decision | long d@1.0 | usefulness | crossing |",
                   "| --- | --- | --- | --- | --- | --- | --- | --- |"]
            out += [f"| {r['member_id']} | {r['stratum'] or r['provenance']} | {f(r['smoke_delta_b1'])} | "
                    f"{f(r['smoke_delta_b2'])} | {r['decision']} | {f(r['long_delta_final'])} | {r['usefulness']} | "
                    f"{r['crossing'] or '-'} |" for r in rows]
            return out

        def fraction(x):
            return f"{x['numerator']}/{x['denominator']}" + (f" = {x['value']:.2f}" if x["value"] is not None
                                                             else " (undefined)")

        def stats(s):
            return [f"- n = {s['n']}; decisions {s['decisions']}; usefulness {s['usefulness']}; "
                    f"crossings {s['crossings']}",
                    f"- Spearman(smoke delta at b2, long delta at 1.0) = {r3(s['spearman_smoke_b2_vs_long_final'])}; "
                    f"at b1 = {r3(s['spearman_smoke_b1_vs_long_final'])}",
                    f"- missed-useful fraction = {fraction(s['missed_useful_fraction'])}",
                    f"- useful-among-rejects fraction = {fraction(s['useful_among_rejects_fraction'])}"]

        lines = [f"# Calibration cohort `{self.name}`", "",
                 f"Smoke checkpoints at {b1:g} and {b2:g} of a {self.spec['horizon_flops']:.3g}-FLOP horizon "
                 f"(seeds {self.spec['smoke_seeds']}); long runs to 1.0 with fresh seeds {self.spec['long_seeds']}. "
                 "Deltas are candidate minus baseline dev loss at the same checkpoint and seeds (negative is better).",
                 "", f"Spec {rep['spec_hash'][:12]}, enrollment {rep['enrollment_hash'][:12]}, screening "
                 f"{rep['screening_hash'][:12]} (frozen before any long run).", "",
                 "## Agent candidates", ""] + stats(rep["agents"]) + [""]
        for stratum, s in rep["agents_by_stratum"].items():
            lines += [f"**{stratum}**", ""] + stats(s) + [""]
        lines += table([r for r in rep["rows"] if r["provenance"] == "agent"])
        lines += ["", "## Known controls (reported separately)", ""] + stats(rep["controls"]) + [""]
        lines += table([r for r in rep["rows"] if r["provenance"] == "control"])
        lines += ["", "## Rules (frozen in the spec)", "",
                  f"- screening: {json.dumps(rep['rules']['screening'])}",
                  f"- usefulness: {json.dumps(rep['rules']['usefulness'])}", "", "## Caveats", ""]
        lines += [f"- {c}" for c in rep["caveats"]]
        return "\n".join(lines) + "\n"

    def _plot(self, rows):
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return None
        pts = [r for r in rows if r["smoke_delta_b2"] is not None and r["long_delta_final"] is not None]
        if not pts:
            return None
        groups = [("agent: reformulation", "o", "#c2410c", lambda r: r["stratum"] == "reformulation"),
                  ("agent: ordinary", "o", "#2563eb", lambda r: r["provenance"] == "agent"
                   and r["stratum"] != "reformulation"),
                  ("known control", "s", "#475569", lambda r: r["provenance"] == "control")]
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        for label, marker, color, member_of in groups:
            sub = [r for r in pts if member_of(r)]
            if not sub:
                continue
            ax.scatter([r["smoke_delta_b2"] for r in sub], [r["long_delta_final"] for r in sub],
                       marker=marker, color=color, label=label, zorder=3)
            for r in sub:
                ax.annotate(r["member_id"], (r["smoke_delta_b2"], r["long_delta_final"]), fontsize=7,
                            xytext=(3, 3), textcoords="offset points")
        ax.axhline(0, color="#999", lw=0.8)
        ax.axvline(0, color="#999", lw=0.8)
        ax.axhline(-self.spec["usefulness"]["delta"], color="#16a34a", lw=0.8, ls="--", label="usefulness threshold")
        ax.set_xlabel(f"smoke delta at b2 = {self.spec['smoke_fracs'][1]:g} (nats)")
        ax.set_ylabel("long-run delta at 1.0 (nats)")
        ax.set_title("Smoke test vs long run (negative = better than baseline)", fontsize=9)
        ax.legend(fontsize=7)
        fig.tight_layout()
        path = self.dir / "smoke_vs_long.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    # ---- driver ----------------------------------------------------------------
    def run(self, until="report", log=print):
        if until not in PHASES:
            raise ValueError(f"until must be one of {PHASES}")
        for phase in PHASES[:PHASES.index(until) + 1]:
            if phase == "enroll":
                self.enroll(log=log)
            elif phase == "screen":
                self.screen(log=log)
            elif phase == "report":
                self.report(log=log)
            elif not self.phase_done(phase):
                log(f"== {phase} phase")
                self.run_phase_runs(phase, log=log)

    def status(self) -> str:
        lines = [f"cohort {self.name} (spec {self.spec_hash[:12]})",
                 "phases: " + ", ".join(f"{p}={'done' if self.phase_done(p) else '-'}" for p in PHASES)]
        for m in self.members():
            smoke, _ = self.run_of(m["member_id"], "smoke")
            long_, _ = self.run_of(m["member_id"], "long")
            decision = self.decision_of(m["member_id"])
            lines.append(f"  {m['member_id']:<22} {m['provenance']:<9} {m['stratum'] or '-':<14} "
                         f"smoke={smoke['status'] if smoke else '-':<8} "
                         f"decision={decision['decision'] if decision else '-':<8} "
                         f"long={long_['status'] if long_ else '-'}")
        return "\n".join(lines)


def _control_path(control):
    p = Path(control)
    if not p.is_dir():
        p = CONTROLS_DIR / control
    if not (p / "model.py").is_file():
        raise ValueError(f"control {control!r} is not a bundle directory (no model.py)")
    return p.resolve()
