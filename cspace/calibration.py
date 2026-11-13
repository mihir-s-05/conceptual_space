"""Stage A calibration cohort: do cheap smoke tests predict longer-run outcomes?

Procedure (each phase is idempotent and must follow the previous one):

  enroll  Pick up to `cap` agent candidates from source campaigns by a fixed,
          outcome-blind rule: creation order, engineering-valid only, stratified
          so about `reform_quota` of slots go to reformulations (classified before
          results by the agent's submitted change_type, or by researcher overrides).
          Only `checks.passed` is read from source experiments, never a loss.
          Known controls and the baseline are added and labelled separately.
  smoke   Every member and the baseline get the smoke test: checkpoints at
          `smoke_fracs` of the horizon, with the smoke seeds.
  screen  A screening rule frozen in the spec turns smoke deltas into
          advance / hold / reject. Decisions are hashed and frozen before any
          long run exists.
  long    Every member and the baseline get a longer run with fresh seeds,
          regardless of its screening decision (no selection on the short score).
  report  Paired smoke-vs-long results, rank correlations, the screening-error
          table with explicit denominators, learning-curve crossings, and controls.

A cohort lives in archive/cohorts/<name>/ with its own content-addressed archive,
separate from every campaign, so agents never see long-run outcomes.
"""

import csv
import json
import math
import tempfile
import time
from pathlib import Path

from .archive import Archive
from .hashing import canonical_json, sha256_bytes
from .paths import ARCHIVE_DIR, SEEDS_DIR

COHORTS_DIR = ARCHIVE_DIR / "cohorts"
CONTROLS_DIR = SEEDS_DIR / "controls"
PHASES = ("enroll", "smoke", "screen", "long", "report")
AGENT_MODES = ("challenge", "continuation", "reconstruction")

DEFAULT_SPEC = dict(
    sources=[],                       # campaign names (under archive/campaigns) supplying agent candidates
    campaigns_root=None,              # optional override of the campaigns directory
    baseline=str(SEEDS_DIR / "baseline_decoder"),
    controls=["recurrent_depth", "depth_weighted", "wide_shallow"],  # names under seeds/controls, or paths
    cap=10,                           # agent candidates enrolled
    reform_quota=0.5,                 # share of agent slots reserved for reformulations
    stratum_overrides={},             # source candidate id -> "reformulation" | "ordinary" (set before results)
    task=None,
    horizon_flops=5.5e15,             # the planned full run B; schedules always span it
    smoke_fracs=[0.05, 0.1],          # b1, b2
    long_fracs=[0.05, 0.1, 0.25, 0.5, 1.0],
    eval_tokens=1 << 19,
    smoke_seeds={"init": 0, "data": 0},
    long_seeds={"init": 1, "data": 1},  # fresh seeds: the long run is an independent run, not a continuation
    screening=dict(advance_max_delta=0.0, reject_min_delta=0.05, reject_requires_no_closing=True),
    usefulness=dict(delta=0.02, uncertain_band=0.01),
    device="auto",
    throttle=0.0,
    temp_limit_c=None,
    eval_time_limit_s=7200.0,
    max_params=60_000_000,
    executor={"kind": "local", "slots": 1},  # cspace.executor spec; slots > 1 runs members concurrently
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
# statistics
def _ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(x, y):
    """Spearman rank correlation with average ranks for ties; None if undefined."""
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None
             and math.isfinite(a) and math.isfinite(b)]
    if len(pairs) < 3:
        return None
    rx, ry = _ranks([p[0] for p in pairs]), _ranks([p[1] for p in pairs])
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx, syy = sum((a - mx) ** 2 for a in rx), sum((b - my) ** 2 for b in ry)
    if sxx == 0 or syy == 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def screen_decision(deltas, status, rule):
    """advance / hold / reject from smoke deltas (candidate minus baseline) at b1, b2."""
    if status != "ok" or not deltas:
        return "reject", f"smoke run did not complete (status {status})"
    d1, d2 = deltas[0], deltas[-1]
    if d2 <= rule["advance_max_delta"]:
        return "advance", f"delta at b2 {d2:+.4f} <= {rule['advance_max_delta']:+.4f}"
    closing = d2 - d1 < 0
    if d2 >= rule["reject_min_delta"] and not (rule.get("reject_requires_no_closing", True) and closing):
        return "reject", f"delta at b2 {d2:+.4f} >= {rule['reject_min_delta']:+.4f}" + (
            ", gap not closing" if rule.get("reject_requires_no_closing", True) else "")
    return "hold", f"delta at b2 {d2:+.4f}" + (" (gap closing from b1)" if closing else "")


def usefulness_label(delta, status, rule):
    if status != "ok" or delta is None:
        return "failed"
    if delta <= -rule["delta"]:
        return "useful"
    if delta >= -rule["delta"] + rule["uncertain_band"]:
        return "not_useful"
    return "uncertain"


# ---------------------------------------------------------------------------
class Cohort:
    def __init__(self, name, root=None):
        self.name = name
        self.dir = Path(root or COHORTS_DIR) / name
        spec_path = self.dir / "cohort.json"
        if not spec_path.is_file():
            raise FileNotFoundError(f"cohort {name!r} not found in {self.dir.parent}; run `cohort init` first")
        stored = json.loads(spec_path.read_text(encoding="utf-8"))
        self.spec = stored["spec"]
        if sha256_bytes(canonical_json(self.spec)) != stored["spec_hash"]:
            raise CohortError("cohort.json was edited after creation; the spec is frozen. Create a new cohort.")
        self.spec_hash = stored["spec_hash"]
        self.archive = Archive(self.dir / "archive")
        self.db = self.archive.db
        self.db.executescript(_SQL)
        from .task import load_task

        self.task = load_task(self.spec["task"])

    @classmethod
    def create(cls, name, root=None, **overrides):
        spec = json.loads(json.dumps(DEFAULT_SPEC))
        for k, v in overrides.items():
            if v is None:
                continue
            if k not in spec:
                raise ValueError(f"unknown cohort setting {k!r}")
            spec[k] = v
        if spec["task"] is None:
            from .paths import DATA_DIR

            tasks = sorted(p.parent for p in DATA_DIR.glob("*/task.json"))
            if len(tasks) != 1:
                raise ValueError(f"specify task (found {len(tasks)} under {DATA_DIR})")
            spec["task"] = str(tasks[0])
        if not spec["sources"] and not spec["controls"]:
            raise ValueError("a cohort needs at least one source campaign or control")
        fr = spec["smoke_fracs"]
        if len(fr) != 2 or not 0 < fr[0] < fr[1] <= 1:
            raise ValueError("smoke_fracs must be two increasing fractions (b1, b2)")
        if max(spec["long_fracs"]) != 1.0:
            raise ValueError("long_fracs must end at 1.0 (the full horizon)")
        spec["controls"] = [str(cls._control_path(c)) for c in spec["controls"]]
        spec["created"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        d = Path(root or COHORTS_DIR) / name
        if (d / "cohort.json").exists():
            raise FileExistsError(f"cohort {name!r} already exists")
        d.mkdir(parents=True, exist_ok=True)
        (d / "cohort.json").write_text(json.dumps({"spec": spec, "spec_hash": sha256_bytes(canonical_json(spec))},
                                                  indent=2), encoding="utf-8")
        return cls(name, root)

    @staticmethod
    def _control_path(c):
        p = Path(c)
        if not p.is_dir():
            p = CONTROLS_DIR / c
        if not (p / "model.py").is_file():
            raise ValueError(f"control {c!r} is not a bundle directory (no model.py)")
        return p.resolve()

    def close(self):
        self.archive.close()

    def _get(self, key, default=None):
        row = self.db.execute("SELECT value FROM cohort_meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    def _set(self, key, value):
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO cohort_meta VALUES (?, ?)", (key, str(value)))

    def members(self, provenance=None):
        q = "SELECT * FROM members" + (" WHERE provenance = ?" if provenance else "") + " ORDER BY enroll_order"
        return [dict(r) for r in self.db.execute(q, (provenance,) if provenance else ()).fetchall()]

    def run_of(self, member_id, phase):
        row = self.db.execute("SELECT * FROM runs WHERE member_id = ? AND phase = ?", (member_id, phase)).fetchone()
        return (self.archive.get_experiment(row["experiment_id"]), row["reused_from"]) if row else (None, None)

    def phase_done(self, phase):
        return self._get(f"done_{phase}") is not None

    # ---- enroll ------------------------------------------------------------
    def _eligible(self, source):
        """Agent candidates of one source campaign in creation order, with strata.
        Reads candidate provenance, change_type, and checks.passed only (outcome-blind)."""
        from .agents.campaign import Campaign

        camp = Campaign(source, root=self.spec.get("campaigns_root"))
        try:
            rows = camp.db.execute(
                "SELECT c.id, c.created, c.action_mode, cm.change_type, cm.smoke_experiment_id "
                "FROM candidates c JOIN candidate_meta cm ON cm.candidate_id = c.id "
                # creation order = (wave, job id, submission time): stable even when a wave's jobs ran in parallel
                "ORDER BY cm.creation_wave, cm.job_id, c.created, c.id").fetchall()
            out, skipped = [], []
            for r in rows:
                if r["action_mode"] not in AGENT_MODES:
                    continue
                exp = camp.archive.get_experiment(r["smoke_experiment_id"]) if r["smoke_experiment_id"] else None
                passed = bool(exp and (exp.get("checks") or {}).get("passed"))
                if not passed:
                    skipped.append(r["id"])
                    continue
                override = self.spec["stratum_overrides"].get(r["id"])
                stratum = override or ("reformulation" if r["change_type"] == "formulation_change" else "ordinary")
                out.append(dict(source=source, source_candidate_id=r["id"], change_type=r["change_type"],
                                stratum=stratum, bundle_hash=camp.archive.get_candidate(r["id"])["bundle_hash"],
                                camp=camp))
            return out, skipped, camp
        except Exception:
            camp.close()
            raise

    def enroll(self, log=print):
        if self.phase_done("enroll"):
            return self.members()
        pools, skipped, opened = [], [], []
        for source in self.spec["sources"]:
            eligible, sk, camp = self._eligible(source)
            pools += eligible
            skipped += sk
            opened.append(camp)
        try:
            seen, unique = set(), []
            for c in pools:  # exact duplicates across sources count once
                if c["bundle_hash"] not in seen:
                    seen.add(c["bundle_hash"])
                    unique.append(c)
            cap = int(self.spec["cap"])
            n_ref = round(cap * float(self.spec["reform_quota"]))
            ref = [c for c in unique if c["stratum"] == "reformulation"]
            ordn = [c for c in unique if c["stratum"] != "reformulation"]
            chosen = ref[:n_ref] + ordn[:cap - n_ref]
            shortfall = cap - len(chosen)
            if shortfall > 0:  # fill from whichever stratum has candidates left, still in creation order
                rest = [c for c in unique if c not in chosen]
                chosen += rest[:shortfall]
            chosen.sort(key=lambda c: unique.index(c))
            order = 0
            with tempfile.TemporaryDirectory(prefix="cspace_cohort_") as tmp:
                for i, c in enumerate(chosen, 1):
                    bundle = c["camp"].archive.materialize(c["bundle_hash"], Path(tmp) / c["source_candidate_id"])
                    rec, _ = self.archive.add_candidate(bundle, action_mode="manual",
                                                        notes=f"from campaign {c['source']} ({c['source_candidate_id']})")
                    order += 1
                    self._add_member(f"m{i:02d}", rec["id"], "agent", c["stratum"], c["source"],
                                     c["source_candidate_id"], c["change_type"], order)
            for path in self.spec["controls"]:
                rec, _ = self.archive.add_candidate(path, action_mode="control", notes=f"known control {Path(path).name}")
                order += 1
                self._add_member(f"k_{Path(path).name}", rec["id"], "control", None, "controls", None, None, order)
            rec, _ = self.archive.add_candidate(self.spec["baseline"], action_mode="seed", notes="baseline")
            self._add_member("baseline", rec["id"], "baseline", None, "seed", None, None, 0)
        finally:
            for camp in opened:
                camp.close()
        record = {
            "rule": "creation order (wave, job, submission); engineering-valid (checks.passed) only; stratified by pre-result change_type "
                    "or researcher override; exact duplicates once; losses never read",
            "cap": cap, "reformulation_slots": n_ref,
            "available": {"reformulation": len(ref), "ordinary": len(ordn)},
            "shortfall_filled_from_other_stratum": max(0, len(chosen) - min(len(ref), n_ref) - min(len(ordn), cap - n_ref)),
            "skipped_engineering_invalid": skipped,
            "members": self.members(),
        }
        (self.dir / "enrollment.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
        self._set("enrollment_hash", sha256_bytes(canonical_json(record)))
        self._set("done_enroll", time.time())
        agents = self.members("agent")
        log(f"enrolled {len(agents)} agent candidates "
            f"({sum(m['stratum'] == 'reformulation' for m in agents)} reformulation), "
            f"{len(self.members('control'))} controls, plus the baseline; {len(skipped)} engineering-invalid skipped")
        if len(agents) < cap:
            log(f"  only {len(agents)} eligible candidates were available for {cap} slots")
        return self.members()

    def _add_member(self, mid, cid, provenance, stratum, source, source_cid, change_type, order):
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO members VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                            (mid, cid, provenance, stratum, source, source_cid, change_type, order))

    # ---- running -----------------------------------------------------------
    def _budget(self, phase):
        fracs = self.spec["smoke_fracs"] if phase == "smoke" else self.spec["long_fracs"]
        return dict(horizon_flops=self.spec["horizon_flops"], checkpoint_fracs=list(fracs),
                    eval_tokens=self.spec["eval_tokens"])

    def _seeds(self, phase):
        return dict(self.spec["smoke_seeds"] if phase == "smoke" else self.spec["long_seeds"])

    def _reusable_smoke(self, member):
        """A finished smoke experiment in the source campaign with identical settings, if any."""
        if member["provenance"] != "agent":
            return None
        from .agents.campaign import Campaign
        from .evaluator.version import evaluator_version

        camp = Campaign(member["source"], root=self.spec.get("campaigns_root"))
        try:
            budget, seeds, ev = self._budget("smoke"), self._seeds("smoke"), evaluator_version()
            for e in camp.archive.list_experiments(member["source_candidate_id"]):
                if (e["status"] == "ok" and e.get("budget") == budget and e.get("seeds") == seeds
                        and e.get("evaluator_version") == ev and e.get("task_hash") == self.task.task_hash):
                    return e, f"{member['source']}/{e['id']}"
        finally:
            camp.close()
        return None

    def _request(self, member, phase, tmp):
        """Evaluation request for a member (materializes its bundle; call on the owning thread)."""
        cand = self.archive.get_candidate(member["candidate_id"])
        bundle = self.archive.materialize(cand["bundle_hash"], Path(tmp) / f"{member['member_id']}_{phase}")
        return dict(bundle_dir=str(bundle), task_dir=self.spec["task"], split="dev",
                    budget=self._budget(phase), seeds=self._seeds(phase), device=self.spec["device"],
                    time_limit_s=self.spec["eval_time_limit_s"], max_params=int(self.spec["max_params"]),
                    throttle=self.spec["throttle"], temp_limit_c=self.spec["temp_limit_c"])

    def run_phase_runs(self, phase, log=print):
        assert phase in ("smoke", "long")
        if phase == "long" and not self.phase_done("screen"):
            raise CohortError("screening decisions must be frozen before any long run (run `screen` first)")
        members = sorted(self.members(), key=lambda m: m["enroll_order"])  # baseline (order 0) first
        pending = []
        for m in members:
            exp, _ = self.run_of(m["member_id"], phase)
            if exp is not None:
                continue
            reused = self._reusable_smoke(m) if phase == "smoke" else None
            if reused:
                src_exp, ref = reused
                result = {k: v for k, v in src_exp.items() if k not in ("id", "candidate_id", "replay_of", "purpose")}
                self._record_run(m, phase, result, ref, log)
                log(f"  {m['member_id']}: reused smoke run {ref}")
            else:
                pending.append(m)
        # fresh runs go through the executor concurrently (bounded by its slots); results are
        # recorded on this thread as they finish, so an interruption keeps every finished run
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from .executor import get_executor

        ex = get_executor(self.spec.get("executor"))
        timeout = self.spec["eval_time_limit_s"] + 600
        with tempfile.TemporaryDirectory(prefix="cspace_bundle_") as tmp,                 ThreadPoolExecutor(max_workers=max(1, min(ex.slots, len(pending) or 1))) as pool:
            futures = {}
            for m in pending:  # archive access stays on this thread; workers only run the executor
                log(f"  {m['member_id']} ({m['provenance']}): {phase} run queued")
                futures[pool.submit(ex.run, self._request(m, phase, tmp), timeout)] = m
            for f in as_completed(futures):
                self._record_run(futures[f], phase, f.result(), None, log)
        self._set(f"done_{phase}", time.time())

    def _record_run(self, m, phase, result, ref, log):
        record = self.archive.add_experiment(result, m["candidate_id"], purpose=phase + (f" (reused {ref})" if ref else ""))
        if not ref:
            last = (record.get("checkpoints") or [{}])[-1]
            log(f"  {m['member_id']}: {record['status']}" + (f", dev loss {last['dev_loss']:.4f}" if "dev_loss" in last else "")
                + (f" ({(record.get('error') or '').splitlines()[0][:120]})" if record["status"] != "ok" else ""))
        with self.db:
            self.db.execute("INSERT INTO runs VALUES (?, ?, ?, ?)", (m["member_id"], phase, record["id"], ref))

    def _deltas(self, member_id, phase):
        """Per-checkpoint (frac, candidate loss - baseline loss) for a member's run."""
        exp, _ = self.run_of(member_id, phase)
        base, _ = self.run_of("baseline", phase)
        if exp is None or base is None:
            return exp, []
        b = {round(c["frac"], 6): c["dev_loss"] for c in base["checkpoints"]}
        return exp, [(c["frac"], c["dev_loss"] - b[round(c["frac"], 6)]) for c in exp["checkpoints"]
                     if round(c["frac"], 6) in b]

    # ---- screen ------------------------------------------------------------
    def screen(self, log=print):
        if self.phase_done("screen"):
            return
        if not self.phase_done("smoke"):
            raise CohortError("run the smoke phase first")
        base, _ = self.run_of("baseline", "smoke")
        if base is None or base["status"] != "ok":
            raise CohortError("the baseline smoke run is missing or failed; cannot screen")
        rule = self.spec["screening"]
        decisions = []
        for m in self.members():
            if m["provenance"] == "baseline":
                continue
            exp, deltas = self._deltas(m["member_id"], "smoke")
            decision, reason = screen_decision([d for _, d in deltas], exp["status"], rule)
            decisions.append({"member_id": m["member_id"], "decision": decision, "reason": reason,
                              "smoke_deltas": deltas})
            with self.db:
                self.db.execute("INSERT OR REPLACE INTO decisions VALUES (?, ?, ?, ?)",
                                (m["member_id"], decision, reason, json.dumps(deltas)))
        record = {"rule": rule, "spec_hash": self.spec_hash, "enrollment_hash": self._get("enrollment_hash"),
                  "decisions": decisions, "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        (self.dir / "screening.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
        self._set("screening_hash", sha256_bytes(canonical_json(record)))
        self._set("done_screen", time.time())
        counts = {k: sum(d["decision"] == k for d in decisions) for k in ("advance", "hold", "reject")}
        log(f"screening frozen: {counts}")

    # ---- report ------------------------------------------------------------
    def report(self, log=print) -> dict:
        if not self.phase_done("long"):
            raise CohortError("run the long phase first")
        rows = []
        for m in self.members():
            if m["provenance"] == "baseline":
                continue
            s_exp, s_d = self._deltas(m["member_id"], "smoke")
            l_exp, l_d = self._deltas(m["member_id"], "long")
            dec = self.db.execute("SELECT decision, reason FROM decisions WHERE member_id = ?",
                                  (m["member_id"],)).fetchone()
            final = l_d[-1][1] if l_d and l_d[-1][0] == 1.0 else None
            label = usefulness_label(final, l_exp["status"] if l_exp else "missing", self.spec["usefulness"])
            smoke_b2 = s_d[-1][1] if s_d else None
            crossing = None
            if smoke_b2 is not None and final is not None:
                if smoke_b2 > 0 > final:
                    crossing = "late_bloomer"
                elif smoke_b2 < 0 < final:
                    crossing = "early_fader"
            rows.append({
                "member_id": m["member_id"], "provenance": m["provenance"], "stratum": m["stratum"],
                "change_type": m["change_type"], "source": m["source"], "source_candidate_id": m["source_candidate_id"],
                "candidate_id": m["candidate_id"],
                "smoke_status": s_exp["status"] if s_exp else None,
                "smoke_delta_b1": s_d[0][1] if s_d else None, "smoke_delta_b2": smoke_b2,
                "decision": dec["decision"], "decision_reason": dec["reason"],
                "long_status": l_exp["status"] if l_exp else None,
                "long_deltas": {f"{f:g}": round(d, 4) for f, d in l_d},
                "long_delta_final": final, "usefulness": label, "crossing": crossing,
                "smoke_train_flops": (s_exp.get("costs") or {}).get("train_flops") if s_exp else None,
                "long_train_flops": (l_exp.get("costs") or {}).get("train_flops") if l_exp else None,
                "long_probes": (l_exp or {}).get("probes"),
            })

        def stats(sub):
            useful = [r for r in sub if r["usefulness"] == "useful"]
            rejected = [r for r in sub if r["decision"] == "reject"]
            missed = [r for r in useful if r["decision"] == "reject"]

            def frac(a, b):
                return {"numerator": len(a), "denominator": len(b),
                        "value": (len(a) / len(b)) if b else None}

            return {
                "n": len(sub),
                "spearman_smoke_b2_vs_long_final": spearman([r["smoke_delta_b2"] for r in sub],
                                                            [r["long_delta_final"] for r in sub]),
                "spearman_smoke_b1_vs_long_final": spearman([r["smoke_delta_b1"] for r in sub],
                                                            [r["long_delta_final"] for r in sub]),
                "missed_useful_fraction": frac(missed, useful),
                "useful_among_rejects_fraction": frac(missed, rejected),
                "decisions": {k: sum(r["decision"] == k for r in sub) for k in ("advance", "hold", "reject")},
                "usefulness": {k: sum(r["usefulness"] == k for r in sub)
                               for k in ("useful", "uncertain", "not_useful", "failed")},
                "crossings": {k: sum(r["crossing"] == k for r in sub) for k in ("late_bloomer", "early_fader")},
            }

        agents = [r for r in rows if r["provenance"] == "agent"]
        controls = [r for r in rows if r["provenance"] == "control"]
        base_s, _ = self.run_of("baseline", "smoke")
        base_l, _ = self.run_of("baseline", "long")
        report = {
            "cohort": self.name, "spec_hash": self.spec_hash,
            "enrollment_hash": self._get("enrollment_hash"), "screening_hash": self._get("screening_hash"),
            "baseline": {"smoke": [(c["frac"], c["dev_loss"]) for c in base_s["checkpoints"]],
                         "long": [(c["frac"], c["dev_loss"]) for c in base_l["checkpoints"]],
                         "long_probes": base_l.get("probes")},
            "agents": stats(agents),
            "agents_by_stratum": {s: stats([r for r in agents if r["stratum"] == s])
                                  for s in ("reformulation", "ordinary")},
            "controls": stats(controls),
            "rows": rows,
            "rules": {"screening": self.spec["screening"], "usefulness": self.spec["usefulness"]},
            "caveats": [
                "Descriptive cohort statistics, not population estimates: n is small, candidates share ancestry "
                "and controls, so candidates are not independent samples.",
                "Usefulness uses a single long-run seed against the baseline at the same seeds (point estimate); "
                "repeat promising or ambiguous members with extra seeds before relying on them.",
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

    @staticmethod
    def _fmt(v, signed=True):
        if v is None:
            return "-"
        return f"{v:+.4f}" if signed else f"{v:.3f}"

    def _markdown(self, rep):
        f = self._fmt
        b1, b2 = self.spec["smoke_fracs"]
        lines = [f"# Calibration cohort `{self.name}`", "",
                 f"Smoke checkpoints at {b1:g} and {b2:g} of a {self.spec['horizon_flops']:.3g}-FLOP horizon "
                 f"(seeds {self.spec['smoke_seeds']}); long runs to 1.0 with fresh seeds {self.spec['long_seeds']}. "
                 "Deltas are candidate minus baseline dev loss at the same checkpoint and seeds (negative is better).",
                 "", f"Spec {rep['spec_hash'][:12]}, enrollment {rep['enrollment_hash'][:12]}, screening frozen "
                 f"{rep['screening_hash'][:12]} before any long run.", ""]

        def table(rows):
            out = ["| member | stratum | smoke d@b1 | smoke d@b2 | decision | long d@1.0 | usefulness | crossing |",
                   "| --- | --- | --- | --- | --- | --- | --- | --- |"]
            for r in rows:
                out.append(f"| {r['member_id']} | {r['stratum'] or r['provenance']} | {f(r['smoke_delta_b1'])} | "
                           f"{f(r['smoke_delta_b2'])} | {r['decision']} | {f(r['long_delta_final'])} | "
                           f"{r['usefulness']} | {r['crossing'] or '-'} |")
            return out

        def stat_lines(s):
            mu, ur = s["missed_useful_fraction"], s["useful_among_rejects_fraction"]
            return [f"- n = {s['n']}; decisions {s['decisions']}; usefulness {s['usefulness']}; "
                    f"crossings {s['crossings']}",
                    f"- Spearman(smoke delta at b2, long delta at 1.0) = {f(s['spearman_smoke_b2_vs_long_final'], False)}; "
                    f"at b1 = {f(s['spearman_smoke_b1_vs_long_final'], False)}",
                    f"- missed-useful fraction = {mu['numerator']}/{mu['denominator']}"
                    + (f" = {mu['value']:.2f}" if mu["value"] is not None else " (undefined)"),
                    f"- useful-among-rejects fraction = {ur['numerator']}/{ur['denominator']}"
                    + (f" = {ur['value']:.2f}" if ur["value"] is not None else " (undefined)")]

        agents = [r for r in rep["rows"] if r["provenance"] == "agent"]
        controls = [r for r in rep["rows"] if r["provenance"] == "control"]
        lines += ["## Agent candidates", ""] + stat_lines(rep["agents"]) + [""]
        for s, st in rep["agents_by_stratum"].items():
            lines += [f"**{s}**", ""] + stat_lines(st) + [""]
        lines += table(agents) + ["", "## Known controls (reported separately)", ""] + stat_lines(rep["controls"])
        lines += [""] + table(controls) + ["", "## Rules (frozen in the spec)", "",
                  f"- screening: {json.dumps(rep['rules']['screening'])}",
                  f"- usefulness: {json.dumps(rep['rules']['usefulness'])}", "", "## Caveats", ""]
        lines += [f"- {c}" for c in rep["caveats"]] + [""]
        return "\n".join(lines)

    def _plot(self, rows):
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            return None
        pts = [r for r in rows if r["smoke_delta_b2"] is not None and r["long_delta_final"] is not None]
        if not pts:
            return None
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        styles = {("agent", "reformulation"): ("o", "#c2410c"), ("agent", "ordinary"): ("o", "#2563eb"),
                  ("control", None): ("s", "#475569")}
        for (prov, strat), (marker, color) in styles.items():
            sub = [r for r in pts if r["provenance"] == prov and (strat is None or r["stratum"] == strat)]
            if sub:
                label = f"{prov}: {strat}" if strat else "known control"
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

    # ---- driver ------------------------------------------------------------
    def run(self, until="report", log=print):
        if until not in PHASES:
            raise ValueError(f"until must be one of {PHASES}")
        for phase in PHASES:
            if phase == "enroll":
                self.enroll(log=log)
            elif phase in ("smoke", "long"):
                if not self.phase_done(phase):
                    log(f"== {phase} phase")
                    self.run_phase_runs(phase, log=log)
            elif phase == "screen":
                self.screen(log=log)
            elif phase == "report":
                self.report(log=log)
            if phase == until:
                break

    def status(self) -> str:
        out = [f"cohort {self.name} (spec {self.spec_hash[:12]})",
               "phases: " + ", ".join(f"{p}={'done' if self.phase_done(p) else '-'}" for p in PHASES)]
        for m in self.members():
            s, _ = self.run_of(m["member_id"], "smoke")
            l, _ = self.run_of(m["member_id"], "long")
            dec = self.db.execute("SELECT decision FROM decisions WHERE member_id = ?", (m["member_id"],)).fetchone()
            out.append(f"  {m['member_id']:<22} {m['provenance']:<9} {m['stratum'] or '-':<14} "
                       f"smoke={s['status'] if s else '-':<8} decision={dec['decision'] if dec else '-':<8} "
                       f"long={l['status'] if l else '-'}")
        return "\n".join(out)
