"""Campaign state and the fixed scheduler.

Each campaign has its own archive directory (separate archives per condition),
seeded with the starting decoder and a baseline control run at the smoke budget.

Per wave (four job slots, all selections made from a frozen evidence snapshot):
  slot 0  continuation of the oldest eligible queued follow-up, else a fallback
          challenge of the development incumbent
  slot 1  challenge from the development incumbent
  slot 2  challenge from the valid archived state expanded least recently (ties by id)
  slot 3  reconstruction from a clean harness
Origins: each challenge/reconstruction dispatch gets a new origin with a fixed
continuation allowance; continuations keep their origin and consume allowance.
"""

import json
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..archive import Archive
from ..hashing import canonical_json, sha256_bytes
from ..paths import ARCHIVE_DIR, ROOT, SEEDS_DIR
from ..task import load_task
from .space_ops import render_flat, space_id

CAMPAIGNS_DIR = ARCHIVE_DIR / "campaigns"
DEFAULT_BOX = ROOT / "boxes" / "lm_state_compute_v0.json"

DEFAULT_CONFIG = dict(
    condition="dag",                 # "dag" or "flat"
    model="gpt-5.4-mini",
    reasoning_effort=None,           # e.g. "low" | "medium" | "high" for reasoning models
    box=str(DEFAULT_BOX),
    task=None,                       # task directory; default: the only one under data/
    horizon_flops=5.5e15,            # schedule horizon (about 20 tokens/param for the 6M baseline)
    smoke_fracs=[0.05, 0.1],         # smoke checkpoints (fractions of the horizon)
    eval_tokens=1 << 19,
    seeds={"init": 0, "data": 0},
    max_turns=16,                    # LLM calls per job
    max_repairs=2,                   # implementation repairs per job
    continuation_allowance=2,
    eval_time_limit_s=1800.0,
    max_params=60_000_000,
    device="auto",                   # "auto" | "cuda" | "cpu"
    throttle=0.0,                    # fraction of wall-clock the GPU idles between steps (0 = off)
    temp_limit_c=None,               # pause training when the GPU reaches this temperature
    prices=None,                     # optional USD per 1M tokens: {"input", "cached_input", "output"}
)

_SQL = """
CREATE TABLE IF NOT EXISTS campaign_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS spaces (id TEXT PRIMARY KEY, json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS candidate_meta (
    candidate_id TEXT PRIMARY KEY, job_id TEXT, creation_wave INTEGER NOT NULL,
    last_expanded_wave INTEGER NOT NULL, origin_id TEXT, change_type TEXT,
    space_id TEXT, register_text TEXT, space_id_post TEXT, register_post TEXT,
    smoke_experiment_id TEXT, smoke_status TEXT, smoke_loss REAL);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY, wave INTEGER NOT NULL, slot INTEGER NOT NULL, mode TEXT NOT NULL,
    status TEXT NOT NULL, record TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS origins (id TEXT PRIMARY KEY, allowance INTEGER NOT NULL, created_wave INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS followups (
    id INTEGER PRIMARY KEY AUTOINCREMENT, origin_id TEXT NOT NULL, candidate_id TEXT NOT NULL,
    job_id TEXT NOT NULL, created_wave INTEGER NOT NULL, dispatched_wave INTEGER, followup TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS snapshots (id TEXT PRIMARY KEY, wave INTEGER NOT NULL, json TEXT NOT NULL);
"""


@dataclass
class JobPlan:
    id: str
    wave: int
    slot: int
    mode: str                      # continuation | challenge | reconstruction
    origin_id: str
    research_parent_id: str | None
    evidence_snapshot_id: str
    followup_id: int | None = None
    followup: dict | None = None
    note: str = ""


@dataclass
class JobRecord:
    plan: dict
    status: str = "planned"        # planned | completed | implementation_failure | agent_failure
    candidates: list = field(default_factory=list)   # every submitted candidate id, in order
    final_candidate_id: str | None = None
    ablation_candidate_id: str | None = None
    experiments: dict = field(default_factory=dict)  # candidate id -> experiment id
    proposal: dict | None = None
    change_type: str | None = None
    summary: dict | None = None    # observations shown to the agent
    reflection: dict | None = None
    retrievals: list = field(default_factory=list)   # [{"turn", "tool", "target"}]
    space_report: dict | None = None
    usage: dict = field(default_factory=dict)
    transcript: str | None = None  # archive object hash of the full message history
    error: str | None = None
    started: float | None = None
    finished: float | None = None


class Campaign:
    def __init__(self, name: str, root=None):
        self.name = name
        self.dir = Path(root or CAMPAIGNS_DIR) / name
        if not (self.dir / "campaign.json").is_file():
            raise FileNotFoundError(f"campaign {name!r} not found in {self.dir.parent}; run `campaign init` first")
        self.config = json.loads((self.dir / "campaign.json").read_text(encoding="utf-8"))
        self.archive = Archive(self.dir / "archive")
        self.db = self.archive.db
        self.db.executescript(_SQL)
        self.task = load_task(self.config["task"])

    # ---- creation ----------------------------------------------------------
    @classmethod
    def create(cls, name, root=None, **overrides):
        cfg = {**DEFAULT_CONFIG, **{k: v for k, v in overrides.items() if v is not None}}
        if cfg["condition"] not in ("dag", "flat"):
            raise ValueError("condition must be 'dag' or 'flat'")
        if cfg["task"] is None:
            from ..paths import DATA_DIR

            tasks = sorted(p.parent for p in DATA_DIR.glob("*/task.json"))
            if len(tasks) != 1:
                raise ValueError(f"specify task (found {len(tasks)} under {DATA_DIR})")
            cfg["task"] = str(tasks[0])
        d = Path(root or CAMPAIGNS_DIR) / name
        if (d / "campaign.json").exists():
            raise FileExistsError(f"campaign {name!r} already exists")
        d.mkdir(parents=True, exist_ok=True)
        box = json.loads(Path(cfg["box"]).read_text(encoding="utf-8"))
        from ..box import validate_space

        validate_space(box)
        cfg["box_hash"] = sha256_bytes(canonical_json(box))
        cfg["created"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        (d / "campaign.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        camp = cls(name, root)
        seed, _ = camp.archive.add_candidate(SEEDS_DIR / "baseline_decoder", action_mode="seed",
                                             notes="Starting formulation: Llama-style decoder.")
        sid = camp.put_space(box)
        with camp.db:
            camp.db.execute("INSERT OR IGNORE INTO candidate_meta (candidate_id, job_id, creation_wave, "
                            "last_expanded_wave, origin_id, change_type, space_id, register_text) "
                            "VALUES (?, NULL, 0, 0, NULL, 'seed', ?, ?)",
                            (seed["id"], sid if cfg["condition"] == "dag" else None,
                             render_flat(box) if cfg["condition"] == "flat" else None))
            camp._meta_set("seed_id", seed["id"])
            camp._meta_set("reference_space_id", sid)
            camp._meta_set("next_wave", "1")
        return camp

    def close(self):
        self.archive.close()

    # ---- small helpers -----------------------------------------------------
    def _meta_get(self, key, default=None):
        row = self.db.execute("SELECT value FROM campaign_meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    def _meta_set(self, key, value):
        self.db.execute("INSERT OR REPLACE INTO campaign_meta VALUES (?, ?)", (key, str(value)))

    @property
    def seed_id(self):
        return self._meta_get("seed_id")

    @property
    def budget(self):
        return dict(horizon_flops=self.config["horizon_flops"], checkpoint_fracs=list(self.config["smoke_fracs"]),
                    eval_tokens=self.config["eval_tokens"])

    def put_space(self, space) -> str:
        sid = space_id(space)
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO spaces VALUES (?, ?)", (sid, json.dumps(space)))
        return sid

    def get_space(self, sid):
        if sid is None:
            return None
        row = self.db.execute("SELECT json FROM spaces WHERE id = ?", (sid,)).fetchone()
        return json.loads(row["json"]) if row else None

    def meta(self, cid):
        row = self.db.execute("SELECT * FROM candidate_meta WHERE candidate_id = ?", (cid,)).fetchone()
        return dict(row) if row else None

    def formulation_of(self, cid):
        """(space or None, register text or None) a child of `cid` inherits (post-reflection if revised)."""
        m = self.meta(cid) or {}
        sid = m.get("space_id_post") or m.get("space_id")
        reg = m.get("register_post") or m.get("register_text")
        return self.get_space(sid), reg

    def log_usage(self, usage: dict, job_id: str):
        prices = self.config.get("prices")
        meta = {"model": usage.get("model"), "reasoning_tokens": usage.get("reasoning_tokens", 0)}
        cached = usage.get("cached_tokens", 0)
        self.archive.log_cost("inference", usage["prompt_tokens"] - cached, "input_tokens", job_id, meta)
        self.archive.log_cost("inference", cached, "cached_input_tokens", job_id, meta)
        self.archive.log_cost("inference", usage["completion_tokens"], "output_tokens", job_id, meta)
        if prices:
            usd = ((usage["prompt_tokens"] - cached) * prices["input"] + cached * prices.get("cached_input", prices["input"])
                   + usage["completion_tokens"] * prices["output"]) / 1e6
            self.archive.log_cost("inference", usd, "usd", job_id, meta)

    # ---- evaluation --------------------------------------------------------
    def evaluate(self, candidate_id, purpose, log=print):
        from ..evaluator.runner import run_isolated

        cand = self.archive.get_candidate(candidate_id)
        with tempfile.TemporaryDirectory(prefix="cspace_bundle_") as tmp:
            bundle = self.archive.materialize(cand["bundle_hash"], tmp)
            request = dict(bundle_dir=str(bundle), task_dir=self.config["task"], split="dev", budget=self.budget,
                           seeds=self.config["seeds"], device=self.config.get("device", "auto"),
                           time_limit_s=self.config["eval_time_limit_s"], max_params=int(self.config["max_params"]),
                           throttle=self.config.get("throttle", 0.0), temp_limit_c=self.config.get("temp_limit_c"))
            result = run_isolated(request, timeout_s=self.config["eval_time_limit_s"] + 600, stream_output=False)
        record = self.archive.add_experiment(result, candidate_id, purpose=purpose)
        log(f"    {candidate_id}: {record['status']}"
            + (f", dev loss {record['checkpoints'][-1]['dev_loss']:.4f}" if record.get("checkpoints") else "")
            + (f" ({(record.get('error') or '').splitlines()[0][:120]})" if record["status"] != "ok" else ""))
        return record

    def control(self, log=print):
        """Seed baseline at the campaign's smoke budget and seeds (run once, then cached)."""
        from ..evaluator.version import evaluator_version

        eid = self._meta_get("control_experiment_id")
        if eid:
            cached = self.archive.get_experiment(eid)
            if cached.get("evaluator_version") == evaluator_version():
                return cached
            log(f"baseline control {eid} was measured by evaluator {cached.get('evaluator_version')}, "
                f"current is {evaluator_version()}; re-running it")
        log("running baseline control ...")
        record = self.evaluate(self.seed_id, "control", log=log)
        if record["status"] != "ok":
            raise RuntimeError(f"baseline control failed: {record.get('error')}")
        with self.db:
            self._meta_set("control_experiment_id", record["id"])
            self.db.execute("UPDATE candidate_meta SET smoke_experiment_id = ?, smoke_status = 'ok', smoke_loss = ? "
                            "WHERE candidate_id = ?", (record["id"], record["checkpoints"][-1]["dev_loss"], self.seed_id))
        return record

    def summarize(self, record, ablation=None):
        """Observation summary of a smoke experiment relative to the control."""
        control = self.control()
        base = {round(c["frac"], 6): c["dev_loss"] for c in control["checkpoints"]}

        def one(r):
            out = {"candidate_id": r["candidate_id"], "status": r["status"]}
            if r.get("error"):
                out["error"] = r["error"].splitlines()[0][:300]
            if r.get("params"):
                out["params_total"], out["params_nonembedding"] = r["params"]["total"], r["params"]["nonembedding"]
            costs = r.get("costs") or {}
            if costs.get("flops_history"):
                out["gflops_per_train_step"] = round(costs["flops_history"][0][1] / 1e9, 2)
                out["fwd_mflops_per_token"] = round(costs["fwd_flops_per_token"] / 1e6, 2)
            out["checkpoints"] = [
                {"frac": c["frac"], "step": c["step"], "tokens": c["tokens"], "dev_loss": round(c["dev_loss"], 4),
                 "baseline_dev_loss": round(base.get(round(c["frac"], 6), float("nan")), 4),
                 "delta_vs_baseline": round(c["dev_loss"] - base.get(round(c["frac"], 6), float("nan")), 4)}
                for c in r.get("checkpoints") or []]
            unused = ((r.get("checks") or {}).get("checks") or {}).get("unused_parameters")
            if unused and unused.get("passed") is False:
                out["warning"] = f"{unused['value']} parameter tensors received no gradient"
            return out

        summary = one(record)
        summary["baseline_steps_at_checkpoints"] = [c["step"] for c in control["checkpoints"]]
        if record.get("probes") and "copy" in record["probes"]:
            summary["probes"] = record["probes"]
            summary["baseline_probes"] = control.get("probes")
        if ablation is not None:
            summary["ablation"] = one(ablation)
        return summary

    # ---- snapshot and bulletin ---------------------------------------------
    def jobs(self, wave=None):
        q = "SELECT record FROM jobs" + (" WHERE wave = ?" if wave is not None else "") + " ORDER BY wave, slot"
        rows = self.db.execute(q, (wave,) if wave is not None else ()).fetchall()
        return [json.loads(r["record"]) for r in rows]

    def build_snapshot(self, wave) -> tuple:
        control = self.control()
        entries = [{"candidate_id": self.seed_id, "wave": 0, "mode": "seed", "parent": None, "status": "ok",
                    "observations": self.summarize(control), "change_type": "seed",
                    "interpretation": "Starting formulation (baseline control)."}]
        for job in self.jobs():
            if job["status"] == "planned" or job["plan"]["wave"] >= wave:
                continue
            entries.append({
                "candidate_id": job["final_candidate_id"], "job_id": job["plan"]["id"], "wave": job["plan"]["wave"],
                "mode": job["plan"]["mode"], "parent": job["plan"]["research_parent_id"], "status": job["status"],
                "observations": job["summary"],
                "change_type": job["change_type"],
                "proposal": job["proposal"],
                "interpretation": (job["reflection"] or {}).get("interpretation"),
                "evidence_class": (job["reflection"] or {}).get("evidence_class"),
                "next_question": (job["reflection"] or {}).get("next_question"),
            })
        snap = {"wave": wave, "entries": entries}
        sid = "snap_" + sha256_bytes(canonical_json(snap))[:12]
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO snapshots VALUES (?, ?, ?)", (sid, wave, json.dumps(snap)))
        return sid, snap

    def get_snapshot(self, sid):
        row = self.db.execute("SELECT json FROM snapshots WHERE id = ?", (sid,)).fetchone()
        return json.loads(row["json"])

    @staticmethod
    def render_bulletin(snap, observations_only=False) -> str:
        lines = []
        for e in snap["entries"]:
            obs = e.get("observations") or {}
            cks = obs.get("checkpoints") or []
            nums = ", ".join(f"@{c['frac']:g}: {c['dev_loss']:.4f} (delta {c['delta_vs_baseline']:+.4f})" for c in cks)
            head = f"- {e['candidate_id']} [wave {e['wave']}, {e['mode']}, parent {e['parent']}] status {e['status']}"
            if obs.get("params_total"):
                head += f", {obs['params_total'] / 1e6:.2f}M params, {obs.get('gflops_per_train_step', '?')} GFLOP/step"
            lines.append(head + (f"; dev loss {nums}" if nums else ""))
            if obs.get("error"):
                lines.append(f"    error: {obs['error'][:200]}")
            if obs.get("ablation", {}).get("checkpoints"):
                ab = obs["ablation"]["checkpoints"][-1]
                lines.append(f"    ablation {obs['ablation']['candidate_id']}: final {ab['dev_loss']:.4f} "
                             f"(delta {ab['delta_vs_baseline']:+.4f})")
            if not observations_only:
                if e.get("proposal"):
                    lines.append(f"    change ({e.get('change_type')}): {e['proposal'].get('change', '')[:300]}")
                if e.get("interpretation"):
                    lines.append(f"    interpretation [{e.get('evidence_class')}]: {e['interpretation'][:400]}")
                if e.get("next_question"):
                    lines.append(f"    open question: {e['next_question'][:200]}")
        return "\n".join(lines) if lines else "(no results yet)"

    # ---- scheduler ---------------------------------------------------------
    def valid_states(self):
        rows = self.db.execute("SELECT cm.* FROM candidate_meta cm JOIN candidates c ON c.id = cm.candidate_id "
                               "WHERE cm.smoke_status = 'ok' AND c.action_mode != 'control'").fetchall()
        return [dict(r) for r in rows]

    def incumbent(self):
        states = self.valid_states()
        return min(states, key=lambda s: (s["smoke_loss"], s["candidate_id"]))["candidate_id"]

    def _new_origin(self, wave, slot):
        oid = f"o{wave:03d}_{slot}"
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO origins VALUES (?, ?, ?)",
                            (oid, int(self.config["continuation_allowance"]), wave))
        return oid

    def _expand(self, cid, wave):
        with self.db:
            self.db.execute("UPDATE candidate_meta SET last_expanded_wave = ? WHERE candidate_id = ?", (wave, cid))

    def plan_wave(self, wave) -> list:
        existing = self.jobs(wave)
        if existing:
            return [JobPlan(**j["plan"]) for j in existing]
        snap_id, _ = self.build_snapshot(wave)
        incumbent = self.incumbent()
        plans = []

        # slot 0: continuation or fallback challenge
        fu = self.db.execute(
            "SELECT f.* FROM followups f JOIN origins o ON o.id = f.origin_id "
            "WHERE f.dispatched_wave IS NULL AND o.allowance > 0 ORDER BY f.id LIMIT 1").fetchone()
        if fu:
            with self.db:
                self.db.execute("UPDATE followups SET dispatched_wave = ? WHERE id = ?", (wave, fu["id"]))
                self.db.execute("UPDATE origins SET allowance = allowance - 1 WHERE id = ?", (fu["origin_id"],))
            plans.append(JobPlan(f"j{wave:03d}_0", wave, 0, "continuation", fu["origin_id"], fu["candidate_id"],
                                 snap_id, fu["id"], json.loads(fu["followup"]), "queued follow-up"))
            self._expand(fu["candidate_id"], wave)
        else:
            plans.append(JobPlan(f"j{wave:03d}_0", wave, 0, "challenge", self._new_origin(wave, 0), incumbent,
                                 snap_id, note="fallback challenge of incumbent"))
            self._expand(incumbent, wave)

        # slot 1: challenge from the incumbent
        plans.append(JobPlan(f"j{wave:03d}_1", wave, 1, "challenge", self._new_origin(wave, 1), incumbent,
                             snap_id, note="incumbent"))
        self._expand(incumbent, wave)

        # slot 2: challenge from the least recently expanded valid state
        states = self.valid_states()
        pick = min(states, key=lambda s: (s["last_expanded_wave"], s["candidate_id"]))["candidate_id"]
        plans.append(JobPlan(f"j{wave:03d}_2", wave, 2, "challenge", self._new_origin(wave, 2), pick,
                             snap_id, note="least-recent expansion"))
        self._expand(pick, wave)

        # slot 3: reconstruction
        plans.append(JobPlan(f"j{wave:03d}_3", wave, 3, "reconstruction", self._new_origin(wave, 3), None,
                             snap_id, note="clean harness"))

        for p in plans:
            self.save_job(JobRecord(plan=asdict(p)))
        return plans

    def save_job(self, job: JobRecord):
        p = job.plan
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO jobs VALUES (?, ?, ?, ?, ?, ?)",
                            (p["id"], p["wave"], p["slot"], p["mode"], job.status, json.dumps(asdict(job))))

    def record_candidate(self, cid, job_id, wave, origin_id, change_type, space_sid, register_text):
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO candidate_meta (candidate_id, job_id, creation_wave, "
                            "last_expanded_wave, origin_id, change_type, space_id, register_text) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                            (cid, job_id, wave, wave, origin_id, change_type, space_sid, register_text))

    def record_smoke(self, cid, record):
        loss = record["checkpoints"][-1]["dev_loss"] if record["status"] == "ok" and record["checkpoints"] else None
        with self.db:
            self.db.execute("UPDATE candidate_meta SET smoke_experiment_id = ?, smoke_status = ?, smoke_loss = ? "
                            "WHERE candidate_id = ?", (record["id"], record["status"], loss, cid))

    def record_revision(self, cid, space_sid=None, register_text=None):
        with self.db:
            self.db.execute("UPDATE candidate_meta SET space_id_post = COALESCE(?, space_id_post), "
                            "register_post = COALESCE(?, register_post) WHERE candidate_id = ?",
                            (space_sid, register_text, cid))

    def enqueue_followup(self, origin_id, candidate_id, job_id, wave, followup) -> bool:
        row = self.db.execute("SELECT allowance FROM origins WHERE id = ?", (origin_id,)).fetchone()
        if row is None or row["allowance"] <= 0:
            return False
        with self.db:
            self.db.execute("INSERT INTO followups (origin_id, candidate_id, job_id, created_wave, followup) "
                            "VALUES (?, ?, ?, ?, ?)", (origin_id, candidate_id, job_id, wave, json.dumps(followup)))
        return True

    # ---- driver ------------------------------------------------------------
    def run(self, waves: int, llm, log=print):
        from .worker import Worker

        self.control(log=log)
        start = int(self._meta_get("next_wave", "1"))
        for wave in range(start, start + waves):
            plans = self.plan_wave(wave)
            log(f"=== wave {wave}: " + ", ".join(f"{p.mode}<-{p.research_parent_id}" for p in plans))
            done = {j["plan"]["id"] for j in self.jobs(wave) if j["status"] != "planned"}
            for plan in plans:
                if plan.id in done:
                    continue
                log(f"  job {plan.id} ({plan.mode}, parent {plan.research_parent_id}, origin {plan.origin_id})")
                job = Worker(self, llm, plan, log=log).run()
                self.save_job(job)
                log(f"  job {plan.id}: {job.status}"
                    + (f", candidate {job.final_candidate_id}" if job.final_candidate_id else "")
                    + (f", evidence {job.reflection.get('evidence_class')}" if job.reflection else ""))
            with self.db:
                self._meta_set("next_wave", wave + 1)
