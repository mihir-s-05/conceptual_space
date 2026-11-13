"""Seed replication and sealed confirmation of selected candidates.

A confirmation evaluates frozen candidates from a campaign against the campaign's
baseline at several training seeds (paired: candidate and baseline share each seed),
on the dev split (replicating promising or ambiguous findings) or the sealed confirm
split (final evaluation of development-selected incumbents).

Results live in archive/confirmations/<name>/ with their own archive, never in the
campaign archive, so agents cannot see them. Candidates are fixed at creation
(`incumbent` resolves to the campaign's development incumbent at that moment) and
the spec is frozen by hash.

Per candidate: paired deltas d_s = loss(candidate, s) - loss(baseline, s), their mean,
and a t-interval. "Reliably better by at least `threshold`" needs the interval's upper
bound on d to be <= -threshold (equivalently, the lower bound on baseline minus
candidate exceeds the threshold). With k candidates each interval is at level
1 - alpha/k (Bonferroni), the joint procedure.
"""

import json
import math
import tempfile
import time
from pathlib import Path

from .archive import Archive
from .frozen import apply_overrides, read_frozen, write_frozen
from .paths import ARCHIVE_DIR

CONFIRM_DIR = ARCHIVE_DIR / "confirmations"
DEFAULT_SPEC = dict(
    campaign=None, campaigns_root=None,
    candidates=["incumbent"],       # candidate ids or "incumbent"; resolved and frozen at creation
    split="dev",                    # "dev" (replication) or "confirm" (sealed final evaluation)
    seeds=[1, 2, 3],                # init and data seed per replicate (paired with the baseline)
    fracs=[1.0],                    # checkpoints (fractions of the campaign horizon); the last is analysed
    horizon_flops=None,             # default: the campaign's horizon
    eval_tokens=None,               # default: the campaign's
    threshold=0.02,                 # practical improvement threshold (nats)
    alpha=0.05,
    device="auto", throttle=0.0, temp_limit_c=None,
    eval_time_limit_s=7200.0,
    executor={"kind": "local", "slots": 1},
)


# ---------------------------------------------------------------------------
# Student t without scipy: Simpson integration of the density plus bisection
def _t_cdf(x, nu, steps=2000):
    norm = math.exp(math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2)) / math.sqrt(nu * math.pi)
    pdf = lambda t: norm * (1 + t * t / nu) ** (-(nu + 1) / 2)
    h = abs(x) / steps
    area = h / 3 * (pdf(0) + pdf(abs(x)) + sum((4 if i % 2 else 2) * pdf(i * h) for i in range(1, steps)))
    return 0.5 + math.copysign(area, x)


def t_quantile(p, nu):
    """Inverse Student-t CDF for p in (0.5, 1)."""
    lo, hi = 0.0, 1000.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _t_cdf(mid, nu) < p else (lo, mid)
    return (lo + hi) / 2


def paired_interval(deltas, alpha):
    n = len(deltas)
    mean = sum(deltas) / n if n else None
    if n < 2:
        return {"n": n, "mean": mean, "sd": None, "lower": None, "upper": None, "level": 1 - alpha}
    sd = math.sqrt(sum((d - mean) ** 2 for d in deltas) / (n - 1))
    half = t_quantile(1 - alpha / 2, n - 1) * sd / math.sqrt(n)
    return {"n": n, "mean": mean, "sd": sd, "lower": mean - half, "upper": mean + half, "level": 1 - alpha}


def verdict(interval, threshold, complete):
    if not complete:
        return "incomplete"
    if interval["upper"] is None:
        return "single seed: no interval"
    if interval["upper"] <= -threshold:
        return "meets threshold"
    if interval["lower"] >= 0:
        return "worse than baseline"
    return "not established"


# ---------------------------------------------------------------------------
class Confirmation:
    def __init__(self, name, root=None):
        self.name = name
        self.dir = Path(root or CONFIRM_DIR) / name
        if not (self.dir / "confirmation.json").is_file():
            raise FileNotFoundError(f"confirmation {name!r} not found in {self.dir.parent}")
        self.spec, self.spec_hash = read_frozen(self.dir / "confirmation.json")
        self.archive = Archive(self.dir / "archive")
        self.db = self.archive.db
        self.db.executescript("CREATE TABLE IF NOT EXISTS runs (member TEXT NOT NULL, seed INTEGER NOT NULL, "
                              "experiment_id TEXT NOT NULL, PRIMARY KEY (member, seed));")

    @classmethod
    def create(cls, name, root=None, **overrides):
        from .agents.campaign import Campaign

        spec = apply_overrides(DEFAULT_SPEC, overrides)
        if not spec["campaign"]:
            raise ValueError("specify the source campaign")
        if spec["split"] not in ("dev", "confirm"):
            raise ValueError("split must be dev or confirm")
        if not spec["seeds"] or len(set(spec["seeds"])) != len(spec["seeds"]):
            raise ValueError("seeds must be distinct")
        d = Path(root or CONFIRM_DIR) / name
        if (d / "confirmation.json").exists():
            raise FileExistsError(f"confirmation {name!r} already exists")
        with Campaign(spec["campaign"], root=spec["campaigns_root"]) as camp:
            candidates = []
            for c in spec["candidates"]:
                cid = camp.incumbent() if c == "incumbent" else c
                if camp.archive.get_candidate(cid) is None:
                    raise ValueError(f"candidate {cid} is not in campaign {spec['campaign']}")
                if cid == camp.seed_id:
                    raise ValueError("the baseline is always included; do not list it as a candidate")
                if cid not in candidates:
                    candidates.append(cid)
            cfg = camp.config
            spec.update(candidates=candidates, baseline=camp.seed_id, task=cfg["task"], max_params=cfg["max_params"],
                        horizon_flops=spec["horizon_flops"] or cfg["horizon_flops"],
                        eval_tokens=spec["eval_tokens"] or cfg["eval_tokens"],
                        created=time.strftime("%Y-%m-%dT%H:%M:%S"))
            with Archive(d / "archive") as archive, tempfile.TemporaryDirectory(prefix="cspace_conf_") as tmp:
                for cid in [camp.seed_id] + candidates:
                    bundle = camp.archive.materialize(camp.archive.get_candidate(cid)["bundle_hash"], Path(tmp) / cid)
                    archive.add_candidate(bundle, action_mode="seed" if cid == camp.seed_id else "manual",
                                          notes=f"from campaign {spec['campaign']}")
        write_frozen(d / "confirmation.json", spec)
        return cls(name, root)

    def close(self):
        self.archive.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def members(self):
        return [self.spec["baseline"]] + list(self.spec["candidates"])

    def run_of(self, member, seed):
        row = self.db.execute("SELECT experiment_id FROM runs WHERE member = ? AND seed = ?", (member, seed)).fetchone()
        return self.archive.get_experiment(row["experiment_id"]) if row else None

    def run(self, log=print):
        from .executor import build_request, get_executor

        s = self.spec
        budget = dict(horizon_flops=s["horizon_flops"], checkpoint_fracs=list(s["fracs"]), eval_tokens=s["eval_tokens"])
        with tempfile.TemporaryDirectory(prefix="cspace_conf_run_") as tmp:
            jobs = []
            for seed in s["seeds"]:
                for member in self.members():
                    if self.run_of(member, seed) is None:
                        bundle = self.archive.materialize(self.archive.get_candidate(member)["bundle_hash"],
                                                          Path(tmp) / f"{member}_{seed}")
                        jobs.append(((member, seed), build_request(
                            bundle, s["task"], s, budget, {"init": seed, "data": seed}, split=s["split"],
                            allow_sealed=s["split"] == "confirm")))

            def record(key, result):
                member, seed = key
                rec = self.archive.add_experiment(result, member, purpose=f"confirmation {s['split']} seed {seed}")
                with self.db:
                    self.db.execute("INSERT INTO runs VALUES (?, ?, ?)", (member, seed, rec["id"]))
                log(f"  {member} seed {seed}: {rec['status']}"
                    + (f", loss {rec['final']['loss']:.4f}" if rec.get("final") else ""))

            get_executor(s["executor"]).run_all(jobs, record)
        return self.report(log=log)

    def _candidate_result(self, cid, alpha):
        s = self.spec
        per_seed, failed = [], []
        for seed in s["seeds"]:
            c, b = self.run_of(cid, seed), self.run_of(s["baseline"], seed)
            if c is None or b is None or c["status"] != "ok" or b["status"] != "ok":
                failed.append({"seed": seed, "candidate": c and c["status"], "baseline": b and b["status"]})
                continue
            per_seed.append({"seed": seed, "candidate": c["final"]["loss"], "baseline": b["final"]["loss"],
                             "delta": c["final"]["loss"] - b["final"]["loss"]})
        interval = paired_interval([p["delta"] for p in per_seed], alpha)
        return {"candidate_id": cid, "per_seed": per_seed, "failed": failed, "interval": interval,
                "verdict": verdict(interval, s["threshold"], complete=not failed)}

    def report(self, log=print):
        s = self.spec
        k = max(1, len(s["candidates"]))
        alpha = s["alpha"] / k
        rows = [self._candidate_result(cid, alpha) for cid in s["candidates"]]
        rep = {"confirmation": self.name, "spec_hash": self.spec_hash, "split": s["split"], "campaign": s["campaign"],
               "seeds": s["seeds"], "frac": max(s["fracs"]), "threshold": s["threshold"],
               "joint_procedure": f"Bonferroni over {k} candidate(s): each interval at level {1 - alpha:.4f}",
               "candidates": rows}
        (self.dir / "report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
        f = lambda v: "-" if v is None else f"{v:+.4f}"
        lines = [f"# Confirmation `{self.name}` ({s['split']} split, campaign `{s['campaign']}`)", "",
                 f"Paired seeds {s['seeds']} at {max(s['fracs']):g} of a {s['horizon_flops']:.3g}-FLOP horizon. "
                 f"delta = candidate - baseline loss (negative is better). {rep['joint_procedure']}. "
                 f"A claim requires upper bound <= -{s['threshold']}.", "",
                 "| candidate | n | mean delta | sd | interval | verdict |", "| --- | --- | --- | --- | --- | --- |"]
        for r in rows:
            iv = r["interval"]
            sd = "-" if iv["sd"] is None else f"{iv['sd']:.4f}"
            lines.append(f"| {r['candidate_id']} | {iv['n']} | {f(iv['mean'])} | {sd} | "
                         f"[{f(iv['lower'])}, {f(iv['upper'])}] | {r['verdict']} |")
        (self.dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        log(f"wrote {self.dir / 'report.md'}")
        return rep
