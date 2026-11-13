"""Seed replication and sealed confirmation of selected candidates.

A confirmation evaluates frozen candidates from a campaign against the campaign's
baseline at several training seeds (paired: candidate and baseline share each seed),
on the dev split (replication of promising or ambiguous findings) or the sealed
confirm split (final evaluation of development-selected incumbents).

Results live in archive/confirmations/<name>/ with their own archive, never in the
campaign archive, so agents cannot see them. Candidates are fixed when the
confirmation is created (`incumbent` resolves to the campaign's development
incumbent at that moment) and the spec is frozen by hash.

Per candidate: paired deltas d_s = loss(candidate, s) - loss(baseline, s), their
mean, and a t-interval. The claim "reliably better by at least `threshold`" needs
the upper bound of the interval on d to be <= -threshold, i.e. the lower bound on
baseline-minus-candidate above the threshold. With k candidates the interval uses
a Bonferroni-adjusted level (alpha / k) as the joint procedure.
"""

import json
import math
import tempfile
import time
from pathlib import Path

from .archive import Archive
from .hashing import canonical_json, sha256_bytes
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


class ConfirmError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
def _t_pdf(x, nu):
    return math.exp(math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2)) / math.sqrt(nu * math.pi) * \
        (1 + x * x / nu) ** (-(nu + 1) / 2)


def _t_cdf(x, nu, steps=4000):
    if x == 0:
        return 0.5
    h = abs(x) / steps
    s = _t_pdf(0, nu) + _t_pdf(abs(x), nu) + sum((4 if i % 2 else 2) * _t_pdf(i * h, nu) for i in range(1, steps))
    area = s * h / 3
    return 0.5 + area if x > 0 else 0.5 - area


def t_quantile(p, nu):
    """Inverse Student-t CDF by bisection (p in (0.5, 1))."""
    lo, hi = 0.0, 1000.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if _t_cdf(mid, nu) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def paired_interval(deltas, alpha):
    n = len(deltas)
    mean = sum(deltas) / n if n else None
    if n < 2:
        return {"n": n, "mean": mean, "sd": None, "lower": None, "upper": None, "level": 1 - alpha}
    sd = math.sqrt(sum((d - mean) ** 2 for d in deltas) / (n - 1))
    half = t_quantile(1 - alpha / 2, n - 1) * sd / math.sqrt(n)
    return {"n": n, "mean": mean, "sd": sd, "lower": mean - half, "upper": mean + half, "level": 1 - alpha}


# ---------------------------------------------------------------------------
class Confirmation:
    def __init__(self, name, root=None):
        self.name = name
        self.dir = Path(root or CONFIRM_DIR) / name
        path = self.dir / "confirmation.json"
        if not path.is_file():
            raise FileNotFoundError(f"confirmation {name!r} not found in {self.dir.parent}")
        stored = json.loads(path.read_text(encoding="utf-8"))
        self.spec, self.spec_hash = stored["spec"], stored["spec_hash"]
        if sha256_bytes(canonical_json(self.spec)) != self.spec_hash:
            raise ConfirmError("confirmation.json was edited after creation; create a new confirmation")
        self.archive = Archive(self.dir / "archive")
        self.db = self.archive.db
        self.db.executescript("CREATE TABLE IF NOT EXISTS runs (member TEXT NOT NULL, seed INTEGER NOT NULL, "
                              "experiment_id TEXT NOT NULL, PRIMARY KEY (member, seed));")

    @classmethod
    def create(cls, name, root=None, **overrides):
        from .agents.campaign import Campaign

        spec = json.loads(json.dumps(DEFAULT_SPEC))
        for k, v in overrides.items():
            if v is None:
                continue
            if k not in spec:
                raise ValueError(f"unknown confirmation setting {k!r}")
            spec[k] = v
        if not spec["campaign"]:
            raise ValueError("specify the source campaign")
        if spec["split"] not in ("dev", "confirm"):
            raise ValueError("split must be dev or confirm")
        if not spec["seeds"] or len(set(spec["seeds"])) != len(spec["seeds"]):
            raise ValueError("seeds must be distinct")
        d = Path(root or CONFIRM_DIR) / name
        if (d / "confirmation.json").exists():
            raise FileExistsError(f"confirmation {name!r} already exists")
        camp = Campaign(spec["campaign"], root=spec["campaigns_root"])
        try:
            resolved = []
            for c in spec["candidates"]:
                cid = camp.incumbent() if c == "incumbent" else c
                if camp.archive.get_candidate(cid) is None:
                    raise ValueError(f"candidate {cid} is not in campaign {spec['campaign']}")
                if cid == camp.seed_id:
                    raise ValueError("the baseline is always included; do not list it as a candidate")
                if cid not in resolved:
                    resolved.append(cid)
            spec["candidates"] = resolved
            spec["baseline"] = camp.seed_id
            spec["task"] = camp.config["task"]
            spec["horizon_flops"] = spec["horizon_flops"] or camp.config["horizon_flops"]
            spec["eval_tokens"] = spec["eval_tokens"] or camp.config["eval_tokens"]
            spec["max_params"] = camp.config["max_params"]
            d.mkdir(parents=True, exist_ok=True)
            archive = Archive(d / "archive")
            with tempfile.TemporaryDirectory(prefix="cspace_conf_") as tmp:
                for cid in [spec["baseline"]] + resolved:
                    bundle = camp.archive.materialize(camp.archive.get_candidate(cid)["bundle_hash"], Path(tmp) / cid)
                    archive.add_candidate(bundle, action_mode="seed" if cid == spec["baseline"] else "manual",
                                          notes=f"from campaign {spec['campaign']}")
            archive.close()
        finally:
            camp.close()
        spec["created"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        (d / "confirmation.json").write_text(json.dumps({"spec": spec, "spec_hash": sha256_bytes(canonical_json(spec))},
                                                        indent=2), encoding="utf-8")
        return cls(name, root)

    def close(self):
        self.archive.close()

    def members(self):
        return [self.spec["baseline"]] + list(self.spec["candidates"])

    def run_of(self, member, seed):
        row = self.db.execute("SELECT experiment_id FROM runs WHERE member = ? AND seed = ?", (member, seed)).fetchone()
        return self.archive.get_experiment(row["experiment_id"]) if row else None

    def _request(self, member, seed, tmp):
        cand = self.archive.get_candidate(member)
        bundle = self.archive.materialize(cand["bundle_hash"], Path(tmp) / f"{member}_{seed}")
        s = self.spec
        return dict(bundle_dir=str(bundle), task_dir=s["task"], split=s["split"], allow_sealed=s["split"] == "confirm",
                    budget=dict(horizon_flops=s["horizon_flops"], checkpoint_fracs=list(s["fracs"]),
                                eval_tokens=s["eval_tokens"]),
                    seeds={"init": seed, "data": seed}, device=s["device"], time_limit_s=s["eval_time_limit_s"],
                    max_params=int(s["max_params"]), throttle=s["throttle"], temp_limit_c=s["temp_limit_c"])

    def run(self, log=print):
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from .executor import get_executor

        ex = get_executor(self.spec.get("executor"))
        todo = [(m, s) for s in self.spec["seeds"] for m in self.members() if self.run_of(m, s) is None]
        with tempfile.TemporaryDirectory(prefix="cspace_conf_run_") as tmp, \
                ThreadPoolExecutor(max_workers=max(1, min(ex.slots, len(todo) or 1))) as pool:
            futures = {pool.submit(ex.run, self._request(m, s, tmp), self.spec["eval_time_limit_s"] + 600): (m, s)
                       for m, s in todo}
            for f in as_completed(futures):
                m, s = futures[f]
                rec = self.archive.add_experiment(f.result(), m, purpose=f"confirmation {self.spec['split']} seed {s}")
                with self.db:
                    self.db.execute("INSERT INTO runs VALUES (?, ?, ?)", (m, s, rec["id"]))
                log(f"  {m} seed {s}: {rec['status']}" + (f", loss {rec['final']['loss']:.4f}" if rec.get("final") else ""))
        return self.report(log=log)

    def report(self, log=print):
        s = self.spec
        k = max(1, len(s["candidates"]))
        alpha = s["alpha"] / k
        rows = []
        for cid in s["candidates"]:
            deltas, per_seed, failed = [], [], []
            for seed in s["seeds"]:
                c, b = self.run_of(cid, seed), self.run_of(s["baseline"], seed)
                if c is None or b is None:
                    failed.append({"seed": seed, "status": "missing"})
                    continue
                if c["status"] != "ok" or b["status"] != "ok":
                    failed.append({"seed": seed, "status": f"candidate {c['status']}, baseline {b['status']}"})
                    continue
                d = c["final"]["loss"] - b["final"]["loss"]
                deltas.append(d)
                per_seed.append({"seed": seed, "candidate": c["final"]["loss"], "baseline": b["final"]["loss"],
                                 "delta": d})
            iv = paired_interval(deltas, alpha)
            if failed:
                verdict = "incomplete"
            elif iv["upper"] is None:
                verdict = "single seed: no interval"
            elif iv["upper"] <= -s["threshold"]:
                verdict = "meets threshold"
            elif iv["lower"] >= 0:
                verdict = "worse than baseline"
            else:
                verdict = "not established"
            rows.append({"candidate_id": cid, "per_seed": per_seed, "failed": failed, "interval": iv,
                         "verdict": verdict})
        rep = {"confirmation": self.name, "spec_hash": self.spec_hash, "split": s["split"], "campaign": s["campaign"],
               "seeds": s["seeds"], "frac": max(s["fracs"]), "threshold": s["threshold"],
               "joint_procedure": f"Bonferroni over {k} candidate(s): each interval at level {1 - alpha:.4f}",
               "candidates": rows}
        (self.dir / "report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
        f = lambda v: "-" if v is None else f"{v:+.4f}"
        lines = [f"# Confirmation `{self.name}` ({s['split']} split, campaign `{s['campaign']}`)", "",
                 f"Paired seeds {s['seeds']} at {max(s['fracs']):g} of a {s['horizon_flops']:.3g}-FLOP horizon. "
                 f"delta = candidate - baseline loss (negative is better). {rep['joint_procedure']}. "
                 f"Claim requires upper bound <= -{s['threshold']}.", "",
                 "| candidate | n | mean delta | sd | interval | verdict |", "| --- | --- | --- | --- | --- | --- |"]
        for r in rows:
            iv = r["interval"]
            sd = "-" if iv["sd"] is None else f"{iv['sd']:.4f}"
            lines.append(f"| {r['candidate_id']} | {iv['n']} | {f(iv['mean'])} | {sd} | "
                         f"[{f(iv['lower'])}, {f(iv['upper'])}] | {r['verdict']} |")
        (self.dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        log(f"wrote {self.dir / 'report.md'}")
        return rep
