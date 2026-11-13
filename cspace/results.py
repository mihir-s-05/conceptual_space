"""Small helpers for reading evaluator results."""


def by_frac(exp) -> dict:
    """Checkpoints of an experiment keyed by rounded horizon fraction."""
    return {round(c["frac"], 6): c for c in (exp or {}).get("checkpoints") or []}


def deltas_vs(exp, base) -> list:
    """[(frac, dev loss minus base dev loss)] at the checkpoints both runs reached."""
    ref = by_frac(base)
    return [(c["frac"], c["dev_loss"] - ref[round(c["frac"], 6)]["dev_loss"])
            for c in (exp or {}).get("checkpoints") or [] if round(c["frac"], 6) in ref]


def first_line(text, limit=120) -> str:
    return (text or "").splitlines()[0][:limit] if text else ""
