"""Specs that are fixed when a study object (cohort, confirmation) is created.

The spec is stored with its hash; loading refuses a spec edited afterwards, so
rules such as screening thresholds cannot drift once results start arriving.
"""

import json
from pathlib import Path

from .hashing import canonical_json, sha256_bytes


class FrozenSpecError(RuntimeError):
    pass


def apply_overrides(defaults: dict, overrides: dict) -> dict:
    """Copy of `defaults` with every non-None override applied; unknown keys are an error."""
    spec = json.loads(json.dumps(defaults))
    for key, value in overrides.items():
        if value is None:
            continue
        if key not in spec:
            raise ValueError(f"unknown setting {key!r}")
        spec[key] = value
    return spec


def spec_hash(spec: dict) -> str:
    return sha256_bytes(canonical_json(spec))


def write_frozen(path: Path, spec: dict) -> str:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    h = spec_hash(spec)
    path.write_text(json.dumps({"spec": spec, "spec_hash": h}, indent=2), encoding="utf-8")
    return h


def read_frozen(path: Path):
    """(spec, hash) from a frozen spec file."""
    stored = json.loads(Path(path).read_text(encoding="utf-8"))
    if spec_hash(stored["spec"]) != stored["spec_hash"]:
        raise FrozenSpecError(f"{path} was edited after creation; specs are frozen, so create a new one")
    return stored["spec"], stored["spec_hash"]
