"""Evaluator identity and locking.

evaluator_version() hashes every source file that can affect a measurement.
`cspace freeze-evaluator` writes archive/evaluator_lock.json; afterwards,
evaluations refuse to run if the evaluator source differs from the lock.
"""

import json
import time

from ..hashing import sha256_bytes, sha256_file
from ..paths import ARCHIVE_DIR, PACKAGE_DIR

LOCK_PATH = ARCHIVE_DIR / "evaluator_lock.json"


class EvaluatorChanged(RuntimeError):
    pass


def evaluator_files():
    files = sorted((PACKAGE_DIR / "evaluator").glob("*.py"))
    files += [PACKAGE_DIR / "data" / "loader.py", PACKAGE_DIR / "task.py", PACKAGE_DIR / "hashing.py"]
    return files


def file_hashes() -> dict:
    return {p.relative_to(PACKAGE_DIR).as_posix(): sha256_file(p) for p in evaluator_files()}


def evaluator_version() -> str:
    lines = "\n".join(f"{k} {v}" for k, v in sorted(file_hashes().items()))
    return sha256_bytes(lines.encode())[:16]


def freeze(lock_path=LOCK_PATH) -> dict:
    lock = {"version": evaluator_version(), "files": file_hashes(), "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    return lock


def check_lock(lock_path=LOCK_PATH):
    """Raise EvaluatorChanged if a lock exists and the evaluator source differs."""
    if not lock_path.is_file():
        return None
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    current = file_hashes()
    if lock["version"] != evaluator_version():
        changed = sorted(k for k in set(current) | set(lock["files"]) if current.get(k) != lock["files"].get(k))
        raise EvaluatorChanged(f"evaluator source differs from lock {lock['version']}: {changed}")
    return lock["version"]
