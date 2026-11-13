"""Frozen task definition: corpus splits, tokenizer, context length, batch size.

A task directory contains task.json plus the token files it references. The
task hash covers the full task.json (including every file hash), so any change
to data, tokenizer, or resource rules produces a different task_hash in records.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from .hashing import canonical_json, sha256_bytes, sha256_file

SPLITS = ("train", "dev", "confirm")


@dataclass(frozen=True)
class Task:
    root: Path
    spec: dict

    @property
    def corpus(self) -> str:
        return self.spec["corpus"]

    @property
    def vocab_size(self) -> int:
        return int(self.spec["vocab_size"])

    @property
    def eot_id(self) -> int:
        return int(self.spec["eot_id"])

    @property
    def context_length(self) -> int:
        return int(self.spec["context_length"])

    @property
    def batch_size(self) -> int:
        return int(self.spec["batch_size"])

    @property
    def task_hash(self) -> str:
        return sha256_bytes(canonical_json(self.spec))

    def split_path(self, split: str) -> Path:
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}")
        return self.root / self.spec["files"][split]["path"]

    def verify(self) -> list:
        """Return a list of files whose contents no longer match task.json."""
        bad = []
        entries = dict(self.spec["files"])
        if "tokenizer" in self.spec:
            entries["tokenizer"] = self.spec["tokenizer"]
        for name, entry in entries.items():
            path = self.root / entry["path"]
            if not path.is_file() or sha256_file(path) != entry["sha256"]:
                bad.append(name)
        return bad


def load_task(task_dir) -> Task:
    root = Path(task_dir)
    spec = json.loads((root / "task.json").read_text(encoding="utf-8"))
    return Task(root=root, spec=spec)
