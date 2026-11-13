"""Content hashing for data files, candidate bundles, and records."""

import hashlib
import json
from pathlib import Path

# Files that make up a candidate bundle. Anything else in a bundle directory is ignored.
BUNDLE_FILES = ("model.py", "train_config.json", "NOTES.md")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk_size):
            h.update(block)
    return h.hexdigest()


def canonical_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def bundle_manifest(bundle_dir) -> dict:
    """Map bundle file name -> sha256. model.py is required."""
    d = Path(bundle_dir)
    if not (d / "model.py").is_file():
        raise FileNotFoundError(f"candidate bundle {d} has no model.py")
    return {name: sha256_file(d / name) for name in BUNDLE_FILES if (d / name).is_file()}


def bundle_hash(manifest: dict) -> str:
    return sha256_bytes(canonical_json(manifest))
