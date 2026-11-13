import json
import os
import re
from pathlib import Path

# Tests never touch the GPU unless explicitly allowed (CSPACE_GPU_TESTS=1). Hiding the
# device here, before torch initializes CUDA, also covers evaluator subprocesses,
# which inherit this environment.
if os.environ.get("CSPACE_GPU_TESTS") != "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # an empty string does NOT hide the device on Windows

import numpy as np
import pytest

from cspace.hashing import sha256_file
from cspace.paths import SEEDS_DIR

TINY_CONFIG = "CONFIG = dict(d_model=32, n_layers=2, n_heads=2, mlp_mult=8 / 3, rope_base=10000.0)"


def write_bundle(directory: Path, model_source: str, train_config=None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "model.py").write_text(model_source, encoding="utf-8")
    if train_config is not None:
        (directory / "train_config.json").write_text(json.dumps(train_config), encoding="utf-8")
    return directory


def tiny_baseline_source() -> str:
    src = (SEEDS_DIR / "baseline_decoder" / "model.py").read_text(encoding="utf-8")
    new, n = re.subn(r"^CONFIG = .*$", TINY_CONFIG, src, count=1, flags=re.M)
    assert n == 1
    return new


@pytest.fixture
def tiny_baseline(tmp_path):
    return write_bundle(tmp_path / "baseline", tiny_baseline_source(), {"lr": 0.003, "warmup_frac": 0.05})


@pytest.fixture
def tiny_task(tmp_path):
    """Synthetic corpus: documents drawn from 40 fixed patterns, so it is learnable."""
    V, T = 64, 32
    rng = np.random.default_rng(0)
    patterns = [rng.integers(1, V, size=rng.integers(8, 24)) for _ in range(40)]
    root = tmp_path / "task"
    root.mkdir()
    files = {}
    for split, n_docs in (("train", 3000), ("dev", 300), ("confirm", 300)):
        docs = [np.append(patterns[i], 0) for i in rng.integers(0, len(patterns), size=n_docs)]
        arr = np.concatenate(docs).astype(np.uint16)
        path = root / f"{split}.bin"
        arr.tofile(path)
        files[split] = {"path": path.name, "sha256": sha256_file(path), "tokens": int(arr.size)}
    spec = {"corpus": "synthetic", "vocab_size": V, "eot_id": 0, "context_length": T, "batch_size": 8,
            "files": files}
    (root / "task.json").write_text(json.dumps(spec), encoding="utf-8")
    return root
