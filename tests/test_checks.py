"""Each deliberately broken model must be caught by the check designed for it,
and the baseline must pass everything."""

import pytest

from cspace.evaluator.candidate import load_candidate_module
from cspace.evaluator.checks import run_checks
from cspace.evaluator.lint import lint_source

from conftest import write_bundle

V, T = 64, 32

HEADER = """
import torch
import torch.nn as nn
import torch.nn.functional as F
"""

BIDIRECTIONAL = HEADER + """
class M(nn.Module):
    def __init__(self, V):
        super().__init__()
        self.emb = nn.Embedding(V, 32)
        self.qkv = nn.Linear(32, 96)
        self.out = nn.Linear(32, V)
    def forward(self, idx):
        x = self.emb(idx)
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        return self.out(x + F.scaled_dot_product_attention(q, k, v, is_causal=False))

def build_model(vocab_size, context_length):
    return M(vocab_size)
"""

CAUSAL_SIMPLE = BIDIRECTIONAL.replace("is_causal=False", "is_causal=True")

LENGTH_DEPENDENT = CAUSAL_SIMPLE.replace(
    "x = self.emb(idx)", "x = self.emb(idx) * (idx.shape[1] / 32.0)")

BATCH_COUPLED = CAUSAL_SIMPLE.replace(
    "x = self.emb(idx)", "x = self.emb(idx)\n        x = x + x.mean(dim=0, keepdim=True)")

STATEFUL = CAUSAL_SIMPLE.replace(
    "self.out = nn.Linear(32, V)",
    "self.out = nn.Linear(32, V)\n        self.register_buffer('mem', torch.zeros(32))",
).replace(
    "x = self.emb(idx)",
    "x = self.emb(idx) + self.mem\n        self.mem.copy_(0.5 * self.mem + 0.5 * x.detach().mean(dim=(0, 1)))",
)

NOT_A_MODULE = HEADER + """
def build_model(vocab_size, context_length):
    return 42
"""


def _checks(tmp_path, source):
    bundle = write_bundle(tmp_path / "cand", source)
    module = load_candidate_module(bundle / "model.py")
    return run_checks(module.build_model, V, T, "cpu", seed=0)


def _failed(report):
    return {k for k, v in report["checks"].items() if v["passed"] is False}


def test_baseline_passes_all(tiny_baseline):
    module = load_candidate_module(tiny_baseline / "model.py")
    report = run_checks(module.build_model, V, T, "cpu")
    assert report["passed"], _failed(report)
    assert report["checks"]["causal_gradient"]["passed"] is True
    assert report["checks"]["unused_parameters"]["passed"] is True


def test_simple_causal_model_passes(tmp_path):
    report = _checks(tmp_path, CAUSAL_SIMPLE)
    assert report["passed"], _failed(report)


@pytest.mark.parametrize("source, expected", [
    (BIDIRECTIONAL, {"causal_perturbation", "causal_gradient"}),
    (LENGTH_DEPENDENT, {"prefix_consistency"}),
    (BATCH_COUPLED, {"batch_independence"}),
    (STATEFUL, {"eval_stateless"}),
    (NOT_A_MODULE, {"build"}),
])
def test_broken_models_are_caught(tmp_path, source, expected):
    report = _checks(tmp_path, source)
    assert not report["passed"]
    assert expected <= _failed(report), _failed(report)


def test_length_dependent_model_passes_perturbation(tmp_path):
    # It never reads future tokens, so only the prefix check should catch it.
    report = _checks(tmp_path, LENGTH_DEPENDENT)
    assert report["checks"]["causal_perturbation"]["passed"] is True
    assert report["checks"]["prefix_consistency"]["passed"] is False


@pytest.mark.parametrize("snippet", [
    "import os",
    "import subprocess",
    "from pathlib import Path",
    "x = open('data/dev.bin', 'rb')",
    "torch.manual_seed(0)",
    "F.cross_entropy = lambda *a, **k: 0",
    "torch.nn.functional.cross_entropy = None",
    "setattr(F, 'cross_entropy', None)",
    "state = torch.load('weights.pt')",
    "g = (1).__class__.__subclasses__()",
])
def test_lint_rejects(snippet):
    assert lint_source(HEADER + snippet + "\n"), snippet


def test_lint_accepts_baseline(tiny_baseline):
    assert lint_source((tiny_baseline / "model.py").read_text()) == []
