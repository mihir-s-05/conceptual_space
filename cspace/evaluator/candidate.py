"""Loading candidate code and interpreting its outputs."""

import importlib.util
import sys
import uuid
from pathlib import Path

import torch
import torch.nn.functional as F


class CandidateError(Exception):
    """The candidate violates the model contract."""


def load_candidate_module(model_py):
    model_py = Path(model_py)
    name = f"cspace_candidate_{uuid.uuid4().hex[:12]}"
    spec = importlib.util.spec_from_file_location(name, str(model_py))
    if spec is None or spec.loader is None:
        raise CandidateError(f"cannot import {model_py}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses and pickling look modules up here
    spec.loader.exec_module(module)
    if not callable(getattr(module, "build_model", None)):
        raise CandidateError("model.py must define build_model(vocab_size, context_length)")
    return module


def split_output(out):
    """Return (logits, aux_loss or None) from a candidate forward output."""
    if isinstance(out, torch.Tensor):
        return out, None
    if isinstance(out, dict) and isinstance(out.get("logits"), torch.Tensor):
        aux = out.get("aux_loss")
        if aux is not None and (not isinstance(aux, torch.Tensor) or aux.numel() != 1):
            raise CandidateError("aux_loss must be a scalar tensor")
        return out["logits"], aux
    raise CandidateError("forward must return a logits tensor or a dict with a 'logits' tensor")


def forward_loss(model, x, y):
    """Return (training objective, next-token cross-entropy)."""
    logits, aux = split_output(model(x))
    ce = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1))
    total = ce if aux is None else ce + aux.float().reshape(())
    return total, ce
