"""Engineering checks: implementation validity only, not scientific evidence.

Required checks (any failure makes the candidate invalid):
  build              build_model returns an nn.Module within the parameter limit
  init_deterministic same seed -> identical initial parameters (needed for replay)
  output_shape       logits [B, T, V] for full and shorter lengths
  finite_outputs     no NaN/Inf in logits
  causal_perturbation  changing tokens after t never changes logits at <= t (eval and train mode)
  causal_gradient    d logits[t] / d embedding[t' > t] == 0 (when a token embedding exists)
  prefix_consistency logits for idx[:, :t+1] alone match the full-sequence logits at <= t
                     (predictions may not depend on how many tokens follow)
  batch_independence other sequences in the batch never affect a sequence's logits
  eval_stateless     eval-mode forward passes do not carry state between calls
  finite_gradients   no NaN/Inf gradients
  tiny_overfit       can fit one small fixed batch
Diagnostic (reported, not required):
  unused_parameters  parameters that receive no gradient
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .candidate import split_output
from .metrics import count_params

STRICT = dict(atol=1e-5, rtol=1e-5)   # same shapes: should be (near) bitwise identical
LOOSE = dict(atol=1e-4, rtol=1e-4)    # different shapes may select different kernels


class _Recorder:
    def __init__(self):
        self.items = {}

    def add(self, name, passed, value=None, detail="", required=True):
        self.items[name] = {"passed": passed, "value": value, "detail": detail, "required": required}

    @property
    def passed(self):
        return all(v["passed"] is not False for v in self.items.values() if v["required"])

    def as_dict(self):
        return {"passed": self.passed, "checks": self.items}


def _logits(model, x, seed):
    torch.manual_seed(seed)  # identical dropout masks across compared calls
    logits, _ = split_output(model(x))
    return logits.float()


def _compare(a, b, atol, rtol):
    diff = (a - b).abs().max().item()
    return diff <= atol + rtol * b.abs().max().item(), diff


def _randint(gen, high, shape, device, low=0):
    return torch.randint(low, high, shape, generator=gen).to(device)


def run_checks(build_model, vocab_size, context_length, device, seed=0, max_params=None,
               overfit_steps=200, overfit_lr=2e-3) -> dict:
    rec = _Recorder()
    prev = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        _run_all(rec, build_model, vocab_size, context_length, torch.device(device), seed,
                 max_params, overfit_steps, overfit_lr)
    except Exception as e:  # candidate code raised during a check
        rec.add("unexpected_error", False, detail=f"{type(e).__name__}: {e}")
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = prev
    return rec.as_dict()


def _build(build_model, V, T, device, seed):
    torch.manual_seed(seed)
    model = build_model(V, T)
    if not isinstance(model, nn.Module):
        raise TypeError(f"build_model returned {type(model).__name__}, not nn.Module")
    return model.to(device).float()


def _run_all(rec, build_model, V, T, device, seed, max_params, overfit_steps, overfit_lr):
    gen = torch.Generator().manual_seed(seed)

    # build
    try:
        model = _build(build_model, V, T, device, seed)
    except Exception as e:
        rec.add("build", False, detail=f"{type(e).__name__}: {e}")
        return
    params = count_params(model)
    if params["total"] == 0:
        rec.add("build", False, params, "model has no parameters")
        return
    if max_params is not None and params["total"] > max_params:
        rec.add("build", False, params, f"{params['total']:,} parameters exceeds limit {max_params:,}")
        return
    rec.add("build", True, params)

    # init determinism
    twin = _build(build_model, V, T, device, seed)
    sd_a, sd_b = model.state_dict(), twin.state_dict()
    same = sd_a.keys() == sd_b.keys() and all(torch.equal(sd_a[k], sd_b[k]) for k in sd_a)
    rec.add("init_deterministic", same, detail="" if same else "initialization differs under the same seed")
    del twin

    # shapes and finiteness
    x = _randint(gen, V, (2, T), device)
    model.eval()
    with torch.no_grad():
        base = _logits(model, x, seed)
        t_short = max(2, T // 3)
        short = _logits(model, x[:, :t_short], seed)
    shape_ok = tuple(base.shape) == (2, T, V) and tuple(short.shape) == (2, t_short, V)
    rec.add("output_shape", shape_ok, [list(base.shape), list(short.shape)],
            "" if shape_ok else f"expected [2, {T}, {V}] and [2, {t_short}, {V}]")
    if not shape_ok:
        return
    finite = bool(torch.isfinite(base).all())
    rec.add("finite_outputs", finite)
    if not finite:
        return

    positions = sorted({t for t in (0, T // 4, T // 2, T - 2) if 0 <= t < T - 1})

    # causal perturbation, eval and train mode
    worst, failures = 0.0, []
    for mode in ("eval", "train"):
        model.train(mode == "train")
        with torch.no_grad():
            ref = _logits(model, x, seed)
            for t in positions:
                xp = x.clone()
                tail = xp[:, t + 1:]
                xp[:, t + 1:] = (tail + _randint(gen, V, tail.shape, device, low=1)) % V
                ok, diff = _compare(_logits(model, xp, seed)[:, :t + 1], ref[:, :t + 1], **STRICT)
                worst = max(worst, diff)
                if not ok:
                    failures.append(f"{mode} t={t} diff={diff:.3g}")
    rec.add("causal_perturbation", not failures, worst, "; ".join(failures))

    # causal gradient through token embeddings
    model.eval()
    captured = []

    def hook(_module, _inp, out):
        if isinstance(out, torch.Tensor) and out.requires_grad:
            out.retain_grad()
            captured.append(out)

    handles = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, nn.Embedding)]
    try:
        t = T // 2
        with torch.enable_grad():
            logits = _logits(model, x, seed)
            w = torch.randn(logits[:, t].shape, generator=gen).to(device)
            (logits[:, t] * w).sum().backward()
    finally:
        for h in handles:
            h.remove()
    usable = [c for c in captured if c.dim() >= 2 and tuple(c.shape[:2]) == tuple(x.shape) and c.grad is not None]
    if not usable:
        rec.add("causal_gradient", None, detail="no token-embedding output of shape [B, T, ...]; skipped",
                required=False)
    else:
        leak = max(c.grad[:, t + 1:].abs().max().item() for c in usable)
        ref_mag = max(c.grad[:, :t + 1].abs().max().item() for c in usable)
        ok = leak <= 1e-6 * ref_mag if ref_mag > 0 else leak == 0.0
        rec.add("causal_gradient", ok, leak, "" if ok else f"gradient from future positions: {leak:.3g}")
    model.zero_grad(set_to_none=True)

    # prefix consistency
    model.eval()
    worst, failures = 0.0, []
    with torch.no_grad():
        for t in positions:
            ok, diff = _compare(_logits(model, x[:, :t + 1], seed), base[:, :t + 1], **LOOSE)
            worst = max(worst, diff)
            if not ok:
                failures.append(f"t={t} diff={diff:.3g}")
    rec.add("prefix_consistency", not failures, worst, "; ".join(failures))

    # batch independence, eval and train mode
    x4 = _randint(gen, V, (4, T), device)
    worst, failures = 0.0, []
    for mode in ("eval", "train"):
        model.train(mode == "train")
        with torch.no_grad():
            ref = _logits(model, x4, seed)[0]
            xp = x4.clone()
            xp[1:] = _randint(gen, V, xp[1:].shape, device)
            ok, diff = _compare(_logits(model, xp, seed)[0], ref, **STRICT)
            worst = max(worst, diff)
            if not ok:
                failures.append(f"{mode} diff={diff:.3g}")
    rec.add("batch_independence", not failures, worst, "; ".join(failures))

    # eval-mode statelessness
    model.eval()
    with torch.no_grad():
        buffers = {k: v.clone() for k, v in model.named_buffers()}
        first = _logits(model, x, seed)
        _logits(model, _randint(gen, V, (2, T), device), seed)
        _logits(model, _randint(gen, V, (3, T // 2 + 1), device), seed)
        again = _logits(model, x, seed)
        ok, diff = _compare(again, first, **STRICT)
        changed = [k for k, v in model.named_buffers() if k in buffers and not torch.equal(v, buffers[k])]
    rec.add("eval_stateless", ok and not changed, diff,
            "; ".join(filter(None, ["" if ok else f"output changed by {diff:.3g}",
                                    f"buffers changed: {changed}" if changed else ""])))

    # gradients
    model.train()
    y = _randint(gen, V, (2, T), device)
    torch.manual_seed(seed)
    logits, aux = split_output(model(x))
    loss = F.cross_entropy(logits.float().reshape(-1, V), y.reshape(-1))
    if aux is not None:
        loss = loss + aux.float().reshape(())
    loss.backward()
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    nonfinite = [n for n, p in named if p.grad is not None and not torch.isfinite(p.grad).all()]
    unused = [n for n, p in named if p.grad is None or p.grad.abs().max().item() == 0.0]
    rec.add("finite_gradients", not nonfinite, detail=", ".join(nonfinite[:10]))
    rec.add("unused_parameters", not unused, len(unused), ", ".join(unused[:10]), required=False)
    model.zero_grad(set_to_none=True)
    del model

    # tiny-batch overfit on a fresh copy
    fresh = _build(build_model, V, T, device, seed)
    fresh.train()
    opt = torch.optim.Adam(fresh.parameters(), lr=overfit_lr)
    seq = _randint(gen, V, (4, min(64, T) + 1), device)
    xb, yb = seq[:, :-1], seq[:, 1:]
    losses = []
    for _ in range(overfit_steps):
        logits, aux = split_output(fresh(xb))
        loss = F.cross_entropy(logits.float().reshape(-1, V), yb.reshape(-1))
        if aux is not None:
            loss = loss + aux.float().reshape(())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    initial, final = losses[0], min(losses[-10:])
    ok = final == final and final < 0.5 * initial
    rec.add("tiny_overfit", ok, {"initial": initial, "final": final, "steps": overfit_steps},
            "" if ok else "did not halve the loss on a fixed batch")
