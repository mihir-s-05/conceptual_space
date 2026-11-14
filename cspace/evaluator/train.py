"""Compute-capped training with a fixed-horizon schedule.

A Budget defines a horizon in training FLOPs (the planned full run) and a set
of checkpoint fractions. The learning-rate schedule is always computed for the
full horizon; a smoke test simply stops at an earlier checkpoint, so short and
long runs share the same schedule prefix (no schedule confound).
"""

import gc
import math
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass

import numpy as np
import torch

from ..data.loader import TokenStream
from .candidate import forward_loss
from .metrics import measure_forward_flops, measure_step_flops
from .pacing import Overheat, Pacer

DEFAULT_RECIPE = dict(lr=2e-3, weight_decay=0.1, beta1=0.9, beta2=0.95, warmup_frac=0.05, min_lr_frac=0.1)
RECIPE_BOUNDS = dict(lr=(1e-5, 1e-2), weight_decay=(0.0, 0.3), beta1=(0.8, 0.99), beta2=(0.9, 0.999),
                     warmup_frac=(0.0, 0.2), min_lr_frac=(0.0, 1.0))
GRAD_CLIP = 1.0


def resolve_recipe(overrides: dict) -> dict:
    """Merge candidate train_config overrides into the default recipe, enforcing bounds."""
    unknown = set(overrides) - set(DEFAULT_RECIPE)
    if unknown:
        raise ValueError(f"train_config has unsupported keys: {sorted(unknown)}")
    recipe = {**DEFAULT_RECIPE, **overrides}
    for key, (lo, hi) in RECIPE_BOUNDS.items():
        value = recipe[key]
        if not isinstance(value, (int, float)) or not lo <= value <= hi:
            raise ValueError(f"train_config {key}={value!r} outside allowed range [{lo}, {hi}]")
        recipe[key] = float(value)
    return recipe


@dataclass
class Budget:
    horizon_flops: float
    checkpoint_fracs: tuple = (1.0,)
    eval_tokens: int = 1 << 20

    def __post_init__(self):
        self.horizon_flops = float(self.horizon_flops)
        self.checkpoint_fracs = tuple(float(f) for f in self.checkpoint_fracs)
        if self.horizon_flops <= 0:
            raise ValueError("horizon_flops must be positive")
        fr = self.checkpoint_fracs
        if not fr or any(not 0 < f <= 1 for f in fr) or list(fr) != sorted(set(fr)):
            raise ValueError("checkpoint_fracs must be increasing values in (0, 1]")

    def to_dict(self):
        d = asdict(self)
        d["checkpoint_fracs"] = list(self.checkpoint_fracs)
        return d

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


def lr_at(step: int, total_steps: int, recipe: dict) -> float:
    peak, floor = recipe["lr"], recipe["lr"] * recipe["min_lr_frac"]
    warmup = int(recipe["warmup_frac"] * total_steps)
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * progress))


def build_optimizer(model, recipe, device):
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    groups = [{"params": decay, "weight_decay": recipe["weight_decay"]},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=recipe["lr"], betas=(recipe["beta1"], recipe["beta2"]),
                             fused=device.type == "cuda")


def autocast_for(device):
    if device.type == "cuda":
        return lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext


@torch.no_grad()
def to_device(t, device):
    """Host-to-device copy that does not stall the GPU queue: copies from pageable memory
    synchronize the stream first, so the GPU would idle while the CPU queued each step."""
    if device.type == "cuda":
        return t.pin_memory().to(device, non_blocking=True)
    return t.to(device)


def eval_loss(model, stream, batch_size, max_tokens, device, autocast_ctx, kernel_ctx=nullcontext):
    was_training = model.training
    model.eval()
    losses = []  # (per-batch mean loss tensor, tokens); read back once, summed in the same order
    try:
        for x, y in stream.eval_batches(batch_size, max_tokens):
            x, y = to_device(x, device), to_device(y, device)
            with kernel_ctx(), autocast_ctx():
                _, ce = forward_loss(model, x, y)
            losses.append((ce.detach(), y.numel()))
    finally:
        model.train(was_training)
    total, count = 0.0, 0
    for ce, n in losses:
        total += ce.item() * n
        count += n
    return total / count, count


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class GraphedStep:
    """Forward+backward recorded as a CUDA graph and replayed with one CPU call per step.

    Replay runs the same kernels in the same order as eager execution, so results are
    bitwise identical; it removes the per-kernel CPU launch cost that dominates small models
    on a throttled CPU. Gradients live in static memory that each replay overwrites, so the
    caller must not reset them to None between steps."""

    def __init__(self, graph, x, y, total, ce, grads):
        self.graph, self.x, self.y, self.total, self.ce, self.grads = graph, x, y, total, ce, grads

    def relink(self, model):
        """Point .grad back at the graph's static gradient buffers (after anything set them to None)."""
        for p, g in zip(model.parameters(), self.grads):
            p.grad = g

    def __call__(self, x, y):
        self.x.copy_(x, non_blocking=True)
        self.y.copy_(y, non_blocking=True)
        self.graph.replay()
        return self.total, self.ce


def capture_step(model, batches, device, autocast_ctx, kernel_ctx, warmup=3):
    """Record forward+backward as a CUDA graph if replay is bitwise identical to eager on every
    batch in `batches` (several, so frozen data-dependent paths or per-call Python state are
    caught). Returns (GraphedStep or None, reason). Leaves gradients set to None on failure."""
    if device.type != "cuda":
        return None, "not a CUDA device"

    def eager(x, y):
        model.zero_grad(set_to_none=True)
        with kernel_ctx(), autocast_ctx():
            total, ce = forward_loss(model, x, y)
        total.backward()
        grads = [None if p.grad is None else p.grad.clone() for p in model.parameters()]
        return total.detach().clone(), ce.detach().clone(), grads

    try:
        expected = [eager(x, y) for x, y in batches]
        sx, sy = batches[0][0].clone(), batches[0][1].clone()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(warmup):
                model.zero_grad(set_to_none=True)
                with kernel_ctx(), autocast_ctx():
                    total, _ = forward_loss(model, sx, sy)
                total.backward()
        torch.cuda.current_stream().wait_stream(side)
        model.zero_grad(set_to_none=True)
        # nothing may free pinned host memory or touch the device from Python GC during capture
        torch.cuda.synchronize(device)
        gc.collect()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, capture_error_mode="relaxed"):
            with kernel_ctx(), autocast_ctx():
                total, ce = forward_loss(model, sx, sy)
            total.backward()
        step = GraphedStep(graph, sx, sy, total, ce, [p.grad for p in model.parameters()])
        for (x, y), (e_total, e_ce, e_grads) in zip(batches, expected):
            g_total, g_ce = step(x, y)
            same = torch.equal(g_total, e_total) and torch.equal(g_ce, e_ce) and all(
                (eg is None and p.grad is None) or (eg is not None and p.grad is not None and torch.equal(p.grad, eg))
                for p, eg in zip(model.parameters(), e_grads))
            if not same:
                model.zero_grad(set_to_none=True)
                return None, "graph replay was not bitwise identical to eager execution"
        return step, "ok"
    except Exception as e:  # capture is not possible for every candidate (syncs, dynamic shapes, ...)
        torch.cuda.synchronize(device)
        model.zero_grad(set_to_none=True)
        return None, f"capture failed: {type(e).__name__}: {str(e).splitlines()[0][:200]}"


def select_attention_backend(model, x, y, device, autocast_ctx):
    """Choose a deterministic scaled-dot-product-attention kernel for this candidate.

    The default memory-efficient kernel has a nondeterministic backward pass on
    CUDA, which breaks replay. cuDNN attention is deterministic in our tests and
    nearly as fast; candidates whose attention calls it cannot serve fall back to
    the (slower) math kernel. Returns (name, context-manager factory).
    """
    if device.type != "cuda":
        return "default", nullcontext
    from torch.nn.attention import SDPBackend, sdpa_kernel

    def probe():
        model.zero_grad(set_to_none=True)
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            with autocast_ctx():
                total, _ = forward_loss(model, x[:2], y[:2])
            total.backward()
        model.zero_grad(set_to_none=True)

    try:
        probe()
        name = "cudnn"
    except RuntimeError:
        model.zero_grad(set_to_none=True)
        name = "math"
    return name, attention_context(name)


def attention_context(name):
    """Context-manager factory pinning the named SDPA kernel ('default' = no pinning)."""
    if name == "default":
        return nullcontext
    from torch.nn.attention import SDPBackend, sdpa_kernel

    backend = {"cudnn": SDPBackend.CUDNN_ATTENTION, "math": SDPBackend.MATH}[name]
    return lambda: sdpa_kernel(backend)


def train_candidate(model, task, recipe, budget: Budget, seeds: dict, device,
                    time_limit_s=None, remeasure_every=200, log=print, pacer=None, cuda_graph=True) -> dict:
    device = torch.device(device)
    pacer = pacer or Pacer(device, log=log)
    ctx = autocast_for(device)
    B, T = task.batch_size, task.context_length
    train = TokenStream(task.split_path("train"), T)
    dev = TokenStream(task.split_path("dev"), T)
    data_rng = np.random.default_rng(seeds["data"])
    probe_rng = np.random.default_rng(seeds["data"] + 1_000_003)  # keeps data order independent of probing

    model.to(device).train()
    opt = build_optimizer(model, recipe, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    px, py = (t.to(device) for t in train.random_batch(B, probe_rng))
    # Attention kernel and CUDA graph come before FLOP counting: capture fails after the counter has run.
    attention_backend, attention_ctx = select_attention_backend(model, px, py, device, ctx)
    graphed, graph_reason = None, "disabled"
    if cuda_graph:
        check = [(px, py)] + [tuple(t.to(device) for t in train.random_batch(B, probe_rng)) for _ in range(2)]
        graphed, graph_reason = capture_step(model, check, device, ctx, attention_ctx)
    # FLOPs are an accounting convention: always measured under the math kernel,
    # independent of the kernel pinned for execution.
    flops_per_step = measure_step_flops(model, px, py, ctx)
    if flops_per_step <= 0:
        raise RuntimeError("FLOP counter measured zero FLOPs; the model uses no counted operations")
    fwd_flops_per_token = measure_forward_flops(model, px, ctx) / (B * T)
    if graphed:
        graphed.relink(model)  # FLOP counting reset the gradients to None
    horizon_steps = max(1, math.ceil(budget.horizon_flops / flops_per_step))
    targets = [f * budget.horizon_flops for f in budget.checkpoint_fracs]
    log(f"  {flops_per_step / 1e9:.2f} GFLOP/step, horizon {horizon_steps} steps, "
        f"checkpoints at {[round(f, 4) for f in budget.checkpoint_fracs]}, attention kernel {attention_backend}, "
        f"CUDA graph {'on' if graphed else 'off (' + graph_reason + ')'}")

    flops_history = [[0, flops_per_step]]
    checkpoints, loss_trace = [], []
    cum_flops, step, tokens = 0.0, 0, 0
    status, next_ck = "ok", 0
    loss_sum, loss_n = torch.zeros((), device=device), 0
    eval_seconds = 0.0
    _sync(device)
    t0 = time.perf_counter()

    while next_ck < len(targets):
        lr = lr_at(step, horizon_steps, recipe)
        for group in opt.param_groups:
            group["lr"] = lr
        x, y = (to_device(t, device) for t in train.random_batch(B, data_rng))
        pacer.step_start()
        if graphed:
            total, ce = graphed(x, y)  # overwrites the static gradients
        else:
            with attention_ctx(), ctx():  # backward reuses the kernel chosen in forward
                total, ce = forward_loss(model, x, y)
            total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        if not graphed:
            opt.zero_grad(set_to_none=True)

        loss_sum += ce.detach()
        loss_n += 1
        step += 1
        tokens += B * T
        cum_flops += flops_per_step
        try:
            pacer.step_end(step)
        except Overheat as e:
            log(f"  {e}")
            status = "overheat"
            break

        if step % 25 == 0 or step >= horizon_steps:
            mean_loss = (loss_sum / loss_n).item()
            loss_sum.zero_()
            loss_n = 0
            if not math.isfinite(mean_loss):
                status = "diverged"
                break
            loss_trace.append([step, cum_flops, mean_loss])

        if remeasure_every and step % remeasure_every == 0 and step < horizon_steps:
            if not graphed:  # a replayed graph executes exactly the captured kernels: FLOPs are constant
                flops_per_step = measure_step_flops(model, px, py, ctx)
            flops_history.append([step, flops_per_step])

        while next_ck < len(targets) and (cum_flops >= targets[next_ck] * (1 - 1e-9) or step >= horizon_steps):
            _sync(device)
            train_seconds = time.perf_counter() - t0 - eval_seconds - pacer.paused_seconds
            e0 = time.perf_counter()
            dev_loss, n_eval = eval_loss(model, dev, B, budget.eval_tokens, device, ctx, attention_ctx)
            eval_seconds += time.perf_counter() - e0
            checkpoints.append(dict(frac=budget.checkpoint_fracs[next_ck], target_flops=targets[next_ck],
                                    flops=cum_flops, step=step, tokens=tokens, dev_loss=dev_loss,
                                    eval_tokens=n_eval, train_seconds=train_seconds))
            log(f"  checkpoint {budget.checkpoint_fracs[next_ck]:.4g}: step {step}, "
                f"{cum_flops:.3g} FLOPs, dev loss {dev_loss:.4f}, {train_seconds:.0f}s")
            next_ck += 1

        if time_limit_s is not None and time.perf_counter() - t0 > time_limit_s and next_ck < len(targets):
            status = "timeout"
            break

    _sync(device)
    elapsed = time.perf_counter() - t0
    costs = dict(
        train_flops=cum_flops,
        train_seconds=elapsed - eval_seconds - pacer.paused_seconds,
        eval_seconds=eval_seconds,
        pacing=pacer.report(),
        tokens_processed=tokens,
        steps=step,
        horizon_steps=horizon_steps,
        flops_history=flops_history,
        fwd_flops_per_token=fwd_flops_per_token,
        peak_mem_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        cuda_graph={"used": graphed is not None, "reason": graph_reason},
    )
    return dict(status=status, checkpoints=checkpoints, loss_trace=loss_trace, costs=costs,
                attention_backend=attention_backend)
