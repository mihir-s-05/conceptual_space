"""Parameter and FLOP accounting.

FLOPs are measured by executing the model under torch's FlopCounterMode, so
recurrent or adaptive computation is counted as actually executed. Only ops
with registered FLOP formulas (matmuls, attention, convolutions) are counted;
elementwise work is ignored, as in standard 6ND-style accounting.
"""

import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

from .candidate import forward_loss, split_output


def count_params(model: nn.Module) -> dict:
    seen = set()
    total = embedding = 0
    for module in model.modules():
        for p in module.parameters(recurse=False):
            if id(p) in seen:
                continue
            seen.add(id(p))
            total += p.numel()
            if isinstance(module, nn.Embedding):
                embedding += p.numel()
    return {"total": total, "embedding": embedding, "nonembedding": total - embedding}


def _counting_kernel():
    """Count attention under the math SDPA kernel on every device. Fused kernels have
    device-specific (or missing) FLOP formulas: the CPU flash kernel is not counted at all,
    which made CPU budgets about 30% larger than GPU budgets. The math kernel decomposes
    into matmuls that are counted identically everywhere."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    return sdpa_kernel(SDPBackend.MATH)


def measure_step_flops(model, x, y, autocast_ctx) -> int:
    """FLOPs of one forward+backward pass on (x, y). Leaves no gradients behind."""
    model.zero_grad(set_to_none=True)
    with FlopCounterMode(display=False) as counter:
        with _counting_kernel(), autocast_ctx():
            total, _ = forward_loss(model, x, y)
        total.backward()
    model.zero_grad(set_to_none=True)
    return int(counter.get_total_flops())


@torch.no_grad()
def measure_forward_flops(model, x, autocast_ctx) -> int:
    was_training = model.training
    model.eval()
    try:
        with FlopCounterMode(display=False) as counter:
            with _counting_kernel(), autocast_ctx():
                split_output(model(x))
    finally:
        model.train(was_training)
    return int(counter.get_total_flops())
