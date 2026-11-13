"""Mechanism probes: cheap, evaluation-only diagnostics on a trained model.

copy:           in-context copying. A random token sequence r of length d is shown
                twice; the first copy is unpredictable, the second can be predicted
                by retrieving what followed each token earlier (induction). Reported
                per delay d: loss on each half, copy_gain = first - second, and
                argmax accuracy on the second half. Tests sequence memory and recall.
position_loss:  dev loss averaged within context-position buckets. Falling loss at
                later positions means the model uses longer context.

Probes are diagnostics, not definitions of a mechanism or of usefulness.
"""

import torch
import torch.nn.functional as F

from .candidate import split_output

COPY_DELAYS = (8, 32, 128, 256)
POSITION_EDGES = (0, 16, 64, 256, 512, 1024, 2048)


@torch.no_grad()
def copy_probe(model, vocab_size, eot_id, context_length, device, autocast_ctx, kernel_ctx,
               n_seqs=32, seed=1234):
    gen = torch.Generator().manual_seed(seed)
    delays = sorted({min(d, context_length // 2) for d in COPY_DELAYS if min(d, context_length // 2) >= 2})
    out = []
    for d in delays:
        r = torch.randint(0, vocab_size - 1, (n_seqs, d), generator=gen)
        r = r + (r >= eot_id).long()  # never the end-of-text token
        seq = torch.cat([r, r], dim=1).to(device)
        x, y = seq[:, :-1], seq[:, 1:]
        with kernel_ctx(), autocast_ctx():
            logits, _ = split_output(model(x))
        logits = logits.float()
        nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none").view(n_seqs, -1)
        first, second = nll[:, : d - 1], nll[:, d:]
        acc = (logits[:, d:].argmax(-1) == y[:, d:]).float().mean().item()
        out.append({"delay": d, "first_half_loss": round(first.mean().item(), 4),
                    "second_half_loss": round(second.mean().item(), 4),
                    "copy_gain": round((first.mean() - second.mean()).item(), 4),
                    "copy_accuracy": round(acc, 4)})
    return out


@torch.no_grad()
def position_loss(model, stream, batch_size, max_tokens, device, autocast_ctx, kernel_ctx):
    T = stream.T
    edges = [e for e in POSITION_EDGES if e < T] + [T]
    sums = torch.zeros(len(edges) - 1, dtype=torch.float64)
    counts = torch.zeros(len(edges) - 1, dtype=torch.float64)
    for x, y in stream.eval_batches(batch_size, max_tokens):
        x, y = x.to(device), y.to(device)
        with kernel_ctx(), autocast_ctx():
            logits, _ = split_output(model(x))
        nll = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1),
                              reduction="none").view(x.shape).double().cpu()
        for i in range(len(edges) - 1):
            block = nll[:, edges[i]:edges[i + 1]]
            sums[i] += block.sum()
            counts[i] += block.numel()
    return [{"positions": f"{edges[i]}-{edges[i + 1] - 1}", "loss": round((sums[i] / counts[i]).item(), 4)}
            for i in range(len(edges) - 1) if counts[i] > 0]


def run_probes(model, task, dev_stream, device, autocast_ctx, kernel_ctx, max_tokens=1 << 18) -> dict:
    was_training = model.training
    model.eval()
    try:
        return {
            "copy": copy_probe(model, task.vocab_size, task.eot_id, task.context_length, device,
                               autocast_ctx, kernel_ctx),
            "position_loss": position_loss(model, dev_stream, task.batch_size, max_tokens, device,
                                           autocast_ctx, kernel_ctx),
        }
    finally:
        model.train(was_training)
