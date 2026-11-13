"""Seed candidate: Llama-style causal decoder (RoPE, RMSNorm, QK-norm, SwiGLU, tied embeddings).

Candidate contract
------------------
build_model(vocab_size, context_length) -> torch.nn.Module
forward(idx: LongTensor[B, T]) -> logits FloatTensor[B, T, vocab_size]
    or a dict {"logits": ..., "aux_loss": scalar tensor (optional)}.
T may be any length from 1 to context_length. Predictions at position t may
depend only on idx[:, :t+1] of the same sequence, and eval-mode forward passes
must not carry state between calls.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

CONFIG = dict(d_model=256, n_layers=6, n_heads=4, mlp_mult=8 / 3, rope_base=10000.0)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return xf.type_as(x) * self.weight


def rope_tables(length, head_dim, base):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.outer(torch.arange(length).float(), inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    # x: [B, H, T, hd]; rotate interleaved pairs.
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)
    return out.type_as(x)


class Attention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.q_norm = RMSNorm(self.head_dim)  # QK-norm: bounds attention logits for stable training
        self.k_norm = RMSNorm(self.head_dim)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class SwiGLU(nn.Module):
    def __init__(self, d_model, mult):
        super().__init__()
        hidden = int(64 * math.ceil(d_model * mult / 64))
        self.w12 = nn.Linear(d_model, 2 * hidden, bias=False)
        self.w3 = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x):
        a, b = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(a) * b)


class Block(nn.Module):
    def __init__(self, d_model, n_heads, mlp_mult):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = Attention(d_model, n_heads)
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, mlp_mult)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class Decoder(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, n_layers, n_heads, mlp_mult, rope_base):
        super().__init__()
        assert d_model % n_heads == 0 and (d_model // n_heads) % 2 == 0
        self.emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(Block(d_model, n_heads, mlp_mult) for _ in range(n_layers))
        self.norm = RMSNorm(d_model)
        cos, sin = rope_tables(context_length, d_model // n_heads, rope_base)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.apply(self._init_weights)
        resid_std = 0.02 / math.sqrt(2 * n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, std=resid_std)
            nn.init.normal_(block.mlp.w3.weight, std=resid_std)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, idx):
        T = idx.shape[1]
        cos, sin = self.cos[:T], self.sin[:T]
        x = self.emb(idx)
        for block in self.blocks:
            x = block(x, cos, sin)
        return F.linear(self.norm(x), self.emb.weight)  # tied output head


def build_model(vocab_size, context_length):
    return Decoder(vocab_size, context_length, **CONFIG)
