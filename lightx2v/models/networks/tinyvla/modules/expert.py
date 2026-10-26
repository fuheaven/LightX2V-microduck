# Copyright AlexWortega and LightX2V contributors. SPDX-License-Identifier: Apache-2.0
# Adapted from AlexWortega/tinyvla @ 603f5045b45126af011418befc899e3c67d04346.
"""Flow-matching action expert.

A small transformer over noisy action tokens with interleaved
cross-attention (to the conditioning set) and self-attention blocks,
following the SmolVLA expert pattern. Predicts the flow velocity.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sincos_time_embedding(t: torch.Tensor, dim: int, min_period: float = 4e-3, max_period: float = 4.0) -> torch.Tensor:
    """t: (B,) in [0, 1] -> (B, dim) sine-cosine embedding."""
    half = dim // 2
    freqs = torch.exp(torch.linspace(math.log(min_period), math.log(max_period), half, device=t.device))
    angles = 2 * math.pi * t[:, None] / freqs[None, :]
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


class _MHA(nn.Module):
    """Drop-in replacement for torch's batch-first multi-head attention, routed
    through F.scaled_dot_product_attention (flash / mem-efficient kernels, and
    far friendlier to torch.compile than the eager fallback).

    Parameter names are kept identical (in_proj_weight / in_proj_bias /
    out_proj.weight / out_proj.bias) so existing checkpoints load unchanged.
    `key_padding_mask` keeps torch semantics: True = ignore that key.
    """

    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        assert dim % n_heads == 0
        self.dim, self.n_heads, self.head_dim = dim, n_heads, dim // n_heads
        self.in_proj_weight = nn.Parameter(torch.empty(3 * dim, dim))
        self.in_proj_bias = nn.Parameter(torch.zeros(3 * dim))
        self.out_proj = nn.Linear(dim, dim)
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=False):
        b, lq, d = query.shape
        w_q, w_k, w_v = self.in_proj_weight.split(d, dim=0)
        b_q, b_k, b_v = self.in_proj_bias.split(d, dim=0)
        q = F.linear(query, w_q, b_q)
        k = F.linear(key, w_k, b_k)
        v = F.linear(value, w_v, b_v)

        def split(t):
            return t.view(b, -1, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = split(q), split(k), split(v)
        attn_mask = None
        if key_padding_mask is not None:
            # torch convention: True = ignore. SDPA bool mask: True = attend.
            attn_mask = (~key_padding_mask)[:, None, None, :]
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        o = o.transpose(1, 2).reshape(b, lq, d)
        return self.out_proj(o), None


class _MLP(nn.Module):
    def __init__(self, dim: int, ratio: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * ratio),
            nn.GELU(),
            nn.Linear(dim * ratio, dim),
        )

    def forward(self, x):
        return self.net(x)


class SelfAttnBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _MHA(dim, n_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _MLP(dim)

    def forward(self, x, cond, cond_mask):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class CrossAttnBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm_cond = nn.LayerNorm(dim)
        self.attn = _MHA(dim, n_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _MLP(dim)

    def forward(self, x, cond, cond_mask):
        h = self.norm1(x)
        c = self.norm_cond(cond)
        # key_padding_mask: True = ignore
        kpm = ~cond_mask if cond_mask is not None else None
        h, _ = self.attn(h, c, c, key_padding_mask=kpm, need_weights=False)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class FlowMatchingExpert(nn.Module):
    def __init__(
        self,
        action_dim: int = 8,
        chunk_size: int = 50,
        d_model: int = 512,
        n_layers: int = 12,
        n_heads: int = 8,
        min_period: float = 4e-3,
        max_period: float = 4.0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.d_model = d_model
        self.min_period = min_period
        self.max_period = max_period

        self.action_in = nn.Linear(action_dim, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, chunk_size, d_model))
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        # even blocks cross-attend to the conditioning set, odd blocks self-attend
        self.blocks = nn.ModuleList([(CrossAttnBlock if i % 2 == 0 else SelfAttnBlock)(d_model, n_heads) for i in range(n_layers)])
        self.norm_out = nn.LayerNorm(d_model)
        self.action_out = nn.Linear(d_model, action_dim)

        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        nn.init.zeros_(self.action_out.weight)
        nn.init.zeros_(self.action_out.bias)

    def forward(
        self,
        noisy_actions: torch.Tensor,  # (B, T, A)
        time: torch.Tensor,  # (B,) in [0, 1]
        cond: torch.Tensor,  # (B, N, d_model)
        cond_mask: torch.Tensor | None = None,  # (B, N) True = valid
    ) -> torch.Tensor:
        x = self.action_in(noisy_actions) + self.pos_emb[:, : noisy_actions.shape[1]]
        t_emb = self.time_mlp(sincos_time_embedding(time, self.d_model, self.min_period, self.max_period))
        x = x + t_emb[:, None, :]
        for block in self.blocks:
            x = block(x, cond, cond_mask)
        return self.action_out(self.norm_out(x))
