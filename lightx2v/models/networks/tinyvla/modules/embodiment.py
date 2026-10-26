# Copyright AlexWortega and LightX2V contributors. SPDX-License-Identifier: Apache-2.0
# Adapted from AlexWortega/tinyvla @ 603f5045b45126af011418befc899e3c67d04346.
"""Cross-embodiment harmonization: pad-to-max vectors + embodiment embedding."""

from __future__ import annotations

import torch
import torch.nn as nn


def pad_vector(x: torch.Tensor, target_dim: int) -> torch.Tensor:
    """Zero-pad the last dim of (..., D) to (..., target_dim)."""
    if x.shape[-1] == target_dim:
        return x
    if x.shape[-1] > target_dim:
        raise ValueError(f"dim {x.shape[-1]} exceeds max {target_dim}")
    pad = x.new_zeros(*x.shape[:-1], target_dim - x.shape[-1])
    return torch.cat([x, pad], dim=-1)


def dim_validity_mask(orig_dim: int, target_dim: int, ref: torch.Tensor) -> torch.Tensor:
    """(target_dim,) bool mask, True for real dims, False for padding."""
    mask = torch.zeros(target_dim, dtype=torch.bool, device=ref.device)
    mask[:orig_dim] = True
    return mask


class EmbodimentEmbedding(nn.Module):
    """One learned token per source dataset / robot embodiment (variants A, B)."""

    def __init__(self, num_embodiments: int, dim: int):
        super().__init__()
        self.emb = nn.Embedding(num_embodiments, dim)

    def forward(self, embodiment_id: torch.Tensor) -> torch.Tensor:
        """(B,) int -> (B, 1, dim)."""
        return self.emb(embodiment_id)[:, None, :]


# Descriptor fields (hand-normalized to O(1)), order fixed — see configs/morphology/*.yaml
MORPH_FIELDS = [
    "arm_dof",  # /10
    "reach_m",  # meters ×2
    "gripper_width_m",  # meters ×10
    "num_cameras",  # /3
    "is_mobile",  # ±1
    "control_hz",  # /30
    "joint_lo_mean",  # rad, /pi
    "joint_hi_mean",  # rad, /pi
    "workspace_x",  # meters ×2
    "workspace_y",
    "workspace_z",
    "payload_kg",  # /5
    "ee_type_parallel",  # ±1 (parallel jaw)
    "ee_type_multi",  # ±1 (>2 finger)
    "base_holonomic",  # ±1
    "reserved",
]
MORPH_DIM = len(MORPH_FIELDS)  # 16


class MorphologyEncoder(nn.Module):
    """Explicit robot descriptor -> a few conditioning tokens (variant C).

    Cheap alternative to a per-robot learned ID: a fixed 16-number descriptor
    (arm DOF, reach, gripper width, mobility, control rate, joint limits, ...)
    through an MLP -> `n_tokens` morphology tokens. Gives a shot at unseen-robot
    transfer (a new robot = write its descriptor, no new learned parameters).
    """

    def __init__(self, dim: int, n_tokens: int = 2, in_dim: int = MORPH_DIM):
        super().__init__()
        self.n_tokens = n_tokens
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim * n_tokens),
        )

    def forward(self, descriptor: torch.Tensor) -> torch.Tensor:
        """(B, in_dim) float -> (B, n_tokens, dim)."""
        b = descriptor.shape[0]
        return self.mlp(descriptor).view(b, self.n_tokens, -1)


class DemoConditioner(nn.Module):
    """In-context few-shot demo conditioning: K (observation, action-chunk) exemplar
    pairs -> conditioning tokens, no gradient update on the new robot.

    v3 (high-capacity): v1/v2 squeezed each exemplar through mean-pool(64 spatial
    tokens) + a 3-point action summary into ONE token via a 2-layer MLP — a
    bottleneck so tight the model learned to ignore the channel (real ≈ garbage
    exemplars). This version widens every stage:
      - vision: `n_visual_groups` spatially-grouped pools instead of one global mean
      - action: `n_action_keys` evenly-spaced chunk keyframes instead of 3 points
      - MLP: 3 layers at `hidden_mult`x width, LayerNorm, -> `n_tokens_per_example` tokens
    """

    def __init__(self, dim: int, visual_dim: int, action_dim: int, n_tokens_per_example: int = 4, n_visual_groups: int = 4, n_action_keys: int = 8, hidden_mult: int = 4):
        super().__init__()
        self.n_tokens_per_example = n_tokens_per_example
        self.n_visual_groups = n_visual_groups
        self.n_action_keys = n_action_keys
        in_dim = visual_dim * n_visual_groups + action_dim * n_action_keys
        hidden = dim * hidden_mult
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim * n_tokens_per_example),
        )

    def forward(self, obs_tokens: torch.Tensor, action_chunks: torch.Tensor) -> torch.Tensor:
        """obs_tokens: (B, K, N, visual_dim) per-exemplar spatial tokens (N=64), or
        (B, K, visual_dim) if already pooled (falls back to repeating it).
        action_chunks: (B, K, T, action_dim).
        Returns (B, K*n_tokens_per_example, dim).
        """
        b, k, t, a = action_chunks.shape
        if obs_tokens.dim() == 4:  # (B, K, N, C) -> spatial group pools
            n = obs_tokens.shape[2]
            g = self.n_visual_groups
            per = max(1, n // g)
            groups = [obs_tokens[:, :, i * per : (i + 1) * per].mean(dim=2) for i in range(g)]
            vis = torch.cat(groups, dim=-1)  # (B, K, C*g)
        else:
            vis = obs_tokens.repeat(1, 1, self.n_visual_groups)

        keys = torch.linspace(0, t - 1, self.n_action_keys, device=action_chunks.device).long()
        act = action_chunks[:, :, keys].reshape(b, k, -1)  # (B, K, n_keys*a)

        feat = torch.cat([vis, act], dim=-1)
        tok = self.mlp(feat)
        return tok.view(b, k * self.n_tokens_per_example, -1)
