# Copyright AlexWortega and LightX2V contributors. SPDX-License-Identifier: Apache-2.0
"""TinyVLA policy config (LeRobot plugin, --policy.type=tinyvla)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class FeatureType(str, Enum):
    VISUAL = "VISUAL"
    STATE = "STATE"
    ACTION = "ACTION"


class NormalizationMode(str, Enum):
    IDENTITY = "IDENTITY"
    MEAN_STD = "MEAN_STD"


@dataclass
class PolicyFeature:
    type: FeatureType
    shape: tuple[int, ...]


@dataclass
class TinyVLAConfig:
    input_features: dict = field(
        default_factory=lambda: {
            "observation.images.cam0": PolicyFeature(FeatureType.VISUAL, (3, 256, 256)),
            "observation.images.cam1": PolicyFeature(FeatureType.VISUAL, (3, 256, 256)),
            "observation.state": PolicyFeature(FeatureType.STATE, (256,)),
        }
    )
    output_features: dict = field(default_factory=lambda: {"action": PolicyFeature(FeatureType.ACTION, (64,))})
    backbone_revision: str | None = None
    vision_backend: str = "fixed_eager"
    # Legacy is kept for reference checkpoint reproduction. Project models
    # explicitly opt into batch-independent Qwen linear-attention padding.
    semantic_padding_policy: str = "legacy"
    # Input / output structure
    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Cross-embodiment padding
    max_state_dim: int = 256
    max_action_dim: int = 64
    num_embodiments: int = 16
    max_cameras: int = 3

    # TinyVLA-2 A/B/C variant switch
    #   conditioning "id"    -> learned embodiment embedding (variants A, B)
    #   conditioning "morph" -> descriptor MLP -> morphology tokens (variant C)
    conditioning: str = "id"  # "id" | "morph" (MLP over numeric fields) | "morph_qwen" (text via shared Qwen)
    morph_tokens: int = 2
    morph_to_slow: bool = False  # also inject morphology into the slow Qwen prefix
    num_morph_readout: int = 4  # for morph_qwen: readout slots on the robot-text encode
    morph_text_max_len: int = 32
    # in-context demo conditioning: K (obs, action) exemplar pairs, no gradient
    # updates for the new robot at deployment. Composable with morph/morph_qwen.
    use_demo_conditioning: bool = False
    n_support: int = 3
    # sample demo exemplars from a DIFFERENT task of the same robot, so the
    # exemplar carries robot-dynamics info the identity channels don't duplicate
    support_other_task: bool = False
    # demo-encoder capacity (v3): wider path so exemplar content isn't destroyed
    demo_tokens_per_example: int = 4
    demo_visual_groups: int = 4
    demo_action_keys: int = 8
    demo_hidden_mult: int = 4
    # Rich-slow mode: demos + robot text + instruction + image all go into the LM
    # sequence (so demos live in the ATTENTION STREAM — the fix for v1-v3's
    # "placed alongside → ignored" failure), and the LM emits latent tokens that
    # condition the fast path. DUAL-RATE IS PRESERVED: the fast path still has its
    # own spatial tokens + proprio + expert, and the LM latents are cacheable
    # exactly like the old semantic latent.
    vlm_native: bool = False
    num_action_readout: int = 16
    demo_action_keys_lm: int = 4
    # DEMO-ONLY: strip every other robot-identity channel (embodiment-ID token,
    # morphology descriptor/text, and the per-source normalization that leaks
    # identity) so the demo exemplars are the ONLY way to know which robot this
    # is. Four prior attempts failed because identity was available elsewhere and
    # the gradient had no reason to read the demos.
    demo_only: bool = False
    # action_space "native" (A) vs "canonical" (B, C: base-frame EE deltas)
    action_space: str = "native"

    # Vision (shared Qwen3.5 vision tower for both paths)
    image_size: int = 256
    freeze_vision_encoder: bool = True  # Stage 1; unfreeze at 0.1x lr in Stage 2

    # Slow path (Qwen3.5 VLM with native vision tower)
    lm_model_name: str = "Qwen/Qwen3.5-0.8B"
    lm_num_layers: int = 12  # L/2 layer skip (24-layer language stack)
    num_readout_tokens: int = 8
    freeze_lm: bool = True
    # Positive N overrides freeze_lm with "only the final N LM blocks train".
    # Zero preserves the author's all-frozen/all-unfrozen boolean behavior.
    lm_unfreeze_last_n: int = 0
    tokenizer_max_length: int = 48
    pad_language_to: str = "longest"

    # Fast path (action expert)
    expert_dim: int = 512
    expert_layers: int = 12
    expert_heads: int = 8

    # Ablation switches (Stage-2 gate baselines)
    use_semantic_latent: bool = True
    use_spatial_tokens: bool = True

    # Spatial distillation aux task (Stage 3, XS-VLA style)
    spatial_vocab: int = 1024  # 32x32 grid cells
    spatial_loss_weight: float = 0.0  # 0 disables the aux head/loss

    # Flow matching
    num_denoise_steps: int = 10
    flow_beta_alpha: float = 1.5
    flow_beta_beta: float = 1.0
    min_period: float = 4e-3
    max_period: float = 4.0

    # Staleness augmentation (Stage 2b): probability of conditioning on a latent
    # computed from an earlier frame. Consumed by the training loop, not here.
    staleness_prob: float = 0.0
    staleness_max_s: float = 2.0

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 10.0
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 40_000
    scheduler_decay_lr: float = 2.5e-6

    def __post_init__(self):
        for attr in ("input_features", "output_features"):
            converted = {}
            for key, value in getattr(self, attr).items():
                converted[key] = value if isinstance(value, PolicyFeature) else PolicyFeature(FeatureType(value["type"]), tuple(value["shape"]))
            setattr(self, attr, converted)
        if self.n_action_steps > self.chunk_size or self.n_action_steps <= 0:
            raise ValueError("n_action_steps must be in [1, chunk_size]")
        if self.num_denoise_steps <= 0 or self.expert_dim % self.expert_heads:
            raise ValueError("invalid denoise steps or expert head dimensions")
        if self.vision_backend not in ("hf", "fixed_eager", "fixed_compile"):
            raise ValueError("unsupported vision backend")
        if self.semantic_padding_policy not in ("legacy", "batch_consistent_v1"):
            raise ValueError("unsupported semantic_padding_policy")
        if isinstance(self.lm_unfreeze_last_n, bool) or not isinstance(self.lm_unfreeze_last_n, int) or not 0 <= self.lm_unfreeze_last_n <= self.lm_num_layers:
            raise ValueError("lm_unfreeze_last_n must be between zero and lm_num_layers")
        image_keys = [k for k, v in self.input_features.items() if v.type == FeatureType.VISUAL]
        if not image_keys or len(image_keys) > self.max_cameras:
            raise ValueError("invalid number of image features")
        if self.output_features["action"].shape[0] > self.max_action_dim:
            raise ValueError("action feature exceeds padding width")

    @classmethod
    def from_dict(cls, payload):
        from dataclasses import fields

        names = {field.name for field in fields(cls)}
        return cls(**{k: v for k, v in payload.items() if k in names})

    @classmethod
    def from_pretrained(cls, directory):
        return cls.from_dict(json.loads((Path(directory) / "config.json").read_text()))

    def to_dict(self):
        from dataclasses import asdict

        return {"type": "tinyvla", **asdict(self)}
