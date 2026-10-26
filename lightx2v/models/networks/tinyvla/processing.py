"""Explicit MicroDuck preprocessing; no environment or ROS dependencies."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .modules.embodiment import MORPH_FIELDS

DUCK_TEXT = "Robot: MicroDuck, a tiny bipedal duck robot with two legs, a neck and a head, 14 joints."
# Compatibility descriptor from the pinned author's eval_microduck.py. These
# are training features, not a claim about calibrated hardware dimensions.
DUCK_MORPH = dict(
    arm_dof=2,
    reach_m=0.08,
    gripper_width_m=0.0,
    num_cameras=2,
    is_mobile=1,
    control_hz=50,
    joint_lo_mean=-1.5,
    joint_hi_mean=1.5,
    workspace_x=0.1,
    workspace_y=0.1,
    workspace_z=0.15,
    payload_kg=0.05,
    ee_type_parallel=-1,
    ee_type_multi=-1,
    base_holonomic=-1,
    reserved=0,
)
MORPH_SCALE = dict(
    arm_dof=0.1, reach_m=2, gripper_width_m=10, num_cameras=1 / 3, control_hz=1 / 30, joint_lo_mean=1 / 3.1416, joint_hi_mean=1 / 3.1416, workspace_x=2, workspace_y=2, workspace_z=2, payload_kg=0.2
)


class MicroDuckNormalizer:
    def __init__(self, payload):
        stats = payload.get("norm_stats", payload)
        for name, size in (("state", 61), ("action", 14)):
            mean = np.asarray(stats[name]["mean"], dtype=np.float32)
            std = np.asarray(stats[name]["std"], dtype=np.float32)
            if mean.shape != (size,) or std.shape != (size,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std < 0).any():
                raise ValueError(f"invalid {name} normalization: expected finite {size}D mean/std")
            setattr(self, f"{name}_mean", mean)
            setattr(self, f"{name}_std", np.maximum(std, 1e-6))

    @classmethod
    def from_file(cls, path):
        return cls(json.loads(Path(path).read_text()))

    def normalize_state(self, state):
        return (np.asarray(state, np.float32) - self.state_mean) / self.state_std

    def denormalize_action(self, action):
        return np.asarray(action, np.float32) * self.action_std + self.action_mean


class TinyVLAPreprocessor:
    def __init__(self, config, tokenizer, *, normalization=None, device="cpu", support=None):
        self.config, self.tokenizer, self.device = config, tokenizer, torch.device(device)
        self.normalization = normalization if isinstance(normalization, MicroDuckNormalizer) or normalization is None else MicroDuckNormalizer(normalization)
        self.support = support

    @classmethod
    def from_config(cls, config, **kwargs):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(config.lm_model_name, revision=config.backbone_revision)
        return cls(config, tokenizer, **kwargs)

    def prepare(self, image_rgb, state61, instruction, *, normalized=False):
        image = np.asarray(image_rgb)
        if image.dtype != np.uint8 or image.shape != (self.config.image_size, self.config.image_size, 3):
            raise ValueError(f"image must be uint8 RGB {self.config.image_size}x{self.config.image_size}")
        state = np.asarray(state61, np.float32)
        if state.shape != (61,) or not np.isfinite(state).all():
            raise ValueError("state must be a finite raw or normalized 61D vector")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be nonempty text")
        if not normalized:
            if self.normalization is None:
                raise ValueError("raw inference requires the checkpoint's verified normalization")
            state = self.normalization.normalize_state(state)
        if self.config.max_state_dim < 61 or self.config.max_action_dim < 14:
            raise ValueError("this configuration cannot represent MicroDuck without truncation")
        state = np.pad(state, (0, self.config.max_state_dim - 61))
        image_tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float().div(255)[None]
        text = self.tokenizer([instruction], padding="max_length", truncation=True, max_length=self.config.tokenizer_max_length, return_tensors="pt")
        robot = self.tokenizer([DUCK_TEXT], padding="max_length", truncation=True, max_length=self.config.morph_text_max_len, return_tensors="pt")
        batch = {
            "observation.images.cam0": image_tensor,
            "observation.images.cam1": torch.zeros_like(image_tensor),
            "observation.state": torch.from_numpy(state)[None],
            "observation.language.tokens": text["input_ids"],
            "observation.language.attention_mask": text["attention_mask"].bool(),
            "morph_text_ids": robot["input_ids"],
            "morph_text_mask": robot["attention_mask"].bool(),
            "morphology": torch.tensor([[DUCK_MORPH[k] * MORPH_SCALE.get(k, 1) for k in MORPH_FIELDS]], dtype=torch.float32),
            "embodiment_id": torch.zeros(1, dtype=torch.long),
        }
        if self.support is not None:
            images = torch.as_tensor(self.support["images"], dtype=torch.float32)
            actions = torch.as_tensor(self.support["actions"], dtype=torch.float32)
            if images.ndim != 4 or images.shape[0] < 1 or images.shape[1:] != (3, self.config.image_size, self.config.image_size):
                raise ValueError("support images must have shape K,3,H,W in [0,1] with K >= 1")
            if actions.shape != (images.shape[0], self.config.chunk_size, self.config.max_action_dim):
                raise ValueError("support actions must be normalized padded K,H,A chunks")
            if not torch.isfinite(images).all() or not torch.isfinite(actions).all() or (images < 0).any() or (images > 1).any():
                raise ValueError("nonfinite or out-of-range support examples")
            batch["support_images"] = images[None]
            batch["support_actions"] = actions[None]
        return {key: value.to(self.device) for key, value in batch.items()}
