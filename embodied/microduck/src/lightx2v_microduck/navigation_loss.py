"""Explicit token-prefix or action-axis objectives for FastWAM experiments.

The joint forward, noise schedules and video objective follow pinned FastWAM
7faa71108368. Token-prefix and action-axis weighting are separate opt-in recipes.
This objective must be selected explicitly; the author's default is unchanged.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path


def objective_manifest(config=None, *, axes=None):
    if axes is not None:
        if config is not None:
            raise ValueError("axis and prefix objectives must be separate experiments")
        from .navigation_axis_loss import axis_objective_manifest

        return axis_objective_manifest(axes)
    if not config:
        return {"kind": "author_uniform"}
    if set(config) != {"steps", "weight"}:
        raise ValueError("action prefix objective requires exactly steps and weight")
    steps, weight = int(config["steps"]), float(config["weight"])
    if steps != config["steps"] or not 1 <= steps <= 32 or not math.isfinite(weight) or weight < 1:
        raise ValueError("prefix steps must be 1..32 and weight finite and at least one")
    return {"kind": "action_prefix", "steps": steps, "weight": weight,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "upstream_commit": "7faa71108368fbb3b6885649f112af607427a2d4",
            "video_objective_unchanged": True}


def reduce_action_tokens(token_loss, padding, *, steps, weight):
    import torch

    if token_loss.ndim != 2 or not 1 <= steps <= token_loss.shape[1] or not math.isfinite(weight) or weight < 1:
        raise ValueError("invalid prefix objective for action sequence")
    weights = torch.ones_like(token_loss)
    weights[:, :steps] = weight
    if padding is not None:
        if padding.shape != token_loss.shape or padding.dtype != torch.bool:
            raise ValueError("padding must be a boolean action-token mask")
        weights *= (~padding).to(device=weights.device, dtype=weights.dtype)
    return (token_loss * weights).sum(1) / weights.sum(1).clamp(min=1.)


def prefix_training_loss(model, sample, *, steps, weight, tiled=False, axis_weights=None):
    import torch
    from .navigation_axis_loss import action_token_mse, validate_axis_weights

    if axis_weights is not None:
        validate_axis_weights(axis_weights)
        if steps != 1 or weight != 1.:
            raise ValueError("axis weighting requires uniform valid-token supervision")

    inputs = model.build_inputs(sample, tiled=tiled)
    video, action = inputs["input_latents"], inputs["action"]
    noise_video = torch.randn_like(video)
    video_t = model.train_video_scheduler.sample_training_t(video.shape[0], model.device, video.dtype)
    noisy_video = model.train_video_scheduler.add_noise(video, noise_video, video_t)
    target_video = model.train_video_scheduler.training_target(video, noise_video, video_t)
    if inputs["first_frame_latents"] is not None:
        noisy_video[:, :, :1] = inputs["first_frame_latents"]
    noise_action = torch.randn_like(action)
    action_t = model.train_action_scheduler.sample_training_t(video.shape[0], model.device, action.dtype)
    noisy_action = model.train_action_scheduler.add_noise(action, noise_action, action_t)
    target_action = model.train_action_scheduler.training_target(action, noise_action, action_t)
    patch_t, patch_h, patch_w = map(int, model.video_expert.patch_size)
    t, h, w = noisy_video.shape[-3:]
    per_frame = (h // patch_h) * (w // patch_w)
    mask = model._build_mot_attention_mask(video_seq_len=(t // patch_t) * per_frame,
        action_seq_len=noisy_action.shape[1], video_tokens_per_frame=per_frame, device=noisy_video.device)
    pred_video, pred_action = model._joint_denoise_core(latents_video=noisy_video,
        latents_action=noisy_action, timestep_video=video_t, timestep_action=action_t,
        context=inputs["context"], context_mask=inputs["context_mask"], attention_mask=mask,
        fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"], action_condition=action)
    include_first = inputs["first_frame_latents"] is None
    if not include_first:
        pred_video, target_video = pred_video[:, :, 1:], target_video[:, :, 1:]
    video_per_sample = model._compute_video_loss_per_sample(pred_video=pred_video, target_video=target_video,
        image_is_pad=inputs["image_is_pad"], include_initial_video_step=include_first)
    video_weight = model.train_video_scheduler.training_weight(video_t).to(video_per_sample)
    video_loss = (video_per_sample * video_weight).mean()
    token_loss = action_token_mse(pred_action, target_action, axis_weights)
    action_per_sample = reduce_action_tokens(token_loss, inputs["action_is_pad"], steps=steps, weight=weight)
    action_weight = model.train_action_scheduler.training_weight(action_t).to(action_per_sample)
    action_loss = (action_per_sample * action_weight).mean()
    total = model.loss_lambda_video * video_loss + model.loss_lambda_action * action_loss
    return total, {"loss_video": model.loss_lambda_video * float(video_loss.detach()),
                   "loss_action": model.loss_lambda_action * float(action_loss.detach())}
