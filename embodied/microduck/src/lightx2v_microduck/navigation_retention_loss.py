"""Experimental frozen-parent action distillation for navigation rehearsal.

The supervised objective defaults to the pinned author's uniform flow objective;
an explicit axis experiment can reweight action supervision. An explicit mask
adds an unchanged action-velocity constraint only on retained
training items. It does not provide a teacher at inference, change action bounds,
or establish closed-loop retention. No existing training recipe enables this.
"""

import hashlib
import math
from pathlib import Path

from .navigation_loss import reduce_action_tokens


def retention_manifest(config=None):
    if config is None:
        return {"kind": "disabled"}
    expected = {"teacher_weights_sha256", "teacher_receipt_sha256", "weight", "apply_to"}
    if not isinstance(config, dict) or set(config) != expected:
        raise ValueError("retention requires a pinned teacher, weight and explicit item scope")
    for key in ("teacher_weights_sha256", "teacher_receipt_sha256"):
        value = config[key]
        if (not isinstance(value, str) or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)):
            raise ValueError("invalid retention teacher digest")
    weight = config["weight"]
    if (type(weight) not in (int, float) or not math.isfinite(weight) or weight <= 0
            or config["apply_to"] != "retained_training_items"):
        raise ValueError("invalid retention weight or item scope")
    return dict(kind="frozen_parent_action_flow", **config,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                upstream_commit="7faa71108368fbb3b6885649f112af607427a2d4",
                supervised_objective="author_uniform", inference_teacher_required=False)


def _mask(mask, batch_size, device):
    import torch

    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.shape != (batch_size,):
        raise ValueError("retention mask must explicitly identify each batch item")
    return mask.to(device=device)


def retention_penalty(student, teacher, padding, retained, timestep_weight):
    """Average valid action-velocity MSE over eligible items only.

    New demonstrations and padded horizon tokens cannot supply a retention
    gradient. The teacher is always detached, even in a standalone diagnostic.
    """
    import torch
    import torch.nn.functional as functional

    if (student.ndim != 3 or student.shape != teacher.shape or not all(student.shape)
            or student.device != teacher.device):
        raise ValueError("paired action-velocity tensors differ")
    retained = _mask(retained, student.shape[0], student.device)
    # The pinned continuous scheduler returns a scalar for batch size one.
    # Preserve the author's scalar/batch broadcasting instead of rejecting it.
    if (not isinstance(timestep_weight, torch.Tensor)
            or timestep_weight.shape not in ((), (student.shape[0],))
            or not torch.isfinite(timestep_weight).all()
            or (timestep_weight < 0).any()):
        raise ValueError("invalid action-timestep weights")
    if not torch.isfinite(student).all() or not torch.isfinite(teacher).all():
        raise ValueError("nonfinite paired action velocity")
    token_loss = functional.mse_loss(student.float(), teacher.detach().float(), reduction="none").mean(2)
    valid_loss = reduce_action_tokens(token_loss, padding, steps=1, weight=1.)
    if padding is not None and bool((retained & padding.to(retained.device).all(1)).any()):
        raise ValueError("a retained item has no valid action tokens")
    eligible = retained.to(valid_loss.dtype)
    weights = timestep_weight.to(valid_loss).detach()
    return (valid_loss * weights * eligible).sum() / eligible.sum().clamp(min=1.)


def _same_inputs(student, teacher, *, paired_video=False):
    """Only an explicit paired-input stage may change visual conditioning."""
    import torch

    for key in ("input_latents", "first_frame_latents", "action", "action_is_pad",
                "image_is_pad", "context_mask"):
        a, b = student[key], teacher[key]
        if a is None or b is None:
            if a is not b:
                raise ValueError("teacher input differs: " + key)
        else:
            visual_pair = paired_video and key in ('input_latents', 'first_frame_latents')
            if (a.dtype != b.dtype or a.shape != b.shape or a.device != b.device
                    or (not visual_pair and not torch.equal(a, b))):
                raise ValueError("teacher input differs: " + key)
            if visual_pair and (not torch.isfinite(a).all() or not torch.isfinite(b).all()):
                raise ValueError('nonfinite paired visual conditioning')
    if (student["fuse_vae_embedding_in_latents"] != teacher["fuse_vae_embedding_in_latents"]
            or student["context"].shape != teacher["context"].shape
            or student["context"].dtype != teacher["context"].dtype):
        raise ValueError("teacher conditioning contract differs")
    # Pinned FastWAM appends one proprioception token after the text context.
    # Only that token may change when the student's state encoder is trained.
    if (student["context"].ndim != 3 or student["context"].shape[1] < 1
            or not torch.equal(student["context"][:, :-1], teacher["context"][:, :-1])):
        raise ValueError("teacher text conditioning differs")


def retention_training_loss(model, teacher, sample, *, retained, weight, tiled=False,
                            paired_video=False, axis_weights=None):
    """Use shared noise/timesteps for student and frozen parent.

    The caller must bind the teacher checkpoint and the item mask to the data
    release. This numerical helper cannot determine an item's provenance. The
    teacher stays outside the student's registered modules and optimizer.
    """
    import torch
    from .navigation_axis_loss import action_token_mse, validate_axis_weights

    if axis_weights is not None:
        axis_weights = validate_axis_weights(axis_weights)

    if type(weight) not in (int, float) or not math.isfinite(weight) or weight < 0:
        raise ValueError("retention weight must be finite and nonnegative")
    if type(paired_video) is not bool:
        raise ValueError('paired visual retention must be explicitly enabled')
    if 'retention_video' in sample and not paired_video:
        raise ValueError('paired visual data requires its explicit retention objective')
    batch_size = sample["action"].shape[0]
    retained = _mask(retained, batch_size, torch.device(model.device))
    if weight == 0 or not bool(retained.any()):
        # Dispatch to the pinned class implementation. The training driver wraps
        # the instance method, so calling model.training_loss would recurse.
        # Ineligible batches must execute the author's actual objective, not a
        # nominally equivalent reduction copied for another experiment.
        if axis_weights is None or all(w == 1. for w in axis_weights):
            total, detail = type(model).training_loss(model, sample, tiled=tiled)
        else:
            from .navigation_loss import prefix_training_loss

            total, detail = prefix_training_loss(model, sample, steps=1, weight=1.,
                                                tiled=tiled, axis_weights=axis_weights)
        return total, dict(detail, loss_retention=0., retention_items=0)
    if (teacher is model or teacher is None or teacher.training
            or any(p.requires_grad for p in teacher.parameters())):
        raise ValueError("retention teacher must be a separate frozen evaluation model")
    if torch.device(teacher.device) != torch.device(model.device) or teacher.torch_dtype != model.torch_dtype:
        raise ValueError("teacher and student device/precision differ")
    parent_sample = sample
    if paired_video:
        original, current = sample.get('retention_video'), sample.get('video')
        if (not isinstance(original, torch.Tensor) or not isinstance(current, torch.Tensor)
                or original.shape != current.shape or original.dtype != current.dtype
                or original.device != current.device or not torch.isfinite(original).all()):
            raise ValueError('paired retention requires the matching original visual tensor')
        parent_sample = dict(sample, video=original)

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
    mask_args = dict(video_seq_len=(t // patch_t) * per_frame, action_seq_len=noisy_action.shape[1],
                     video_tokens_per_frame=per_frame, device=noisy_video.device)
    attention = model._build_mot_attention_mask(**mask_args)
    core_args = dict(latents_video=noisy_video, latents_action=noisy_action,
                     timestep_video=video_t, timestep_action=action_t, attention_mask=attention,
                     fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
                     action_condition=action)
    pred_video, pred_action = model._joint_denoise_core(**core_args, context=inputs["context"],
                                                      context_mask=inputs["context_mask"])

    # Auxiliary inference must not advance the student's training RNG stream.
    device = torch.device(model.device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices), torch.no_grad():
        teacher_inputs = teacher.build_inputs(parent_sample, tiled=tiled)
        _same_inputs(inputs, teacher_inputs, paired_video=paired_video)
        if not torch.equal(attention, teacher._build_mot_attention_mask(**mask_args)):
            raise ValueError("teacher attention contract differs")
        parent_core_args = {k: v.detach() if isinstance(v, torch.Tensor) else v
                            for k, v in core_args.items()}
        if paired_video:
            # The parent uses its original fixed-history video, with the same
            # sampled video noise/time and identical noisy action targets.
            parent_video = teacher.train_video_scheduler.add_noise(
                teacher_inputs['input_latents'], noise_video, video_t)
            if teacher_inputs['first_frame_latents'] is not None:
                parent_video[:, :, :1] = teacher_inputs['first_frame_latents']
            parent_core_args['latents_video'] = parent_video
        _, teacher_action = teacher._joint_denoise_core(
            **parent_core_args,
            context=teacher_inputs["context"], context_mask=teacher_inputs["context_mask"])

    include_first = inputs["first_frame_latents"] is None
    if not include_first:
        pred_video, target_video = pred_video[:, :, 1:], target_video[:, :, 1:]
    video_per_sample = model._compute_video_loss_per_sample(pred_video=pred_video, target_video=target_video,
        image_is_pad=inputs["image_is_pad"], include_initial_video_step=include_first)
    video_weight = model.train_video_scheduler.training_weight(video_t).to(video_per_sample)
    video_loss = (video_per_sample * video_weight).mean()
    token_loss = action_token_mse(pred_action, target_action, axis_weights)
    action_per_sample = reduce_action_tokens(token_loss, inputs["action_is_pad"], steps=1, weight=1.)
    action_weight = model.train_action_scheduler.training_weight(action_t).to(action_per_sample)
    action_loss = (action_per_sample * action_weight).mean()
    auxiliary = retention_penalty(pred_action, teacher_action, inputs["action_is_pad"], retained, action_weight)
    scaled_auxiliary = weight * model.loss_lambda_action * auxiliary
    total = model.loss_lambda_video * video_loss + model.loss_lambda_action * action_loss + scaled_auxiliary
    return total, dict(loss_video=model.loss_lambda_video * float(video_loss.detach()),
                       loss_action=model.loss_lambda_action * float(action_loss.detach()),
                       loss_retention=float(scaled_auxiliary.detach()), retention_items=int(retained.sum()))
