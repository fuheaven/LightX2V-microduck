"""Explicit training-only weights for the five navigation action dimensions.

The denominator stays five, so increasing lateral supervision does not reduce
the other action losses or the video/retention coefficients. Execution units,
normalization, and the frozen-parent retention penalty remain unchanged.
"""

from collections.abc import Mapping
import hashlib
import math
from pathlib import Path


AXIS_ORDER = ("vx", "vy", "yaw", "head_yaw", "stand")
UPSTREAM = "7faa71108368fbb3b6885649f112af607427a2d4"


def validate_axis_weights(weights):
    if (not isinstance(weights, (tuple, list)) or len(weights) != len(AXIS_ORDER)
            or any(type(w) not in (int, float) or not math.isfinite(w) or w <= 0
                   for w in weights)):
        raise ValueError("action axis weights must be five finite positive numbers")
    return tuple(float(w) for w in weights)


def axis_objective_manifest(config):
    if (not isinstance(config, Mapping) or set(config) != {"axis_order", "weights"}
            or config["axis_order"] != list(AXIS_ORDER)):
        raise ValueError("action axis objective requires the explicit navigation axis order and weights")
    weights = validate_axis_weights(config["weights"])
    source = Path(__file__)
    return dict(kind="action_axis", axis_order=list(AXIS_ORDER), weights=list(weights),
        reduction="mean_of_weighted_dimensions_then_valid_tokens",
        video_objective_unchanged=True, retention_penalty_unchanged=True,
        normalization_unchanged=True, upstream_commit=UPSTREAM,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        joint_loss_source_sha256=hashlib.sha256(source.with_name("navigation_loss.py").read_bytes()).hexdigest())


def verify_recorded_axis_objective(recipe, receipt):
    """Check immutable recipe/receipt semantics without rehashing historical code.

    Source digests belong to the recorded training version. Runtime source
    verification is performed separately against that run's source manifest.
    """
    config = recipe.get("action_axis_loss")
    recorded = receipt.get("training_objective", {"kind": "author_uniform"})
    if config is None and recorded.get("kind") != "action_axis":
        return None
    if config is None or recipe.get("action_prefix_loss") is not None:
        raise ValueError("axis objective recipe is missing or combined with prefix weighting")
    expected = axis_objective_manifest(config)
    digest_keys = ("source_sha256", "joint_loss_source_sha256")
    if (not isinstance(recorded, dict) or set(recorded) != set(expected)
            or any(recorded[k] != expected[k] for k in expected if k not in digest_keys)):
        raise ValueError("action axis objective differs between recipe and checkpoint")
    for key in digest_keys:
        value = recorded[key]
        if not isinstance(value, str) or len(value) != 64 or set(value) - set("0123456789abcdef"):
            raise ValueError("invalid action axis objective source digest")
    return recorded


def action_token_mse(prediction, target, weights=None):
    """Return FP32 per-token losses; padding is handled by the caller."""
    import torch
    import torch.nn.functional as functional

    if weights is None:
        return functional.mse_loss(prediction.float(), target.float(), reduction="none").mean(2)
    values = validate_axis_weights(weights)
    if (prediction.ndim != 3 or prediction.shape != target.shape
            or prediction.shape[-1] != len(AXIS_ORDER)):
        raise ValueError("axis supervision requires matching [batch, horizon, 5] tensors")
    error = functional.mse_loss(prediction.float(), target.float(), reduction="none")
    # Preserve the default reduction exactly, including its backward arithmetic.
    if all(w == 1. for w in values):
        return error.mean(2)
    return (error * torch.tensor(values, dtype=error.dtype, device=error.device)).mean(2)
