"""Bind video/action loss coefficients across complete-state navigation resumes.

The token-reduction objective is recorded separately by navigation_loss.py.
Older receipts omit these coefficients; a hash-bound original saved recipe is
required to resolve them. Weight-only initialization remains a new training stage.
"""

from collections.abc import Mapping
import hashlib
import math
from pathlib import Path


SCHEMA = "microduck_navigation_loss_weights_v1"
KEYS = frozenset(("lambda_video", "lambda_action"))


def loss_weight_manifest(config=None):
    """Resolve the pinned author's explicit defaults, rejecting unknown losses."""
    config = {} if config is None else config
    if not isinstance(config, Mapping) or set(config) - KEYS:
        raise ValueError("loss binding supports only lambda_video and lambda_action")
    values = {}
    for name in sorted(KEYS):
        value = config.get(name, 1.0)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("loss coefficients must be finite nonnegative numbers")
        values[name] = float(value)
    if values["lambda_action"] <= 0:
        raise ValueError("navigation training requires a positive action loss coefficient")
    return {"schema": SCHEMA, **values}


def _validated_manifest(value):
    if (not isinstance(value, dict) or set(value) != KEYS | {"schema"}
            or value["schema"] != SCHEMA):
        raise ValueError("invalid loss-weight receipt")
    return loss_weight_manifest({key: value[key] for key in KEYS})


def verify_resume_loss_weights(receipt, requested, *, state_dir=None,
                               legacy_recipe_sha256=None):
    """Reject objective changes before any optimizer state is restored.

    The caller freezes legacy_recipe_sha256 in its new run configuration. The
    legacy recipe must be the config.yaml adjacent to that checkpoint's original
    checkpoints directory, not a replacement configuration chosen by the caller.
    This verifies coefficients only, not full optimizer/model/RNG restoration.
    """
    requested = _validated_manifest(requested)
    if "loss_weights" in receipt:
        if legacy_recipe_sha256 is not None:
            raise ValueError("explicit loss-weight receipts do not need a legacy recipe override")
        stored = _validated_manifest(receipt["loss_weights"])
        evidence = {"source": "checkpoint_receipt"}
    else:
        digest = legacy_recipe_sha256
        if (not isinstance(digest, str) or len(digest) != 64
                or set(digest) - set("0123456789abcdef")):
            raise ValueError("legacy resume requires resume_loss_recipe_sha256 for its saved recipe")
        if state_dir is None:
            raise ValueError("legacy resume requires its original checkpoint state directory")
        state = Path(state_dir).resolve()
        step = receipt.get("step")
        if (type(step) is not int or step <= 0 or not state.is_dir()
                or state.name != "step_%06d" % step
                or state.parent.name != "state" or state.parent.parent.name != "checkpoints"):
            raise ValueError("unsupported legacy checkpoint layout or step")
        recipe = state.parents[2] / "config.yaml"
        raw = recipe.read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("legacy saved training recipe changed")
        import yaml

        saved = yaml.safe_load(raw)
        if (not isinstance(saved, dict) or not isinstance(saved.get("model"), dict)):
            raise ValueError("legacy recipe lacks the model loss configuration")
        stored = loss_weight_manifest(saved["model"].get("loss"))
        evidence = {"source": "hash_bound_legacy_saved_recipe",
                    "recipe": {"path": str(recipe), "sha256": digest}}
    if stored != requested:
        raise ValueError("resume loss coefficients changed; explicitly initialize a new training stage")
    return {**evidence, "loss_weights": stored, "optimizer_restoration_verified": False}


def verify_runtime_loss_weights(model, expected):
    """Verify the instantiated model actually uses the declared coefficients."""
    expected = _validated_manifest(expected)
    actual = loss_weight_manifest({
        "lambda_video": model.loss_lambda_video,
        "lambda_action": model.loss_lambda_action,
    })
    if actual != expected:
        raise ValueError("instantiated model loss coefficients differ from the training recipe")
    return actual
