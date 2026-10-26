"""Explicit parameter scope for navigation transfer; inference stays unchanged."""
import hashlib
import json


def scope_manifest(kind=None):
    kind = "full" if kind is None else kind
    if not isinstance(kind, str) or kind not in ("full", "action_and_proprio"):
        raise ValueError("training_scope must be full or action_and_proprio")
    return {"schema": "microduck_navigation_training_scope_v1", "kind": kind}


def verify_resume_scope(receipt, scope):
    _kind(scope)
    if receipt.get("training_scope", scope_manifest()) != scope:
        raise ValueError("resume parameter scope changed; initialize a new training stage")


def _kind(scope):
    if not isinstance(scope, dict) or scope != scope_manifest(scope.get("kind")):
        raise ValueError("invalid training parameter-scope manifest")
    return scope["kind"]


def _expected_ids(model, kind):
    dit = {id(p) for p in model.dit.parameters()}
    proprio = getattr(model, "proprio_encoder", None)
    proprio_ids = {id(p) for p in proprio.parameters()} if proprio is not None else set()
    if not dit or dit & proprio_ids:
        raise ValueError("expected separate transformer and proprioception parameters")
    if kind == "full":
        return dit | proprio_ids
    video = getattr(model, "video_expert", None)
    action = getattr(model, "action_expert", None)
    if video is None or action is None or not proprio_ids:
        raise ValueError("action_and_proprio requires separate video/action experts and a state encoder")
    video_ids, action_ids = ({id(p) for p in module.parameters()} for module in (video, action))
    if not video_ids or not action_ids or video_ids & action_ids or dit != video_ids | action_ids:
        raise ValueError("unexpected shared or additional transformer parameters")
    return action_ids | proprio_ids


def apply_training_scope(model, scope):
    """Reapply the same scope before optimizer construction and after evaluation."""
    kind = _kind(scope)
    _expected_ids(model, kind)  # Reject unsupported topology before changing flags.
    model.eval()
    model.requires_grad_(False)
    model.dit.train()
    if kind == "full":
        model.dit.requires_grad_(True)
    else:
        model.video_expert.eval()
        model.action_expert.train().requires_grad_(True)
    proprio = getattr(model, "proprio_encoder", None)
    if proprio is not None:
        proprio.train().requires_grad_(True)
    return audit_training_scope(model, scope)


def audit_training_scope(model, scope):
    expected = _expected_ids(model, _kind(scope))
    actual = {id(p) for p in model.parameters() if p.requires_grad}
    if actual != expected:
        raise ValueError("trainable parameters differ from the declared training scope")
    names, groups = [], {}
    for name, value in model.named_parameters():
        group = groups.setdefault(name.split(".")[0], {"tensors": 0, "elements": 0,
                                                       "trainable_tensors": 0, "trainable_elements": 0})
        group["tensors"] += 1
        group["elements"] += value.numel()
        if value.requires_grad:
            names.append({"name": name, "shape": list(value.shape), "dtype": str(value.dtype)})
            group["trainable_tensors"] += 1
            group["trainable_elements"] += value.numel()
    return {"training_scope": scope, "groups": groups,
            "trainable_tensors": len(actual),
            "trainable_elements": sum(g["trainable_elements"] for g in groups.values()),
            "frozen_elements": sum(g["elements"] - g["trainable_elements"] for g in groups.values()),
            "trainable_names_shapes_sha256": hashlib.sha256(
                json.dumps(names, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}


def select_optimizer_parameters(model, parameters, scope):
    """Filter the author's optimizer before Accelerate creates optimizer state."""
    parameters = list(parameters)
    original = [id(p) for p in parameters]
    if len(original) != len(set(original)) or set(original) != _expected_ids(model, "full"):
        raise ValueError("author optimizer parameter identities or multiplicity changed")
    audit_training_scope(model, scope)
    selected = [p for p in parameters if p.requires_grad]
    if {id(p) for p in selected} != _expected_ids(model, _kind(scope)):
        raise ValueError("optimizer does not cover the declared training scope")
    return selected


def verify_action_scope_weight_update(parent, trained):
    """Independently check complete saved tensors after an action-only preflight.

    Inputs are internally produced checkpoint payloads loaded on CPU (prefer mmap
    for the actual model). This validates saved values, not just requires_grad.
    """
    import torch

    groups = {name: {"checked_tensors": 0, "checked_elements": 0,
                     "changed_tensors": 0, "changed_elements": 0}
              for name in ("video_expert", "action_expert", "proprio_encoder")}
    for section in ("mot", "proprio_encoder"):
        before, after = parent[section], trained[section]
        if not before or before.keys() != after.keys():
            raise ValueError(f"checkpoint tensor names differ: {section}")
        for key, value in after.items():
            reference = before[key]
            if section == "proprio_encoder":
                group = "proprio_encoder"
            elif key.startswith("mixtures.video."):
                group = "video_expert"
            elif key.startswith("mixtures.action."):
                group = "action_expert"
            else:
                raise ValueError(f"unexpected expert tensor: {key}")
            if (not isinstance(value, torch.Tensor) or not isinstance(reference, torch.Tensor)
                    or value.device.type != "cpu" or reference.device.type != "cpu"
                    or value.dtype != reference.dtype or value.shape != reference.shape):
                raise ValueError(f"checkpoint tensor contract differs: {section}.{key}")
            count = 0
            a, b = value.detach().reshape(-1), reference.detach().reshape(-1)
            for start in range(0, value.numel(), 1024 * 1024):
                actual, expected = a[start:start + 1024 * 1024], b[start:start + 1024 * 1024]
                if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
                    raise ValueError(f"nonfinite checkpoint values: {section}.{key}")
                if group == "video_expert":
                    if not torch.equal(actual, expected):
                        raise ValueError(f"frozen video expert changed: {key}")
                else:
                    count += int(torch.count_nonzero(actual != expected))
            row = groups[group]
            row["checked_tensors"] += 1
            row["checked_elements"] += value.numel()
            row["changed_tensors"] += int(count > 0)
            row["changed_elements"] += count
    if any(row["checked_elements"] == 0 for row in groups.values()):
        raise ValueError("checkpoint lacks a required expert or state projection")
    if any(groups[g]["changed_elements"] == 0 for g in ("action_expert", "proprio_encoder")):
        raise ValueError("preflight did not update both the action expert and state projection")
    return {"video_expert_exactly_unchanged": True, "action_and_proprio_updated": True,
            "all_checked_values_finite": True, "groups": groups,
            "formal_acceptance": False}
