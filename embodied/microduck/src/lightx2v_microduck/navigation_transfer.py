"""Explicit robot-policy backbone transfer, separate from navigation resume.

An arm checkpoint does not share the duck's action/state semantics. All three
interface modules are initialized afresh, including shape-compatible biases.
Every other MoT tensor must match exactly; no partial or legacy loading path is
accepted. The frozen duck training statistics remain authoritative.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .navigation_checkpoint import checkpoint_payload, sha256


INTERFACE_MODULES = (
    "mot.mixtures.action.action_encoder",
    "mot.mixtures.action.head",
    "proprio_encoder",
)
INTERFACE_KEYS = frozenset(f"{name}.{part}" for name in INTERFACE_MODULES for part in ("weight", "bias"))


def validate_initialization_modes(*, resume=None, navigation_weights=None, released_backbone=None):
    if sum(bool(value) for value in (resume, navigation_weights, released_backbone)) > 1:
        raise ValueError("choose exactly one of resume, navigation weights, or released backbone transfer")


def tensor_sha256(value):
    import torch

    raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def verify_tensor_values(actual, expected, name):
    """Compare every value, bounding copies of the real 12 GB checkpoint."""
    import torch

    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError(f"transferred tensor metadata differs: {name}")
    actual, expected = actual.detach().reshape(-1), expected.detach().reshape(-1)
    for start in range(0, actual.numel(), 1024 * 1024):
        if not torch.equal(actual[start:start + 1024 * 1024].cpu(), expected[start:start + 1024 * 1024].cpu()):
            raise ValueError(f"transferred tensor values differ: {name}")


def _flatten(payload):
    import torch

    result = {}
    for module in ("mot", "proprio_encoder"):
        tensors = payload[module]
        if not isinstance(tensors, dict) or not tensors:
            raise ValueError(f"missing transfer tensors: {module}")
        for name, tensor in tensors.items():
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"nontensor in transfer checkpoint: {module}.{name}")
            result[f"{module}.{name}"] = tensor
    return result


def initialize_released_backbone(model, weights, expected_sha256, stats_path, *,
                                 expected_stats_sha256, interface_seed):
    import torch
    from .navigation import NAVIGATION_CONTRACT

    if isinstance(interface_seed, bool) or not isinstance(interface_seed, int) or interface_seed < 0:
        raise ValueError("interface initialization requires an explicit nonnegative integer seed")
    if sha256(weights) != expected_sha256 or sha256(stats_path) != expected_stats_sha256:
        raise ValueError("released checkpoint or frozen navigation statistics changed")
    stats = json.loads(Path(stats_path).read_text())
    if stats.get("schema") != NAVIGATION_CONTRACT or stats.get("normalization_split") != "train":
        raise ValueError("transfer requires navigation train-only statistics")
    payload = checkpoint_payload(weights)
    source = _flatten(payload)
    target = _flatten({"mot": model.mot.state_dict(), "proprio_encoder": model.proprio_encoder.state_dict()})
    if source.keys() != target.keys() or not INTERFACE_KEYS.issubset(target):
        raise ValueError("released backbone tensor names differ; partial loading is forbidden")
    modules = {
        "mot.mixtures.action.action_encoder": model.mot.get_submodule("mixtures.action.action_encoder"),
        "mot.mixtures.action.head": model.mot.get_submodule("mixtures.action.head"),
        "proprio_encoder": model.proprio_encoder,
    }
    if any(not isinstance(m, torch.nn.Linear) or m.bias is None for m in modules.values()):
        raise ValueError("robot interfaces must be the declared complete Linear modules")
    encoder, head, proprio = (modules[k] for k in INTERFACE_MODULES)
    if (encoder.in_features, head.out_features, proprio.in_features) != (5, 5, 48):
        raise ValueError("target interfaces are not the 5D-action/48D-state navigation contract")
    expected_source = {
        INTERFACE_MODULES[0] + ".weight": (encoder.out_features, 7),
        INTERFACE_MODULES[0] + ".bias": (encoder.out_features,),
        INTERFACE_MODULES[1] + ".weight": (7, head.in_features),
        INTERFACE_MODULES[1] + ".bias": (7,),
        INTERFACE_MODULES[2] + ".weight": (proprio.out_features, 8),
        INTERFACE_MODULES[2] + ".bias": (proprio.out_features,),
    }
    # Validate the full source before mutating any live parameters.
    for name, tensor in source.items():
        shape = expected_source[name] if name in INTERFACE_KEYS else tuple(target[name].shape)
        if tuple(tensor.shape) != shape or tensor.dtype != target[name].dtype:
            raise ValueError(f"released tensor shape or dtype differs: {name}")
        if tensor.layout != torch.strided or not tensor.is_contiguous():
            raise ValueError(f"unsupported released tensor layout: {name}")
        flat = tensor.reshape(-1)
        for start in range(0, flat.numel(), 1024 * 1024):
            if not torch.isfinite(flat[start:start + 1024 * 1024]).all():
                raise ValueError(f"nonfinite released tensor: {name}")

    fresh = {}
    # Preserve the training RNG, including CUDA RNG. The CPU generator is the
    # only generator touched inside this fork; FP32 initialization is then cast
    # to each target dtype, independently of the arm checkpoint and model load.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(interface_seed)
        for name, module in modules.items():
            new = torch.nn.Linear(module.in_features, module.out_features, bias=True,
                                  device="cpu", dtype=torch.float32)
            for part, value in new.state_dict().items():
                key = f"{name}.{part}"
                fresh[key] = value.to(dtype=target[key].dtype)
    merged = {key: fresh[key] if key in INTERFACE_KEYS else value for key, value in source.items()}
    for module_name in ("mot", "proprio_encoder"):
        prefix = module_name + "."
        getattr(model, module_name).load_state_dict(
            {name[len(prefix):]: value for name, value in merged.items() if name.startswith(prefix)}, strict=True)
    actual = _flatten({"mot": model.mot.state_dict(), "proprio_encoder": model.proprio_encoder.state_dict()})
    for name, value in actual.items():
        verify_tensor_values(value, merged[name], name)
    interfaces = {
        name: {"source_shape": list(source[name].shape), "target_shape": list(actual[name].shape),
               "initialized_sha256": tensor_sha256(fresh[name]),
               "source_sha256": tensor_sha256(source[name]), "copied_from_source": False}
        for name in sorted(INTERFACE_KEYS)
    }
    return {
        "kind": "released_robot_policy_backbone_with_new_navigation_interfaces",
        "weights_sha256": expected_sha256, "source_checkpoint_step": payload.get("step"),
        "stats_sha256": expected_stats_sha256, "normalizer_replaced": False,
        "interface_seed": interface_seed,
        "interface_initializer": "torch.nn.Linear FP32 default initialization, cast to target dtype",
        "interface_tensor_count": len(interfaces), "interfaces": interfaces,
        "backbone_tensor_count": len(source) - len(INTERFACE_KEYS),
        "backbone_parameters": sum(t.numel() for n, t in source.items() if n not in INTERFACE_KEYS),
        "every_loaded_tensor_verified_exact": True,
        "optimizer_restored": False, "step_counter_restarted": True,
        "action_axis_mapping": None, "not_capability_acceptance": True,
    }
