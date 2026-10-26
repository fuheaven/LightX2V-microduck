"""Explicit full-model initialization and evidence for restored training state."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_parent(weights, expected_sha256, parent_receipt, stats_path):
    receipt = json.loads(Path(parent_receipt).read_text())
    if (sha256(weights) != expected_sha256 or receipt["weights_sha256"] != expected_sha256
            or receipt["stats_sha256"] != sha256(stats_path)):
        raise ValueError("parent checkpoint or frozen normalization changed")
    return receipt


def checkpoint_payload(weights):
    import torch

    payload = torch.load(weights, map_location="cpu", weights_only=True, mmap=True)
    if not {"mot", "proprio_encoder"}.issubset(payload):
        raise ValueError("a navigation checkpoint requires both experts and proprioception")
    return payload


def initialize_from_weights(model, weights, expected_sha256, parent_receipt, stats_path):
    receipt = verify_parent(weights, expected_sha256, parent_receipt, stats_path)
    payload = checkpoint_payload(weights)
    # Strict loading deliberately rejects the upstream legacy video-only path.
    model.mot.load_state_dict(payload["mot"], strict=True)
    model.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
    return {"kind": "new_stage_from_complete_weights", "weights_sha256": expected_sha256,
            "parent_step": receipt["step"], "stats_sha256": receipt["stats_sha256"],
            "optimizer_restored": False, "step_counter_restarted": True,
            "restored_modules": ["mot", "proprio_encoder"]}


def _assert_equal(actual, expected, label):
    import torch

    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor) or actual.shape != expected.shape or actual.dtype != expected.dtype:
            raise ValueError(f"restored optimizer tensor differs: {label}; "
                             f"actual={getattr(actual, 'shape', None)}/{getattr(actual, 'dtype', None)}, "
                             f"expected={expected.shape}/{expected.dtype}")
        actual, expected = actual.detach().reshape(-1), expected.reshape(-1)
        # Bound host copies when validating the real 72 GB ZeRO checkpoint.
        for start in range(0, actual.numel(), 1024 * 1024):
            if not torch.equal(actual[start:start + 1024 * 1024].cpu(), expected[start:start + 1024 * 1024].cpu()):
                raise ValueError(f"restored optimizer values differ: {label}")
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError(f"restored optimizer fields differ: {label}")
        for key in expected:
            _assert_equal(actual[key], expected[key], f"{label}.{key}")
    elif isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            raise ValueError(f"restored optimizer groups differ: {label}")
        for index, (a, b) in enumerate(zip(actual, expected)):
            _assert_equal(a, b, f"{label}.{index}")
    elif actual != expected:
        raise ValueError(f"restored optimizer value differs: {label}")


def verify_restored_state(model, optimizer_roots, weights, expected_step, *, optimizer_checkpoint=None):
    """Check every restored trainable tensor and AdamW group before continuation.

    Frozen backbone assets retain their separately pinned provenance. This
    verifies exact restored weights, optimizer clocks and, when supplied, every
    saved AdamW moment and FP32 master value, without taking a gradient step.
    """
    import torch

    payload = checkpoint_payload(weights)
    checked = {}
    for name in ("mot", "proprio_encoder"):
        actual = getattr(model, name).state_dict()
        expected = payload[name]
        if actual.keys() != expected.keys():
            raise ValueError(f"restored {name} has different tensors")
        for key, value in actual.items():
            reference = expected[key]
            value = value.detach().cpu()
            if value.dtype != reference.dtype or not torch.equal(value, reference):
                raise ValueError(f"restored model differs from saved weights: {name}.{key}")
        checked[name] = len(actual)
    optimizer, visited, optimizer_chain = None, set(), []
    for candidate in optimizer_roots:
        while candidate is not None and id(candidate) not in visited:
            visited.add(id(candidate))
            optimizer_chain.append(type(candidate).__name__)
            if isinstance(candidate, torch.optim.Optimizer):
                # Accelerate's wrapper is itself a torch Optimizer, while its
                # state_dict may forward the entire ZeRO checkpoint. Follow
                # the chain to the deepest actual optimizer before comparing
                # with DeepSpeed's saved base_optimizer_state.
                optimizer = candidate
            candidate = getattr(candidate, "optimizer", None)
        if optimizer is not None:
            break
    if optimizer is None:
        raise ValueError("could not inspect the restored underlying optimizer")
    steps = [int(state["step"]) for state in optimizer.state.values() if "step" in state]
    if not steps or any(step != expected_step for step in steps):
        raise ValueError(f"optimizer clocks do not match step {expected_step}: {sorted(set(steps))}")
    for state in optimizer.state.values():
        if "step" in state and not {"exp_avg", "exp_avg_sq"}.issubset(state):
            raise ValueError("AdamW moments are missing from restored state")
    padding_elements = 0
    if optimizer_checkpoint is not None:
        # This is our own recorded training artifact, not a third-party pickle.
        # DeepSpeed's saved loss-scaler and enum need its normal unpickler.
        saved = torch.load(optimizer_checkpoint, map_location="cpu", weights_only=False, mmap=True)["optimizer_state_dict"]
        if any(count != 1 for count in saved["partition_count"]):
            raise ValueError("this resume verifier supports the declared single-GPU ZeRO checkpoint")
        _assert_equal(optimizer.state_dict(), saved["base_optimizer_state"], "AdamW")
        masters = saved["single_partition_of_fp32_groups"]
        paddings = saved.get("group_paddings", [0] * len(masters))
        if len(masters) != len(optimizer.param_groups) or len(paddings) != len(masters):
            raise ValueError("saved FP32 master group count changed")
        for i, (group, master, padding) in enumerate(zip(optimizer.param_groups, masters, paddings)):
            if len(group["params"]) != 1:
                raise ValueError("expected one flat parameter per ZeRO group")
            if isinstance(padding, bool) or int(padding) != padding or padding < 0:
                raise ValueError("invalid saved FP32 padding length")
            actual = group["params"][0]
            if padding:
                # Pinned DeepSpeed strips alignment padding from saved FP32
                # groups, then restores it with _get_padded_tensor (zeros).
                # Compare every real value exactly and verify padding too.
                if actual.ndim != 1 or master.ndim != 1 or actual.numel() != master.numel() + padding:
                    raise ValueError(f"restored FP32 master padding differs: group {i}")
                _assert_equal(actual[:master.numel()], master, f"master.{i}")
                tail = actual[master.numel():]
                _assert_equal(tail, torch.zeros_like(tail), f"master.{i}.padding")
                padding_elements += int(padding)
            else:
                _assert_equal(actual, master, f"master.{i}")
    return {"all_trainable_tensors_exact": True, "checked_tensors": checked,
            "optimizer_state_entries": len(steps), "optimizer_step": expected_step,
            "optimizer_class": type(optimizer).__name__, "optimizer_chain": optimizer_chain,
            "weights_updated": False,
            "restored_fp32_padding_elements_checked": padding_elements if optimizer_checkpoint else None,
            "optimizer_moments_and_fp32_masters_exact": True if optimizer_checkpoint else None}
