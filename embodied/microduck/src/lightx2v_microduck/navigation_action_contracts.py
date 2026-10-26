"""Explicit execution bounds; changing them must not re-interpret old policies.

The extended forward contract is experimental. Supporting its wire format is
not teacher qualification, dataset admission, or a trained-model capability.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

import numpy as np


LEGACY_NAVIGATION_CONTRACT = "microduck_navigation_v1"
FORWARD04_NAVIGATION_CONTRACT = "microduck_navigation_forward04_v1"


@dataclass(frozen=True)
class NavigationActionContract:
    name: str
    lower: tuple[float, ...]
    upper: tuple[float, ...]

    def bounds(self):
        # Return fresh arrays: a caller must not mutate another session's limits.
        return np.asarray(self.lower, np.float32), np.asarray(self.upper, np.float32)

    def validate(self, values):
        lower, upper = self.bounds()
        values = np.asarray(values, dtype=np.float32)
        if values.shape[-1:] != (5,) or not np.isfinite(values).all():
            raise ValueError("navigation decision must have five finite coordinates")
        if np.any(values < lower) or np.any(values > upper):
            raise ValueError(f"navigation decision exceeds declared bounds: {self.name}")
        return values


_LOWER = (-.15, -.15, -1.1, -.45, 0.)
_CONTRACTS = {
    LEGACY_NAVIGATION_CONTRACT: NavigationActionContract(
        LEGACY_NAVIGATION_CONTRACT, _LOWER, (.30, .15, 1.1, .45, 1.)),
    FORWARD04_NAVIGATION_CONTRACT: NavigationActionContract(
        FORWARD04_NAVIGATION_CONTRACT, _LOWER, (.40, .15, 1.1, .45, 1.)),
}


def action_contract(name=LEGACY_NAVIGATION_CONTRACT):
    if not isinstance(name, str) or name not in _CONTRACTS:
        raise ValueError(f"unsupported navigation action contract: {name!r}")
    return _CONTRACTS[name]


def core_action_contract(core, requested=None):
    config = getattr(core, "config", {})
    configured = config.get("navigation_contract") if isinstance(config, Mapping) else None
    declared = getattr(core, "navigation_contract", None)
    if configured is not None and declared is not None and configured != declared:
        raise ValueError("core and config declare different navigation contracts")
    declared = declared or configured or LEGACY_NAVIGATION_CONTRACT
    selected = declared if requested is None else requested
    action_contract(declared)
    action_contract(selected)
    if declared != selected:
        raise ValueError("policy action contract differs from its model core")
    return selected


def verify_artifact_action_contract(manifest, config, stats):
    """Bind execution to frozen normalizer coordinates before loading a GPU.

    The extended executor can retain the old train-only normalizer, but only
    through an explicit, hashed binding in both model metadata and config.
    The statistics themselves keep their original schema and bytes.
    """
    contract = manifest["contract"]
    action_contract(contract)
    if manifest["embodiment"] != "legs":
        raise ValueError("unsupported navigation embodiment")
    if config.get("navigation_contract", LEGACY_NAVIGATION_CONTRACT) != contract:
        raise ValueError("artifact and config navigation contracts differ")
    if stats.get("normalization_split") != "train":
        raise ValueError("navigation normalization must use train-only statistics")
    binding = manifest.get("normalization_binding")
    if binding is None:
        if contract != LEGACY_NAVIGATION_CONTRACT or config.get("normalization_binding") is not None:
            raise ValueError("extended navigation requires an explicit normalization binding")
        if stats.get("schema") != LEGACY_NAVIGATION_CONTRACT:
            raise ValueError("normalization coordinates differ from legacy execution")
    else:
        if not isinstance(binding, dict) or binding != config.get("normalization_binding"):
            raise ValueError("artifact and config normalization bindings differ")
        digest = binding.get("stats_sha256")
        if not isinstance(digest, str) or len(digest) != 64 or set(digest) - set('0123456789abcdef'):
            raise ValueError("normalization binding requires a SHA256 digest")
        if (binding.get("execution_contract") != contract
                or binding.get("coordinate_contract") != stats.get("schema")
                or binding.get("stats_sha256") != manifest["files"]["dataset_stats.json"]):
            raise ValueError("normalization binding does not match the frozen artifact")
        action_contract(binding["coordinate_contract"])
    return contract
