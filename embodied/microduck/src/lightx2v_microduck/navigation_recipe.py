"""Compare complete resolved training recipes across YAML serialization forms."""

import hashlib
import io
import json
from pathlib import Path


def _recipe(path):
    from omegaconf import OmegaConf

    path = Path(path)
    raw = path.read_bytes()
    value = OmegaConf.to_container(
        OmegaConf.load(io.StringIO(raw.decode("utf-8"))),
        resolve=False,
        throw_on_missing=True,
    )
    if not isinstance(value, dict):
        raise ValueError("training recipe must be a mapping")

    def check(node):
        if isinstance(node, dict):
            if any(not isinstance(key, str) for key in node):
                raise ValueError("training recipe keys must be strings")
            return 1 + sum(check(child) for child in node.values())
        if isinstance(node, list):
            return 1 + sum(check(child) for child in node)
        if isinstance(node, str) and "${" in node:
            raise ValueError("training recipe must already be resolved")
        if node is not None and type(node) not in (str, int, float, bool):
            raise ValueError("unsupported training recipe value")
        return 1

    nodes = check(value)
    canonical = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return {
        "path": str(path),
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "values_sha256": hashlib.sha256(canonical).hexdigest(),
        "value_nodes": nodes,
    }, canonical


def verify_saved_recipe(prepared_path, saved_path):
    """Allow formatting/aliases only; compare every typed value and list order.

    Interpolation is rejected rather than re-evaluated in a later environment.
    The caller must also pin both source files to their experiment receipts.
    """
    prepared, before = _recipe(prepared_path)
    saved, after = _recipe(saved_path)
    if before != after:
        raise ValueError("saved training recipe values differ")
    return {
        "schema": "microduck_training_recipe_equivalence_v1",
        "status": "complete",
        "prepared": prepared,
        "saved": saved,
        "all_values_equal": True,
        "byte_identical": prepared["file_sha256"] == saved["file_sha256"],
        "values_sha256": prepared["values_sha256"],
        "value_nodes_checked": prepared["value_nodes"],
        "ignored_fields": [],
        "interpolation_evaluated": False,
    }
