"""Portable behavior identity shared by local and HTTP model adapters."""
import copy
import hashlib
import json


def bundle_sha256(record):
    """Bind weights, statistics, backbone, programs and inference settings."""
    return hashlib.sha256(json.dumps(record, sort_keys=True, allow_nan=False).encode()).hexdigest()


def policy_artifacts(record):
    result = {"checkpoint_sha256": record["checkpoint"]["sha256"], "artifact_bundle_sha256": bundle_sha256(record)}
    for name in ("normalization", "config"):
        if name in record:
            result[name + "_sha256"] = record[name]["sha256"]
    return result


def policy_metadata(record):
    keys = ("acceptance_scope", "dataset_protocol", "trained_tasks", "task_contracts",
            "initial_conditions", "command_program", "uses_preset_command13",
            "claims_visual_language_autonomy", "author_program_reproduction", "counts_as_skill_success")
    return {key: copy.deepcopy(record[key]) for key in keys if key in record}
