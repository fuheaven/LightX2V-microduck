"""Hash-verified raw-action artifact contract; tensor-only parity is separate."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

POLICY_JOINT_NAMES = (
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
)
HOME_POLICY = (0.0, -0.0873, -0.4579, -0.0049, 0.4530, 0.3491, 0.3491, 0.0, 0.0, 0.0, 0.0873, 0.4579, 0.0049, -0.4530)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verified_file(root, record):
    root = Path(root).resolve()
    path = (root / record["path"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("artifact paths must remain inside checkpoint directory")
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"artifact SHA256 mismatch: {path.name}")
    return path


def load_raw_manifest(directory):
    root = Path(directory)
    path = root / "artifact_manifest.json"
    if not path.is_file():
        raise ValueError("raw closed-loop inference requires artifact_manifest.json; use tensor API for normalization-free internal parity only")
    data = json.loads(path.read_text())
    if "inference" in data:
        inference = data["inference"]
        refresh = inference.get("semantic_refresh_steps") if isinstance(inference, dict) else None
        if isinstance(refresh, bool) or not isinstance(refresh, int) or refresh < 1:
            raise ValueError("inference.semantic_refresh_steps must be a positive integer")
        if inference.get("action_padding", "free") not in ("free", "training_noise_path"):
            raise ValueError("unsupported inference.action_padding")
    if data.get("format") != "lightx2v-tinyvla-v1" or data.get("model_variant") not in ("public_duckft_50", "lightx2v_microduck_50"):
        raise ValueError("unsupported TinyVLA artifact; Stage 2.5 is not available")
    for key in ("source_revision", "backbone_revision", "observation_profile", "embodiment_digest"):
        if not isinstance(data.get(key), str) or not data[key]:
            raise ValueError(f"manifest requires {key}")
    for key in ("source_revision", "backbone_revision"):
        if not re.fullmatch(r"[0-9a-f]{40}(?:[0-9a-f]{24})?", data[key]):
            raise ValueError(f"{key} must be an immutable hexadecimal revision")
    if not re.fullmatch(r"[0-9a-f]{64}", data["embodiment_digest"]):
        raise ValueError("embodiment_digest must be a SHA256")
    contract = data["action_contract"]
    # Validate legacy defaults without rewriting the artifact identity. The
    # runtime/executor apply this same default when consuming the contract.
    age = contract.get("max_age_ticks", 10)
    if not isinstance(age, int) or isinstance(age, bool) or not 5 <= age <= 50:
        raise ValueError("max_age_ticks must be between execution prefix 5 and horizon 50")
    required = dict(kind="joint_residual", units="rad", control_hz=50, observation_hz=10, execute_steps=5)
    if any(contract.get(key) != value for key, value in required.items()):
        raise ValueError("unsupported MicroDuck action contract")
    import math

    names, home = contract.get("joint_names", []), contract.get("home_pose", [])
    if len(names) != 14 or len(set(names)) != 14 or len(home) != 14 or not all(math.isfinite(x) for x in home):
        raise ValueError("manifest requires exact fourteen joint names and home pose")
    if tuple(names) != POLICY_JOINT_NAMES or any(abs(a - b) > 1e-7 for a, b in zip(home, HOME_POLICY)):
        raise ValueError("joint order/home pose differs from the MicroDuck policy contract")
    if contract.get("scale") != 1.0:
        raise ValueError("TinyVLA raw offsets require action scale 1.0")
    for key in ("checkpoint", "config", "normalization"):
        verified_file(root, data[key])
    if data["checkpoint"]["path"] != "model.safetensors" or data["config"]["path"] != "config.json":
        raise ValueError("checkpoint/config must use canonical filenames")
    config = json.loads((root / "config.json").read_text())
    semantic = config.get("semantic_padding_policy", "legacy")
    declared = data.get("inference", {}).get("semantic_padding_policy")
    correction = data.get("runtime_correction", {})
    if (semantic not in ("legacy", "batch_consistent_v1") or not isinstance(correction, dict)
            or declared is not None and declared != semantic
            or correction.get("policy", semantic) != semantic
            or semantic != "legacy" and declared is None and not correction.get("policy")):
        raise ValueError("artifact semantic padding declaration differs from verified config")
    if config.get("chunk_size") != 50 or config.get("max_state_dim") != 256 or config.get("max_action_dim") != 64:
        raise ValueError("raw runtime supports the 50x64 public duck architecture only")
    support = data.get("support", {})
    if support.get("mode") == "required":
        verified_file(root, support)
    elif support.get("mode") != "none_trained":
        raise ValueError("support must be required with verified examples, or explicitly none_trained")
    if data["model_variant"] == "public_duckft_50" and support.get("mode") != "required":
        raise ValueError("public duckft author compatibility requires its support examples; do not silently omit them")
    return data


def verify_local_backbone(directory, revision):
    """A local path bypasses HF revision resolution, so require its receipt."""
    root = Path(directory)
    receipt_path = root / "download_receipt.json"
    if not receipt_path.is_file():
        raise ValueError("local backbone requires download_receipt.json with revision and per-file SHA256")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("revision") != revision:
        raise ValueError("local backbone revision differs from model manifest")
    files = receipt.get("files", {})
    required = {"config.json", "tokenizer_config.json", "preprocessor_config.json"}
    if not required.issubset(files) or not ("tokenizer.json" in files or {"vocab.json", "merges.txt"}.issubset(files)):
        raise ValueError("backbone receipt omits required tokenizer/processor/config artifacts")
    for filename, record in files.items():
        verified_file(root, {"path": filename, "sha256": record["sha256"]})
