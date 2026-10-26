"""Cross-language golden fixtures from the actual pinned Rust control crate."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .assets import RESOURCE_DIR, resolve_asset_root, verify_source
from .contracts import HOME_POSE, pack_observation, scatter_action
from .data import file_sha256, write_json
from .policies import POLICY_FILES, OnnxLocomotionPolicy


def validate_rust_contract(asset_root=None, output_dir="runs/rust-contract", *, vectors_per_policy=32, atol=1e-5):
    """Build an extra example; upstream Observation and ONNX code stay intact."""
    import onnxruntime

    from .daemon import prepare_robotd

    if vectors_per_policy < 1 or not np.isfinite(atol) or atol <= 0:
        raise ValueError("positive vector count and finite tolerance required")
    root = resolve_asset_root(asset_root)
    for source in ("microduck", "microduck-policies"):
        receipt = verify_source(source, root)
        if not receipt["ok"]:
            raise ValueError(f"pinned source failed verification: {receipt}")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "validation.json").exists():
        raise FileExistsError("golden validation output already exists")
    build = prepare_robotd(root)
    resource = RESOURCE_DIR / "robotd/contract_vectors.rs"
    example = build / "duck-control/examples/lx_contract_vectors.rs"
    example.parent.mkdir(parents=True, exist_ok=True)
    if example.exists() and file_sha256(example) != file_sha256(resource):
        raise ValueError("existing golden example differs; review it before replacing")
    shutil.copyfile(resource, example)
    environment = dict(os.environ)
    if "ORT_DYLIB_PATH" not in environment:
        libraries = sorted((Path(onnxruntime.__file__).parent / "capi").glob("libonnxruntime.so*"))
        if not libraries:
            raise FileNotFoundError("set ORT_DYLIB_PATH to the installed ONNX Runtime shared library")
        environment["ORT_DYLIB_PATH"] = str(libraries[0])
    cargo = shutil.which("cargo") or str(Path.home() / ".cargo/bin/cargo")
    vectors_path = output / "rust-vectors.jsonl"
    with vectors_path.open("w") as stdout, (output / "cargo.log").open("w") as stderr:
        subprocess.run(
            [cargo, "run", "--locked", "--quiet", "-p", "duck-control", "--example", "lx_contract_vectors", "--", str(root / "microduck-policies"), str(vectors_per_policy)],
            cwd=build,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            check=True,
        )
    sessions, metrics, counts = {}, {}, {}
    for line in vectors_path.read_text().splitlines():
        vector = json.loads(line)
        name = vector["policy"]
        if name not in POLICY_FILES.values():
            raise ValueError(f"unexpected policy in golden output: {name}")
        if name not in sessions:
            sessions[name] = OnnxLocomotionPolicy(root / "microduck-policies" / name)
            metrics[name] = {"observation_max_abs": 0.0, "onnx_same_obs_max_abs": 0.0, "end_to_end_action_max_abs": 0.0, "scatter_max_abs": 0.0}
            counts[name] = 0
        if vector["index"] != counts[name]:
            raise ValueError("missing, repeated or out-of-order golden vector")
        np.testing.assert_array_equal(vector["home_pose"], HOME_POSE)
        observation = pack_observation(vector["gyro"], vector["gravity"], vector["joint_positions"], vector["joint_velocities"], vector["previous_raw_action"], vector["command13"])
        raw = sessions[name].infer(observation)
        shared_raw = sessions[name].infer(np.asarray(vector["observation61"], np.float32))
        targets = scatter_action(vector["raw_action14"])
        pairs = {
            "observation_max_abs": (observation, vector["observation61"]),
            "onnx_same_obs_max_abs": (shared_raw, vector["raw_action14"]),
            "end_to_end_action_max_abs": (raw, vector["raw_action14"]),
            "scatter_max_abs": (targets, vector["joint_targets15"]),
        }
        for key, (actual, expected) in pairs.items():
            error = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
            metrics[name][key] = max(metrics[name][key], error)
        counts[name] += 1
    if set(counts) != set(POLICY_FILES.values()) or any(count != vectors_per_policy for count in counts.values()):
        raise ValueError("golden output does not cover all nine pinned policies")
    passed = all(value <= atol for metric in metrics.values() for value in metric.values())
    report = {
        "schema_version": 1,
        "passed": passed,
        "atol": atol,
        "vectors_per_policy": vectors_per_policy,
        "total_vectors": sum(counts.values()),
        "policies": metrics,
        "source_revision": "5620aa214e6e304eec00d5c376ac81d3001fb7a3",
        "example_sha256": file_sha256(resource),
        "vectors_sha256": file_sha256(vectors_path),
        "checkpoint_sha256": {name: file_sha256(root / "microduck-policies" / name) for name in counts},
        "scope": "Rust Observation::build + Policy::infer + scatter_action versus Python; excludes task behavior and real hardware",
        "hardware_validated": False,
    }
    write_json(output / "validation.json", report)
    if not passed:
        raise AssertionError(f"cross-language contract mismatch; see {output / 'validation.json'}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root")
    parser.add_argument("--output", required=True)
    parser.add_argument("--vectors", type=int, default=32)
    args = parser.parse_args()
    print(json.dumps(validate_rust_contract(args.asset_root, args.output, vectors_per_policy=args.vectors), indent=2))


if __name__ == "__main__":
    main()
