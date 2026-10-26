"""Offline standing-curriculum validation, separate from skills13 acceptance."""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import time

import numpy as np

from .data import file_sha256, write_json
from .evaluation import serializable, validate_action_source, validate_policy_environment
from .foundation import CONDITIONS, FoundationEnv, PROTOCOL, TASK, THRESHOLDS
from .preset_campaign import wilson


OBSERVATION_PROFILE = "microduck_rgb_proprio_neutral_foundation_v1"


def validate_foundation_manifest(manifest):
    required = {"observation_profile": OBSERVATION_PROFILE, "simulation_profile": "official_bam", "embodiment": "legs", "acceptance_scope": "foundation_stability", "counts_as_skill_success": False}
    if any(manifest.get(key) != value for key, value in required.items()):
        raise ValueError("checkpoint is not a declared official-BAM foundation artifact")
    if manifest.get("trained_tasks") != [TASK.task_id] or manifest.get("initial_conditions") != CONDITIONS:
        raise ValueError("foundation task/initial-condition contract differs from the checkpoint")
    if manifest.get("inference", {}).get("semantic_refresh_steps") != 5:
        raise ValueError("foundation evaluation requires fresh semantics every five ticks")


def _load_policy(checkpoint, backbone, *, device, precision):
    from lightx2v.models.networks.tinyvla.runtime import MicroDuckTinyVLAPolicy, TinyVLARuntime
    runtime = TinyVLARuntime.from_directory(checkpoint, backbone_path=backbone, device=device, precision=precision, semantic_refresh_steps=5)
    return MicroDuckTinyVLAPolicy(runtime), runtime


def summarize_foundation(rows):
    if not rows:
        raise ValueError("foundation evaluation requires episodes")
    count = sum(row["success"] for row in rows)
    distinct = len({row["initial_condition_sha256"] for row in rows})
    failures = Counter(check for row in rows for check in row["foundation_result"]["failed_checks"])
    # Shared checkpoint selection prioritizes terminal stability. This explicit
    # composite is a ranking diagnostic; underlying acceptance checks stay intact.
    stable_checks = ("terminal_window_complete", "terminal_height", "terminal_gravity", "terminal_rotation")
    failures["terminal_stable"] = sum(not all(row["foundation_result"]["checks"].get(check, False) for check in stable_checks) for row in rows)
    rate = count / len(rows)
    return {
        "successes": count, "episodes": len(rows), "distinct_conditions": distinct,
        "strict_rate": rate, "wilson_95": wilson(count, len(rows)), "failed_checks": dict(failures),
        "selection_terminal_stable_composite": list(stable_checks),
        "foundation_accepted": len(rows) >= 100 and distinct == len(rows) and rate >= .95,
        "foundation_acceptance_min_rate": .95,
        "offline_accepted": False, "realtime_accepted": False, "counts_as_skill_success": False,
        "takeovers": sum(row.get("recovery_events", 0) for row in rows),
    }


def evaluate_foundation_checkpoint(checkpoint, backbone, *, tasks, seeds, device="cuda", precision="bf16", record_dir=None, asset_root=None, mode="offline", ablation=None):
    if tuple(tasks) != (TASK.task_id,) or mode != "offline" or ablation is not None:
        raise ValueError("foundation evaluator supports only independent stand_still offline validation")
    seed_list = list(seeds)
    if not seed_list or len(set(seed_list)) != len(seed_list):
        raise ValueError("foundation evaluation requires nonempty unique seeds")
    record_dir = Path(record_dir).resolve() if record_dir else None
    policy, runtime = _load_policy(checkpoint, backbone, device=device, precision=precision)
    rows, latencies = [], []
    recorded_failure = False
    try:
        validate_foundation_manifest(runtime.manifest)
        with FoundationEnv(asset_root=asset_root, render=True) as env:
            for seed in seed_list:
                observation = env.reset(seed, TASK.task_id)
                validate_policy_environment(policy.manifest, env, observation)
                policy.reset(observation.episode_id)
                cameras, observers, states, actions = [], [], [], []
                capture = record_dir is not None and not recorded_failure
                while not env.terminated:
                    observation = env.observe()
                    if observation.rgb is None or np.any(observation.state[48:61] != 0):
                        raise ValueError("foundation inputs require current RGB and neutral command13")
                    if observation.camera_tick != observation.tick:
                        raise ValueError("foundation camera frame must match the source control tick")
                    if capture:
                        cameras.append(observation.rgb.copy())
                        observers.append(env.render_observer().copy())
                        states.append(observation.state.copy())
                    start = time.monotonic()
                    action = policy.predict(observation)
                    latencies.append((time.monotonic() - start) * 1000)
                    validate_action_source(action, observation)
                    if len(action.actions) < 5:
                        raise ValueError("foundation model must return at least a five-step action prefix")
                    if capture:
                        actions.append(action.actions[:5].copy())
                    env.render_enabled = False
                    try:
                        for raw in action.actions[:5]:
                            if env.terminated:
                                break
                            env.step(raw)
                    finally:
                        env.render_enabled = True
                state = env.evaluator_state()
                row = serializable({"task": TASK.task_id, "seed": seed, **{name: state[name] for name in ("success", "foundation_result", "initial_condition_sha256", "recovery_events", "collision", "fault", "counts_as_skill_success")}})
                rows.append(row)
                if record_dir is not None:
                    if capture and (seed == seed_list[0] or not row["success"]):
                        from .checkpoint_eval import _save_video
                        destination = record_dir / f"{TASK.task_id}-{seed}"
                        _save_video(destination / "observer.mp4", observers)
                        _save_video(destination / "camera.mp4", cameras)
                        np.savez_compressed(destination / "trajectory.npz", state61=states, action_prefix=actions)
                        write_json(destination / "result.json", row)
                    recorded_failure |= not row["success"]
                    write_json(record_dir / "progress.json", {"protocol": PROTOCOL, "episodes": rows, "complete": False, "counts_as_skill_success": False})
        summary = summarize_foundation(rows)
        report = {
            "checkpoint": str(Path(checkpoint).resolve()), "checkpoint_sha256": file_sha256(Path(checkpoint) / "model.safetensors"),
            "protocol": PROTOCOL, "condition_recipe": dict(CONDITIONS), "thresholds": dict(THRESHOLDS),
            "tasks": {TASK.task_id: summary}, "episodes": rows, "mean_strict_rate": summary["strict_rate"],
            "inference_p95_ms": float(np.percentile(latencies, 95)) if latencies else None,
            "mode": "offline", "physics_waits_for_model": True, "ablation": None,
            "semantic_refresh_steps": 5, "execute_steps": 5, "observation_hz": 10,
            "action_padding": runtime.action_padding,
            "acceptance_scope": "foundation_stability", "counts_as_skill_success": False,
            "offline_accepted": False, "realtime_accepted": False,
            "foundation_accepted": summary["foundation_accepted"], "hardware_unvalidated": True,
        }
        if record_dir is not None:
            write_json(record_dir / "progress.json", {"protocol": PROTOCOL, "episodes": rows, "complete": True, "counts_as_skill_success": False})
        return report
    finally:
        del policy, runtime
        gc.collect()
        if str(device).startswith("cuda"):
            import torch
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--asset-root")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--seed-start", type=int, default=900000)
    parser.add_argument("--seeds", type=int, default=100)
    args = parser.parse_args()
    report = evaluate_foundation_checkpoint(args.checkpoint, args.backbone, tasks=(TASK.task_id,), seeds=range(args.seed_start, args.seed_start + args.seeds), device=args.device, precision=args.precision, record_dir=Path(args.output).with_suffix(""), asset_root=args.asset_root)
    write_json(args.output, report)
    print(json.dumps(report["tasks"], indent=2))


if __name__ == "__main__":
    main()
