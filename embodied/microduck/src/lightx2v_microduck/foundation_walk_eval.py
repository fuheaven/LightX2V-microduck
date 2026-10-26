"""Fresh-observation walk/stop curriculum evaluation, never skills13 acceptance."""
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
from .foundation_walk import ACCEPTANCE_SCOPE, COMMAND_PROGRAM, CONDITIONS, OBSERVATION_PROFILE, PROTOCOL, TASK, FoundationWalkEnv, command_context
from .preset_campaign import wilson


def validate_walk_manifest(manifest):
    required = {"observation_profile": OBSERVATION_PROFILE, "simulation_profile": "official_bam", "embodiment": "legs",
                "acceptance_scope": ACCEPTANCE_SCOPE, "counts_as_skill_success": False, "trained_tasks": [TASK.task_id],
                "initial_conditions": CONDITIONS, "command_program": COMMAND_PROGRAM}
    if any(manifest.get(key) != value for key, value in required.items()):
        raise ValueError("checkpoint does not match the independent walk curriculum contract")
    if manifest.get("inference", {}).get("semantic_refresh_steps") != 5:
        raise ValueError("walk evaluation requires fresh semantics every five control ticks")


def _load_policy(checkpoint, backbone, *, device, precision):
    from lightx2v.models.networks.tinyvla.runtime import MicroDuckTinyVLAPolicy, TinyVLARuntime
    runtime = TinyVLARuntime.from_directory(checkpoint, backbone_path=backbone, device=device, precision=precision, semantic_refresh_steps=5)
    return MicroDuckTinyVLAPolicy(runtime), runtime


def summarize_walk(rows):
    if not rows:
        raise ValueError("walk evaluation requires a nonempty population")
    successes = sum(row["success"] for row in rows)
    distinct = len({row["initial_condition_sha256"] for row in rows})
    failures = Counter(check for row in rows for check in row["foundation_walk_result"]["failed_checks"])
    no_falls = all(row["foundation_walk_result"]["checks"].get("no_fall", False) for row in rows)
    timing_rows = [row["foundation_walk_result"].get("diagnostics") for row in rows]
    timing_rows = [value for value in timing_rows if value and value.get("protocol") == "walk_failure_timing_v1"]
    fall_times = [value["first_fall_time_s"] for value in timing_rows if value["fall_observed"]]
    timing = {"protocol": "walk_failure_timing_v1", "episodes_with_timing": len(timing_rows),
              "episodes_without_timing": len(rows) - len(timing_rows),
              "observed_falls": len(fall_times), "no_fall_observed": len(timing_rows) - len(fall_times),
              "first_fall_median_s_among_fallen": float(np.median(fall_times)) if fall_times else None,
              "first_fall_phase_counts": dict(Counter(value["first_fall_phase"] for value in timing_rows if value["fall_observed"])),
              "used_for_acceptance": False, "used_for_checkpoint_selection": False}
    return {"episodes": len(rows), "successes": successes, "distinct_conditions": distinct,
            "strict_rate": successes / len(rows), "wilson_95": wilson(successes, len(rows)), "failed_checks": dict(failures),
            "no_falls": no_falls, "foundation_walk_acceptance_min_rate": .9, "zero_falls_required": True,
            "foundation_walk_accepted": len(rows) >= 100 and distinct == len(rows) and successes / len(rows) >= .9 and no_falls,
            "counts_as_skill_success": False, "offline_accepted": False, "realtime_accepted": False,
            "takeovers": sum(row.get("recovery_events", 0) for row in rows), "failure_timing": timing}


def evaluate_walk_checkpoint(checkpoint, backbone, *, tasks, seeds, device="cuda", precision="bf16", record_dir=None,
                             asset_root=None, mode="offline", ablation=None, deadline=None):
    if tuple(tasks) != (TASK.task_id,) or mode != "offline" or ablation not in (None, "zero_flow_noise"):
        raise ValueError("walk evaluator supports only independent walk_forward_stop offline validation")
    seeds = list(seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("walk evaluation needs distinct nonempty seeds")
    destination = Path(record_dir).resolve() if record_dir else None
    policy, runtime = _load_policy(checkpoint, backbone, device=device, precision=precision)
    rows, latencies = [], []
    recorded_failure = False
    try:
        validate_walk_manifest(runtime.manifest)
        if runtime.precision != precision:
            raise ValueError("loaded inference precision differs from the requested evaluation")
        # Diagnostic only: hold the initial flow sample at its Gaussian mean.
        # This changes the sampler, so it can never grant curriculum acceptance.
        noise = (np.zeros((1, runtime.model.config.chunk_size, runtime.model.config.max_action_dim), np.float32)
                 if ablation == "zero_flow_noise" else None)
        with FoundationWalkEnv(asset_root=asset_root, render=True) as env:
            for seed in seeds:
                observation = env.reset(seed, TASK.task_id)
                validate_policy_environment(policy.manifest, env, observation)
                policy.reset(observation.episode_id)
                cameras, observers, states, actions = [], [], [], []
                capture = destination is not None and not recorded_failure
                while not env.terminated:
                    if deadline is not None and time.monotonic() >= deadline:
                        raise TimeoutError("walk evaluation exhausted the shared pilot wall-time budget")
                    observation = env.observe()
                    if observation.rgb is None or observation.camera_tick != observation.tick:
                        raise ValueError("walk evaluation requires an RGB image from the source control tick")
                    if not np.array_equal(observation.state[48:61], command_context(observation.tick)[1]):
                        raise ValueError("walk command13 differs from the prescribed same-tick command")
                    if capture:
                        cameras.append(observation.rgb.copy())
                        observers.append(env.render_observer().copy())
                        states.append(observation.state.copy())
                    start = time.monotonic()
                    action = policy.predict(observation, noise=noise) if noise is not None else policy.predict(observation)
                    latencies.append((time.monotonic() - start) * 1000)
                    validate_action_source(action, observation)
                    if len(action.actions) < 5:
                        raise ValueError("walk model must provide at least five actions per fresh observation")
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
                row = serializable({"task": TASK.task_id, "seed": seed, **{key: state[key] for key in (
                    "success", "foundation_walk_result", "initial_condition_sha256", "recovery_events", "collision", "fault", "ever_fallen", "counts_as_skill_success")}})
                rows.append(row)
                if destination is not None:
                    if capture and (seed == seeds[0] or not row["success"]):
                        from .checkpoint_eval import _save_video
                        folder = destination / f"{TASK.task_id}-{seed}"
                        _save_video(folder / "observer.mp4", observers)
                        _save_video(folder / "camera.mp4", cameras)
                        np.savez_compressed(folder / "trajectory.npz", state61=states, action_prefix=actions)
                        write_json(folder / "result.json", row)
                    recorded_failure |= not row["success"]
                    write_json(destination / "progress.json", {"protocol": PROTOCOL, "episodes": rows, "complete": False, "counts_as_skill_success": False})
        grade = summarize_walk(rows)
        if ablation is not None:
            grade["foundation_walk_accepted"] = False
        report = {"checkpoint": str(Path(checkpoint).resolve()), "checkpoint_sha256": file_sha256(Path(checkpoint) / "model.safetensors"),
                  "protocol": PROTOCOL, "condition_recipe": CONDITIONS, "command_program": COMMAND_PROGRAM,
                  "tasks": {TASK.task_id: grade}, "episodes": rows, "mean_strict_rate": grade["strict_rate"],
                  "mode": "offline", "ablation": ablation, "sampling": "zero_flow_noise" if noise is not None else "seeded_gaussian",
                  "inference_precision": runtime.precision,
                  "physics_waits_for_model": True, "semantic_refresh_steps": 5, "execute_steps": 5, "observation_hz": 10,
                  "inference_p95_ms": float(np.percentile(latencies, 95)) if latencies else None, "action_padding": runtime.action_padding,
                  "acceptance_scope": ACCEPTANCE_SCOPE, "counts_as_skill_success": False, "offline_accepted": False, "realtime_accepted": False,
                  "foundation_walk_accepted": grade["foundation_walk_accepted"], "uses_preset_command13": True,
                  "claims_visual_language_autonomy": False, "hardware_unvalidated": True}
        if destination is not None:
            write_json(destination / "progress.json", {"protocol": PROTOCOL, "episodes": rows, "complete": True, "counts_as_skill_success": False})
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
    parser.add_argument("--ablation", choices=("zero_flow_noise",))
    parser.add_argument("--seed-start", type=int, default=910000)
    parser.add_argument("--seeds", type=int, default=100)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() or output.with_suffix("").exists():
        raise FileExistsError("walk evaluation outputs already exist")
    report = evaluate_walk_checkpoint(args.checkpoint, args.backbone, tasks=(TASK.task_id,), seeds=range(args.seed_start, args.seed_start + args.seeds),
                                      device=args.device, precision=args.precision, record_dir=output.with_suffix(""), asset_root=args.asset_root, ablation=args.ablation)
    write_json(output, report)
    print(json.dumps(report["tasks"], indent=2))


if __name__ == "__main__":
    main()
