"""Independent offline evaluation of versioned project tasks, never author13.

Validation/development results can select checkpoints but cannot grant release.
The fixed final population is reserved; reference teachers are CPU diagnostics.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .data import file_sha256, write_json
from .evaluation import serializable, validate_action_source, validate_policy_environment
from .preset_campaign import wilson
from .qualified_teachers import ACCEPTANCE_SCOPE, OBSERVATION_PROFILE, make_environment, make_teacher, program_context, training_contracts
from .qualified_training import DATASET_PROTOCOL

PROTOCOL = "project_qualified_evaluation_v2"
FINAL_SEED_START = 980000
FINAL_SEEDS = tuple(range(FINAL_SEED_START, FINAL_SEED_START + 100))
PURPOSES = ("validation", "development", "final_test", "teacher_reference")
# Private final-comparison runner only, after the frozen-candidate and completed
# student-final gates in qualified_teacher_compare.py. Not a diagnostic CLI mode.
FINAL_TEACHER_PURPOSE = "teacher_final_comparison"


def _canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _sha(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def validate_population(tasks, seeds, *, purpose, ablation=None, record_dir=None):
    tasks, seeds = list(tasks), list(seeds)
    if not tasks or len(set(tasks)) != len(tasks) or any(task not in training_contracts() for task in tasks):
        raise ValueError("qualified evaluation requires unique registered project tasks (ground_touch is excluded)")
    if not seeds or len(set(seeds)) != len(seeds) or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("qualified evaluation requires unique nonnegative integer seeds")
    if purpose not in PURPOSES or ablation not in (None, "black_image", "language", "command"):
        raise ValueError("unknown evaluation purpose or ablation")
    if purpose == "final_test":
        if tuple(seeds) != FINAL_SEEDS or ablation is not None or record_dir is None:
            raise ValueError("final_test requires the fixed 100 final conditions, no ablation, and a record directory")
    elif set(seeds).intersection(FINAL_SEEDS):
        raise ValueError("reserved final conditions cannot be used for validation, development or teacher diagnostics")
    if purpose == "teacher_reference" and ablation is not None:
        raise ValueError("teacher reference is not a student input ablation")
    return tasks, seeds


def validate_qualified_manifest(manifest, tasks):
    required = {"acceptance_scope": ACCEPTANCE_SCOPE, "observation_profile": OBSERVATION_PROFILE,
                "dataset_protocol": DATASET_PROTOCOL, "simulation_profile": "official_bam",
                "counts_as_skill_success": False, "author_program_reproduction": False}
    if any(manifest.get(key) != value for key, value in required.items()):
        raise ValueError("checkpoint is not an explicitly scoped qualified project artifact")
    expected = training_contracts()
    trained, contracts = manifest.get("trained_tasks"), manifest.get("task_contracts")
    if (not isinstance(trained, list) or not trained or len(set(trained)) != len(trained)
            or not isinstance(contracts, dict) or set(contracts) != set(trained) or not set(tasks).issubset(trained)):
        raise ValueError("checkpoint task coverage and per-task contracts do not match")
    for task in trained:
        if task not in expected or _canonical(contracts[task]) != _canonical(expected[task]):
            raise ValueError(f"checkpoint teacher/program/initial conditions differ from registered task: {task}")
        if contracts[task]["task_spec"]["embodiment"] != manifest.get("embodiment"):
            raise ValueError("checkpoint cannot mix leg and roller embodiments")
    if manifest.get("initial_conditions") is not None:
        raise ValueError("qualified initial conditions must be declared per task, never as a global fallback")
    if manifest.get("inference", {}).get("semantic_refresh_steps") != 5:
        raise ValueError("qualified evaluation requires fresh semantics every five ticks")
    return {task: copy.deepcopy(contracts[task]) for task in tasks}


def _expected_command(env, observation, task):
    if task == "ground_lower":
        from .ground_lower import command_context
        return command_context(observation.tick)[1]
    return program_context(env.teacher_version, observation.tick, observation.state, copy.deepcopy(env._phase_state),
                           rollers=env.task.embodiment == "rollers")[1]


def _episode_result(env, task, seed, contract):
    if hasattr(env, "condition_recipe") and _canonical(env.condition_recipe) != _canonical(contract["initial_conditions"]):
        raise ValueError("executed initial conditions differ from artifact contract")
    return episode_result_from_state(env.evaluator_state(), task, seed, contract)


def episode_result_from_state(state, task, seed, contract):
    """Score the same captured evaluator state for offline and real-daemon runs."""
    initial = state.get("initial_condition")
    if not initial or _canonical(initial.get("recipe")) != _canonical(contract["initial_conditions"]):
        raise ValueError("episode omitted its actual initial condition recipe")
    strict = state.get("ground_lower_result" if task == "ground_lower" else "qualified_teacher_result")
    if not isinstance(strict, dict) or strict.get("protocol") != contract["protocol"]:
        raise ValueError("episode returned an incompatible task success protocol")
    if state.get("acceptance_scope") != contract["acceptance_scope"] or state.get("success_protocol") != contract["protocol"]:
        raise ValueError("episode acceptance scope changed during execution")
    if _canonical(state.get("command_program")) != _canonical(contract["command_program"]):
        raise ValueError("executed command program differs from artifact contract")
    if task != "ground_lower" and state.get("teacher_version") != contract["teacher_version"]:
        raise ValueError("executed teacher version differs from artifact contract")
    checks = dict(strict.get("checks", {}))
    if not checks or any(not isinstance(value, (bool, np.bool_)) for value in checks.values()):
        raise ValueError("task scorer must provide explicit boolean checks")
    if bool(strict.get("success")) != all(checks.values()):
        raise ValueError("task scorer success contradicts its physical checks")
    independent = not state.get("recovery_events") and not state.get("collision") and not state.get("fault")
    condition_sha = _sha(initial)
    if condition_sha != state.get("initial_condition_sha256"):
        raise ValueError("initial condition hash is invalid")
    return serializable({"task": task, "seed": seed, "success": bool(state["success"] and strict["success"] and independent),
                         "strict_preset_result": strict, "task_result": strict, "author_success": None,
                         "acceptance_scope": contract["acceptance_scope"], "success_protocol": contract["protocol"],
                         "teacher_version": contract["teacher_version"], "command_program_sha256": contract["command_program_sha256"],
                         "initial_condition": initial, "initial_condition_sha256": condition_sha,
                         "recovery_events": state.get("recovery_events", 0), "collision": state.get("collision", False),
                         "fault": state.get("fault"), "termination_reason": state.get("termination_reason"),
                         "counts_as_skill_success": False, "counts_as_ground_touch_success": False})


def summarize_task(rows, contract, *, purpose, evaluator_role="student", ablation=None):
    if not rows:
        raise ValueError("cannot summarize an empty task population")
    successes = sum(bool(row["success"]) for row in rows)
    distinct = len({row["initial_condition_sha256"] for row in rows})
    rate = successes / len(rows)
    accepted = (purpose == "final_test" and evaluator_role == "student" and ablation is None
                and [row["seed"] for row in rows] == list(FINAL_SEEDS) and distinct == len(FINAL_SEEDS)
                and rate >= .9)
    return {"episodes": len(rows), "successes": successes, "distinct_conditions": distinct, "strict_rate": rate,
            "wilson_95": wilson(successes, len(rows)), "author_rate": None,
            "failed_checks": dict(Counter(check for row in rows for check in row["task_result"].get("failed_checks", []))),
            "takeovers": sum(row.get("recovery_events", 0) for row in rows),
            "acceptance_scope": contract["acceptance_scope"], "success_protocol": contract["protocol"],
            "teacher_version": contract["teacher_version"], "command_program_sha256": contract["command_program_sha256"],
            "offline_accepted": accepted, "project_task_accepted": accepted, "realtime_accepted": False,
            "counts_as_project_task_success": accepted, "counts_as_skill_success": False,
            "counts_as_ground_touch_success": False}


def _load_policy(checkpoint, backbone, *, device, precision):
    from lightx2v.models.networks.tinyvla.runtime import MicroDuckTinyVLAPolicy, TinyVLARuntime
    runtime = TinyVLARuntime.from_directory(checkpoint, backbone_path=backbone, device=device, precision=precision)
    return MicroDuckTinyVLAPolicy(runtime), runtime


def _evaluate(tasks, seeds, contracts, *, purpose, policy=None, record_dir=None, asset_root=None, ablation=None,
              record_video=False, deadline=None):
    from dataclasses import replace
    destination = Path(record_dir).resolve() if record_dir else None
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=False)
    rows, summary = [], {}
    reference = purpose in ("teacher_reference", FINAL_TEACHER_PURPOSE)
    for task in tasks:
        latencies, video_classes = [], set()
        with make_environment(task, asset_root=asset_root, render=not reference) as env:
            if hasattr(env, "condition_recipe") and _canonical(env.condition_recipe) != _canonical(contracts[task]["initial_conditions"]):
                raise ValueError("environment factory no longer matches registered initial conditions")
            selected_policy = make_teacher(env) if reference else policy
            for seed in seeds:
                obs = env.reset(seed, task)
                validate_policy_environment(selected_policy.manifest, env, obs)
                selected_policy.reset(obs.episode_id)
                state_ticks, states, action_ticks, actions, targets = [], [], [], [], []
                query_ticks, query_states, cameras, observers = [], [], [], []
                capture = destination is not None and record_video and not reference and len(video_classes) < 2
                while not env.terminated:
                    if deadline is not None and time.monotonic() >= deadline:
                        raise TimeoutError("qualified evaluation exceeded its wall-time budget")
                    obs = env.observe()
                    if not np.array_equal(obs.state[48:], _expected_command(env, obs, task)):
                        raise ValueError("observation command13 differs from the same-tick registered program")
                    if not reference and (obs.rgb is None or obs.camera_tick != obs.tick):
                        raise ValueError("qualified model requires fresh source-tick RGB")
                    if ablation == "black_image":
                        obs = replace(obs, rgb=np.zeros_like(obs.rgb))
                    elif ablation == "language":
                        obs = replace(obs, instruction="Stand still.")
                    elif ablation == "command":
                        state = obs.state.copy()
                        state[48:] = 0
                        obs = replace(obs, state=state)
                    query_ticks.append(obs.tick)
                    query_states.append(obs.state.copy())
                    if capture:
                        cameras.append(obs.rgb.copy())
                        observers.append(env.render_observer().copy())
                    start = time.monotonic()
                    action = selected_policy.predict(obs)
                    latencies.append((time.monotonic() - start) * 1000)
                    validate_action_source(action, obs)
                    execute = 1 if reference else 5
                    if len(action.actions) < execute:
                        raise ValueError("qualified model must return at least five executable actions")
                    env.render_enabled = False
                    try:
                        for raw in action.actions[:execute]:
                            if env.terminated:
                                break
                            state_ticks.append(env.tick)
                            states.append(env.observe().state.copy())
                            action_ticks.append(env.tick)
                            actions.append(raw.copy())
                            env.step(raw)
                            targets.append(env.physics.last_targets.copy())
                    finally:
                        env.render_enabled = not reference
                row = _episode_result(env, task, seed, contracts[task])
                if destination is not None:
                    folder = destination / f"{task}-{seed}"
                    folder.mkdir()
                    trace = env._ground_trace if task == "ground_lower" else env._trajectory
                    np.savez_compressed(folder / "trajectory.npz", state_tick=state_ticks, state61=states,
                                        action_tick=action_ticks, raw_action14=actions, joint_targets=targets,
                                        query_tick=query_ticks, query_state61=query_states, physical_trace=trace)
                    row.update(trajectory_path=str(folder / "trajectory.npz"), trajectory_sha256=file_sha256(folder / "trajectory.npz"))
                    if capture and row["success"] not in video_classes:
                        from .checkpoint_eval import _save_video
                        _save_video(folder / "camera.mp4", cameras)
                        _save_video(folder / "observer.mp4", observers)
                        video_classes.add(row["success"])
                    write_json(folder / "result.json", row)
                rows.append(row)
                if destination is not None:
                    write_json(destination / "progress.json", {"protocol": PROTOCOL, "purpose": purpose, "episodes": rows, "complete": False})
        selected = [row for row in rows if row["task"] == task]
        summary[task] = summarize_task(selected, contracts[task], purpose=purpose,
                                       evaluator_role="official_teacher_reference" if reference else "student", ablation=ablation)
        summary[task]["inference_p95_ms"] = float(np.percentile(latencies, 95)) if latencies else None
    report = {"schema_version": 1, "complete": True, "protocol": PROTOCOL, "acceptance_scope": ACCEPTANCE_SCOPE, "purpose": purpose,
              "evaluator_role": "official_teacher_reference" if reference else "student", "independent_final_test": purpose == "final_test",
              "task_contracts": contracts, "task_contracts_sha256": _sha(contracts), "tasks": summary, "episodes": rows,
              "mean_strict_rate": float(np.mean([value["strict_rate"] for value in summary.values()])),
              "mode": "offline", "ablation": ablation, "physics_waits_for_model": True,
              "semantic_refresh_steps": None if reference else 5, "execute_steps": 1 if reference else 5,
              "nominal_observation_hz": 50 if reference else 10, "counts_as_skill_success": False,
              "offline_accepted": all(value["offline_accepted"] for value in summary.values()), "realtime_accepted": False,
              "complete_author13_accepted": False, "claims_visual_language_autonomy": False, "author_program_reproduction": False,
              "uses_prescribed_command13": True, "hardware_unvalidated": True,
              "evaluator_source_sha256": file_sha256(Path(__file__))}
    if destination is not None:
        write_json(destination / "progress.json", {"protocol": PROTOCOL, "purpose": purpose, "episodes": rows, "complete": True})
    return report


def evaluate_qualified_checkpoint(checkpoint, backbone, *, tasks, seeds, device="cuda", precision="bf16", record_dir=None,
                                  asset_root=None, mode="offline", ablation=None, purpose="validation", record_video=False, deadline=None):
    if mode != "offline" or purpose == "teacher_reference":
        raise ValueError("qualified checkpoint evaluator supports offline student evidence only")
    tasks, seeds = validate_population(tasks, seeds, purpose=purpose, ablation=ablation, record_dir=record_dir)
    checkpoint = Path(checkpoint).resolve()
    manifest = json.loads((checkpoint / "artifact_manifest.json").read_text())
    contracts = validate_qualified_manifest(manifest, tasks)
    policy, runtime = _load_policy(checkpoint, backbone, device=device, precision=precision)
    try:
        validate_qualified_manifest(runtime.manifest, tasks)
        if _canonical(runtime.manifest) != _canonical(manifest) or runtime.refresh_steps != 5:
            raise ValueError("runtime differs from its verified evaluation artifact")
        report = _evaluate(tasks, seeds, contracts, purpose=purpose, policy=policy, record_dir=record_dir, asset_root=asset_root,
                           ablation=ablation, record_video=record_video, deadline=deadline)
        report.update(checkpoint=str(checkpoint), checkpoint_sha256=file_sha256(checkpoint / "model.safetensors"),
                      artifact_manifest_sha256=file_sha256(checkpoint / "artifact_manifest.json"), artifact_bundle_sha256=_sha(manifest),
                      action_padding=runtime.action_padding, precision=runtime.precision,
                      padding_policy=runtime.model.config.semantic_padding_policy)
        return report
    finally:
        del policy, runtime
        gc.collect()
        if str(device).startswith("cuda"):
            import torch
            torch.cuda.empty_cache()


def evaluate_teacher_reference(*, tasks, seeds, record_dir=None, asset_root=None, deadline=None):
    tasks, seeds = validate_population(tasks, seeds, purpose="teacher_reference", record_dir=record_dir)
    contracts = training_contracts()
    return _evaluate(tasks, seeds, {task: contracts[task] for task in tasks}, purpose="teacher_reference",
                     record_dir=record_dir, asset_root=asset_root, deadline=deadline)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint")
    parser.add_argument("--backbone")
    parser.add_argument("--teacher-reference", action="store_true")
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--purpose", choices=PURPOSES[:3], default="validation")
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--seeds", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--asset-root")
    parser.add_argument("--ablation", choices=("black_image", "language", "command"))
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists() or output.with_suffix("").exists():
        raise FileExistsError("evaluation outputs already exist; use a new explicit directory")
    if args.teacher_reference:
        if args.checkpoint or args.purpose != "validation" or args.ablation:
            parser.error("teacher reference cannot be combined with a checkpoint, final purpose or student ablation")
        start = args.seed_start if args.seed_start is not None else 970000
        count = args.seeds if args.seeds is not None else 2
        report = evaluate_teacher_reference(tasks=args.tasks, seeds=range(start, start + count),
                                            asset_root=args.asset_root, record_dir=output.with_suffix(""))
    else:
        if not args.checkpoint:
            parser.error("a student checkpoint is required")
        start = args.seed_start if args.seed_start is not None else (FINAL_SEED_START if args.purpose == "final_test" else 500000)
        count = args.seeds if args.seeds is not None else (100 if args.purpose == "final_test" else 5)
        report = evaluate_qualified_checkpoint(args.checkpoint, args.backbone, tasks=args.tasks, seeds=range(start, start + count),
                                                purpose=args.purpose, device=args.device, precision=args.precision, record_dir=output.with_suffix(""),
                                                asset_root=args.asset_root, ablation=args.ablation, record_video=args.record_video)
    write_json(output, report)
    print(json.dumps(report["tasks"], indent=2))


if __name__ == "__main__":
    main()
