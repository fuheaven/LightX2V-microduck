"""Teacher continuations from actual student states, without changing roll-in.

A separate MuJoCo environment receives the complete physics, BAM, policy,
evaluator, captured camera and RNG snapshot. The student's simulator and renderer
are left untouched. The teacher is copied with its observed-target/search
history. Only successful continuations from training layouts are eligible.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from .navigation import DECISION_TICKS, NavigationBuffer, VisualSearchTeacher, intent_from_action, student_state
from .navigation_action_contracts import LEGACY_NAVIGATION_CONTRACT, action_contract
from .wam_data import split_for_seed, write_json


def teacher_branch(env, teacher, *, max_decisions=300, label_horizon=32):
    from .env import MicroDuckEnv

    if not isinstance(teacher, VisualSearchTeacher) or teacher.env is not env:
        raise ValueError("branching requires this environment's visibility-gated teacher")
    if env.daemon_backend is not None:
        raise ValueError("teacher branches require an independent local simulation")
    if env.terminated or env.tick % DECISION_TICKS:
        raise ValueError("branch at a live navigation decision boundary")
    if max_decisions < 1 or label_horizon < 1:
        raise ValueError("positive branch and label horizons are required")
    snapshot = env.clone_state()
    with MicroDuckEnv(profile=env.profile, scene=env.scene, asset_root=env.asset_root,
                      render=False, image_size=env.physics.image_size,
                      action_delay_ticks=env.action_delay_ticks) as branch_env:
        branch_env.restore_state(snapshot)
        # Restoring an already captured RGB leaves the renderer uninitialized.
        # The teacher also queries segmentation, so it needs its own context;
        # do not rerender or replace the student's original first observation.
        branch_env.physics.renderer = branch_env.physics.mj.Renderer(
            branch_env.physics.model, height=branch_env.physics.image_size,
            width=branch_env.physics.image_size)
        branch_teacher = copy.deepcopy(teacher, {id(env): branch_env})
        return _teacher_continuation(branch_env, branch_teacher,
                                     max_decisions=max_decisions, label_horizon=label_horizon)


def _teacher_continuation(env, branch_teacher, *, max_decisions, label_horizon):
    """Consume only the disposable branch environment, including on failure."""
    assert branch_teacher.env is env and branch_teacher.planner.env is env
    source_tick = env.tick
    previously_observed = branch_teacher.last_seen_position is not None
    frames, states, ticks, actions, diagnostics = [], [], [], [], []
    trace_states, trace_actions, trace_commands, trace_skills, trace_qpos = [], [], [], [], []
    env.render_enabled = True
    observation = env.observe()
    frames.append(observation.rgb.copy())
    states.append(student_state(observation).copy())
    ticks.append(observation.tick)
    for _ in range(max_decisions):
        action = branch_teacher.predict(observation)
        diagnostic = copy.deepcopy(branch_teacher.last_diagnostics)
        start_tick = env.tick
        env.render_enabled = False
        for _ in range(DECISION_TICKS):
            trace_states.append(observation.state.copy())
            trace_qpos.append(env.physics.data.qpos.copy())
            intent = intent_from_action(action, episode_id=env.episode_id, tick=env.tick)
            trace_commands.append(intent.command.copy())
            trace_skills.append(1 if intent.skill == "walk" else 0)
            observation = env.step(intent)
            trace_actions.append(env.last_action.copy())
            if env.terminated:
                break
        env.render_enabled = True
        observation = env.observe()
        if env.tick - start_tick == DECISION_TICKS:
            actions.append(action.copy())
            diagnostics.append(diagnostic)
            frames.append(observation.rgb.copy())
            states.append(student_state(observation).copy())
            ticks.append(observation.tick)
        if env.terminated:
            break
    result = copy.deepcopy(env.evaluator_state())
    unsafe = any(d.get("planner_phase", d.get("phase")) in ("recover", "no_path") for d in diagnostics)
    qualified = bool(result["success"] and result["strict_success"] and not unsafe and not result["collision"]
                     and not result["ever_fallen"] and result["recovery_events"] == 0)
    split = split_for_seed(env.seed)
    count = min(len(actions), label_horizon)
    # Keep complete teacher recovery separately; training's action/video
    # targets must start at the exact student state, never a new reset.
    arrays = {
        "rgb": np.stack(frames[:count + 1]),
        "proprio": np.stack(states[:count + 1]),
        "action": np.asarray(actions[:count], np.float32).reshape(-1, 5),
        "ticks": np.asarray(ticks[:count + 1], np.int64),
        "continuation_action": np.asarray(actions, np.float32).reshape(-1, 5),
        "continuation_state61": np.asarray(trace_states, np.float32).reshape(-1, 61),
        "continuation_raw_action": np.asarray(trace_actions, np.float32).reshape(-1, 14),
        # Include incomplete final decisions. These diagnostic tensors allow a
        # second simulator to replay recorded commands without querying a teacher.
        "continuation_command13": np.asarray(trace_commands, np.float32).reshape(-1, 13),
        "continuation_skill_id": np.asarray(trace_skills, np.int8),
        "continuation_qpos": np.asarray(trace_qpos, np.float64).reshape(-1, env.physics.model.nq),
    }
    report = {
        "task": env.task.task_id, "seed": env.seed, "split": split,
        "source_tick": source_tick, "instruction": env.task.instruction,
        "label_decisions": count, "continuation_decisions": len(actions),
        "teacher_continuation_success": qualified,
        "eligible_training_supervision": bool(qualified and split == "train" and count > 0),
        "initial_teacher_target_previously_observed": previously_observed,
        "branch_isolation": "separate_environment_and_renderer",
        "teacher_diagnostics": diagnostics, "result": result,
        "scope": "teacher branch from student state; not student task success",
        "hardware_validated": False,
    }
    return arrays, report


def collect_student_rollout(env, policy, output, *, task, seed, branch_ticks,
                             max_ticks=1500, branch_max_decisions=300, record_only=False,
                             training_admission=None):
    """Offline student roll-in with successful teacher labels in separate files.

    This low-level collector enforces train splits. Its CLI additionally checks
    the qualified dataset and the permanently reserved independent test seeds.
    """
    if training_admission is None:
        if split_for_seed(seed) != "train":
            raise ValueError("held-out layout cannot be used for training roll-in")
        if (env.profile != "official_bam" or (env.scene != "room_clear"
                and not (record_only is True and env.scene == "room_curriculum_zero"))):
            raise ValueError("obstacle roll-in requires an explicitly qualified training admission")
    else:
        from .navigation_rollin_admission import GuardedTrainingCase
        if not isinstance(training_admission, GuardedTrainingCase):
            raise ValueError("invalid guarded training admission")
        training_admission.validate_request(task=task, seed=seed, scene=env.scene,
            profile=env.profile, contract=policy.metadata.get('contract'), record_only=record_only)
    if getattr(policy, "metadata", {}).get("uses_privileged_state") is not False:
        raise ValueError("student must declare sensor-only inputs")
    if type(record_only) is not bool:
        raise ValueError("record_only must be an explicit boolean")
    contract = action_contract(policy.metadata.get('contract', LEGACY_NAVIGATION_CONTRACT)).name
    if contract != LEGACY_NAVIGATION_CONTRACT and not record_only:
        raise ValueError("extended-contract roll-in must be record-only; teacher labels require separate replay")
    if record_only and (branch_ticks or max_ticks != 1500):
        raise ValueError("record-only collection requires no branches and the complete 1500-tick task budget")
    if ((not branch_ticks and not record_only) or len(branch_ticks) != len(set(branch_ticks))
            or any(t < 0 or t % 5 or t >= max_ticks for t in branch_ticks)):
        raise ValueError("unique branch decision ticks within the rollout are required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    env.render_enabled = False
    env.reset(seed, task)
    for _ in range(50):
        env.step(intent_from_action([0, 0, 0, 0, 1], episode_id=env.episode_id, tick=env.tick))
    env.begin_task_at_current_pose()
    if training_admission is not None:
        training_admission.verify_initial(env)
    policy.reset(env.episode_id)
    teacher = None if record_only else VisualSearchTeacher(env)
    buffer = NavigationBuffer(max_wall_age_s=1e9, contract=contract)  # declared offline roll-in
    buffer.reset(env.episode_id)
    branches, trace, frames, camera_ticks, positions = [], [], [], [], []
    proprio, predictions = [], []
    while not env.terminated and env.tick < max_ticks:
        env.render_enabled = env.tick % DECISION_TICKS == 0
        observation = env.observe()
        positions.append(env.physics.data.qpos.copy())
        if env.tick % DECISION_TICKS == 0:
            frames.append(observation.rgb.copy())
            camera_ticks.append(env.tick)
            proprio.append(student_state(observation))
            if env.tick in branch_ticks:
                arrays, report = teacher_branch(env, teacher, max_decisions=branch_max_decisions)
                path = output / f"branch-{env.tick:04d}"
                path.mkdir()
                np.savez_compressed(path / "navigation.npz", **arrays)
                report.update(student_model=policy.metadata, source_rollout=output.name,
                              navigation_sha256=hashlib.sha256((path / "navigation.npz").read_bytes()).hexdigest())
                report = json.loads(json.dumps(report, default=lambda v: v.tolist()))
                write_json(path / "report.json", report)
                branches.append({"path": path.name, **{k: report[k] for k in (
                    "source_tick", "teacher_continuation_success", "eligible_training_supervision",
                    "label_decisions", "continuation_decisions", "navigation_sha256")}})
            if teacher is not None:
                teacher.predict(observation)  # update observation history only
            chunk = policy.predict(observation)
            if (chunk.episode_id, chunk.source_tick, chunk.source_monotonic_ns) != (observation.episode_id, observation.tick, observation.monotonic_ns):
                raise ValueError("student response does not match its current observation")
            if not buffer.submit(chunk, observation):
                raise ValueError(f"student roll-in action rejected: {buffer.last_reason}")
            predictions.append({"source_tick": chunk.source_tick,
                                "source_monotonic_ns": chunk.source_monotonic_ns,
                                "episode_id": chunk.episode_id,
                                "instruction": observation.instruction,
                                 "valid_until_tick": chunk.valid_until_tick,
                                 "action_contract": chunk.contract,
                                "observed_proprio48": proprio[-1].tolist(),
                                "submitted_actions": chunk.actions.tolist(), "accepted": True})
            for key in ('history_source_ticks', 'history_padding'):
                if key in chunk.diagnostics:
                    predictions[-1][key] = chunk.diagnostics[key]
        intent, evidence = buffer.consume(observation)
        trace.append({"tick": env.tick, "skill": intent.skill, "command": intent.command.tolist(), **evidence})
        env.render_enabled = False
        env.step(intent)
    result = env.evaluator_state()
    np.savez_compressed(output / "recording.npz", rgb=np.stack(frames), camera_ticks=np.asarray(camera_ticks),
                        proprio48=np.stack(proprio), qpos=np.stack(positions), physics_ticks=np.arange(len(positions)))
    write_json(output / "control.json", trace)
    write_json(output / "predictions.json", predictions)
    report = {"status": "complete", "mode": "offline", "task": task, "seed": seed, "split": "train",
              "instruction": env.task.instruction,
              "action_contract": contract,
              "policy": policy.metadata, "evaluator": result, "branches": branches,
              "student_task_success": bool(result["success"] and buffer.fallback_ticks == 0),
              "fallback_control_ticks": buffer.fallback_ticks,
              "eligible_branches": sum(b["eligible_training_supervision"] for b in branches),
              "collection_mode": "student_record_only" if record_only else "student_and_teacher_branches",
              "task_terminated": bool(env.terminated),
              "maximum_control_ticks": max_ticks,
              "teacher_queries_during_rollin": 0 if record_only else len(predictions),
               "evidence": "training data collection; not validation or acceptance", "hardware_validated": False}
    if training_admission is not None:
        report['training_admission'] = training_admission.binding()
    report = json.loads(json.dumps(report, default=lambda v: v.tolist()))
    write_json(output / "report.json", report)
    return report
