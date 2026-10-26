"""Prepared second curriculum stage: prescribed low-speed walk, then stand.

This is the project_v028_stand25 program verified in CPU diagnosis. It is not
an author preset reproduction or evidence of language/visual autonomy. Nothing
is registered in TASKS/PROGRAMS. Its teacher gate and collection remain scoped
to this independent curriculum.
"""
from __future__ import annotations

import copy
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np

from .contracts import JointResidualChunk, PolicyManifest, TaskSpec
from .data import CollectionConfig, collect_dataset, file_sha256, write_json
from .env import MicroDuckEnv
from .physics import PROFILES
from .preset_campaign import CONDITIONS as PRESET_INITIAL_CONDITIONS
from .preset_campaign import wilson
from .tasks import strict_preset_result


PROTOCOL = "foundation_walk_v1"
TASK = TaskSpec("walk_forward_stop", "Walk slowly forward, then stop and remain upright.", "foundation_walk", duration_s=5.)
COMMAND_PROGRAM = {"version": "project_v028_stand25", "control_hz": 50, "walk_until_tick": 125, "end_tick": 250, "forward_command": .28, "walking_policy": "walk", "stopping_policy": "stand"}
CONDITIONS = {**PRESET_INITIAL_CONDITIONS, "version": "foundation_walk_initial_v1", "command": "preset_command13", "command_program": copy.deepcopy(COMMAND_PROGRAM)}
OBSERVATION_PROFILE = "microduck_rgb_proprio_preset_foundation_walk_v1"
ACCEPTANCE_SCOPE = "foundation_walk"


def command_context(tick, *, warming_up=False):
    """One control clock supplies both observed command13 and teacher policy."""
    if isinstance(tick, bool) or not isinstance(tick, (int, np.integer)) or tick < 0:
        raise ValueError("foundation walk tick must be a nonnegative integer")
    command = np.zeros(13, np.float32)
    if not warming_up and tick < COMMAND_PROGRAM["walk_until_tick"]:
        command[0] = COMMAND_PROGRAM["forward_command"]
        return "walk", command
    return "stand", command


def score_foundation_walk(trajectory, *, collision=False, recoveries=0, fault="", initialization_failed=False):
    """Keep all walk_forward physical thresholds unchanged and explicit."""
    trace = np.asarray(trajectory)
    if trace.ndim != 2 or trace.shape[1] != 12 or not len(trace) or not np.isfinite(trace).all() or np.any(np.diff(trace[:, 0]) <= 0):
        raise ValueError("foundation walk scoring requires a finite physical trajectory with increasing timestamps")
    if trace[-1, 0] < TASK.duration_s - 1e-7:
        raise ValueError("foundation walk scoring requires the complete five-second trajectory")
    result = strict_preset_result("walk_forward", trajectory, set(), collision=collision, recoveries=recoveries)
    checks = {**result["checks"], "no_fall": bool(np.all((trace[:, 3] >= .073) & (trace[:, 4] <= -.58))),
              "no_fault": not bool(fault), "initialization_valid": not initialization_failed}
    # Diagnostic timing uses exactly the acceptance predicate. It never changes
    # success, checkpoint selection, or the required zero-fall population.
    fallen = np.flatnonzero((trace[:, 3] < .073) | (trace[:, 4] > -.58))
    first_fall_s = float(trace[fallen[0], 0]) if len(fallen) else None
    diagnostics = {"protocol": "walk_failure_timing_v1", "first_fall_time_s": first_fall_s,
                   "observed_duration_s": float(trace[-1, 0]), "fall_observed": bool(len(fallen)),
                   "first_fall_phase": ("walk" if first_fall_s < COMMAND_PROGRAM["walk_until_tick"] / COMMAND_PROGRAM["control_hz"] - 1e-7 else "stop") if len(fallen) else None,
                   "minimum_height_m": float(trace[:, 3].min()), "maximum_gravity_z": float(trace[:, 4].max()),
                   "used_for_acceptance": False, "used_for_checkpoint_selection": False}
    return {"protocol": PROTOCOL, "reference_strict_result": result, "checks": checks,
            "success": all(checks.values()), "failed_checks": [key for key, passed in checks.items() if not passed],
            "curriculum_protocol": PROTOCOL, "reference_task": "walk_forward", "counts_as_skill_success": False,
            "uses_preset_command13": True, "claims_visual_language_autonomy": False, "diagnostics": diagnostics}


class FoundationWalkEnv(MicroDuckEnv):
    def __init__(self, *, asset_root=None, render=True, image_size=256):
        super().__init__(profile="official_bam", scene="flat", asset_root=asset_root, render=render, image_size=image_size, action_delay_ticks=0)
        self.preset_clock = "program"
        self.initial_condition = None
        self._walk_warming_up = False

    def reset(self, seed=0, task="walk_forward_stop", options=None):
        if isinstance(task, TaskSpec):
            if (task.task_id, task.suite, task.embodiment, task.duration_s) != (TASK.task_id, TASK.suite, TASK.embodiment, TASK.duration_s):
                raise ValueError("FoundationWalkEnv only accepts its walk_forward_stop curriculum task")
            spec = task
        elif task == TASK.task_id:
            spec = TASK
        else:
            raise ValueError("FoundationWalkEnv accepts walk_forward_stop, not stand_still or author presets")
        self._walk_warming_up = True
        self.initial_condition = None
        super().reset(seed, spec, options)
        rng = np.random.default_rng(seed)
        xy = rng.uniform(-CONDITIONS["xy_half_range_m"], CONDITIONS["xy_half_range_m"], 2)
        yaw = float(rng.uniform(-CONDITIONS["yaw_half_range_rad"], CONDITIONS["yaw_half_range_rad"]))
        joints = rng.uniform(-CONDITIONS["joint_half_range_rad"], CONDITIONS["joint_half_range_rad"], 14)
        light = float(rng.uniform(1 - CONDITIONS["light_half_range"], 1 + CONDITIONS["light_half_range"]))
        self.physics.reset(robot_xy=xy, robot_yaw=yaw, seed=seed)
        self.physics.data.qpos[self.physics.qpos_indices] += joints
        controller = self.physics.bam_controller
        controller.reset(self.physics.data.qpos)
        controller.last_ts = self.physics.data.time
        controller.q_target[:] = self.physics.last_targets
        self.physics.model.light_diffuse[:] *= light
        self.physics.mj.mj_forward(self.physics.model, self.physics.data)
        rendering, self.render_enabled = self.render_enabled, False
        try:
            for _ in range(CONDITIONS["warmup_ticks"]):
                self.step(self.bank.infer("stand", self.observe().state))
        finally:
            self.render_enabled = rendering
        self._walk_warming_up = False
        if self._ever_fallen or self._collision or self._fault or self._recovery_events:
            raise RuntimeError("foundation walk initialization failed; refusing to erase an invalid warmup")
        # Only the task/scoring clock restarts. Physical time, raw action
        # history, BAM memory and sensor RNG continue through initialization.
        self.begin_task_at_current_pose()
        self.initial_condition = {"recipe": copy.deepcopy(CONDITIONS), "task": TASK.task_id, "xy": xy.tolist(), "yaw": yaw,
                                  "joint_offset": joints.tolist(), "light_scale": light}
        return self.observe()

    def observe(self):
        observation = super().observe()
        _, command = command_context(observation.tick, warming_up=self._walk_warming_up)
        state = observation.state.copy()
        state[48:61] = command
        return replace(observation, state=state)

    def _update_task(self):
        current = self._trajectory[-1]
        self._ever_fallen |= bool(current[3] < .073 or current[4] > -.58)
        if self._walk_warming_up:
            return
        if self.tick >= COMMAND_PROGRAM["end_tick"]:
            self._strict_result = score_foundation_walk(self._trajectory, collision=self._collision, recoveries=self._recovery_events, fault=self._fault)
            self._success = self._strict_result["success"]
            self._terminated = True
            self._termination_reason = "foundation_walk_pass" if self._success else "foundation_walk_failed"

    def evaluator_state(self):
        if self.physics is None:
            raise RuntimeError("call reset before evaluator_state")
        _, gravity, _, _ = self.physics.sensors()
        data, initial = self.physics.data, copy.deepcopy(self.initial_condition)
        return {
            "episode_id": self.episode_id, "tick": self.tick, "sim_time_s": self.sim_time_s,
            "evaluation_elapsed_s": self.sim_time_s - self._evaluation_origin_s,
            "robot_position": data.xpos[self.physics.trunk].copy(), "robot_yaw": self._trajectory[-1][5],
            "robot_velocity": data.qvel[self.physics.trunk_qvel:self.physics.trunk_qvel + 3].copy(),
            "projected_gravity": gravity.copy(), "joint_targets": self.physics.last_targets.copy(),
            "applied_action": self.applied_action.copy(), "previous_raw_action": self.last_action.copy(),
            "success": self._success, "strict_success": self._success, "terminated": self._terminated,
            "fallen": bool(gravity[2] > -.58 or data.xpos[self.physics.trunk, 2] < .073),
            "ever_fallen": self._ever_fallen, "collision": self._collision, "fault": self._fault,
            "recovery_events": self._recovery_events, "recovery_active": False, "task_start_tick": 0,
            "task_context": command_context(self.tick, warming_up=self._walk_warming_up),
            "termination_reason": self._termination_reason, "success_protocol": PROTOCOL,
            "foundation_walk_result": copy.deepcopy(self._strict_result), "strict_preset_result": copy.deepcopy(self._strict_result),
            "author_success": None, "counts_as_skill_success": False, "acceptance_scope": "foundation_walk",
            "uses_preset_command13": True, "claims_visual_language_autonomy": False,
            "command_program": copy.deepcopy(COMMAND_PROGRAM), "initial_condition": initial,
            "initial_condition_sha256": hashlib.sha256(json.dumps(initial, sort_keys=True).encode()).hexdigest() if initial else None,
            "model_signature": self._signature, "robot_signature": self.physics.robot_signature(),
            "actuator_profile": PROFILES[self.profile].actuator, "action_delay_ticks": self.action_delay_ticks,
            "scene": self.description.as_dict(), "hardware_validated": False,
        }

    def clone_state(self):
        snapshot = super().clone_state()
        snapshot["foundation_walk"] = {"protocol": PROTOCOL, "command_program": copy.deepcopy(COMMAND_PROGRAM),
                                       "initial_condition": copy.deepcopy(self.initial_condition), "warming_up": self._walk_warming_up}
        return snapshot

    snapshot = clone_state

    def restore_state(self, snapshot):
        metadata = snapshot.get("foundation_walk", {})
        if metadata.get("protocol") != PROTOCOL or metadata.get("command_program") != COMMAND_PROGRAM:
            raise ValueError("foundation walk snapshot belongs to another task or command program")
        super().restore_state(snapshot)
        self.initial_condition = copy.deepcopy(metadata["initial_condition"])
        self._walk_warming_up = metadata["warming_up"]
        return self.observe()

    restore = restore_state


class FoundationWalkTeacher:
    def __init__(self, env):
        if not isinstance(env, FoundationWalkEnv):
            raise ValueError("FoundationWalkTeacher requires FoundationWalkEnv")
        self.env = env
        self.manifest = PolicyManifest("official_walk_stop_foundation_teacher", "joint_residual", "official_61d", "official_bam",
                                       metadata={"protocol": PROTOCOL, "command_program": copy.deepcopy(COMMAND_PROGRAM),
                                                 "uses_preset_command13": True, "claims_visual_language_autonomy": False,
                                                 "counts_as_skill_success": False})

    def reset(self, episode_id=""):
        self.episode_id = str(episode_id)

    def predict(self, observation):
        if observation.episode_id != self.env.episode_id or observation.tick != self.env.tick:
            raise ValueError("foundation walk teacher requires its current environment observation")
        role, command = command_context(observation.tick, warming_up=self.env._walk_warming_up)
        if not np.array_equal(observation.state[48:61], command):
            raise ValueError("foundation walk command13 differs from the prescribed episode clock")
        raw = self.env.bank.infer(role, observation.state)
        return JointResidualChunk(raw[None], observation.episode_id, observation.tick, observation.sim_time_s,
                                  diagnostics={"teacher_policy": role, "teacher_command": command.tolist(),
                                               "phase": "foundation_walk" if role == "walk" else "foundation_stop",
                                               "protocol": PROTOCOL, "counts_as_skill_success": False,
                                               "uses_preset_command13": True, "claims_visual_language_autonomy": False})


def teacher_episode(job):
    seed, asset_root, split = job
    with FoundationWalkEnv(asset_root=asset_root, render=False) as env:
        observation = env.reset(seed)
        teacher = FoundationWalkTeacher(env)
        while not env.terminated:
            observation = env.step(teacher.predict(observation))
        status = env.evaluator_state()
        return {"seed": seed, "split": split, "success": status["success"], "result": status["foundation_walk_result"],
                "initial_condition_sha256": status["initial_condition_sha256"], "model_signature": status["model_signature"],
                "ever_fallen": status["ever_fallen"], "counts_as_skill_success": False}


def validate_teacher(output, *, asset_root=None, diagnostic=20, validation=100, workers=4):
    if not 20 <= diagnostic <= 1000 or validation < 100 or workers < 1:
        raise ValueError("walk teacher gate requires 20 diagnostic and 100 distinct validation conditions")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    jobs = [(seed, asset_root, split) for split, seeds in (("diagnostic", range(1000, 1000 + diagnostic)), ("validation", range(2000, 2000 + validation))) for seed in seeds]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(teacher_episode, jobs))
    validation_rows = [row for row in rows if row["split"] == "validation"]
    passed = sum(row["success"] for row in validation_rows)
    distinct = len({row["initial_condition_sha256"] for row in validation_rows})
    report = {"protocol": PROTOCOL, "task": TASK.task_id, "profile": "official_bam", "conditions": copy.deepcopy(CONDITIONS),
              "command_program": copy.deepcopy(COMMAND_PROGRAM), "diagnostic_episodes": diagnostic, "validation_episodes": validation,
              "validation_successes": passed, "validation_success_rate": passed / validation, "distinct_validation_conditions": distinct,
              "wilson_95": wilson(passed, validation), "eligible_for_foundation_walk_collection": passed / validation >= .95 and distinct == validation,
              "counts_as_skill_success": False, "hardware_unvalidated": True, "episodes": rows,
              "source_sha256": file_sha256(__file__), "original_author_thresholds_changed": False}
    write_json(output / "teacher-gate.json", report)
    return report


def validate_collection_gate(gate_path):
    gate = json.loads(Path(gate_path).read_text())
    total, passed = gate.get("validation_episodes", 0), gate.get("validation_successes", 0)
    if (gate.get("protocol") != PROTOCOL or gate.get("profile") != "official_bam" or gate.get("task") != TASK.task_id
            or gate.get("conditions") != CONDITIONS or gate.get("command_program") != COMMAND_PROGRAM
            or gate.get("counts_as_skill_success") is not False or not gate.get("eligible_for_foundation_walk_collection")
            or total < 100 or gate.get("distinct_validation_conditions") != total or not .95 <= passed / max(total, 1) <= 1
            or gate.get("validation_success_rate") != passed / max(total, 1)):
        raise ValueError("matching walk teacher must pass its independent 100-condition gate")
    return gate


def collect_walk(output, gate_path, *, asset_root=None, episodes=200, seed=20000, mode="teacher", student=None, teacher_rollin_probability=0.0):
    validate_collection_gate(gate_path)
    if mode not in ("teacher", "dagger") or (mode == "dagger" and student is None):
        raise ValueError("walk collection requires a teacher, or an explicit student for real MuJoCo DAgger")
    result = collect_dataset(
        lambda: FoundationWalkEnv(asset_root=asset_root, render=True), FoundationWalkTeacher, output,
        teacher_env_factory=lambda: FoundationWalkEnv(asset_root=asset_root, render=False), student=student,
        config=CollectionConfig(tasks=(TASK.task_id,), episodes=episodes, max_steps=250, action_horizon=50, execute_steps=5,
                                seed=seed, mode=mode, teacher_rollin_probability=teacher_rollin_probability),
        provenance={"protocol": PROTOCOL, "acceptance_scope": ACCEPTANCE_SCOPE, "counts_as_skill_success": False,
                    "teacher_gate_sha256": file_sha256(gate_path), "initial_conditions": copy.deepcopy(CONDITIONS), "command_program": copy.deepcopy(COMMAND_PROGRAM)},
    )
    result.update(protocol=PROTOCOL, acceptance_scope=ACCEPTANCE_SCOPE, counts_as_skill_success=False,
                  observation="RGB + official 61D; prescribed walk/stop command13 from the same control tick", observation_profile=OBSERVATION_PROFILE)
    for row in result["episodes"]:
        row.update(counts_as_skill_success=False, acceptance_scope=ACCEPTANCE_SCOPE)
    write_json(Path(output) / "index.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("teachers", "collect"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--asset-root")
    parser.add_argument("--gate")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20000)
    args = parser.parse_args()
    if args.command == "teachers":
        result = validate_teacher(args.output, asset_root=args.asset_root, workers=args.workers)
        print(json.dumps({key: value for key, value in result.items() if key != "episodes"}, indent=2))
    else:
        if not args.gate:
            parser.error("collect requires --gate")
        result = collect_walk(args.output, args.gate, asset_root=args.asset_root, episodes=args.episodes, seed=args.seed)
        print(json.dumps({"episodes": len(result["episodes"]), "counts_as_skill_success": False}))


if __name__ == "__main__":
    main()
