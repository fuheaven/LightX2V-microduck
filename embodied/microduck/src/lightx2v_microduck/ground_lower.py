"""Official near-ground motion, independently scored without claiming touch.

The fixed official ground-pick policy deliberately avoids head contact. Its
entire four-second phase cycle is followed by two seconds of standing. These
project acceptance thresholds are separate from author ``ground_touch``.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import copy
from dataclasses import replace
import hashlib
import json
import math
import multiprocessing
from pathlib import Path

import numpy as np

from .contracts import JointResidualChunk, PolicyManifest, TaskSpec
from .data import write_json
from .env import MicroDuckEnv
from .foundation import CONDITIONS as FOUNDATION_CONDITIONS
from .physics import PROFILES
from .preset_campaign import wilson


PROTOCOL = "ground_lower_v1"
TASK = TaskSpec("ground_lower", "Lower your beak close to the floor without touching it, then stand upright.", "official_ground_lower", duration_s=6.)
COMMAND_PROGRAM = {"version": "official_ground_pick_full_cycle_v1", "control_hz": 50, "ground_pick_period_s": 4., "ground_pick_until_tick": 200, "end_tick": 300, "final_policy": "stand"}
CONDITIONS = {**FOUNDATION_CONDITIONS, "version": "ground_lower_initial_v1", "command": "preset_command13", "command_program": dict(COMMAND_PROGRAM)}
THRESHOLDS = {
    "duration_s": 6., "near_window_start_s": 1., "near_window_end_s": 2.,
    "minimum_mouth_lowering_m": .05, "minimum_mouth_clearance_m": .005,
    "maximum_near_mouth_height_m": .04, "minimum_down_alignment": .5,
    "minimum_continuous_near_s": .08, "maximum_body_tilt_gravity_z": -.58,
    "minimum_body_height_m": .04, "maximum_planar_displacement_m": .08,
    "terminal_window_s": 1., "terminal_mouth_recovery_tolerance_m": .02,
    "terminal_min_height_m": .09, "terminal_max_gravity_z": -.82,
    "terminal_speed_p95_m_s": .06, "terminal_yaw_span_rad": .20,
    "terminal_roll_span_rad": .20, "terminal_pitch_span_rad": .20,
}
OBSERVATION_PROFILE = "microduck_rgb_proprio_preset_ground_lower_v1"
OFFICIAL_RL_REVISION = "2b581c641406a48346e696212930ea881c222c52"


def command_context(tick, *, warming_up=False):
    if isinstance(tick, bool) or not isinstance(tick, (int, np.integer)) or tick < 0:
        raise ValueError("ground_lower tick must be a nonnegative integer")
    command = np.zeros(13, np.float32)
    if not warming_up and tick < COMMAND_PROGRAM["ground_pick_until_tick"]:
        phase = 2 * math.pi * tick / COMMAND_PROGRAM["ground_pick_until_tick"]
        command[:2] = math.cos(phase), math.sin(phase)
        return "ground_pick", command
    return "stand", command


def score_ground_lower(trajectory, *, head_floor_contact=False, collision=False, recoveries=0, fault="", initialization_failed=False):
    """Trace is the base physical 12 columns plus mouth-axis down alignment."""
    trace = np.asarray(trajectory, dtype=np.float64)
    if trace.ndim != 2 or trace.shape[1] != 13 or not len(trace) or not np.isfinite(trace).all() or not np.allclose(np.diff(trace[:, 0]), .02, atol=1e-7, rtol=0):
        raise ValueError("ground_lower requires a finite 13-column physical trace sampled at 50 Hz")
    near = ((trace[:, 0] >= THRESHOLDS["near_window_start_s"] - 1e-7) & (trace[:, 0] <= THRESHOLDS["near_window_end_s"] + 1e-7)
            & (trace[:, 10] >= THRESHOLDS["minimum_mouth_clearance_m"]) & (trace[:, 10] <= THRESHOLDS["maximum_near_mouth_height_m"])
            & (trace[0, 10] - trace[:, 10] >= THRESHOLDS["minimum_mouth_lowering_m"]) & (trace[:, 12] >= THRESHOLDS["minimum_down_alignment"]))
    longest, start = 0., None
    for row, qualifies in zip(trace, near):
        if qualifies:
            start = row[0] if start is None else start
            longest = max(longest, float(row[0] - start))
        else:
            start = None
    terminal = trace[trace[:, 0] >= THRESHOLDS["duration_s"] - THRESHOLDS["terminal_window_s"] - 1e-7]
    window_complete = bool(len(terminal) >= 50 and terminal[-1, 0] - terminal[0, 0] >= .98 - 1e-7)
    metrics = {
        "duration_s": float(trace[-1, 0]), "initial_mouth_height_m": float(trace[0, 10]),
        "minimum_mouth_height_m": float(trace[:, 10].min()),
        "maximum_mouth_lowering_m": float(trace[0, 10] - trace[:, 10].min()),
        "minimum_mouth_down_alignment": float(trace[:, 12].min()),
        "maximum_mouth_down_alignment": float(trace[:, 12].max()),
        "longest_qualified_near_interval_s": longest, "terminal_samples": len(terminal),
        "maximum_planar_displacement_m": float(np.linalg.norm(trace[:, 1:3] - trace[0, 1:3], axis=1).max()),
    }
    if len(terminal):
        metrics.update(terminal_min_mouth_height_m=float(terminal[:, 10].min()), terminal_min_height_m=float(terminal[:, 3].min()),
                       terminal_max_gravity_z=float(terminal[:, 4].max()), terminal_speed_p95_m_s=float(np.quantile(terminal[:, 9], .95)),
                       terminal_yaw_span_rad=float(np.ptp(np.unwrap(terminal[:, 5]))), terminal_roll_span_rad=float(np.ptp(np.unwrap(terminal[:, 7]))),
                       terminal_pitch_span_rad=float(np.ptp(terminal[:, 8])))
    checks = {
        "duration_complete": bool(trace[-1, 0] >= THRESHOLDS["duration_s"] - 1e-7),
        "initialization_valid": not initialization_failed,
        "mouth_lowered": metrics["maximum_mouth_lowering_m"] >= THRESHOLDS["minimum_mouth_lowering_m"],
        "mouth_near_floor_facing_down": longest >= THRESHOLDS["minimum_continuous_near_s"] - 1e-7,
        "mouth_clear_of_floor": metrics["minimum_mouth_height_m"] >= THRESHOLDS["minimum_mouth_clearance_m"],
        "no_head_floor_contact": not head_floor_contact,
        "no_uncontrolled_tip": bool(np.all((trace[:, 3] >= THRESHOLDS["minimum_body_height_m"]) & (trace[:, 4] <= THRESHOLDS["maximum_body_tilt_gravity_z"]))),
        "no_collision": not collision, "no_takeover": recoveries == 0, "no_fault": not bool(fault),
        "limited_displacement": metrics["maximum_planar_displacement_m"] <= THRESHOLDS["maximum_planar_displacement_m"],
        "terminal_window_complete": window_complete,
        "mouth_recovered": window_complete and metrics["terminal_min_mouth_height_m"] >= metrics["initial_mouth_height_m"] - THRESHOLDS["terminal_mouth_recovery_tolerance_m"],
        "terminal_height": window_complete and metrics["terminal_min_height_m"] > THRESHOLDS["terminal_min_height_m"],
        "terminal_gravity": window_complete and metrics["terminal_max_gravity_z"] < THRESHOLDS["terminal_max_gravity_z"],
        "terminal_speed": window_complete and metrics["terminal_speed_p95_m_s"] < THRESHOLDS["terminal_speed_p95_m_s"],
        "terminal_rotation": window_complete and all(metrics[key] < THRESHOLDS[key] for key in ("terminal_yaw_span_rad", "terminal_roll_span_rad", "terminal_pitch_span_rad")),
    }
    return {"protocol": PROTOCOL, "success": all(checks.values()), "checks": checks, "failed_checks": [key for key, value in checks.items() if not value],
            "metrics": metrics, "thresholds": dict(THRESHOLDS), "counts_as_skill_success": False, "counts_as_ground_touch_success": False,
            "uses_preset_command13": True, "claims_visual_language_autonomy": False}


class GroundLowerEnv(MicroDuckEnv):
    def __init__(self, *, asset_root=None, render=True, image_size=256):
        super().__init__(profile="official_bam", scene="flat", asset_root=asset_root, render=render, image_size=image_size, action_delay_ticks=0)
        self.preset_clock = "program"
        self.initial_condition = None
        self._ground_warming_up = False
        self._ground_initialization_failed = False

    def reset(self, seed=0, task="ground_lower", options=None):
        if isinstance(task, TaskSpec):
            if (task.task_id, task.suite, task.embodiment, task.duration_s) != (TASK.task_id, TASK.suite, TASK.embodiment, TASK.duration_s):
                raise ValueError("GroundLowerEnv requires its independent ground_lower task")
            spec = task
        elif task == TASK.task_id:
            spec = TASK
        else:
            raise ValueError("GroundLowerEnv requires ground_lower; ground_touch remains separate")
        self._ground_warming_up, self._ground_initialization_failed = True, False
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
        self.begin_task_at_current_pose()
        rendering, self.render_enabled = self.render_enabled, False
        try:
            for _ in range(CONDITIONS["warmup_ticks"]):
                self.step(self.bank.infer("stand", self.observe().state))
        finally:
            self.render_enabled = rendering
        self._ground_initialization_failed = bool(self._ever_fallen or self._head_floor_events or self._collision or self._fault or self._recovery_events)
        self._ground_warming_up = False
        self.begin_task_at_current_pose()
        self.initial_condition = {"recipe": copy.deepcopy(CONDITIONS), "task": TASK.task_id, "xy": xy.tolist(), "yaw": yaw,
                                  "joint_offset": joints.tolist(), "light_scale": light}
        return self.observe()

    def begin_task_at_current_pose(self):
        self._ground_trace, self._head_floor_events = [], set()
        super().begin_task_at_current_pose()

    def _record_metrics(self, *, include_contacts=True, integrate_wheels=True):
        super()._record_metrics(include_contacts=include_contacts, integrate_wheels=integrate_wheels)
        if self._mouth_tip < 0:
            raise ValueError("ground_lower evaluation requires the official mouth_tip site")
        alignment = -float(self.physics.data.site_xmat[self._mouth_tip].reshape(3, 3)[2, 0])
        self._ground_trace.append((*self._trajectory[-1], alignment))
        if include_contacts:
            model, mj = self.physics.model, self.physics.mj
            neck = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, self.description.robot_prefix + "neck")
            if neck < 0:
                raise ValueError("ground_lower evaluation requires the official neck subtree")
            for pair in self.physics.last_contacts:
                for robot_geom, floor_geom in (pair, pair[::-1]):
                    if model.geom_type[floor_geom] != mj.mjtGeom.mjGEOM_PLANE:
                        continue
                    body = int(model.geom_bodyid[robot_geom])
                    while body > 0:
                        if body == neck:
                            self._head_floor_events.add((self.tick, int(robot_geom), int(floor_geom)))
                            break
                        body = int(model.body_parentid[body])

    def observe(self):
        observation = super().observe()
        _, command = command_context(observation.tick, warming_up=self._ground_warming_up)
        state = observation.state.copy()
        state[48:61] = command
        return replace(observation, state=state)

    def _update_task(self):
        row = self._trajectory[-1]
        self._ever_fallen |= bool(row[3] < THRESHOLDS["minimum_body_height_m"] or row[4] > THRESHOLDS["maximum_body_tilt_gravity_z"])
        if self._ground_warming_up:
            return
        if self.tick >= COMMAND_PROGRAM["end_tick"]:
            self._strict_result = score_ground_lower(self._ground_trace, head_floor_contact=bool(self._head_floor_events), collision=self._collision,
                                                     recoveries=self._recovery_events, fault=self._fault, initialization_failed=self._ground_initialization_failed)
            self._success, self._terminated = self._strict_result["success"], True
            self._termination_reason = "ground_lower_pass" if self._success else "ground_lower_failed"

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
            "fallen": bool(gravity[2] > THRESHOLDS["maximum_body_tilt_gravity_z"] or data.xpos[self.physics.trunk, 2] < THRESHOLDS["minimum_body_height_m"]),
            "ever_fallen": self._ever_fallen, "collision": self._collision, "fault": self._fault,
            "head_floor_contacts": len(self._head_floor_events), "recovery_events": self._recovery_events,
            "recovery_active": False, "task_start_tick": 0, "task_context": command_context(self.tick, warming_up=self._ground_warming_up),
            "termination_reason": self._termination_reason, "success_protocol": PROTOCOL,
            "ground_lower_result": copy.deepcopy(self._strict_result), "author_success": None,
            "counts_as_skill_success": False, "counts_as_ground_touch_success": False, "acceptance_scope": TASK.suite,
            "uses_preset_command13": True, "claims_visual_language_autonomy": False,
            "initial_condition": initial, "initial_condition_sha256": hashlib.sha256(json.dumps(initial, sort_keys=True).encode()).hexdigest() if initial else None,
            "command_program": dict(COMMAND_PROGRAM), "model_signature": self._signature, "robot_signature": self.physics.robot_signature(),
            "actuator_profile": PROFILES[self.profile].actuator, "action_delay_ticks": self.action_delay_ticks, "scene": self.description.as_dict(), "hardware_validated": False,
        }

    def clone_state(self):
        snapshot = super().clone_state()
        snapshot["ground_lower"] = {"protocol": PROTOCOL, "command_program": dict(COMMAND_PROGRAM), "conditions": copy.deepcopy(CONDITIONS),
                                    "initial_condition": copy.deepcopy(self.initial_condition), "warming_up": self._ground_warming_up,
                                    "initialization_failed": self._ground_initialization_failed, "trajectory": copy.deepcopy(self._ground_trace),
                                    "head_floor_events": copy.deepcopy(self._head_floor_events)}
        return snapshot

    snapshot = clone_state

    def restore_state(self, snapshot):
        metadata = snapshot.get("ground_lower", {})
        if metadata.get("protocol") != PROTOCOL or metadata.get("command_program") != COMMAND_PROGRAM or metadata.get("conditions") != CONDITIONS:
            raise ValueError("ground_lower snapshot task, program or initial-condition version differs")
        super().restore_state(snapshot)
        self.initial_condition = copy.deepcopy(metadata["initial_condition"])
        self._ground_warming_up, self._ground_initialization_failed = metadata["warming_up"], metadata["initialization_failed"]
        self._ground_trace, self._head_floor_events = copy.deepcopy(metadata["trajectory"]), copy.deepcopy(metadata["head_floor_events"])
        return self.observe()

    restore = restore_state


class GroundLowerTeacher:
    def __init__(self, env):
        if not isinstance(env, GroundLowerEnv):
            raise ValueError("GroundLowerTeacher requires GroundLowerEnv")
        self.env = env
        self.manifest = PolicyManifest("official_ground_lower_teacher", "joint_residual", "official_61d", "official_bam",
                                       metadata={"protocol": PROTOCOL, "command_program": dict(COMMAND_PROGRAM), "uses_preset_command13": True,
                                                 "counts_as_skill_success": False, "counts_as_ground_touch_success": False, "claims_visual_language_autonomy": False})

    def reset(self, episode_id=""):
        self.episode_id = str(episode_id)

    def predict(self, observation):
        if observation.episode_id != self.env.episode_id or observation.tick != self.env.tick:
            raise ValueError("ground_lower teacher requires its current environment observation")
        role, command = command_context(observation.tick, warming_up=self.env._ground_warming_up)
        if not np.array_equal(observation.state[48:61], command):
            raise ValueError("ground_lower command13 differs from the prescribed phase clock")
        raw = self.env.bank.infer(role, observation.state)
        return JointResidualChunk(raw[None], observation.episode_id, observation.tick, observation.sim_time_s,
                                  diagnostics={"teacher_policy": role, "teacher_command": command.tolist(), "phase": "ground_lower" if role == "ground_pick" else "stand",
                                               "protocol": PROTOCOL, "counts_as_ground_touch_success": False})


def teacher_episode(job):
    seed, asset_root, split = job
    with GroundLowerEnv(asset_root=asset_root, render=False) as env:
        observation = env.reset(seed)
        teacher = GroundLowerTeacher(env)
        while not env.terminated:
            observation = env.step(teacher.predict(observation))
        status = env.evaluator_state()
        return {"seed": seed, "split": split, "success": status["success"], "result": status["ground_lower_result"],
                "initial_condition_sha256": status["initial_condition_sha256"], "model_signature": status["model_signature"],
                "head_floor_contacts": status["head_floor_contacts"], "counts_as_ground_touch_success": False}


def validate_teacher(output, *, asset_root=None, diagnostic=20, validation=100, workers=4):
    output = Path(output).resolve()
    if not 1 <= diagnostic <= 1000 or validation < 1 or workers < 1:
        raise ValueError("positive disjoint diagnostic/validation populations and worker count required")
    if output.exists():
        raise FileExistsError("use a new immutable ground_lower diagnostic directory")
    output.mkdir(parents=True)
    source = Path(__file__).resolve()
    source_text = source.read_bytes()
    (output / "ground_lower_source.py").write_bytes(source_text)
    specification = {"protocol": PROTOCOL, "task": TASK.task_id, "profile": "official_bam", "conditions": copy.deepcopy(CONDITIONS),
                     "command_program": dict(COMMAND_PROGRAM), "thresholds": dict(THRESHOLDS), "official_rl_revision": OFFICIAL_RL_REVISION,
                     "source_sha256": hashlib.sha256(source_text).hexdigest(), "acceptance_scope": TASK.suite,
                     "counts_as_skill_success": False, "counts_as_ground_touch_success": False, "uses_preset_command13": True,
                     "claims_visual_language_autonomy": False, "hardware_unvalidated": True}
    write_json(output / "specification.json", specification)
    rows = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as executor:
        # Commit the diagnostic results before starting independent validation;
        # the specification/thresholds are fixed before either population runs.
        for split, seeds in (("diagnostic", range(7000, 7000 + diagnostic)), ("validation", range(8000, 8000 + validation))):
            current = list(executor.map(teacher_episode, [(seed, asset_root, split) for seed in seeds]))
            rows.extend(current)
            write_json(output / f"{split}.json", {"episodes": current, "successes": sum(row["success"] for row in current), "count": len(current)})
    validation_rows = [row for row in rows if row["split"] == "validation"]
    passed = sum(row["success"] for row in validation_rows)
    distinct = len({row["initial_condition_sha256"] for row in validation_rows})
    result = {**specification, "diagnostic_episodes": diagnostic, "validation_episodes": validation, "validation_successes": passed,
              "validation_success_rate": passed / validation, "distinct_validation_conditions": distinct, "wilson_95": wilson(passed, validation),
              "eligible_for_ground_lower_collection": validation >= 100 and distinct == validation and passed / validation >= .95,
              "failure_checks": dict(Counter(check for row in validation_rows for check in row["result"]["failed_checks"])),
              "episodes": rows, "formal_collection_started": False, "gpu_used": False}
    write_json(output / "teacher-gate.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--asset-root")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--diagnostic", type=int, default=20)
    parser.add_argument("--validation", type=int, default=100)
    args = parser.parse_args()
    result = validate_teacher(args.output, asset_root=args.asset_root, diagnostic=args.diagnostic, validation=args.validation, workers=args.workers)
    print(json.dumps({key: value for key, value in result.items() if key != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
