"""Independent standing curriculum diagnostic, never a thirteenth-skill score.

The official BAM body, raw14 actuator path and official stand actor are reused.
No entry is added to TASKS or PROGRAMS. The 20/100-seed teacher gate runs without
rendering or model inference; collection separately enables head-camera RGB.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .contracts import JointResidualChunk, PolicyManifest, TaskSpec
from .data import CollectionConfig, collect_dataset, file_sha256, write_json
from .env import MicroDuckEnv
from .physics import PROFILES
from .preset_campaign import wilson


PROTOCOL = "foundation_stability_v1"
TASK = TaskSpec("stand_still", "Stand still and maintain a stable upright posture.", "foundation_stability", duration_s=5.0)
CONDITIONS = {"version": "foundation_initial_v1", "xy_half_range_m": .02, "yaw_half_range_rad": .04, "joint_half_range_rad": .003, "light_half_range": .05, "warmup_ticks": 50, "command": "neutral"}
THRESHOLDS = {"duration_s": 5.0, "terminal_window_s": 1.0, "fall_height_m": .073, "fall_gravity_z": -.58, "terminal_min_height_m": .09, "terminal_max_gravity_z": -.82, "terminal_speed_p95_m_s": .06, "terminal_yaw_span_rad": .20, "terminal_roll_span_rad": .20, "terminal_pitch_span_rad": .20, "maximum_planar_displacement_m": .08}


def score_foundation(trajectory, *, collision=False, recoveries=0, fault="", initialization_failed=False):
    """Score the 12-column physical trace independently of author presets."""
    trace = np.asarray(trajectory, dtype=np.float64)
    if trace.ndim != 2 or trace.shape[1] != 12 or not len(trace) or not np.isfinite(trace).all():
        raise ValueError("foundation scoring requires a finite physical 12-column trajectory")
    if np.any(np.diff(trace[:, 0]) <= 0):
        raise ValueError("foundation timestamps must increase")
    terminal = trace[trace[:, 0] >= THRESHOLDS["duration_s"] - THRESHOLDS["terminal_window_s"] - 1e-7]
    window_complete = bool(len(terminal) >= 50 and terminal[-1, 0] - terminal[0, 0] >= .98 - 1e-7)
    displacement = float(np.max(np.linalg.norm(trace[:, 1:3] - trace[0, 1:3], axis=1)))
    metrics = {"duration_s": float(trace[-1, 0]), "maximum_planar_displacement_m": displacement, "terminal_samples": len(terminal)}
    if len(terminal):
        metrics.update(
            terminal_min_height_m=float(terminal[:, 3].min()),
            terminal_max_gravity_z=float(terminal[:, 4].max()),
            terminal_speed_p95_m_s=float(np.quantile(terminal[:, 9], .95)),
            terminal_yaw_span_rad=float(np.ptp(np.unwrap(terminal[:, 5]))),
            terminal_roll_span_rad=float(np.ptp(np.unwrap(terminal[:, 7]))),
            terminal_pitch_span_rad=float(np.ptp(terminal[:, 8])),
        )
    checks = {
        "duration_complete": bool(trace[-1, 0] >= THRESHOLDS["duration_s"] - 1e-7),
        "initialization_valid": not initialization_failed,
        "no_fall": bool(np.all((trace[:, 3] >= THRESHOLDS["fall_height_m"]) & (trace[:, 4] <= THRESHOLDS["fall_gravity_z"]))),
        "no_collision": not collision,
        "no_takeover": recoveries == 0,
        "no_fault": not bool(fault),
        "limited_displacement": displacement <= THRESHOLDS["maximum_planar_displacement_m"],
        "terminal_window_complete": window_complete,
        "terminal_height": window_complete and metrics["terminal_min_height_m"] > THRESHOLDS["terminal_min_height_m"],
        "terminal_gravity": window_complete and metrics["terminal_max_gravity_z"] < THRESHOLDS["terminal_max_gravity_z"],
        "terminal_speed": window_complete and metrics["terminal_speed_p95_m_s"] < THRESHOLDS["terminal_speed_p95_m_s"],
        "terminal_rotation": window_complete and all(metrics[key] < THRESHOLDS[key] for key in ("terminal_yaw_span_rad", "terminal_roll_span_rad", "terminal_pitch_span_rad")),
    }
    return {"protocol": PROTOCOL, "counts_as_skill_success": False, "success": all(checks.values()), "checks": checks, "failed_checks": [name for name, passed in checks.items() if not passed], "metrics": metrics, "thresholds": dict(THRESHOLDS)}


class FoundationEnv(MicroDuckEnv):
    """Five-second neutral-command stand task with a full-state clone contract."""
    def __init__(self, *, asset_root=None, render=True, image_size=256):
        super().__init__(profile="official_bam", scene="flat", asset_root=asset_root, render=render, image_size=image_size, action_delay_ticks=0)
        self.preset_clock = "program"
        self.initial_condition = None
        self._foundation_initialization_failed = False
        self._foundation_warming_up = False

    def reset(self, seed=0, task="stand_still", options=None):
        if isinstance(task, TaskSpec):
            if (task.task_id, task.suite, task.embodiment, task.duration_s) != (TASK.task_id, TASK.suite, TASK.embodiment, TASK.duration_s):
                raise ValueError("FoundationEnv only accepts its independent stand_still task")
            spec = task
        elif task == TASK.task_id:
            spec = TASK
        else:
            raise ValueError("FoundationEnv only accepts stand_still; author presets remain separate")
        self._foundation_initialization_failed = False
        self._foundation_warming_up = True
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
        rendering = self.render_enabled
        self.render_enabled = False
        try:
            for _ in range(CONDITIONS["warmup_ticks"]):
                state = self.observe().state
                self.step(self.bank.infer("stand", state))
        finally:
            self.render_enabled = rendering
        self._foundation_initialization_failed = bool(self._ever_fallen or self._collision or self._fault or self._recovery_events)
        self._foundation_warming_up = False
        # This is initialization, not an episode recovery. Preserve physics,
        # raw action history, policy state, BAM memory and sensor RNG.
        self.begin_task_at_current_pose()
        self.initial_condition = {"recipe": dict(CONDITIONS), "task": TASK.task_id, "xy": xy.tolist(), "yaw": yaw, "joint_offset": joints.tolist(), "light_scale": light}
        return self.observe()

    def _update_task(self):
        current = self._trajectory[-1]
        fallen = current[3] < THRESHOLDS["fall_height_m"] or current[4] > THRESHOLDS["fall_gravity_z"]
        self._ever_fallen |= bool(fallen)
        # Continue warmup to record its outcome; never silently erase a fall.
        if self._foundation_warming_up:
            return
        self._terminated = bool(self.tick >= 250 or self._ever_fallen or self._collision or self._fault or self._recovery_events)
        if self._terminated:
            self._strict_result = score_foundation(self._trajectory, collision=self._collision, recoveries=self._recovery_events, fault=self._fault, initialization_failed=self._foundation_initialization_failed)
            self._success = self._strict_result["success"]
            self._termination_reason = "foundation_pass" if self._success else "foundation_failed"

    def evaluator_state(self):
        if self.physics is None:
            raise RuntimeError("call reset before evaluator_state")
        _, gravity, _, _ = self.physics.sensors()
        data = self.physics.data
        initial = copy.deepcopy(self.initial_condition)
        result = {
            "episode_id": self.episode_id, "tick": self.tick, "sim_time_s": self.sim_time_s,
            "evaluation_elapsed_s": self.sim_time_s - self._evaluation_origin_s,
            "robot_position": data.xpos[self.physics.trunk].copy(), "robot_yaw": self._trajectory[-1][5],
            "robot_velocity": data.qvel[self.physics.trunk_qvel:self.physics.trunk_qvel + 3].copy(),
            "projected_gravity": gravity.copy(), "joint_targets": self.physics.last_targets.copy(),
            "applied_action": self.applied_action.copy(), "previous_raw_action": self.last_action.copy(),
            "success": self._success, "strict_success": self._success, "terminated": self._terminated,
            "fallen": bool(gravity[2] > THRESHOLDS["fall_gravity_z"] or data.xpos[self.physics.trunk, 2] < THRESHOLDS["fall_height_m"]),
            "ever_fallen": self._ever_fallen, "collision": self._collision, "fault": self._fault,
            "recovery_events": self._recovery_events, "recovery_active": False, "task_start_tick": 0,
            "termination_reason": self._termination_reason, "success_protocol": PROTOCOL,
            "foundation_result": copy.deepcopy(self._strict_result), "author_success": None,
            "counts_as_skill_success": False, "acceptance_scope": "foundation_stability",
            "initial_condition": initial,
            "initial_condition_sha256": hashlib.sha256(json.dumps(initial, sort_keys=True).encode()).hexdigest() if initial else None,
            "model_signature": self._signature, "robot_signature": self.physics.robot_signature(),
            "actuator_profile": PROFILES[self.profile].actuator, "action_delay_ticks": self.action_delay_ticks,
            "scene": self.description.as_dict(), "hardware_validated": False,
        }
        return result

    def clone_state(self):
        snapshot = super().clone_state()
        snapshot["foundation"] = {"protocol": PROTOCOL, "initial_condition": copy.deepcopy(self.initial_condition), "initialization_failed": self._foundation_initialization_failed, "warming_up": self._foundation_warming_up}
        return snapshot

    snapshot = clone_state

    def restore_state(self, snapshot):
        metadata = snapshot["foundation"]
        if metadata["protocol"] != PROTOCOL:
            raise ValueError("foundation snapshot belongs to another protocol")
        super().restore_state(snapshot)
        self.initial_condition = copy.deepcopy(metadata["initial_condition"])
        self._foundation_initialization_failed = metadata["initialization_failed"]
        self._foundation_warming_up = metadata["warming_up"]
        return self.observe()

    restore = restore_state


class FoundationTeacher:
    def __init__(self, env):
        self.env = env
        self.manifest = PolicyManifest("official_stand_foundation_teacher", "joint_residual", "official_61d", "official_bam", metadata={"protocol": PROTOCOL, "counts_as_skill_success": False, "teacher_role": "stand"})

    def reset(self, episode_id=""):
        # The ONNX policy's own state is owned and cloned by the environment.
        self.episode_id = str(episode_id)

    def predict(self, observation):
        if observation.episode_id != self.env.episode_id or observation.tick != self.env.tick:
            raise ValueError("foundation teacher requires its current environment observation")
        if np.any(observation.state[48:61] != 0):
            raise ValueError("foundation standing requires neutral command13")
        raw = self.env.bank.infer("stand", observation.state)
        return JointResidualChunk(raw[None], observation.episode_id, observation.tick, observation.sim_time_s, diagnostics={"teacher_policy": "stand", "teacher_command": [0.] * 13, "phase": "foundation_stand", "protocol": PROTOCOL, "counts_as_skill_success": False})


def teacher_episode(job):
    seed, asset_root, split = job
    with FoundationEnv(asset_root=asset_root, render=False) as env:
        obs = env.reset(seed)
        teacher = FoundationTeacher(env)
        while not env.terminated:
            obs = env.step(teacher.predict(obs))
        status = env.evaluator_state()
        return {"seed": seed, "split": split, "success": status["success"], "result": status["foundation_result"], "initial_condition_sha256": status["initial_condition_sha256"], "model_signature": status["model_signature"], "counts_as_skill_success": False}


def validate_teacher(output, *, asset_root=None, diagnostic=20, validation=100, workers=4):
    output = Path(output).resolve()
    if diagnostic < 1 or validation < 1 or workers < 1 or diagnostic > 1000:
        raise ValueError("positive population/worker counts required; diagnostic seeds must not overlap validation")
    if (output / "teacher-gate.json").exists():
        raise FileExistsError("foundation gate exists; use a new immutable output directory")
    output.mkdir(parents=True, exist_ok=True)
    jobs = [(seed, asset_root, split) for split, seeds in (("diagnostic", range(1000, 1000 + diagnostic)), ("validation", range(2000, 2000 + validation))) for seed in seeds]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        rows = list(executor.map(teacher_episode, jobs))
    validation_rows = [row for row in rows if row["split"] == "validation"]
    passed = sum(row["success"] for row in validation_rows)
    distinct = len({row["initial_condition_sha256"] for row in validation_rows})
    result = {"protocol": PROTOCOL, "task": TASK.task_id, "profile": "official_bam", "conditions": dict(CONDITIONS), "thresholds": dict(THRESHOLDS), "diagnostic_episodes": diagnostic, "validation_episodes": validation, "validation_successes": passed, "validation_success_rate": passed / validation, "distinct_validation_conditions": distinct, "wilson_95": wilson(passed, validation), "eligible_for_foundation_collection": validation >= 100 and distinct == validation and passed / validation >= .95, "counts_as_skill_success": False, "hardware_unvalidated": True, "episodes": rows}
    write_json(output / "teacher-gate.json", result)
    return result


def collect_foundation(output, gate_path, *, asset_root=None, episodes=200, seed=10000):
    gate = json.loads(Path(gate_path).read_text())
    if gate.get("protocol") != PROTOCOL or gate.get("profile") != "official_bam" or gate.get("conditions") != CONDITIONS or gate.get("thresholds") != THRESHOLDS or not gate.get("eligible_for_foundation_collection"):
        raise ValueError("matching foundation teacher must pass its independent gate before collection")
    result = collect_dataset(
        lambda: FoundationEnv(asset_root=asset_root, render=True), FoundationTeacher, output,
        teacher_env_factory=lambda: FoundationEnv(asset_root=asset_root, render=False),
        config=CollectionConfig(tasks=(TASK.task_id,), episodes=episodes, max_steps=250, action_horizon=50, execute_steps=5, seed=seed, goal_tasks=(TASK.task_id,)),
        provenance={"protocol": PROTOCOL, "acceptance_scope": "foundation_stability", "counts_as_skill_success": False, "teacher_gate_sha256": file_sha256(gate_path), "initial_conditions": dict(CONDITIONS)},
    )
    result.update(protocol=PROTOCOL, acceptance_scope="foundation_stability", counts_as_skill_success=False,
                  observation="RGB + official 61D; foundation command13 is neutral",
                  observation_profile="microduck_rgb_proprio_neutral_foundation_v1")
    for row in result["episodes"]:
        row.update(acceptance_scope="foundation_stability", counts_as_skill_success=False)
    write_json(Path(output) / "index.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("teachers", "collect"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--asset-root")
    parser.add_argument("--gate")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--diagnostic", type=int, default=20)
    parser.add_argument("--validation", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=10000)
    args = parser.parse_args()
    if args.command == "teachers":
        result = validate_teacher(args.output, asset_root=args.asset_root, diagnostic=args.diagnostic, validation=args.validation, workers=args.workers)
        print(json.dumps({key: value for key, value in result.items() if key != "episodes"}, indent=2))
    else:
        if not args.gate:
            parser.error("collect requires --gate")
        result = collect_foundation(args.output, args.gate, asset_root=args.asset_root, episodes=args.episodes, seed=args.seed)
        print(json.dumps({"episodes": len(result["episodes"]), "counts_as_skill_success": False, "output": str(Path(args.output).resolve())}))


if __name__ == "__main__":
    main()
