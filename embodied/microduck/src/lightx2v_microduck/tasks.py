"""Versioned task programs, separate from robot physics and policy execution.

The thirteen preset instructions and command schedules reproduce AlexWortega's
vla-tasks.js at 1545800ded064082a67766fbc87f5e25aad71c1e. They do not imply that
the author's private Stage 2.5 weights have been reproduced.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .contracts import TaskSpec

AUTHOR_TASK_REVISION = "1545800ded064082a67766fbc87f5e25aad71c1e"
TASKS_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
AUTHOR_TASK_SOURCE = f"https://huggingface.co/spaces/AlexWortega/microduck-vla-simulator/blob/{AUTHOR_TASK_REVISION}/app/src/game/vla-tasks.js"


@dataclass(frozen=True)
class Stage:
    duration_s: float
    policy: str
    values: tuple[float, ...] = ()
    waveform: str = "fixed"
    period_s: float = 1.0

    def command(self, elapsed_s: float) -> np.ndarray:
        value = np.zeros(13, dtype=np.float32)
        value[: len(self.values)] = self.values
        if self.waveform == "phase":
            phase = 2 * math.pi * elapsed_s / self.period_s
            value[:2] = math.cos(phase), math.sin(phase)
        elif self.waveform == "sway":
            # Preserve the author's exact indices, even though these are the
            # body z/roll slots, rather than silently 'correcting' the task.
            value[9:11] = 0.12 * math.sin(2 * math.pi * elapsed_s), 0.12 * math.cos(2 * math.pi * elapsed_s)
        return value


@dataclass(frozen=True)
class TaskProgram:
    spec: TaskSpec
    stages: tuple[Stage, ...]
    recovery_after_s: float = 0.0
    ball_side: str | None = None

    def context(self, elapsed_s: float) -> tuple[str, np.ndarray]:
        if elapsed_s >= 3.5:
            return "stand", np.zeros(13, dtype=np.float32)
        for stage in self.stages:
            if elapsed_s < stage.duration_s:
                return stage.policy, stage.command(elapsed_s)
            elapsed_s -= stage.duration_s
        return "stand", np.zeros(13, dtype=np.float32)


def _program(task_id, instruction, stages, *, rollers=False, recovery=0.0, ball_side=None):
    return TaskProgram(TaskSpec(task_id, instruction, "author_presets", embodiment="rollers" if rollers else "legs"), tuple(stages), recovery, ball_side)


PROGRAMS = {
    p.spec.task_id: p
    for p in (
        _program("walk_forward", "Walk steadily forward, then stop and balance.", [Stage(3.5, "walk", (0.2, 0, 0))]),
        _program("circle_left", "Dance by walking in a circle to the left.", [Stage(3.5, "walk", (0.16, 0, 1.25))]),
        _program("figure_eight", "Dance in a figure-eight path.", [Stage(1.75, "walk", (0.16, 0, 1.25)), Stage(1.75, "walk", (0.16, 0, -1.25))]),
        _program("zigzag", "Dance forward in a zigzag path.", [Stage(1.15, "walk", (0.17, 0, 1)), Stage(1.15, "walk", (0.17, 0, -1)), Stage(1.2, "walk", (0.17, 0, 0))]),
        _program("sway_circle", "Sway, then dance in a small circle.", [Stage(1.0, "stand", waveform="sway"), Stage(2.5, "walk", (0.15, 0, 1.15))]),
        _program(
            "kick_left",
            "Approach the ball and kick it with your left foot.",
            [Stage(0.3, "walk", (0.15, 0, 0)), Stage(0.2, "stand"), Stage(2.5, "kick_left"), Stage(0.5, "stand")],
            recovery=3.5,
            ball_side="left",
        ),
        _program(
            "kick_right",
            "Approach the ball and kick it with your right foot.",
            [Stage(0.3, "walk", (0.15, 0, 0)), Stage(0.2, "stand"), Stage(2.5, "kick_right"), Stage(0.5, "stand")],
            recovery=3.5,
            ball_side="right",
        ),
        _program("ground_touch", "Lower your beak to touch the ground, then stand back up.", [Stage(2.8, "ground_pick", waveform="phase", period_s=4.0), Stage(0.7, "stand")], recovery=2.8),
        _program("sit_walk", "Sit down once, rise, and walk forward.", [Stage(2.0, "sitstand", (1.0,)), Stage(1.5, "walk", (0.14, 0, 0))], recovery=2.0),
        _program("roll_circle", "Perform a forward roll, then move in a small circle.", [Stage(2.0, "roulade"), Stage(1.5, "walk", (0.13, 0, 1.0))], recovery=2.0),
        _program("roller_glide", "Roll forward on the skates, then stop upright.", [Stage(3.5, "roller", (0.35, 0, 0))], rollers=True),
        _program(
            "roller_crouch",
            "Crouch once on the rollers, then glide forward.",
            [Stage(2.0, "roller_crouch", waveform="phase", period_s=5.0), Stage(1.5, "roller", (0.35, 0, 0))],
            rollers=True,
            recovery=2.0,
        ),
        _program(
            "roller_slalom",
            "Glide through a gentle left-right slalom.",
            [Stage(1.2, "roller", (0.32, 0, 0)), Stage(1.1, "roller_crouch", waveform="phase", period_s=5.0), Stage(1.2, "roller", (0.38, 0, 0))],
            rollers=True,
            recovery=2.3,
        ),
    )
}

TASKS: dict[str, TaskSpec] = {name: program.spec for name, program in PROGRAMS.items()}
TASKS["run_roll_recover"] = TaskSpec("run_roll_recover", "Walk forward, roll once, and recover to standing.", "community_tricks", duration_s=8.1)
TASKS["cone_slalom"] = TaskSpec("cone_slalom", "Weave past all eight physical cones without falling or contact.", "community_tricks", duration_s=31., embodiment="rollers")
TASKS["track_red_ball"] = TaskSpec("track_red_ball", "Find the initially leftmost red ball. Keep tracking that same ball, approach it, then stop.", "visual_goal", duration_s=30., target_color="red")
TASKS["detect_duck"] = TaskSpec("detect_duck", "Remain standing and detect the other duck in the camera image.", "community_perception", duration_s=5.)
TASKS['climb_desk'] = TaskSpec('climb_desk', 'Climb the 27-tread ladder, recover, and stand on the desk for ten seconds.', 'community_climb', duration_s=60.)
for _color in ("red", "blue", "yellow"):
    _id = f"find_{_color}_ball"
    TASKS[_id] = TaskSpec(_id, f"Find the {_color} ball, approach it without hitting obstacles, then stop and wait.", "visual_goal", duration_s=30.0, target_color=_color)

TASKS["ball_apex"] = TaskSpec("ball_apex", "Balance upright on top of the free basketball for ten seconds.", "community_basketball", duration_s=10.0)
TASKS["community_follow"] = TaskSpec(
    "community_follow", "Approach the standing person and hold a comfortable following distance.", "community_follow", duration_s=20.0, stop_distance_m=(0.45, 0.95), stop_hold_s=2.0
)


def get_task(task: str | TaskSpec) -> TaskSpec:
    if isinstance(task, TaskSpec):
        return task
    aliases = {"find_ball": "find_red_ball", "visual_find_ball": "find_red_ball"}
    key = aliases.get(task, task)
    try:
        return TASKS[key]
    except KeyError as exc:
        raise ValueError(f"unknown task {task!r}; available: {', '.join(TASKS)}") from exc


def task_command(task: str | TaskSpec, tick: int, *, recovery: bool = False, task_start_tick: int = 0) -> np.ndarray:
    spec = get_task(task)
    if spec.suite != "author_presets" or recovery or tick >= 175:
        return np.zeros(13, dtype=np.float32)
    return PROGRAMS[spec.task_id].context(max(0, tick - task_start_tick) / 50.0)[1]


def teacher_context(task: str | TaskSpec, tick: int, *, recovery: bool = False, task_start_tick: int = 0) -> tuple[str, np.ndarray]:
    spec = get_task(task)
    if recovery or tick >= round(spec.duration_s * 50) or (spec.suite == "author_presets" and tick >= 175):
        return "stand", np.zeros(13, dtype=np.float32)
    if spec.suite != "author_presets":
        return "walk", np.zeros(13, dtype=np.float32)
    return PROGRAMS[spec.task_id].context(max(0, tick - task_start_tick) / 50.0)


def strict_preset_result(task_id: str, trajectory: np.ndarray, events: set[str], *, collision: bool, recoveries: int, ball_displacement_m: float = 0.0) -> dict:
    """Physical task predicates, explicitly separate from the author's score.

    Trace columns: t,x,y,z,gravity_z,yaw,gyro_y,roll,pitch,planar_speed,
    mouth_tip_z,integrated_absolute_wheel_rotation. These are evaluator-only.
    Thresholds define our v1 benchmark, not an assertion made by upstream.
    """
    v = np.asarray(trajectory, dtype=np.float64)
    if v.ndim != 2 or v.shape[1] != 12 or len(v) < 2:
        raise ValueError("strict preset evaluation needs a physical 12-column trajectory")
    yaw = np.unwrap(v[:, 5])

    def sample(seconds):
        return int(np.clip(np.searchsorted(v[:, 0], seconds), 0, len(v) - 1))

    def yaw_change(a, b):
        return float(yaw[sample(b)] - yaw[sample(a)])

    def path(a, b):
        p = v[sample(a) : sample(b) + 1, 1:3]
        return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())

    terminal = v[v[:, 0] >= 4.25]
    stable = bool(len(terminal) and np.mean((terminal[:, 4] < -0.72) & (terminal[:, 3] > 0.09)) >= 0.8 and terminal[-1, 4] < -0.72)
    stopped = bool(len(terminal) and np.quantile(terminal[:, 9], 0.95) < 0.06)
    checks = {"terminal_stable": stable, "terminal_stopped": stopped, "no_collision": not collision, "no_recovery_takeover": recoveries == 0}
    if task_id == "walk_forward":
        checks.update(forward_progress=v[sample(3.5), 1] - v[0, 1] >= 0.12, limited_lateral_drift=abs(v[sample(3.5), 2] - v[0, 2]) <= 0.10)
    elif task_id == "circle_left":
        checks.update(left_turn=yaw_change(0, 3.5) >= 1.0, travelled=path(0, 3.5) >= 0.12)
    elif task_id == "figure_eight":
        checks.update(left_arc=yaw_change(0, 1.75) >= 0.3, right_arc=yaw_change(1.75, 3.5) <= -0.3, travelled=path(0, 3.5) >= 0.12)
    elif task_id == "zigzag":
        checks.update(left_leg=yaw_change(0, 1.15) >= 0.15, right_leg=yaw_change(1.15, 2.3) <= -0.15, final_forward=path(2.3, 3.5) >= 0.025)
    elif task_id == "sway_circle":
        checks.update(sway=np.ptp(v[: sample(1.0) + 1, 7]) >= 0.025, circle=yaw_change(1.0, 3.5) >= 0.4, travelled=path(1.0, 3.5) >= 0.06)
    elif task_id in ("kick_left", "kick_right"):
        side = task_id.split("_")[1]
        other = "right" if side == "left" else "left"
        checks.update(correct_foot_contact=f"{side}_foot_ball" in events, no_wrong_foot_contact=f"{other}_foot_ball" not in events, ball_moved=ball_displacement_m > 0.03)
    elif task_id == "ground_touch":
        checks.update(beak_touched_floor="mouth_ground" in events or np.min(v[:, 10]) <= 0.02)
    elif task_id == "sit_walk":
        checks.update(sat_down=np.min(v[: sample(2.0) + 1, 3]) < v[0, 3] - 0.015, walked_after_sit=path(2.0, 3.5) >= 0.04)
    elif task_id == "roll_circle":
        checks.update(rolled=np.sum(np.abs(v[: sample(2.0) + 1, 6])) * 0.02 > 2.5, circle_after_roll=yaw_change(2.0, 3.5) >= 0.3, travelled_after_roll=path(2.0, 3.5) >= 0.06)
    elif task_id == "roller_glide":
        checks.update(wheels_rolled=v[-1, 11] >= 0.3, glided=path(0, 3.5) >= 0.15)
    elif task_id == "roller_crouch":
        checks.update(crouched=np.min(v[: sample(2.0) + 1, 3]) < v[0, 3] - 0.012, glided_after_crouch=path(2.0, 3.5) >= 0.10, wheels_rolled=v[-1, 11] >= 0.3)
    elif task_id == "roller_slalom":
        differences = np.diff(yaw[: sample(3.5) + 1])
        checks.update(
            left_turn=float(differences[differences > 0].sum()) >= 0.1, right_turn=float(-differences[differences < 0].sum()) >= 0.1, glided=path(0, 3.5) >= 0.15, wheels_rolled=v[-1, 11] >= 0.3
        )
    else:
        raise ValueError(f"unknown preset {task_id}")
    checks = {name: bool(value) for name, value in checks.items()}
    return {"protocol": "microduck_strict_presets_v1", "success": all(checks.values()), "checks": checks, "failed_checks": [name for name, ok in checks.items() if not ok]}
