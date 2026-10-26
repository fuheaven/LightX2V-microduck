"""The actor boundary; privileged simulator state must never be added here.

Joint names and observation layout follow pollen-robotics/microduck 5620aa214e6e,
duck-control/src/{model,obs}.rs (Apache-2.0). Models/assets have separate licenses.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, TypeAlias

import numpy as np

JOINT_NAMES = (
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "mouth",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
)
MOUTH_INDEX = 9
POLICY_JOINT_NAMES = tuple(n for n in JOINT_NAMES if n != "mouth")
POLICY_INDICES = tuple(i for i, n in enumerate(JOINT_NAMES) if n != "mouth")
HOME_POSE = (0.0, -0.0873, -0.4579, -0.0049, 0.4530, 0.3491, 0.3491, 0.0, 0.0, 0.0, 0.0, 0.0873, 0.4579, 0.0049, -0.4530)
HOME_POLICY = tuple(HOME_POSE[i] for i in POLICY_INDICES)
OBS_DIM, ACTION_DIM, COMMAND_DIM = 61, 14, 13
CONTROL_DT_S, PHYSICS_DT_S = 0.020, 0.005


def finite_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32).copy()
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape {shape}, got {result.shape}")
    result.setflags(write=False)
    return result


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class RobotContract:
    schema_version: int = 1
    model: str = "microduck"
    embodiment: str = "legs"
    joint_names: tuple[str, ...] = JOINT_NAMES
    policy_joint_names: tuple[str, ...] = POLICY_JOINT_NAMES
    home_pose: tuple[float, ...] = HOME_POSE
    observation_layout: tuple[tuple[str, int], ...] = (("gyro", 3), ("gravity", 3), ("joint_offset", 14), ("joint_velocity", 14), ("previous_raw_action", 14), ("command", 13))
    command_layout: tuple[tuple[str, int], ...] = (("twist", 3), ("head", 4), ("body", 6))
    action_kind: str = "joint_residual"
    control_dt_s: float = CONTROL_DT_S
    physics_dt_s: float = PHYSICS_DT_S
    upstream_runtime: str = "5620aa214e6e304eec00d5c376ac81d3001fb7a3"
    upstream_rl: str = "2b581c641406a48346e696212930ea881c222c52"

    @property
    def sha256(self) -> str:
        return digest(asdict(self))


def pack_observation(gyro, gravity, joint_positions, joint_velocities, previous_raw_action, command) -> np.ndarray:
    """Pack the official 61D contract, never including the mouth or world pose."""
    positions = finite_array(joint_positions, (15,), "joint_positions")
    velocities = finite_array(joint_velocities, (15,), "joint_velocities")
    return finite_array(
        np.concatenate(
            (
                finite_array(gyro, (3,), "gyro"),
                finite_array(gravity, (3,), "gravity"),
                positions[list(POLICY_INDICES)] - np.asarray(HOME_POLICY, dtype=np.float32),
                velocities[list(POLICY_INDICES)],
                finite_array(previous_raw_action, (14,), "previous_raw_action"),
                finite_array(command, (13,), "command"),
            )
        ),
        (61,),
        "state",
    )


def scatter_action(action, *, mouth: float = 0.0, scale: float = 1.0) -> np.ndarray:
    raw = finite_array(action, (14,), "action")
    if not np.isfinite([mouth, scale]).all() or scale <= 0:
        raise ValueError("mouth and positive action scale must be finite")
    result = np.asarray(HOME_POSE, dtype=np.float32).copy()
    result[list(POLICY_INDICES)] += raw * scale
    result[MOUTH_INDEX] = mouth
    return result


@dataclass(frozen=True)
class Observation:
    episode_id: str
    tick: int
    sim_time_s: float
    state: np.ndarray
    rgb: np.ndarray | None = None
    instruction: str = ""
    task_id: str = ""
    camera_tick: int | None = None
    monotonic_ns: int = 0
    tof: np.ndarray | None = None
    camera_metadata: dict[str, Any] = field(default_factory=dict)
    tof_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.tick < 0 or not np.isfinite(self.sim_time_s) or self.sim_time_s < 0:
            raise ValueError("observation tick/time must be nonnegative")
        object.__setattr__(self, "state", finite_array(self.state, (61,), "state"))
        if self.rgb is not None:
            rgb = np.asarray(self.rgb)
            if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
                raise ValueError("rgb must be uint8 HxWx3")
            rgb = rgb.copy()
            rgb.setflags(write=False)
            object.__setattr__(self, "rgb", rgb)
        if self.tof is not None:
            object.__setattr__(self, "tof", finite_array(self.tof, (8, 8), "tof"))


@dataclass(frozen=True)
class JointResidualChunk:
    actions: np.ndarray
    episode_id: str
    source_tick: int
    source_time_s: float = 0.0
    control_dt_s: float = CONTROL_DT_S
    valid_until_tick: int | None = None
    action_profile: str = "raw_residual"
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        actions = np.asarray(self.actions)
        if actions.ndim != 2 or actions.shape[1] != 14 or not 1 <= len(actions) <= 256:
            raise ValueError("actions must have shape (1..256, 14)")
        if self.source_tick < 0 or not np.isfinite(self.source_time_s) or self.source_time_s < 0:
            raise ValueError("source tick/time must be nonnegative")
        if abs(self.control_dt_s - CONTROL_DT_S) > 1e-9:
            raise ValueError("MicroDuck actions must run at 50Hz")
        if self.valid_until_tick is not None and self.valid_until_tick <= self.source_tick:
            raise ValueError("valid_until_tick must follow source_tick")
        object.__setattr__(self, "actions", finite_array(actions, actions.shape, "actions"))


@dataclass(frozen=True)
class SkillIntent:
    skill: str = "walk"
    command: np.ndarray = field(default_factory=lambda: np.zeros(13, dtype=np.float32))
    episode_id: str = ""
    source_tick: int = 0
    duration_ticks: int = 5

    def __post_init__(self):
        object.__setattr__(self, "command", finite_array(self.command, (13,), "command"))
        if self.source_tick < 0 or self.duration_ticks < 1:
            raise ValueError("skill source tick/duration invalid")


PolicyAction: TypeAlias = JointResidualChunk | SkillIntent


@dataclass(frozen=True)
class PolicyManifest:
    policy_id: str
    action_kind: str
    observation_profile: str
    control_profile: str
    embodiment: str = "legs"
    horizon: int = 1
    execution_prefix: int = 1
    recurrent: bool = False
    uses_privileged_state: bool = False
    hardware_validated: bool = False
    artifacts: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.action_kind not in ("joint_residual", "skill_intent"):
            raise ValueError("unsupported action kind")
        if not 1 <= self.execution_prefix <= self.horizon <= 256:
            raise ValueError("invalid action horizon/execution prefix")


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    instruction: str
    suite: str
    duration_s: float = 5.0
    embodiment: str = "legs"
    target_color: str | None = None
    stop_distance_m: tuple[float, float] = (0.25, 0.40)
    stop_hold_s: float = 2.0
    allowed_sensors: tuple[str, ...] = ("rgb", "proprioception")


class PolicyAdapter(Protocol):
    manifest: PolicyManifest

    def reset(self, episode_id: str) -> None: ...
    def predict(self, observation: Observation) -> PolicyAction: ...
