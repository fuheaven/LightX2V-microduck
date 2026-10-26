"""High-level navigation above the pinned, 50 Hz official locomotion actors.

The model never emits joint targets and never receives planner/world truth.
Each decision is [vx, vy, yaw_rate, head_yaw, stand_probability]; SI units,
body-frame velocities, with stand selected at probability >= .5.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path

import numpy as np

from .contracts import Observation, SkillIntent
from .navigation_action_contracts import action_contract


NAVIGATION_CONTRACT = "microduck_navigation_v1"
NAVIGATION_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
ACTION_NAMES = ("vx_mps", "vy_mps", "yaw_rate_rps", "head_yaw_rad", "stand_probability")
ACTION_LOW = np.array([-.15, -.15, -1.1, -.45, 0.], dtype=np.float32)
ACTION_HIGH = np.array([.30, .15, 1.1, .45, 1.], dtype=np.float32)
DECISION_TICKS = 5  # 10 Hz decisions, official ONNX still observes/acts at 50 Hz.
STATE_DIM = 48  # gyro, gravity, joints, joint velocities, previous raw action.


def navigation_expiry_limits(max_action_age_ms=500, *, realtime=True):
    """One declared TTL for the 50 Hz physical clock and the execution clock.

    Historical callers retain 500 ms. Offline evaluation disables only the
    wall-clock check, since its physics intentionally waits for inference.
    """
    # Explicit simulation ablation: no observation-age rejection. The finite
    # source-aligned action horizon, session and ordering checks still apply.
    if max_action_age_ms is None:
        return {"max_age_ticks": None, "max_wall_age_s": None}
    if type(max_action_age_ms) is not int or max_action_age_ms <= 0 or max_action_age_ms % 20:
        raise ValueError("max_action_age_ms must be a positive multiple of the 20 ms control tick")
    if max_action_age_ms // 20 > 256 * DECISION_TICKS:
        raise ValueError("max_action_age_ms exceeds the maximum navigation chunk horizon")
    return {"max_age_ticks": max_action_age_ms // 20,
            "max_wall_age_s": max_action_age_ms / 1000 if realtime else 1e9}


def student_state(observation: Observation) -> np.ndarray:
    """Exclude command13, including any task/teacher navigation answer."""
    return observation.state[:STATE_DIM].copy()


@dataclass(frozen=True)
class NavigationChunk:
    actions: np.ndarray
    episode_id: str
    source_tick: int
    source_monotonic_ns: int
    valid_until_tick: int
    action_interval_ticks: int = DECISION_TICKS
    diagnostics: dict = field(default_factory=dict)
    contract: str = NAVIGATION_CONTRACT

    def __post_init__(self):
        actions = np.asarray(self.actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 5 or not 1 <= len(actions) <= 256:
            raise ValueError("navigation actions must have shape (1..256, 5)")
        action_contract(self.contract).validate(actions)
        if not self.episode_id or self.source_tick < 0 or self.source_monotonic_ns <= 0:
            raise ValueError("navigation actions need episode, source tick and capture time")
        if self.action_interval_ticks != DECISION_TICKS:
            raise ValueError("navigation_v1 decisions must use 5 control ticks (10 Hz)")
        if not self.source_tick < self.valid_until_tick <= self.source_tick + len(actions) * self.action_interval_ticks:
            raise ValueError("navigation validity must fit the predicted horizon")
        actions = actions.copy()
        actions.setflags(write=False)
        object.__setattr__(self, "actions", actions)


def intent_from_action(action, *, episode_id: str, tick: int, duration_ticks=1,
                       contract=NAVIGATION_CONTRACT) -> SkillIntent:
    """Decode one decision; the env recomputes the ONNX actor every control tick."""
    values = np.asarray(action, dtype=np.float32)
    if values.shape != (5,):
        raise ValueError("invalid navigation decision")
    action_contract(contract).validate(values)
    command = np.zeros(13, dtype=np.float32)
    stand = values[4] >= .5
    if not stand:
        command[:3] = values[:3]
    command[5] = values[3]
    return SkillIntent("stand" if stand else "walk", command, episode_id, tick, duration_ticks)


class NavigationBuffer:
    """Source-aligned latest plan with both physical-time and wall-time expiry."""

    def __init__(self, *, max_age_ticks=25, max_wall_age_s=.5, contract=NAVIGATION_CONTRACT):
        self.age_rejection_enabled = max_age_ticks is not None or max_wall_age_s is not None
        if self.age_rejection_enabled and (max_age_ticks is None or max_wall_age_s is None
                or max_age_ticks < 1 or not np.isfinite(max_wall_age_s) or max_wall_age_s <= 0):
            raise ValueError("navigation expiry must be positive")
        self.max_age_ticks = int(max_age_ticks) if self.age_rejection_enabled else None
        self.max_wall_age_ns = int(max_wall_age_s * 1e9) if self.age_rejection_enabled else None
        self.contract = action_contract(contract).name
        self.reset("")

    def reset(self, episode_id):
        self.episode_id = str(episode_id)
        self.chunk = None
        self.last_source_tick = -1
        self.rejected = 0
        self.fallback_ticks = 0
        self.last_reason = "no_plan"

    def expiry_reason(self, chunk, observation, age_ns):
        if not self.age_rejection_enabled:
            return "horizon_exhausted" if observation.tick >= chunk.valid_until_tick else None
        if observation.tick >= min(chunk.valid_until_tick, chunk.source_tick + self.max_age_ticks) or age_ns >= self.max_wall_age_ns:
            return "expired"
        return None

    def submit(self, chunk: NavigationChunk, observation: Observation) -> bool:
        age_ns = observation.monotonic_ns - chunk.source_monotonic_ns
        reason = None
        if observation.episode_id != self.episode_id or chunk.episode_id != self.episode_id:
            reason = "wrong_episode"
        elif chunk.contract != self.contract:
            reason = "wrong_contract"
        elif chunk.source_tick > observation.tick or age_ns < 0:
            reason = "future_observation"
        elif chunk.source_tick <= self.last_source_tick:
            reason = "out_of_order"
        else:
            reason = self.expiry_reason(chunk, observation, age_ns)
        if reason:
            self.rejected += 1
            self.last_reason = reason
            return False
        self.chunk = chunk
        self.last_source_tick = chunk.source_tick
        self.last_reason = "accepted"
        return True

    def consume(self, observation: Observation) -> tuple[SkillIntent, dict]:
        chunk = self.chunk
        reason = "no_plan"
        if observation.episode_id != self.episode_id:
            reason = "wrong_episode"
        elif chunk is not None:
            elapsed = observation.tick - chunk.source_tick
            age_ns = observation.monotonic_ns - chunk.source_monotonic_ns
            if elapsed < 0 or age_ns < 0:
                reason = "future_observation"
            elif self.expiry_reason(chunk, observation, age_ns):
                reason = self.expiry_reason(chunk, observation, age_ns)
            else:
                index = elapsed // chunk.action_interval_ticks
                return intent_from_action(chunk.actions[index], episode_id=self.episode_id,
                                          tick=observation.tick, contract=self.contract), {
                    "execution_source": "navigation_model", "source_tick": chunk.source_tick,
                    "action_index": index, "observation_age_ms": age_ns / 1e6,
                }
        self.chunk = None
        self.fallback_ticks += 1
        self.last_reason = reason
        return intent_from_action([0, 0, 0, 0, 1], episode_id=observation.episode_id,
                                  tick=observation.tick, contract=self.contract), {
            "execution_source": "official_stand_fallback", "reason": reason,
        }


class NavigationTeacher:
    """Privileged label source, deliberately separate from the student API."""

    def __init__(self, env):
        from .teachers import ClassicalTeacher

        self.env = env
        self.planner = ClassicalTeacher(env)
        self.last_diagnostics = {}
        self._stop_latched = False

    def predict(self, observation: Observation) -> np.ndarray:
        if observation.episode_id != self.env.episode_id or observation.tick != self.env.tick:
            raise ValueError("teacher requires its current simulator observation")
        if self.env.task.suite != "visual_goal":
            raise ValueError("navigation teacher supports visual_goal tasks only")
        return self._predict_world(observation, self.env.evaluator_state())

    def _predict_world(self, observation, world):
        distance = np.linalg.norm(np.asarray(world["balls"][self.env.task.target_color])[:2]
                                  - np.asarray(world["robot_position"])[:2]) - .035
        lower, upper = self.env.task.stop_distance_m
        if self._stop_latched and lower <= distance <= upper:
            role, command, diagnostics = "stand", np.zeros(13, np.float32), {"phase": "stop", "stop_hysteresis": True}
        else:
            self._stop_latched = False
            role, command, diagnostics = self.planner._goal_command(world)
            self._stop_latched = diagnostics.get("phase") == "stop"
        if diagnostics.get("phase") == "approach" and command[0] == 0 and abs(command[2]) > .9:
            # Use the best verified diagnostic controller. Lateral stepping
            # reduced isolated turning radius but regressed search in closed
            # loop; those experiments are retained under navigation artifacts.
            command[0] = .15
            diagnostics = {**diagnostics, "turning_arc": True}
        if role not in ("walk", "stand") or np.any(command[[3, 4, 6, 7, 8, 9, 10, 11, 12]]):
            raise ValueError("teacher command is not representable by navigation_v1")
        action = np.asarray([*command[:3], command[5], float(role == "stand")], dtype=np.float32)
        intent_from_action(action, episode_id=observation.episode_id, tick=observation.tick)
        self.last_diagnostics = {**diagnostics, "teacher_role": role, "uses_privileged_state": True}
        return action


class VisualSearchTeacher(NavigationTeacher):
    """Search independently of hidden target coordinates until RGB observes it.

    Planning still uses a privileged obstacle map. After a visual observation,
    the teacher may remember that target location but cannot track it through
    the simulator when hidden. This is an expert label source, not a learned
    navigation result and not an RGB-only baseline.
    """

    def __init__(self, env, *, obstacle_aware=False):
        super().__init__(env)
        # This is an opt-in teacher experiment. Keep qualified clear-room
        # demonstrations reproducible with their original controller.
        self.obstacle_aware = bool(obstacle_aware)
        self.last_seen_position = None
        self.observed_balls = {}
        self.coverage = None
        self._segmentation_tick = -1

    def predict(self, observation):
        if observation.episode_id != self.env.episode_id or observation.tick != self.env.tick:
            raise ValueError("search teacher requires its current simulator observation")
        if observation.rgb is None or observation.camera_tick != observation.tick:
            raise ValueError("visual search teacher requires fresh RGB")
        world = self.env.evaluator_state()
        color = self.env.task.target_color
        pixels_by_color = {name: self._visible_pixels(name) for name in world["balls"]}
        for name, pixels in pixels_by_color.items():
            if pixels >= 16:
                self.observed_balls[name] = np.asarray(world["balls"][name]).copy()
        visible_pixels = pixels_by_color[color]
        visible = visible_pixels >= 16
        if visible:
            self.last_seen_position = self.observed_balls[color]
        if self.obstacle_aware:
            # Only positions acquired through the actual camera are available.
            # Hidden distractors are neither discovered nor refreshed from
            # simulator truth; the search cannot learn where the target is
            # through the obstacle list.
            world["obstacles"] = list(world["obstacles"]) + [
                {"name": f"observed_ball_{name}", "xy": point[:2].tolist(),
                 "half_size": [.035, .035], "height": .07}
                for name, point in self.observed_balls.items() if name != color]
        if self.last_seen_position is None:
            from .navigation_search import SearchCoverage

            if self.coverage is None:
                self.coverage = SearchCoverage(world)
            self.coverage.obstacles = world["obstacles"]
            action, diagnostics = self.coverage.command(world, self.env.physics, observation.tick)
            self.last_diagnostics = {**diagnostics, "planner_phase": diagnostics["phase"],
                                     "teacher_role": "walk", "target_visible_pixels": visible_pixels,
                                     "target_previously_observed": False,
                                     "uses_privileged_state": True, "hidden_target_coordinates_used": False}
            return self._guard_turn(action, world)
        # Other balls and the true hidden target are not passed to the planner.
        world["balls"] = {color: self.last_seen_position}
        action = self._predict_world(observation, world)
        self.last_diagnostics.update(target_visible=visible, target_previously_observed=self.last_seen_position is not None,
                                     hidden_target_coordinates_used=False,
                                     planner_phase=self.last_diagnostics["phase"], target_visible_pixels=visible_pixels)
        return self._guard_turn(action, world)

    def _guard_turn(self, action, world):
        if not self.obstacle_aware:
            return action
        position = np.asarray(world["robot_position"][:2])
        clearance = min((float(np.linalg.norm(np.maximum(
            np.abs(position - obstacle["xy"]) - obstacle["half_size"], 0)))
            for obstacle in world["obstacles"]), default=float("inf"))
        def predicted_clearance(vx):
            # A coarse short motion estimate based on recorded BAM
            # locomotion rather than interpreting commanded speed as exact
            # realized speed. Final teacher quality still requires rollouts.
            point = position.copy()
            yaw = float(world["robot_yaw"])
            minimum = clearance
            for _ in range(12):
                point += .1 * .6 * vx * np.array([np.cos(yaw), np.sin(yaw)])
                yaw += .1 * .65 * float(action[2])
                minimum = min(minimum, *(float(np.linalg.norm(np.maximum(
                    np.abs(point - obstacle["xy"]) - obstacle["half_size"], 0)))
                    for obstacle in world["obstacles"]))
            return minimum

        forward_clearance = predicted_clearance(float(action[0])) if world["obstacles"] else clearance
        guarded = bool(action[4] < .5 and action[0] > 0 and abs(action[2]) > .9
                       and clearance < .23 and forward_clearance < min(.16, clearance - .01))
        if guarded:
            action = action.copy()
            # Move away while turning when possible. Pure yaw from stand
            # reaches a deadband, so it is only a last-resort safe command.
            action[0] = -.15 if predicted_clearance(-.15) >= min(.14, clearance - .005) else 0
        self.last_diagnostics.update(
            obstacle_aware_teacher=True, turn_translation_guard=guarded,
            predicted_forward_clearance_m=forward_clearance if np.isfinite(forward_clearance) else None,
            known_obstacle_clearance_m=clearance if np.isfinite(clearance) else None,
            observed_distractors=sorted(name for name in self.observed_balls if name != self.env.task.target_color))
        return action

    def _visible_pixels(self, color):
        """Privileged segmentation gate from the student's actual head camera.

        This avoids confusing a conservative RGB baseline's missed detections
        with absence of visual evidence. Segmentation is never a student input.
        """
        physics = self.env.physics
        if self._segmentation_tick != self.env.tick:
            self.env.render()
            renderer = physics.renderer
            renderer.enable_segmentation_rendering()
            try:
                renderer.update_scene(physics.data, camera=physics.camera)
                self._segmentation = renderer.render().copy()
                self._segmentation_tick = self.env.tick
            finally:
                renderer.disable_segmentation_rendering()
        segmentation = self._segmentation
        geom = physics.mj.mj_name2id(physics.model, physics.mj.mjtObj.mjOBJ_GEOM, f"lx_ball_{color}_geom")
        if geom < 0:
            raise ValueError("missing visual-goal target geometry")
        return int(((segmentation[:, :, 0] == geom) &
                    (segmentation[:, :, 1] == int(physics.mj.mjtObj.mjOBJ_GEOM))).sum())
