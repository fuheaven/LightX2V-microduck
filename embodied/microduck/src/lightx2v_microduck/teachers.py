"""Privileged teachers for labels and explicitly marked oracle baselines.

These controllers use world state for planning; their success must never be
reported as RGB-only VLA performance. Actions still come from the pinned ONNX
locomotion policies and execute through the same MuJoCo actuator as students.
"""

from __future__ import annotations

import hashlib
import heapq
import math
from pathlib import Path

import numpy as np

from .contracts import JointResidualChunk, Observation, PolicyManifest
from .tasks import PROGRAMS, teacher_context

TEACHER_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _angle(value):
    return (value + math.pi) % (2 * math.pi) - math.pi


def plan_grid_path(start, goal, obstacles, bounds=(-1.5, 1.5, -1.5, 1.5), *, resolution=0.08, clearance=0.12):
    """Deterministic 8-connected A* with inflated actual box obstacles."""
    start, goal = np.asarray(start)[:2], np.asarray(goal)[:2]
    low = np.array((bounds[0], bounds[2])) + clearance
    high = np.array((bounds[1], bounds[3])) - clearance
    shape = np.floor((high - low) / resolution).astype(int) + 1

    def cell(point):
        return tuple(np.clip(np.rint((point - low) / resolution).astype(int), 0, shape - 1))

    def world(index):
        return low + np.asarray(index) * resolution

    def blocked(index):
        if any(index[i] < 0 or index[i] >= shape[i] for i in range(2)):
            return True
        point = world(index)
        return any(np.all(np.abs(point - np.asarray(o["xy"])) <= np.asarray(o["half_size"]) + clearance) for o in obstacles)

    def nearest_free(index):
        if not blocked(index):
            return index
        # Quantization can put a physically clear start just inside an inflated
        # cell. Snap only a short distance, never teleport across an obstacle.
        candidates = [(dx * dx + dy * dy, (index[0] + dx, index[1] + dy)) for dx in range(-2, 3) for dy in range(-2, 3)]
        for _, candidate in sorted(candidates):
            if not blocked(candidate):
                return candidate
        return None

    initial, terminal = nearest_free(cell(start)), nearest_free(cell(goal))
    if initial is None or terminal is None:
        return []
    frontier = [(0.0, initial)]
    previous, costs = {}, {initial: 0.0}
    while frontier:
        _, current = heapq.heappop(frontier)
        if current == terminal:
            indices = [current]
            while current in previous:
                current = previous[current]
                indices.append(current)
            return [world(index) for index in reversed(indices)]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            nxt = (current[0] + dx, current[1] + dy)
            if blocked(nxt) or (dx and dy and (blocked((current[0] + dx, current[1])) or blocked((current[0], current[1] + dy)))):
                continue
            candidate = costs[current] + math.hypot(dx, dy)
            if candidate < costs.get(nxt, math.inf):
                costs[nxt], previous[nxt] = candidate, current
                score = candidate + math.hypot(nxt[0] - terminal[0], nxt[1] - terminal[1])
                heapq.heappush(frontier, (score, nxt))
    return []


class ClassicalTeacher:
    def __init__(self, env):
        self.env = env
        self.manifest = PolicyManifest(
            "official_rl_task_teacher",
            "joint_residual",
            "official_61d",
            env.profile,
            uses_privileged_state=True,
            metadata={"planner": "task_program_or_privileged_astar", "not_visual_autonomy": True, "source_sha256": TEACHER_SOURCE_SHA256},
        )
        self.reset("")

    def reset(self, episode_id=""):
        self.episode_id = str(episode_id)
        self.last_diagnostics = {}
        self._alternate_stop_point = None
        self._alternate_target = None

    def _goal_command(self, world):
        command = np.zeros(13, dtype=np.float32)
        task = self.env.task
        position = np.asarray(world["robot_position"][:2])
        target = np.asarray(world["balls"][task.target_color][:2])
        delta = target - position
        distance = float(np.linalg.norm(delta))
        # The task's stop band is measured from the ball surface.
        lower, upper = (v + 0.035 for v in task.stop_distance_m)
        if world["recovery_active"]:
            return "stand", command, {"phase": "recover"}
        if lower <= distance <= upper - 0.04:
            return "stand", command, {"phase": "stop", "target_color": task.target_color}
        if distance < lower + 0.015:
            command[0] = -0.07
            command[2] = np.clip(_angle(math.atan2(delta[1], delta[0]) - world["robot_yaw"]), -0.5, 0.5)
            return "walk", command, {"phase": "backoff", "target_color": task.target_color}
        stop_point = target - delta / distance * (0.5 * (lower + upper))

        def clear(point):
            bounds = world["room_bounds"]
            return (
                bounds[0] + 0.12 < point[0] < bounds[1] - 0.12
                and bounds[2] + 0.12 < point[1] < bounds[3] - 0.12
                and not any(np.all(np.abs(point - np.asarray(o["xy"])) <= np.asarray(o["half_size"]) + 0.12) for o in world["obstacles"])
            )

        if self._alternate_stop_point is not None and np.linalg.norm(target - self._alternate_target) < 0.03:
            stop_point = self._alternate_stop_point
        path = plan_grid_path(position, stop_point, world["obstacles"], world["room_bounds"]) if clear(stop_point) else []
        if not path:
            # An occluding box can contain the nearest point of the stop
            # annulus while the ball itself is reachable. Search other sides
            # of that annulus instead of declaring the whole task unreachable.
            radius = 0.5 * (lower + upper)
            candidates = [target + radius * np.array((math.cos(angle), math.sin(angle))) for angle in np.linspace(-math.pi, math.pi, 24, endpoint=False)]
            for candidate in sorted(candidates, key=lambda point: np.linalg.norm(point - position)):
                if not clear(candidate):
                    continue
                candidate_path = plan_grid_path(position, candidate, world["obstacles"], world["room_bounds"])
                if candidate_path:
                    self._alternate_stop_point, self._alternate_target = candidate.copy(), target.copy()
                    path = candidate_path
                    break
        if not path:
            return "stand", command, {"phase": "no_path", "target_color": task.target_color}
        waypoint = path[-1]
        for point in path[1:]:
            if np.linalg.norm(point - position) >= 0.14:
                waypoint = point
                break
        direction = waypoint - position
        error = _angle(math.atan2(direction[1], direction[0]) - world["robot_yaw"])
        command[2] = np.clip(error * 2.0, -1.1, 1.1)
        # Turn in place before entering a narrow obstacle corridor.
        # The pinned alpha walking actor has a measured straight-line deadband
        # at <= .2 under nominal BAM. A .3 command starts stepping; slowing it
        # below that near the goal can leave the teacher stationary forever.
        command[0] = 0.3 if abs(error) < 0.75 else 0.0
        target_error = _angle(math.atan2(delta[1], delta[0]) - world["robot_yaw"])
        command[5] = np.clip(target_error, -0.45, 0.45)
        return "walk", command, {"phase": "approach", "target_color": task.target_color, "path_nodes": len(path)}

    def predict(self, observation: Observation) -> JointResidualChunk:
        if observation.episode_id != self.env.episode_id or observation.tick != self.env.tick:
            raise ValueError("teacher must observe its own current simulator branch")
        world = self.env.evaluator_state()
        if self.env.task.suite == "visual_goal":
            role, command, diagnostics = self._goal_command(world)
        else:
            role, command = teacher_context(self.env.task, observation.tick, recovery=world["recovery_active"], task_start_tick=world["task_start_tick"])
            diagnostics = {"phase": "recovery" if world["recovery_active"] else "task"}
        teacher_state = observation.state.copy()
        teacher_state[48:61] = command
        if self.env.profile == "tinyvla_author_compat" and observation.tick > 0 and self.env.task.task_id in PROGRAMS:
            previous_role = teacher_context(self.env.task, observation.tick - 1, recovery=world["recovery_active"], task_start_tick=world["task_start_tick"])[0]
            if previous_role != role:
                # This reset is an author-teacher convention only. The student's
                # observation keeps its actual raw action; deployment does too.
                teacher_state[34:48] = 0.0
        raw = self.env.bank.infer(role, teacher_state)
        self.last_diagnostics = {**diagnostics, "teacher_policy": role, "teacher_command": command.tolist(), "uses_privileged_state": True}
        return JointResidualChunk(raw[None], observation.episode_id, observation.tick, observation.sim_time_s, diagnostics=self.last_diagnostics)


OfficialTeacher = ClassicalTeacher
