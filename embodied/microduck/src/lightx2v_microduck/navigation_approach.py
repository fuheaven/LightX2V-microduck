"""Experimental measured-motion control after actual visual target discovery.

The observation gate, remembered target, stop/backoff rules, A* route and
official actor stay unchanged. This candidate is not a qualified data teacher.
"""

import copy
import math

import numpy as np

from .navigation import VisualSearchTeacher
from .navigation_motion import load_motion_calibration
from .teachers import plan_grid_path


class MeasuredApproachTeacher(VisualSearchTeacher):
    def __init__(self, env, *, calibration, expected_calibration_sha256):
        super().__init__(env, obstacle_aware=True)
        self.motion_controller = load_motion_calibration(
            calibration, expected_calibration_sha256, env._robot_signature)

    @classmethod
    def copy_history(cls, teacher, *, calibration, expected_calibration_sha256):
        if not isinstance(teacher, VisualSearchTeacher) or not teacher.obstacle_aware:
            raise ValueError("requires a visibility-gated obstacle-aware teacher")
        result = cls(teacher.env, calibration=calibration,
                     expected_calibration_sha256=expected_calibration_sha256)
        result.__dict__.update(copy.deepcopy(teacher.__dict__, {id(teacher.env): teacher.env}))
        return result

    def _predict_world(self, observation, world):
        action = super()._predict_world(observation, world)
        if self.last_diagnostics["phase"] != "approach":
            return action
        color = self.env.task.target_color
        if self.last_seen_position is None or set(world["balls"]) != {color}:
            raise ValueError("approach requires the actually observed target memory")
        position = np.asarray(world["robot_position"][:2])
        target = np.asarray(world["balls"][color][:2])
        delta = target - position
        radius = float(np.mean(self.env.task.stop_distance_m)) + .035
        stop_point = target - delta / np.linalg.norm(delta) * radius
        if (self.planner._alternate_stop_point is not None
                and np.linalg.norm(target - self.planner._alternate_target) < .03):
            stop_point = self.planner._alternate_stop_point
        path = plan_grid_path(position, stop_point, world["obstacles"], world["room_bounds"])
        if not path:
            raise ValueError("the unchanged approach route could not be reconstructed")
        waypoint = path[-1]
        for point in path[1:]:
            if np.linalg.norm(point - position) >= .14:
                waypoint = point
                break
        look = math.atan2(delta[1], delta[0])
        action, diagnostic = self.motion_controller.command(world, waypoint, look, orienting=False)
        self.last_diagnostics.update(diagnostic, measured_approach=True,
                                     approach_waypoint=np.asarray(waypoint).tolist())
        return action

    def _guard_turn(self, action, world):
        if self.last_diagnostics.get("measured_approach"):
            # The measured controller checks its full one-second physical
            # curve against the same known obstacles, including seen balls.
            return action
        return super()._guard_turn(action, world)
