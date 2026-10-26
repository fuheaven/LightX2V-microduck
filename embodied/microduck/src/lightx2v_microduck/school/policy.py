"""15 learned high-level coefficients. The RGB policy never receives an env."""

import math

import numpy as np

from ..contracts import SkillIntent
from ..pixel_policy import ball_in_trunk, detect_colored_ball
from .simulation import wrap


def initial_weights(course):
    weights = np.zeros((3, 5))
    weights[0, 1] = 0.8
    weights[2, 3] = 2.5
    weights[2, 4] = 1.1  # Search direction when a visual target is absent.
    if course.family == "explore":
        weights[0, 0] = 0.3
    elif course.family == "spin":
        weights[:] = 0
        weights[2, 0] = 1.1 * course.direction
    elif course.family == "circle":
        weights[0, 0] = 0.3
    elif course.family in ("rgb", "ball"):
        # The pinned gait barely translates below ~0.2 m/s commands. Keep a
        # meaningful rule comparator; CEM may refine this command bias.
        weights[0, 0] = 0.25
    return weights.ravel()


class SchoolPolicy:
    def __init__(self, course, weights):
        self.course = course
        self.weights = np.asarray(weights, dtype=float)
        if self.weights.shape != (15,) or not np.isfinite(self.weights).all() or np.max(np.abs(self.weights)) > 10:
            raise ValueError("School weights must be 15 finite bounded coefficients")
        self.weights = self.weights.reshape(3, 5)
        self.last_diagnostics = {}
        self.reset()

    def reset(self):
        self.latched = False
        self.last_diagnostics = {}

    def predict(self, observation, *, truth=None):
        c = self.course
        if not c.privileged and truth is not None:
            raise ValueError("truth is forbidden for RGB courses")
        if c.privileged and truth is None:
            raise ValueError("this course requires declared simulator truth")
        feature = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        stop = self.latched
        head_pitch = 0.0
        if c.family == "rgb":
            if observation.rgb is None:
                raise ValueError("RGB course requires an actual camera frame")
            detection = detect_colored_ball(observation.rgb, c.color)
            self.last_diagnostics = {"uses_privileged_state": False, "color": c.color, "detection": detection, "frame_tick": observation.camera_tick}
            if detection:
                point = ball_in_trunk(detection, observation.camera_metadata)
                surface = float(np.linalg.norm(point[:2])) - 0.035
                bearing = math.atan2(point[1], point[0])
                feature[1:4] = [max(0.0, surface - 0.325), float(point[1]), bearing]
                head_pitch = float(np.clip((0.8 - surface) * 0.6, 0.0, 0.3))
                stop |= 0.27 <= surface <= 0.37 and abs(bearing) < 0.4
                self.last_diagnostics.update(estimated_surface_m=surface, bearing=bearing, phase="approach")
            else:
                feature[4] = 1
                self.last_diagnostics["phase"] = "search"
            self.latched = stop
        elif c.family in ("goto", "ball", "circle"):
            if c.family == "goto":
                target = np.array(truth["layout"]["waypoints"][truth["waypoint"]])
            elif c.family == "ball":
                target = truth["balls"]["red"][:2]
            else:
                ray = truth["xy"] - truth["layout"]["center"]
                angle = math.atan2(ray[1], ray[0]) + 0.4 * c.direction
                target = np.array(truth["layout"]["center"]) + c.radius_m * np.array([np.cos(angle), np.sin(angle)])
            delta = target - truth["xy"]
            bearing = wrap(math.atan2(delta[1], delta[0]) - truth["yaw"])
            feature[1:4] = [float(np.linalg.norm(delta)), float(-np.sin(truth["yaw"]) * delta[0] + np.cos(truth["yaw"]) * delta[1]), bearing]
            stop = c.family == "goto" and truth["waypoint"] == c.waypoints - 1 and np.linalg.norm(delta) < 0.10
        command = np.zeros(13, np.float32)
        command[:3] = np.clip(self.weights @ feature, [-0.2, -0.12, -1.2], [0.35, 0.12, 1.2])
        if abs(feature[3]) > 0.65 or feature[4] == 1:
            command[:2] = 0
        command[4] = head_pitch
        if stop:
            command[:3] = 0
        self.last_diagnostics.update(phase="hold" if stop else self.last_diagnostics.get("phase", "command"), feature=feature.tolist(), command=command[:3].tolist())
        return SkillIntent("stand" if stop else "walk", command, observation.episode_id, observation.tick, 5)
