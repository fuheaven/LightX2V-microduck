"""Experimental pose control from measured, pinned official-actor responses.

This privileged teacher component reads the robot pose and known obstacle map.
It has no access to hidden target positions and is not a learned policy.
"""

import hashlib
import json
from pathlib import Path

import numpy as np


MOTION_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def angle(value):
    return np.arctan2(np.sin(value), np.cos(value))


def load_motion_calibration(directory, expected_sha256, robot_signature):
    directory = Path(directory)
    report_path = directory / "report.json"
    if hashlib.sha256(report_path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("motion calibration report changed")
    report = json.loads(report_path.read_text())
    motion_path = directory / "motion.npz"
    if (report["status"] != "complete" or report["completed_branches"] != 90
            or report["anchor_modes"] != ["stand", "forward_walk"]
            or report["repeat_exact"] != {"stand": True, "forward_walk": True}
            or report["colliding_branches"] or report["fallen_branches"]
            or report["profile"] != "official_bam" or report["robot_signature"] != robot_signature
            or hashlib.sha256(motion_path.read_bytes()).hexdigest() != report["motion_sha256"]):
        raise ValueError("motion calibration is incomplete or incompatible with this robot")
    with np.load(motion_path, allow_pickle=False) as data:
        commands, poses = data["commands"].copy(), data["poses"].copy()
    if commands.shape != (45, 5) or poses.shape != (2, 45, 151, 3):
        raise ValueError("motion calibration has incompatible dimensions")
    return MeasuredMotionController(commands, poses)


class MeasuredMotionController:
    def __init__(self, commands, poses):
        self.commands, self.poses = np.asarray(commands), np.asarray(poses)
        if (self.commands.ndim != 2 or self.commands.shape[1] != 5
                or self.poses.shape != (2, len(self.commands), 151, 3)
                or not np.isfinite(self.commands).all() or not np.isfinite(self.poses).all()):
            raise ValueError("invalid measured motion trajectories")
        self.previous_yaw = None
        self.previous_action = None

    @staticmethod
    def clearance(points, world):
        bounds = world["room_bounds"]
        result = np.minimum.reduce([points[..., 0] - bounds[0], bounds[1] - points[..., 0],
                                    points[..., 1] - bounds[2], bounds[3] - points[..., 1]])
        for obstacle in world["obstacles"]:
            result = np.minimum(result, np.linalg.norm(np.maximum(
                np.abs(points - obstacle["xy"]) - obstacle["half_size"], 0), axis=-1))
        return result

    def command(self, world, waypoint, look, *, orienting=None):
        position = np.asarray(world["robot_position"][:2])
        yaw = world["robot_yaw"]
        angular = 0. if self.previous_yaw is None else abs(angle(yaw - self.previous_yaw)) / .1
        self.previous_yaw = yaw
        moving = np.linalg.norm(world["robot_velocity"][:2]) > .025 or angular > .15
        curves = self.poses[int(moving), :, :51]
        turn = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        positions = position + curves[..., :2] @ turn.T
        delta = np.asarray(waypoint) - position
        near = np.linalg.norm(delta) < .22 if orienting is None else orienting
        desired_yaw = look if near else np.arctan2(delta[1], delta[0])
        scores = (np.linalg.norm(positions[:, -1] - waypoint, axis=1)
                  + (.2 if near else .1) * np.abs(angle(desired_yaw - yaw - curves[:, -1, 2])))
        if self.previous_action is not None:
            scores += .005 * np.abs((self.commands[:, :3] - self.previous_action[:3]) / [.45, .3, 2.2]).sum(1)
        minimum = self.clearance(positions, world).min(axis=1)
        current = float(self.clearance(position, world))
        allowed = minimum >= min(.14, current - .005)
        if current <= 0:
            allowed[:] = False
        if not allowed.any():
            return np.array([0, 0, 0, 0, 1], np.float32), {
                "phase": "no_path", "measured_motion_guard": True, "available_primitives": 0}
        selected = int(np.argmin(np.where(allowed, scores, np.inf)))
        action = self.commands[selected].astype(np.float32, copy=True)
        action[3] = np.clip(angle(look - yaw), -.45, .45)
        self.previous_action = action.copy()
        return action, {"measured_anchor_mode": int(moving), "command_index": selected,
                        "measured_motion_guard": True, "available_primitives": int(allowed.sum()),
                        "predicted_min_clearance_m": float(minimum[selected])}
