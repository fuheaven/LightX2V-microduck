"""Experimental search over measured official-actor motion trajectories.

Only the known obstacle map, robot pose, camera calibration and actually seen
space enter the planner. Predicted visibility ranks motion; it never changes
observed coverage. This is a privileged offline teacher, not a student policy.
"""

from __future__ import annotations

import json

import numpy as np

from .navigation_motion import MeasuredMotionController, angle
from .navigation_search import SearchCoverage


def projected_visibility(poses, cells, camera_mount, fovy, obstacles):
    """Batch the same full camera projection and ray/box test as the baseline."""
    poses, cells = np.asarray(poses), np.asarray(cells)
    offset, mount, body_height = camera_mount
    turns = np.zeros((len(poses), 3, 3))
    c, s = np.cos(poses[:, 2]), np.sin(poses[:, 2])
    turns[:, 0, 0] = turns[:, 1, 1] = c
    turns[:, 1, 0], turns[:, 0, 1], turns[:, 2, 2] = s, -s, 1.
    origins = np.column_stack([poses[:, :2], np.full(len(poses), body_height)]) + turns @ offset
    rotations = turns @ mount
    directions = cells[None] - origins[:, None]
    camera = np.einsum("kni,kij->knj", directions, rotations)
    depth = -camera[:, :, 2]
    extent = depth * np.tan(np.deg2rad(fovy / 2)) * .90
    visible = (depth > .10) & (np.abs(camera[:, :, 0]) < extent) & (np.abs(camera[:, :, 1]) < extent)
    for obstacle in obstacles:
        center = np.array([*obstacle["xy"], obstacle["height"] / 2])
        extent = np.array([*obstacle["half_size"], obstacle["height"] / 2]) + .015
        lo, hi = center - extent, center + extent
        parallel = np.abs(directions) < 1e-10
        denominator = np.where(parallel, 1., directions)
        first, second = (lo - origins[:, None]) / denominator, (hi - origins[:, None]) / denominator
        enter, leave = np.minimum(first, second), np.maximum(first, second)
        outside = parallel & ((origins[:, None] < lo) | (origins[:, None] > hi))
        enter, leave = np.where(parallel, -np.inf, enter), np.where(parallel, np.inf, leave)
        hit = ((np.maximum(enter.max(2), 0) <= np.minimum(leave.min(2), 1)) & ~outside.any(2))
        visible &= ~hit
    return visible


class TrajectorySearchCoverage(SearchCoverage):
    """Two-stage, six-second lookahead with 0.5-second execution commitment.

    The 45 commands and standing/walking responses come from the immutable
    three-second BAM motion calibration. Six promising first stages are
    expanded with all 45 second stages. Actual RGB can interrupt the search
    on every 10 Hz observation, regardless of the motion commitment.
    """

    def __init__(self, world, controller):
        super().__init__(world)
        self.controller = controller
        self.camera_mount = None
        self.previous_yaw, self.previous_tick = None, None
        self.plan_tick, self.plan_action = None, None
        self.plan_map = None
        self.plan_diagnostic = {}

    def _curves(self, position, yaw, moving):
        local = self.controller.poses[int(moving)]
        rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        curves = local.copy()
        curves[:, :, :2] = position + local[:, :, :2] @ rotation.T
        curves[:, :, 2] += yaw
        return curves

    def _rewards(self, curves, unknown, fovy, *, offset_s=0.):
        # Only rank hypothetical views. self.seen is updated in command()
        # from the actual camera and never from these predicted masks.
        samples = curves[:, 25::25]
        visible = projected_visibility(samples.reshape(-1, 3), unknown, self.camera_mount,
                                       fovy, self.obstacles).reshape(len(curves), 6, len(unknown))
        weights = np.exp(-(offset_s + np.arange(1, 7) * .5) / 3.)
        return (visible * weights[None, :, None]).max(1)

    def _allowed(self, curves, world):
        # Check the entire measured 50 Hz curve, not just its endpoints.
        minimum = MeasuredMotionController.clearance(curves[:, :, :2], world).min(1)
        current = float(MeasuredMotionController.clearance(curves[0, 0, :2], world))
        return (minimum >= min(.14, current - .005)) & (current > 0)

    def _plan(self, world, physics, tick, moving):
        unknown = self.cells[~self.seen]
        if not len(unknown):
            self.seen[:] = False
            unknown = self.cells
        curves = self._curves(np.asarray(world["robot_position"][:2]), world["robot_yaw"], moving)
        allowed = self._allowed(curves, world)
        fovy = physics.model.cam_fovy[physics.camera]
        rewards = self._rewards(curves, unknown, fovy)
        scores = np.where(allowed, rewards.sum(1), -np.inf)
        if not allowed.any():
            return np.array([0, 0, 0, 0, 1], np.float32), {
                "phase": "no_path", "available_primitives": 0}
        shortlist = np.argsort(-scores, kind="stable")[:min(6, int(allowed.sum()))]
        best, best_score, second_best = int(shortlist[0]), float(scores[shortlist[0]]), None
        for first_index in shortlist:
            endpoint = curves[first_index, -1]
            speed = np.linalg.norm(curves[first_index, -1, :2] - curves[first_index, -11, :2]) / .2
            angular = abs(angle(curves[first_index, -1, 2] - curves[first_index, -11, 2])) / .2
            following = self._curves(endpoint[:2], endpoint[2], speed > .025 or angular > .15)
            next_allowed = self._allowed(following, world)
            if not next_allowed.any():
                continue
            combined = np.maximum(rewards[first_index], self._rewards(following, unknown, fovy, offset_s=3.)).sum(1)
            next_scores = np.where(next_allowed, combined, -np.inf)
            candidate = int(np.argmax(next_scores))
            if next_scores[candidate] > best_score:
                best, best_score, second_best = int(first_index), float(next_scores[candidate]), candidate
        if best_score <= 0:
            # A bounded local horizon can contain no unseen space. Use the
            # existing global map search without inventing target positions.
            action, diagnostic = super().command(world, physics, tick)
            return action, {**diagnostic, "global_search_fallback": True,
                            "predicted_discounted_gain": best_score}
        return self.controller.commands[best].astype(np.float32, copy=True), {
            "phase": "search", "command_index": best, "second_command_index": second_best,
            "predicted_discounted_gain": best_score, "available_primitives": int(allowed.sum()),
            "measured_anchor_mode": int(moving), "global_search_fallback": False}

    def command(self, world, physics, tick):
        if self.camera_mount is None:
            from .navigation_teacher_experiments import CameraFrustumCoverage

            projection = CameraFrustumCoverage(world)
            projection.capture_camera_mount(world, physics)
            self.camera_mount = projection.camera_mount
        camera = physics.camera
        self.update_camera(physics.data.cam_xpos[camera], physics.data.cam_xmat[camera],
                           physics.model.cam_fovy[camera])
        elapsed = max(1, tick - self.previous_tick) / 50 if self.previous_tick is not None else .1
        angular = abs(angle(world["robot_yaw"] - self.previous_yaw)) / elapsed if self.previous_yaw is not None else 0.
        moving = np.linalg.norm(world["robot_velocity"][:2]) > .025 or angular > .15
        self.previous_yaw, self.previous_tick = world["robot_yaw"], tick
        known_map = json.dumps(self.obstacles, sort_keys=True)
        if self.plan_tick is None or tick - self.plan_tick >= 25 or self.plan_map != known_map:
            self.plan_action, self.plan_diagnostic = self._plan(world, physics, tick, moving)
            self.plan_tick, self.plan_map = tick, known_map
        return self.plan_action.copy(), {
            **self.plan_diagnostic, "search_method": "measured_trajectory_gain",
            "coverage_fraction": float(self.seen.mean()), "plan_source_tick": self.plan_tick,
            "predicted_visibility_is_not_observed": True, "motion_commitment_ticks": 25,
            "hidden_target_coordinates_used": False}
