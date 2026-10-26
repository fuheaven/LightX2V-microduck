"""Privileged obstacle-map exploration, without access to target locations.

This is a demonstration planner, not the learned policy or an RGB baseline.
Visibility accumulates from the real head-camera pose. Search destinations
depend on unobserved free space, never on a simulator's hidden ball position.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

from .teachers import _angle, plan_grid_path

SEARCH_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def clear_sight(origin, targets, obstacles, *, margin=.015):
    """Ray/box intersections for opaque scene obstacles (no target geometry)."""
    origin, targets = np.asarray(origin), np.asarray(targets)
    direction = targets - origin
    visible = np.ones(len(targets), dtype=bool)
    for obstacle in obstacles:
        center = np.array([*obstacle["xy"], obstacle["height"] / 2])
        extent = np.array([*obstacle["half_size"], obstacle["height"] / 2]) + margin
        lo, hi = center - extent, center + extent
        parallel = np.abs(direction) < 1e-10
        denominator = np.where(parallel, 1., direction)
        first, second = (lo - origin) / denominator, (hi - origin) / denominator
        enter, leave = np.minimum(first, second), np.maximum(first, second)
        outside = parallel & ((origin < lo) | (origin > hi))
        enter = np.where(parallel, -np.inf, enter)
        leave = np.where(parallel, np.inf, leave)
        hit = (np.maximum(enter.max(1), 0) <= np.minimum(leave.min(1), 1)) & ~outside.any(1)
        visible &= ~hit
    return visible


class SearchCoverage:
    def __init__(self, world):
        self.obstacles = world["obstacles"]
        self.bounds = tuple(world["room_bounds"])
        x0, x1, y0, y1 = self.bounds
        xy = np.array([(x, y) for x in np.arange(x0 + .2, x1 - .19, .12)
                       for y in np.arange(y0 + .2, y1 - .19, .12)])
        valid = np.ones(len(xy), dtype=bool)
        for obstacle in self.obstacles:
            valid &= ~np.all(np.abs(xy - obstacle["xy"]) <= np.asarray(obstacle["half_size"]) + .04, axis=1)
        self.cells = np.column_stack([xy[valid], np.full(valid.sum(), .035)])
        self.seen = np.zeros(len(self.cells), dtype=bool)
        xy = np.array([(x, y) for x in np.arange(x0 + .35, x1 - .34, .35)
                       for y in np.arange(y0 + .35, y1 - .34, .35)])
        valid = np.ones(len(xy), dtype=bool)
        for obstacle in self.obstacles:
            valid &= ~np.all(np.abs(xy - obstacle["xy"]) <= np.asarray(obstacle["half_size"]) + .19, axis=1)
        self.candidates = xy[valid]
        self.destination = None
        self.look_yaw = 0.
        self.chosen_tick = -1
        self.headings = np.linspace(-np.pi, np.pi, 16, endpoint=False)
        self.visits = np.zeros((len(self.candidates), len(self.headings)), dtype=np.int32)

    def update_camera(self, position, rotation, fovy):
        camera = (self.cells - position) @ np.asarray(rotation).reshape(3, 3)
        depth = -camera[:, 2]
        extent = depth * np.tan(np.deg2rad(fovy / 2)) * .90
        inside = (depth > .10) & (np.abs(camera[:, 0]) < extent) & (np.abs(camera[:, 1]) < extent)
        self.seen |= inside & clear_sight(position, self.cells, self.obstacles)

    def _choose(self, position, yaw, camera_height, fovy, tick):
        unknown = self.cells[~self.seen]
        if not len(unknown):
            # No oracle assertion that the target does not exist: re-observe.
            self.seen[:] = False
            unknown = self.cells
        candidates = []
        for index, point in enumerate(self.candidates):
            distance = float(np.linalg.norm(point - position))
            heading = math.atan2(*(point - position)[::-1])
            bearing = np.arctan2(unknown[:, 1] - point[1], unknown[:, 0] - point[0])
            visible = clear_sight([*point, camera_height], unknown, self.obstacles)
            for heading_index, look_yaw in enumerate(self.headings):
                front = np.cos(bearing - look_yaw) > np.cos(np.deg2rad(fovy * .45))
                gain = int((front & visible).sum())
                # Account for walking and orienting the actual 45-degree
                # camera, rather than crediting a 360-degree view at arrival.
                turn = (abs(_angle(look_yaw - yaw)) if distance < .22 else
                        abs(_angle(heading - yaw)) + max(0, abs(_angle(look_yaw - heading)) - .45))
                cost = 1. + distance / .20 + turn / .65
                score = gain / cost / (1 + self.visits[index, heading_index])
                candidates.append((score, index, heading_index))
        for _, index, heading_index in sorted(candidates, reverse=True):
            path = plan_grid_path(position, self.candidates[index], self.obstacles, self.bounds, clearance=.16)
            if path:
                self.destination = self.candidates[index].copy()
                self.look_yaw = float(self.headings[heading_index])
                self.visits[index, heading_index] += 1
                self.chosen_tick = tick
                return path
        return []

    def command(self, world, physics, tick):
        position = np.asarray(world["robot_position"][:2])
        camera = physics.camera
        camera_position = physics.data.cam_xpos[camera]
        self.update_camera(camera_position, physics.data.cam_xmat[camera], physics.model.cam_fovy[camera])
        yaw = world["robot_yaw"]
        arrived = self.destination is not None and np.linalg.norm(position - self.destination) < .22
        oriented = abs(_angle(self.look_yaw - yaw)) < .35
        if self.destination is None or (arrived and oriented) or tick - self.chosen_tick >= 600:
            path = self._choose(position, yaw, camera_position[2], physics.model.cam_fovy[camera], tick)
        else:
            path = plan_grid_path(position, self.destination, self.obstacles, self.bounds, clearance=.16)
        if not path:
            return np.array([0, 0, 0, 0, 1], dtype=np.float32), {"phase": "no_path"}
        waypoint = path[-1]
        for candidate in path[1:]:
            if np.linalg.norm(candidate - position) >= .18:
                waypoint = candidate
                break
        delta = waypoint - position
        error = _angle((self.look_yaw if np.linalg.norm(position - self.destination) < .22
                        else math.atan2(delta[1], delta[0])) - yaw)
        action = np.array([.3 if abs(error) < .75 else .15, 0,
                           np.clip(2 * error, -1.1, 1.1), np.clip(_angle(self.look_yaw - yaw), -.45, .45), 0], dtype=np.float32)
        return action, {"phase": "search", "coverage_fraction": float(self.seen.mean()),
                        "search_destination": self.destination.tolist(), "search_look_yaw": self.look_yaw,
                        "path_nodes": len(path)}
