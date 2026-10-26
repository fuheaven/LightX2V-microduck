"""Explicit experimental teachers; qualified demonstration controllers stay fixed."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

from .navigation import VisualSearchTeacher
from .navigation_search import SearchCoverage, clear_sight
from .teachers import _angle, plan_grid_path

EXPERIMENT_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class InitialScanCoverage(SearchCoverage):
    """Observe surrounding directions before choosing a distant search point.

    The small reverse/positive-yaw command comes from the recorded official
    BAM locomotion probes. Progress uses actual body rotation, not elapsed
    command time. Hidden target coordinates never enter this controller.
    """

    def __init__(self, world):
        super().__init__(world)
        self.previous_yaw = float(world["robot_yaw"])
        self.swept_yaw = 0.
        self.scan_start_tick = None
        self.scan_done = False

    def command(self, world, physics, tick):
        yaw = float(world["robot_yaw"])
        self.swept_yaw += _angle(yaw - self.previous_yaw)
        self.previous_yaw = yaw
        if self.scan_start_tick is None:
            self.scan_start_tick = tick
        # The camera leads the body by .4 rad; initial RGB was observed before
        # the scan. A bounded scan cannot indefinitely displace exploration.
        self.scan_done |= self.swept_yaw >= 2 * np.pi - .4 or tick - self.scan_start_tick >= 600
        if self.scan_done:
            return super().command(world, physics, tick)
        camera = physics.camera
        self.update_camera(physics.data.cam_xpos[camera], physics.data.cam_xmat[camera],
                           physics.model.cam_fovy[camera])
        return np.array([-.05, 0, 1.1, .4, 0], dtype=np.float32), {
            "phase": "search", "search_method": "initial_scan",
            "coverage_fraction": float(self.seen.mean()), "swept_yaw_rad": self.swept_yaw,
            "scan_elapsed_ticks": tick - self.scan_start_tick,
        }


class InitialScanVisualSearchTeacher(VisualSearchTeacher):
    def __init__(self, env):
        super().__init__(env, obstacle_aware=True)

    def predict(self, observation):
        if self.coverage is None:
            # SearchCoverage reads the map and current robot pose, not balls.
            self.coverage = InitialScanCoverage(self.env.evaluator_state())
        return super().predict(observation)


class OccludedPocketCoverage(SearchCoverage):
    """Prioritize unseen floor regions initially occluded by the known map.

    This experimental prior uses the actual initial camera position and static
    obstacle geometry. It does not read target positions or change the route
    controller, physical observation accounting, or task time limit.
    """

    def __init__(self, world):
        super().__init__(world)
        self.initially_occluded = None

    def _choose(self, position, yaw, camera_height, fovy, tick):
        if self.initially_occluded is None:
            raise ValueError("observe the initial camera before selecting a search viewpoint")
        if self.seen.all():
            self.seen[:] = False
        unknown = self.cells[~self.seen]
        weights = np.where(self.initially_occluded[~self.seen], 4., 1.)
        candidates = []
        for index, point in enumerate(self.candidates):
            distance = float(np.linalg.norm(point - position))
            heading = math.atan2(*(point - position)[::-1])
            bearing = np.arctan2(unknown[:, 1] - point[1], unknown[:, 0] - point[0])
            visible = clear_sight([*point, camera_height], unknown, self.obstacles)
            for heading_index, look_yaw in enumerate(self.headings):
                front = np.cos(bearing - look_yaw) > np.cos(np.deg2rad(fovy * .45))
                gain = float(weights[front & visible].sum())
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
        if self.initially_occluded is None:
            camera_position = physics.data.cam_xpos[physics.camera]
            self.initially_occluded = ~clear_sight(camera_position, self.cells, self.obstacles)
        action, diagnostic = super().command(world, physics, tick)
        diagnostic.update(search_method="occluded_pockets", occluded_cell_weight=4.,
                          initially_occluded_cells=int(self.initially_occluded.sum()),
                          unseen_occluded_cells=int((self.initially_occluded & ~self.seen).sum()))
        return action, diagnostic


class OccludedPocketVisualSearchTeacher(VisualSearchTeacher):
    def __init__(self, env):
        super().__init__(env, obstacle_aware=True)

    def predict(self, observation):
        if self.coverage is None:
            self.coverage = OccludedPocketCoverage(self.env.evaluator_state())
        return super().predict(observation)


class LocalViewCoverage(SearchCoverage):
    """Replan nearby observation points before committing to a long traverse."""

    def _choose(self, position, yaw, camera_height, fovy, tick):
        candidates, visits = self.candidates, self.visits
        nearby = np.flatnonzero(np.linalg.norm(candidates - position, axis=1) <= .7)
        if not len(nearby):
            return super()._choose(position, yaw, camera_height, fovy, tick)
        self.candidates, self.visits = candidates[nearby], visits[nearby].copy()
        try:
            return super()._choose(position, yaw, camera_height, fovy, tick)
        finally:
            visits[nearby] = self.visits
            self.candidates, self.visits = candidates, visits

    def command(self, world, physics, tick):
        if self.destination is not None and tick - self.chosen_tick >= 200:
            self.destination = None
        return super().command(world, physics, tick)


class LocalViewVisualSearchTeacher(VisualSearchTeacher):
    def __init__(self, env):
        super().__init__(env, obstacle_aware=True)

    def predict(self, observation):
        if self.coverage is None:
            self.coverage = LocalViewCoverage(self.env.evaluator_state())
        return super().predict(observation)


class LatchedViewCoverage(SearchCoverage):
    """Finish an arrived viewpoint's orientation despite small walking drift.

    The official actor translates during turns. Switching back to navigation
    whenever it leaves the .22 m arrival radius causes an orbit around the
    observation point. A bounded arrival latch separates these two phases.
    Map coverage still comes only from the actual camera pose.
    """

    def __init__(self, world):
        super().__init__(world)
        self.orienting = False

    def _choose(self, position, yaw, camera_height, fovy, tick):
        self.orienting = False
        return super()._choose(position, yaw, camera_height, fovy, tick)

    def command(self, world, physics, tick):
        position = np.asarray(world["robot_position"][:2])
        camera = physics.camera
        camera_position = physics.data.cam_xpos[camera]
        self.update_camera(camera_position, physics.data.cam_xmat[camera], physics.model.cam_fovy[camera])
        yaw = world["robot_yaw"]
        distance = np.linalg.norm(position - self.destination) if self.destination is not None else np.inf
        self.orienting |= distance < .22
        # A large displacement invalidates the latch instead of allowing an
        # unbounded excursion. This is separate from all task success gates.
        if distance > .45:
            self.orienting = False
        oriented = abs(_angle(self.look_yaw - yaw)) < .35
        if self.destination is None or (self.orienting and oriented) or tick - self.chosen_tick >= 600:
            path = self._choose(position, yaw, camera_position[2], physics.model.cam_fovy[camera], tick)
            self.orienting = self.destination is not None and np.linalg.norm(position - self.destination) < .22
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
        error = _angle((self.look_yaw if self.orienting else math.atan2(delta[1], delta[0])) - yaw)
        action = np.array([.3 if abs(error) < .75 else .15, 0,
                           np.clip(2 * error, -1.1, 1.1), np.clip(_angle(self.look_yaw - yaw), -.45, .45), 0], dtype=np.float32)
        return action, {"phase": "search", "search_method": "latched_view",
                        "coverage_fraction": float(self.seen.mean()),
                        "search_destination": self.destination.tolist(), "search_look_yaw": self.look_yaw,
                        "viewpoint_orientation_latched": self.orienting, "path_nodes": len(path)}


class LatchedViewVisualSearchTeacher(VisualSearchTeacher):
    def __init__(self, env):
        super().__init__(env, obstacle_aware=True)

    def predict(self, observation):
        if self.coverage is None:
            self.coverage = LatchedViewCoverage(self.env.evaluator_state())
        return super().predict(observation)


class PivotViewCoverage(LatchedViewCoverage):
    """Use measured low-displacement turns only while orienting a viewpoint."""

    def command(self, world, physics, tick):
        action, diagnostics = super().command(world, physics, tick)
        if self.orienting and diagnostics["phase"] == "search":
            # Positive and negative turns are asymmetric in the pinned actor.
            # These two choices come from the stored BAM locomotion probes;
            # they still require independent closed-loop teacher validation.
            error = _angle(self.look_yaw - world["robot_yaw"])
            action = action.copy()
            action[:3] = [-.05, 0, 1.1] if error >= 0 else [0, -.15, -1.1]
            diagnostics.update(search_method="pivot_view", measured_pivot=True)
        return action, diagnostics


class PivotViewVisualSearchTeacher(VisualSearchTeacher):
    def __init__(self, env):
        super().__init__(env, obstacle_aware=True)

    def predict(self, observation):
        if self.coverage is None:
            self.coverage = PivotViewCoverage(self.env.evaluator_state())
        return super().predict(observation)


class CameraFrustumCoverage(SearchCoverage):
    """Score candidate views with the settled camera's full 3D projection.

    This changes only the experimental search gain. The original route
    controller, costs and task limits remain fixed. Candidate views are a
    nominal pose approximation; coverage is still credited only through the
    actual camera in SearchCoverage.command.
    """

    def __init__(self, world):
        super().__init__(world)
        self.camera_mount = None

    @staticmethod
    def _rotation(yaw):
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])

    def capture_camera_mount(self, world, physics):
        inverse = self._rotation(-float(world["robot_yaw"]))
        position = np.asarray(world["robot_position"], dtype=float)
        camera = physics.camera
        self.camera_mount = (
            inverse @ (physics.data.cam_xpos[camera] - position),
            inverse @ np.asarray(physics.data.cam_xmat[camera]).reshape(3, 3),
            float(position[2]),
        )

    def candidate_visible(self, point, yaw, cells, fovy):
        if self.camera_mount is None:
            raise ValueError("capture the settled camera before choosing a viewpoint")
        offset, mount_rotation, body_height = self.camera_mount
        turn = self._rotation(yaw)
        origin = np.array([*point, body_height]) + turn @ offset
        rotation = turn @ mount_rotation
        camera_points = (cells - origin) @ rotation
        depth = -camera_points[:, 2]
        extent = depth * np.tan(np.deg2rad(fovy / 2)) * .90
        inside = ((depth > .10) & (np.abs(camera_points[:, 0]) < extent)
                  & (np.abs(camera_points[:, 1]) < extent))
        return inside & clear_sight(origin, cells, self.obstacles)

    def _choose(self, position, yaw, camera_height, fovy, tick):
        unknown = self.cells[~self.seen]
        if not len(unknown):
            self.seen[:] = False
            unknown = self.cells
        candidates = []
        for index, point in enumerate(self.candidates):
            distance = float(np.linalg.norm(point - position))
            heading = math.atan2(*(point - position)[::-1])
            for heading_index, look_yaw in enumerate(self.headings):
                gain = int(self.candidate_visible(point, look_yaw, unknown, fovy).sum())
                if not gain:
                    continue
                turn = (abs(_angle(look_yaw - yaw)) if distance < .22 else
                        abs(_angle(heading - yaw)) + max(0, abs(_angle(look_yaw - heading)) - .45))
                cost = 1. + distance / .20 + turn / .65
                score = gain / cost / (1 + self.visits[index, heading_index])
                candidates.append((score, index, heading_index))
        for _, index, heading_index in sorted(candidates, reverse=True):
            path = plan_grid_path(position, self.candidates[index], self.obstacles,
                                  self.bounds, clearance=.16)
            if path:
                self.destination = self.candidates[index].copy()
                self.look_yaw = float(self.headings[heading_index])
                self.visits[index, heading_index] += 1
                self.chosen_tick = tick
                return path
        return []

    def command(self, world, physics, tick):
        if self.camera_mount is None:
            self.capture_camera_mount(world, physics)
        action, diagnostics = super().command(world, physics, tick)
        diagnostics["search_method"] = "camera_frustum"
        return action, diagnostics


class CameraFrustumVisualSearchTeacher(VisualSearchTeacher):
    def __init__(self, env):
        super().__init__(env, obstacle_aware=True)

    def predict(self, observation):
        if self.coverage is None:
            self.coverage = CameraFrustumCoverage(self.env.evaluator_state())
        return super().predict(observation)


class SpawnSupportCoverage(CameraFrustumCoverage):
    """Favor plausible target area under the declared room generator.

    All geometry comes from the static map and public spawn rules. This does
    not know which cell contains a ball, its color, or the episode seed. The
    route controller, actual-camera coverage and post-discovery behavior stay
    unchanged from the camera-frustum development baseline.
    """

    def __init__(self, world):
        super().__init__(world)
        from .navigation_spawn_support import spawn_support_weights

        self.estimated_support = spawn_support_weights(self.cells, world["obstacles"], self.bounds)
        # The 12 cm grid and finite area quadrature can miss a thin valid
        # sliver at a boundary. Retain every cell with a 5% floor; an estimated
        # zero is a low prior, never a proof that a target cannot be there.
        self.cell_weights = np.maximum(.05, self.estimated_support)

    def _choose(self, position, yaw, camera_height, fovy, tick):
        if not np.any(self.cell_weights[~self.seen] > 0):
            # An empty nominal search region is not evidence of task success
            # or target absence. Re-observe rather than issue a false stop.
            self.seen[:] = False
        unknown, weights = self.cells[~self.seen], self.cell_weights[~self.seen]
        candidates = []
        for index, point in enumerate(self.candidates):
            distance = float(np.linalg.norm(point - position))
            heading = math.atan2(*(point - position)[::-1])
            for heading_index, look_yaw in enumerate(self.headings):
                gain = float(weights[self.candidate_visible(point, look_yaw, unknown, fovy)].sum())
                if not gain:
                    continue
                turn = (abs(_angle(look_yaw - yaw)) if distance < .22 else
                        abs(_angle(heading - yaw)) + max(0, abs(_angle(look_yaw - heading)) - .45))
                cost = 1. + distance / .20 + turn / .65
                score = gain / cost / (1 + self.visits[index, heading_index])
                candidates.append((score, index, heading_index))
        for _, index, heading_index in sorted(candidates, reverse=True):
            path = plan_grid_path(position, self.candidates[index], self.obstacles,
                                  self.bounds, clearance=.16)
            if path:
                self.destination = self.candidates[index].copy()
                self.look_yaw = float(self.headings[heading_index])
                self.visits[index, heading_index] += 1
                self.chosen_tick = tick
                return path
        return []

    def command(self, world, physics, tick):
        action, diagnostic = super().command(world, physics, tick)
        diagnostic.update(search_method="spawn_support", spawn_support_prior=True,
                          estimated_positive_support_cells=int((self.estimated_support > 0).sum()),
                          minimum_search_cell_weight=.05,
                          unseen_support_area_cells=float(self.cell_weights[~self.seen].sum()))
        return action, diagnostic


class SpawnSupportVisualSearchTeacher(VisualSearchTeacher):
    def __init__(self, env):
        if env.description.name != "room_randomized":
            raise ValueError("spawn-support teacher requires the declared randomized-room generator")
        super().__init__(env, obstacle_aware=True)

    def predict(self, observation):
        if self.coverage is None:
            self.coverage = SpawnSupportCoverage(self.env.evaluator_state())
        return super().predict(observation)


class OpportunityWeightedCoverage(CameraFrustumCoverage):
    """Balance view gain toward floor cells with fewer observation positions.

    A fixed map-only prior counts nominal camera views of each floor cell.
    Inverse counts, capped at four times the median-cell weight, change only
    the gain of the existing camera-frustum planner. Real coverage still comes
    from actual camera poses. Hidden object locations never set this prior.
    """

    def __init__(self, world):
        super().__init__(world)
        self.initial_obstacles = list(self.obstacles)
        self.opportunity_counts = None
        self.cell_weights = None

    def _initialize_weights(self, fovy):
        from .navigation_active_search import projected_visibility

        poses = np.array([[*point, heading] for point in self.candidates for heading in self.headings])
        if not len(poses):
            self.opportunity_counts = np.zeros(len(self.cells), dtype=np.int64)
        else:
            self.opportunity_counts = projected_visibility(
                poses, self.cells, self.camera_mount, fovy, self.initial_obstacles).sum(0)
        positive = self.opportunity_counts[self.opportunity_counts > 0]
        median = float(np.median(positive)) if len(positive) else 1.
        self.cell_weights = np.minimum(4., median / np.maximum(1, self.opportunity_counts))
        self.cell_weights[self.opportunity_counts == 0] = 0.

    def _choose(self, position, yaw, camera_height, fovy, tick):
        if self.cell_weights is None:
            self._initialize_weights(fovy)
        if self.seen.all():
            self.seen[:] = False
        unknown, weights = self.cells[~self.seen], self.cell_weights[~self.seen]
        candidates = []
        for index, point in enumerate(self.candidates):
            distance = float(np.linalg.norm(point - position))
            heading = math.atan2(*(point - position)[::-1])
            for heading_index, look_yaw in enumerate(self.headings):
                gain = float(weights[self.candidate_visible(point, look_yaw, unknown, fovy)].sum())
                if not gain:
                    continue
                turn = (abs(_angle(look_yaw - yaw)) if distance < .22 else
                        abs(_angle(heading - yaw)) + max(0, abs(_angle(look_yaw - heading)) - .45))
                cost = 1. + distance / .20 + turn / .65
                score = gain / cost / (1 + self.visits[index, heading_index])
                candidates.append((score, index, heading_index))
        for _, index, heading_index in sorted(candidates, reverse=True):
            path = plan_grid_path(position, self.candidates[index], self.obstacles,
                                  self.bounds, clearance=.16)
            if path:
                self.destination = self.candidates[index].copy()
                self.look_yaw = float(self.headings[heading_index])
                self.visits[index, heading_index] += 1
                self.chosen_tick = tick
                return path
        return []

    def command(self, world, physics, tick):
        action, diagnostic = super().command(world, physics, tick)
        diagnostic.update(search_method="opportunity_weighted", opportunity_weight_cap=4.,
                          floor_cells_without_nominal_view=int((self.opportunity_counts == 0).sum()))
        return action, diagnostic


class OpportunityWeightedVisualSearchTeacher(VisualSearchTeacher):
    def __init__(self, env):
        super().__init__(env, obstacle_aware=True)

    def predict(self, observation):
        if self.coverage is None:
            self.coverage = OpportunityWeightedCoverage(self.env.evaluator_state())
        return super().predict(observation)


class ObservedGainCoverage(SearchCoverage):
    """Finish a view when its projected unknown region has been observed.

    Candidate selection retains the baseline scoring. Completion uses only
    coverage accumulated from actual camera poses; the nominal projection
    defines the requested region but never credits an unseen cell as seen.
    This is an opt-in development experiment, not a qualified data teacher.
    """

    def __init__(self, world):
        super().__init__(world)
        self.camera_projection = CameraFrustumCoverage(world)
        self.gain_mask = None

    def _choose(self, position, yaw, camera_height, fovy, tick):
        path = super()._choose(position, yaw, camera_height, fovy, tick)
        self.camera_projection.obstacles = self.obstacles
        self.gain_mask = (~self.seen & self.camera_projection.candidate_visible(
            self.destination, self.look_yaw, self.cells, fovy)) if path else None
        return path

    def command(self, world, physics, tick):
        if self.camera_projection.camera_mount is None:
            self.camera_projection.capture_camera_mount(world, physics)
        camera = physics.camera
        self.update_camera(physics.data.cam_xpos[camera], physics.data.cam_xmat[camera],
                           physics.model.cam_fovy[camera])
        initial = int(self.gain_mask.sum()) if self.gain_mask is not None else 0
        remaining = int((self.gain_mask & ~self.seen).sum()) if initial else 0
        completed = bool(initial and tick > self.chosen_tick and remaining <= .05 * initial)
        if completed:
            self.destination = None
        action, diagnostic = super().command(world, physics, tick)
        diagnostic.update(search_method="observed_gain", observed_gain_completion=completed,
                          previous_view_projected_gain=initial,
                          previous_view_projected_unseen=remaining, view_seen_fraction_gate=.95)
        return action, diagnostic


class ObservedGainVisualSearchTeacher(VisualSearchTeacher):
    def __init__(self, env):
        super().__init__(env, obstacle_aware=True)

    def predict(self, observation):
        if self.coverage is None:
            self.coverage = ObservedGainCoverage(self.env.evaluator_state())
        return super().predict(observation)


class ProgressLimitedCoverage(SearchCoverage):
    """Replan a search view after sustained lack of actual camera coverage.

    This experimental teacher changes only search-view persistence. It does
    not credit predicted views, change the task deadline, or inspect targets.
    The qualified default teacher and its approach controller are unchanged.
    """

    def __init__(self, world):
        super().__init__(world)
        self.last_gain_tick = None
        self.command_tick = 0
        self.stagnation_replan = False

    def _choose(self, position, yaw, camera_height, fovy, tick):
        path = super()._choose(position, yaw, camera_height, fovy, tick)
        self.last_gain_tick = tick
        return path

    def update_camera(self, position, rotation, fovy):
        before = int(self.seen.sum())
        super().update_camera(position, rotation, fovy)
        if self.last_gain_tick is None or int(self.seen.sum()) > before:
            self.last_gain_tick = self.command_tick
        # Four seconds of commitment, then two seconds without even one new
        # observed cell. The original 12 s maximum still applies otherwise.
        if (self.destination is not None
                and self.command_tick - self.chosen_tick >= 200
                and self.command_tick - self.last_gain_tick >= 100):
            self.destination = None
            self.stagnation_replan = True

    def command(self, world, physics, tick):
        self.command_tick = tick
        self.stagnation_replan = False
        action, diagnostic = super().command(world, physics, tick)
        diagnostic.update(search_method="progress_limited", stagnation_replan=self.stagnation_replan,
                          last_coverage_gain_tick=self.last_gain_tick, view_chosen_tick=self.chosen_tick)
        return action, diagnostic


class ProgressLimitedVisualSearchTeacher(VisualSearchTeacher):
    def __init__(self, env):
        super().__init__(env, obstacle_aware=True)

    def predict(self, observation):
        if self.coverage is None:
            self.coverage = ProgressLimitedCoverage(self.env.evaluator_state())
        return super().predict(observation)


class MeasuredMotionCoverage(SearchCoverage):
    """Keep baseline search viewpoints; change only their motion execution."""

    def __init__(self, world, controller):
        super().__init__(world)
        self.controller = controller

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
            return np.array([0, 0, 0, 0, 1], np.float32), {"phase": "no_path"}
        waypoint = path[-1]
        for candidate in path[1:]:
            if np.linalg.norm(candidate - position) >= .18:
                waypoint = candidate
                break
        action, diagnostic = self.controller.command(world, waypoint, self.look_yaw,
            orienting=np.linalg.norm(position - self.destination) < .22)
        return action, {"phase": "search", "search_method": "measured_motion",
                        "coverage_fraction": float(self.seen.mean()),
                        "search_destination": self.destination.tolist(), "search_look_yaw": self.look_yaw,
                        "path_nodes": len(path), **diagnostic}


class MeasuredMotionVisualSearchTeacher(VisualSearchTeacher):
    def __init__(self, env, *, calibration, expected_calibration_sha256):
        super().__init__(env, obstacle_aware=True)
        from .navigation_motion import load_motion_calibration

        self.motion_controller = load_motion_calibration(calibration, expected_calibration_sha256,
                                                        env._robot_signature)

    def predict(self, observation):
        if self.coverage is None:
            self.coverage = MeasuredMotionCoverage(self.env.evaluator_state(), self.motion_controller)
        return super().predict(observation)

    def _guard_turn(self, action, world):
        if self.last_seen_position is None:
            # Search already rejects measured curves crossing the known map.
            # Preserve the original approach controller after visual discovery.
            return action
        return super()._guard_turn(action, world)


class TrajectorySearchVisualTeacher(MeasuredMotionVisualSearchTeacher):
    """Plan observable coverage directly along measured feasible motion."""

    def predict(self, observation):
        if self.coverage is None:
            from .navigation_active_search import TrajectorySearchCoverage

            self.coverage = TrajectorySearchCoverage(self.env.evaluator_state(), self.motion_controller)
        return super().predict(observation)


class LongTrajectorySearchVisualTeacher(MeasuredMotionVisualSearchTeacher):
    """Opt-in teacher diagnostic using long measured motion forecasts."""

    def predict(self, observation):
        if self.coverage is None:
            from .navigation_long_search import LongTrajectorySearchCoverage

            self.coverage = LongTrajectorySearchCoverage(self.env.evaluator_state(), self.motion_controller)
        return super().predict(observation)


class PersistentTrajectorySearchVisualTeacher(MeasuredMotionVisualSearchTeacher):
    """Retain a measured search route until vision or clearance interrupts."""

    def predict(self, observation):
        if self.coverage is None:
            from .navigation_persistent_search import PersistentTrajectorySearchCoverage

            self.coverage = PersistentTrajectorySearchCoverage(self.env.evaluator_state(), self.motion_controller)
        return super().predict(observation)


class PoseDiversePersistentVisualTeacher(MeasuredMotionVisualSearchTeacher):
    """Opt-in endpoint-diverse search; visual handoff and approach stay unchanged."""

    def predict(self, observation):
        if self.coverage is None:
            from .navigation_persistent_search import PoseDiversePersistentSearchCoverage

            self.coverage = PoseDiversePersistentSearchCoverage(self.env.evaluator_state(), self.motion_controller)
        return super().predict(observation)


class EarlyVisibilityPersistentVisualTeacher(MeasuredMotionVisualSearchTeacher):
    """Prioritize earlier visibility while retaining the measured safe route."""

    def predict(self, observation):
        if self.coverage is None:
            from .navigation_persistent_search import EarlyVisibilityPersistentSearchCoverage

            self.coverage = EarlyVisibilityPersistentSearchCoverage(self.env.evaluator_state(), self.motion_controller)
        return super().predict(observation)


class PoseCheckedPersistentVisualTeacher(MeasuredMotionVisualSearchTeacher):
    """Replan a search route when measured pose invalidates its prediction."""

    def predict(self, observation):
        if self.coverage is None:
            from .navigation_pose_checked_search import PoseCheckedPersistentSearchCoverage

            self.coverage = PoseCheckedPersistentSearchCoverage(self.env.evaluator_state(), self.motion_controller)
        return super().predict(observation)


class TwoViewRouteVisualSearchTeacher(VisualSearchTeacher):
    def __init__(self, env):
        super().__init__(env, obstacle_aware=True)

    def predict(self, observation):
        if self.coverage is None:
            from .navigation_view_route import TwoViewRouteCoverage

            self.coverage = TwoViewRouteCoverage(self.env.evaluator_state())
        return super().predict(observation)


EXPERIMENTAL_TEACHERS = {
    "visual_search_pose_diverse_persistent": PoseDiversePersistentVisualTeacher,
    "visual_search_pose_checked_persistent": PoseCheckedPersistentVisualTeacher,
    "visual_search_early_persistent": EarlyVisibilityPersistentVisualTeacher,
    "visual_search_persistent_trajectory": PersistentTrajectorySearchVisualTeacher,
    "visual_search_long_trajectory": LongTrajectorySearchVisualTeacher,
    "visual_search_two_view_route": TwoViewRouteVisualSearchTeacher,
    "visual_search_initial_scan": InitialScanVisualSearchTeacher,
    "visual_search_occluded_pockets": OccludedPocketVisualSearchTeacher,
    "visual_search_local_views": LocalViewVisualSearchTeacher,
    "visual_search_latched_view": LatchedViewVisualSearchTeacher,
    "visual_search_pivot_view": PivotViewVisualSearchTeacher,
    "visual_search_camera_frustum": CameraFrustumVisualSearchTeacher,
    "visual_search_spawn_support": SpawnSupportVisualSearchTeacher,
    "visual_search_opportunity_weighted": OpportunityWeightedVisualSearchTeacher,
    "visual_search_observed_gain": ObservedGainVisualSearchTeacher,
    "visual_search_progress_limited": ProgressLimitedVisualSearchTeacher,
    "visual_search_measured_motion": MeasuredMotionVisualSearchTeacher,
    "visual_search_trajectory_gain": TrajectorySearchVisualTeacher,
}
