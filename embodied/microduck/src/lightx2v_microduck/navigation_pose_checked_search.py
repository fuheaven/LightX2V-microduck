"""Opt-in teacher experiment: invalidate routes whose nominal pose has drifted.

Robot pose and the known map are privileged teacher inputs. Hidden target
positions never enter this check, and predictions never become observations.
The baseline persistent planner and its safety checks remain unchanged.
"""

import numpy as np

from .navigation_motion import angle
from .navigation_persistent_search import PersistentTrajectorySearchCoverage


class PoseCheckedPersistentSearchCoverage(PersistentTrajectorySearchCoverage):
    max_position_error_m = .15
    max_heading_error_rad = np.deg2rad(20.)

    def __init__(self, world, controller):
        super().__init__(world, controller)
        self.nominal_route_poses = None

    def _nominal_poses(self, world, moving):
        pose = np.array([*world['robot_position'][:2], world['robot_yaw']])
        pieces = [pose[None]]
        for index in self.route:
            curve = self._curves(pose[:2], pose[2], moving)[index]
            pieces.append(curve[1:])
            pose = curve[-1]
            moving = (np.linalg.norm(curve[-1, :2] - curve[-11, :2]) / .2 > .025
                      or abs(angle(curve[-1, 2] - curve[-11, 2])) / .2 > .15)
        return np.concatenate(pieces)

    def _plan(self, world, physics, tick, moving):
        position_error, heading_error, reference_tick = None, None, self.route_start_tick
        drifted = False
        if reference_tick is not None and self.nominal_route_poses is not None:
            elapsed = tick - reference_tick
            if 0 <= elapsed < len(self.nominal_route_poses):
                nominal = self.nominal_route_poses[elapsed]
                position_error = float(np.linalg.norm(np.asarray(world['robot_position'][:2]) - nominal[:2]))
                heading_error = float(abs(angle(world['robot_yaw'] - nominal[2])))
                if not np.isfinite([position_error, heading_error]).all():
                    raise ValueError('nonfinite pose in persistent search route check')
                drifted = (position_error > self.max_position_error_m
                           or heading_error > self.max_heading_error_rad)
                if drifted:
                    self.route = ()
                    self.route_start_tick = None
                    self.nominal_route_poses = None

        action, diagnostic = super()._plan(world, physics, tick, moving)
        if not diagnostic['route_retained']:
            self.nominal_route_poses = self._nominal_poses(world, moving) if self.route else None
        if drifted:
            diagnostic['route_replan_reason'] = 'nominal_pose_drift'
        return action, {
            **diagnostic,
            'pose_reference_route_start_tick': reference_tick,
            'nominal_position_error_m': position_error,
            'nominal_heading_error_rad': heading_error,
            'pose_drift_replanned': drifted,
            'pose_check_position_limit_m': self.max_position_error_m,
            'pose_check_heading_limit_rad': float(self.max_heading_error_rad),
        }

    def command(self, world, physics, tick):
        action, diagnostic = super().command(world, physics, tick)
        diagnostic['search_method'] = 'pose_checked_persistent_trajectory'
        return action, diagnostic
