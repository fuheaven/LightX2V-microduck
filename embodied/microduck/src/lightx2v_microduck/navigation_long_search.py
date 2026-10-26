"""Experimental long-horizon search over measured official locomotion.

The planner accounts for camera views during motion. It remains a privileged
map-based teacher experiment; predicted visibility never becomes observation.
"""
import math

import numpy as np

from .navigation_active_search import TrajectorySearchCoverage, projected_visibility
from .navigation_motion import angle


class LongTrajectorySearchCoverage(TrajectorySearchCoverage):
    horizon_stages = 7  # Measured primitives are three seconds each.
    beam_width = 8

    def _route_rank(self, item, remaining_s):
        times = item[2]
        finite = np.isfinite(times)
        return (int(finite.sum()), float((remaining_s - times[finite]).sum()),
                tuple(-i for i in item[3]))

    def _prune_beam(self, expanded, rank):
        # Preserve the established first-action diversity for the baseline.
        first_actions = {}
        for item in sorted(expanded, key=rank, reverse=True):
            first_actions.setdefault(item[3][0], item)
        return sorted(first_actions.values(), key=rank, reverse=True)[:self.beam_width]

    def command(self, world, physics, tick):
        action, diagnostic = super().command(world, physics, tick)
        diagnostic['search_method'] = 'long_measured_trajectory_gain'
        return action, diagnostic

    def _plan(self, world, physics, tick, moving):
        unknown = self.cells[~self.seen]
        if not len(unknown):
            # Match the existing re-observation behavior, never infer success.
            return super()._plan(world, physics, tick, moving)
        remaining_s = min(21., max(0., 28. - tick / 50.))
        if remaining_s < 3.:
            return super()._plan(world, physics, tick, moving)
        stages = min(self.horizon_stages, int(remaining_s // 3))
        fovy = physics.model.cam_fovy[physics.camera]
        # endpoint, moving, earliest nominal sighting for each unknown cell,
        # command sequence. Infinity means still unseen in this prediction.
        beam = [(np.array([*world['robot_position'][:2], world['robot_yaw']]),
                 moving, np.full(len(unknown), np.inf), ())]
        best, expanded_count = None, 0

        def rank(item):
            return self._route_rank(item, remaining_s)

        for depth in range(stages):
            expanded = []
            for pose, in_motion, previous_times, commands in beam:
                curves = self._curves(pose[:2], pose[2], in_motion)
                allowed = self._allowed(curves, world)
                indices = np.flatnonzero(allowed)
                if not len(indices):
                    continue
                samples = curves[indices, 25::25]
                visible = projected_visibility(samples.reshape(-1, 3), unknown, self.camera_mount,
                                               fovy, self.obstacles).reshape(len(indices), 6, len(unknown))
                sample_times = depth * 3. + np.arange(1, 7) * .5
                earliest = np.where(visible, sample_times[None, :, None], np.inf).min(1)
                earliest = np.minimum(earliest, previous_times[None])
                for n, index in enumerate(indices):
                    endpoint = curves[index, -1]
                    speed = np.linalg.norm(curves[index, -1, :2] - curves[index, -11, :2]) / .2
                    angular = abs(angle(curves[index, -1, 2] - curves[index, -11, 2])) / .2
                    expanded.append((endpoint, speed > .025 or angular > .15,
                                     earliest[n], commands + (int(index),)))
                expanded_count += len(indices)
            if not expanded:
                break
            # Retain competing first actions, rather than eight nearly
            # identical continuations of a single first action.
            beam = self._prune_beam(expanded, rank)
            if best is None or rank(beam[0]) > rank(best):
                best = beam[0]
        if best is None or not np.isfinite(best[2]).any():
            return super()._plan(world, physics, tick, moving)
        finite = np.isfinite(best[2])
        coverage_time_area = float((remaining_s - best[2][finite]).sum())
        if not math.isfinite(coverage_time_area):
            return super()._plan(world, physics, tick, moving)
        selected = best[3][0]
        return self.controller.commands[selected].astype(np.float32, copy=True), {
            'phase': 'search', 'search_planner': 'long_measured_motion_beam',
            'command_index': selected, 'nominal_command_sequence': list(best[3]),
            'nominal_planning_seconds': len(best[3]) * 3., 'beam_width': self.beam_width,
            'expanded_motion_branches': expanded_count,
            'predicted_new_cells': int(finite.sum()), 'predicted_coverage_time_area': coverage_time_area,
            'measured_anchor_mode': int(moving), 'global_search_fallback': False,
            'predicted_visibility_is_not_observed': True,
            'hidden_target_coordinates_used': False,
        }
