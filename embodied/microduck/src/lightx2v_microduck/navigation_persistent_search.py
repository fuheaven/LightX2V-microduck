"""Opt-in search teacher: retain a measured route while checking clearance.

This privileged map-based experiment does not change the student contract.
The outer visual teacher interrupts search as soon as the camera sees target.
Only actual camera observations update coverage, never predicted route views.
"""
import numpy as np

from .navigation_long_search import LongTrajectorySearchCoverage
from .navigation_motion import MeasuredMotionController


class PersistentTrajectorySearchCoverage(LongTrajectorySearchCoverage):
    primitive_ticks = 150  # Same three-second measured calibration primitives.

    def __init__(self, world, controller):
        super().__init__(world, controller)
        self.route = ()
        self.route_start_tick = None
        self.route_diagnostics = {}

    def command(self, world, physics, tick):
        if self.route_start_tick is not None:
            stage = (tick - self.route_start_tick) // self.primitive_ticks
            if stage != self.plan_diagnostic.get('route_stage'):
                # A newly observed obstacle can shift the periodic recheck
                # clock. Still switch primitives at their actual boundaries.
                self.plan_tick = None
        action, diagnostic = super().command(world, physics, tick)
        diagnostic['search_method'] = 'persistent_measured_trajectory'
        return action, diagnostic

    def _plan(self, world, physics, tick, moving):
        reason = 'initial_route'
        if self.route_start_tick is not None:
            elapsed = tick - self.route_start_tick
            stage = elapsed // self.primitive_ticks
            if elapsed >= 0 and stage < len(self.route) and tick < 1400:
                index = self.route[stage]
                remaining = self.primitive_ticks - elapsed % self.primitive_ticks
                # Re-evaluate the remainder from the actual pose and current
                # known map, rather than trusting the initial nominal pose.
                curve = self._curves(np.asarray(world['robot_position'][:2]),
                                     world['robot_yaw'], moving)[index, :remaining + 1]
                current = float(MeasuredMotionController.clearance(
                    np.asarray(world['robot_position'][:2]), world))
                minimum = float(MeasuredMotionController.clearance(curve[:, :2], world).min())
                if current > 0 and minimum >= min(.14, current - .005):
                    return self.controller.commands[index].astype(np.float32, copy=True), {
                        **self.route_diagnostics, 'command_index': index,
                        'route_start_tick': self.route_start_tick, 'route_stage': stage,
                        'route_primitive_ticks': self.primitive_ticks,
                        'remaining_primitive_ticks': remaining,
                        'route_retained': True, 'route_replan_reason': None,
                        'predicted_remaining_clearance_m': minimum,
                        'safety_recheck_ticks': 25,
                    }
                reason = 'remaining_curve_not_safe_on_current_map'
            else:
                reason = 'route_finished_or_clock_changed'
        action, diagnostic = super()._plan(world, physics, tick, moving)
        sequence = diagnostic.get('nominal_command_sequence')
        self.route = tuple(sequence) if sequence else ()
        self.route_start_tick = tick if self.route else None
        self.route_diagnostics = dict(diagnostic)
        return action, {
            **diagnostic, 'route_start_tick': self.route_start_tick, 'route_stage': 0,
            'route_primitive_ticks': self.primitive_ticks,
            'remaining_primitive_ticks': self.primitive_ticks if self.route else None,
            'route_retained': False, 'route_replan_reason': reason,
            'safety_recheck_ticks': 25,
        }


class EarlyVisibilityPersistentSearchCoverage(PersistentTrajectorySearchCoverage):
    """Compare earlier predicted observation with maximal terminal coverage.

    Keep route execution, measured dynamics, safety checks and actual detection
    unchanged. A smaller covered area can win only in this opt-in experiment.
    """

    def _route_rank(self, item, remaining_s):
        count, area, tie = super()._route_rank(item, remaining_s)
        return area, count, tie

    def command(self, world, physics, tick):
        action, diagnostic = super().command(world, physics, tick)
        diagnostic.update(search_method='early_visibility_persistent_trajectory',
                          route_objective='coverage_time_area_then_terminal_coverage')
        return action, diagnostic


class PoseDiversePersistentSearchCoverage(PersistentTrajectorySearchCoverage):
    """Opt-in diagnostic retaining different reachable viewpoints in the beam.

    The baseline keeps only one continuation per first command. This variant
    keeps one per quantized endpoint, viewing direction and motion mode, so
    alternative continuations of the same first command can survive. Scoring,
    measured dynamics, horizon and beam size remain the baseline's. These bins
    match the existing search position/heading grids, not target positions.
    """

    def _prune_beam(self, expanded, rank):
        viewpoints = {}
        for item in sorted(expanded, key=rank, reverse=True):
            pose, moving = item[:2]
            xy = tuple(np.floor(np.asarray(pose[:2]) / .35 + .5).astype(int))
            yaw = float(pose[2]) % (2 * np.pi)
            heading = int(np.floor(yaw / (2 * np.pi / 16) + .5)) % 16
            viewpoints.setdefault((*xy, heading, bool(moving)), item)
        return sorted(viewpoints.values(), key=rank, reverse=True)[:self.beam_width]

    def command(self, world, physics, tick):
        action, diagnostic = super().command(world, physics, tick)
        diagnostic.update(search_method='pose_diverse_persistent_trajectory',
                          beam_diversity='endpoint_xy_0.35m_heading_16_motion_mode')
        return action, diagnostic
