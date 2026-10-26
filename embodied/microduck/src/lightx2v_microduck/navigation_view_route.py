"""Experimental two-view search over the map, never hidden target locations.

Unlike the short motion-primitive lookahead, this ranks spatial observation
destinations connected by collision-aware grid paths. Camera and travel-time
predictions remain approximations; only actual camera poses update coverage.
"""
import math

import numpy as np

from .navigation_teacher_experiments import CameraFrustumCoverage
from .teachers import _angle, plan_grid_path


def view_route_scores(visible, first_cost, second_cost, remaining_s):
    """Area under predicted newly observed coverage until the fixed deadline.

Each cell contributes once, at its earliest predicted observation. Infeasible
transitions use infinite cost. A route can contain only its first view.
"""
    visible = np.asarray(visible, dtype=bool)
    first_cost, second_cost = np.asarray(first_cost, float), np.asarray(second_cost, float)
    if (visible.ndim != 2 or first_cost.shape != (len(visible),)
            or second_cost.shape != (len(visible), len(visible))
            or not np.isfinite(remaining_s) or remaining_s <= 0
            or np.isnan(first_cost).any() or np.isnan(second_cost).any()
            or np.any(first_cost < 0) or np.any(second_cost < 0)):
        raise ValueError("invalid observation-route inputs")
    gain = visible.sum(1)
    first = gain * np.maximum(0., remaining_s - first_cost)
    scores, following = first.copy(), np.full(len(visible), -1, dtype=int)
    for i in range(len(visible)):
        new_gain = (visible & ~visible[i]).sum(1)
        extra = new_gain * np.maximum(0., remaining_s - first_cost[i] - second_cost[i])
        if len(extra) and extra.max() > 0:
            j = int(np.argmax(extra))
            scores[i] += extra[j]
            following[i] = j
    return scores, following


class TwoViewRouteCoverage(CameraFrustumCoverage):
    """Receding two-view route, with original motion and observation accounting."""

    @staticmethod
    def _path_cost(path, start_yaw, end_yaw):
        if not path:
            return float('inf')
        edges = np.diff(np.asarray(path), axis=0)
        edges = edges[np.linalg.norm(edges, axis=1) > 1e-9]
        if not len(edges):
            return 1. + abs(_angle(end_yaw - start_yaw)) / .65
        headings = np.arctan2(edges[:, 1], edges[:, 0])
        turn = abs(_angle(headings[0] - start_yaw)) + max(0., abs(_angle(end_yaw - headings[-1])) - .45)
        # Grid quantization creates alternating diagonals; count only net
        # endpoint turns, and explicitly retain this as a nominal time model.
        return 1. + float(np.linalg.norm(edges, axis=1).sum()) / .20 + turn / .65

    def _choose(self, position, yaw, camera_height, fovy, tick):
        unknown = self.cells[~self.seen]
        if not len(unknown):
            self.seen[:] = False
            unknown = self.cells
        masks, identities, nominal = [], [], []
        for i, point in enumerate(self.candidates):
            distance = float(np.linalg.norm(point - position))
            heading = math.atan2(*(point - position)[::-1])
            for h, look in enumerate(self.headings):
                mask = self.candidate_visible(point, look, unknown, fovy)
                if not mask.any():
                    continue
                turn = (abs(_angle(look - yaw)) if distance < .22 else
                        abs(_angle(heading - yaw)) + max(0., abs(_angle(look - heading)) - .45))
                masks.append(mask)
                identities.append((i, h))
                nominal.append(1. + distance / .20 + turn / .65)
        if not masks:
            return super()._choose(position, yaw, camera_height, fovy, tick)
        masks = np.asarray(masks)
        gain = masks.sum(1)
        # Mix nearby efficient views and globally informative views so the
        # route shortlist is not solely the existing greedy ranking.
        shortlist = sorted(set(np.argsort(-gain / np.asarray(nominal), kind='stable')[:16]) |
                           set(np.argsort(-gain, kind='stable')[:16]))
        choices = [identities[i] for i in shortlist]
        masks = masks[shortlist]
        first_paths, transitions = {}, {}
        for i, _ in choices:
            if i not in first_paths:
                first_paths[i] = plan_grid_path(position, self.candidates[i], self.obstacles, self.bounds, clearance=.16)
        first_cost = [self._path_cost(first_paths[i], yaw, self.headings[h]) for i, h in choices]
        second_cost = np.full((len(choices), len(choices)), np.inf)
        for a, (i, h) in enumerate(choices):
            for b, (j, k) in enumerate(choices):
                if a == b:
                    continue
                if (i, j) not in transitions:
                    transitions[i, j] = plan_grid_path(self.candidates[i], self.candidates[j],
                                                       self.obstacles, self.bounds, clearance=.16)
                second_cost[a, b] = self._path_cost(transitions[i, j], self.headings[h], self.headings[k])
        remaining_s = max(.1, 30. - tick / 50. - 2.)
        scores, following = view_route_scores(masks, first_cost, second_cost, remaining_s)
        scores /= np.array([1 + self.visits[i, h] for i, h in choices])
        if not np.any(scores > 0):
            self.route_diagnostic = {'two_view_budget_fallback': True}
            return super()._choose(position, yaw, camera_height, fovy, tick)
        for choice in np.argsort(-scores, kind='stable'):
            i, h = choices[choice]
            if first_paths[i]:
                self.destination = self.candidates[i].copy()
                self.look_yaw = float(self.headings[h])
                self.visits[i, h] += 1
                self.chosen_tick = tick
                self.route_diagnostic = {'route_shortlist': len(choices),
                    'predicted_coverage_time_area': float(scores[choice]),
                    'nominal_first_view_seconds': float(first_cost[choice]),
                    'second_view_selected': bool(following[choice] >= 0),
                    'predicted_visibility_is_not_observed': True,
                    'travel_time_model': 'grid path length and endpoint yaw approximation'}
                return first_paths[i]
        return super()._choose(position, yaw, camera_height, fovy, tick)

    def command(self, world, physics, tick):
        action, diagnostic = super().command(world, physics, tick)
        diagnostic.update(getattr(self, 'route_diagnostic', {}), search_method='two_view_route')
        return action, diagnostic
