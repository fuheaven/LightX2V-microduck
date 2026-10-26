"""Map-only, finite-horizon viewpoint tour diagnostic.

This plans nominal views, not robot actions. Predicted coverage must never
update an actual observer's seen mask. No target coordinate is an input.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ViewTour:
    indices: tuple[int, ...]
    arrival_seconds: tuple[float, ...]
    covered_cells: int
    coverage_time_area: float


def plan_view_tour(visibility, first_cost, transition_cost, *, deadline_s=22.,
                   max_views=6, beam_width=64):
    """Beam search prioritizing coverage by a fixed deadline, then early views.

    Returns a bounded heuristic, not a proof of optimality or of achievable
    travel times. Each cell contributes to coverage and timing only once.
    Infinite travel cost marks an unreachable transition.
    """
    masks = np.asarray(visibility, dtype=bool)
    first = np.asarray(first_cost, dtype=float)
    costs = np.asarray(transition_cost, dtype=float)
    if (masks.ndim != 2 or first.shape != (len(masks),)
            or costs.shape != (len(masks), len(masks))
            or np.isnan(first).any() or np.isnan(costs).any()
            or np.any(first < 0) or np.any(costs < 0)
            or not np.isfinite(deadline_s) or deadline_s <= 0
            or type(max_views) is not int or max_views < 1
            or type(beam_width) is not int or beam_width < 1):
        raise ValueError('invalid viewpoint graph or planning budget')
    packed = [int.from_bytes(np.packbits(row, bitorder='little').tobytes(), 'little')
              for row in masks]
    # State: route, arrivals, union of actually predicted cells, time-area.
    beam = [((), (), 0, 0.)]
    best = beam[0]

    def rank(state):
        route, arrivals, covered, area = state
        return (covered.bit_count(), area, -(arrivals[-1] if arrivals else 0.),
                tuple(-i for i in route))

    for _ in range(max_views):
        expanded = []
        for route, arrivals, covered, area in beam:
            for i, mask in enumerate(packed):
                if i in route:
                    continue
                arrival = first[i] if not route else arrivals[-1] + costs[route[-1], i]
                gain = (mask & ~covered).bit_count()
                if not gain or not np.isfinite(arrival) or arrival > deadline_s:
                    continue
                expanded.append((route + (i,), arrivals + (float(arrival),),
                                 covered | mask, area + gain * (deadline_s - arrival)))
        if not expanded:
            break
        beam = sorted(expanded, key=rank, reverse=True)[:beam_width]
        if rank(beam[0]) > rank(best):
            best = beam[0]
    return ViewTour(best[0], best[1], best[2].bit_count(), best[3])
