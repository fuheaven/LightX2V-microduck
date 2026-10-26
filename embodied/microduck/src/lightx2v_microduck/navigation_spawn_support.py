"""Map-only support of the declared randomized-room target distribution.

These are teacher-side geometry weights, not student inputs. No seed, target
color, ball coordinate, or realized layout is an argument. The support comes
from scenes.randomized_room_layout; it does not model the generator's preferred
occluded locations or condition on where a target actually spawned.
"""

from __future__ import annotations

import numpy as np


def spawn_support_weights(cells, obstacles, room_bounds, *, cell_width=.12):
    """Estimate valid area in each coverage cell with fixed 5-by-5 quadrature.

    A cell straddling a boundary keeps fractional weight rather than having its
    entire area discarded based on the center. Only static scene obstacles
    belong here; previously observed balls are execution obstacles, not spawn
    exclusions for this prior.
    """
    points = np.asarray(cells, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("coverage cells must be finite 3D points")
    if tuple(room_bounds) != (-1.5, 1.5, -1.5, 1.5):
        raise ValueError("spawn support is specific to the declared randomized room")
    if not np.isfinite(cell_width) or cell_width <= 0:
        raise ValueError("coverage cell width must be positive")
    offsets = (np.arange(5, dtype=float) + .5) / 5 - .5
    offsets = np.array([(x, y) for x in offsets for y in offsets]) * cell_width
    probes = points[:, None, :2] + offsets[None]
    valid = np.all(np.abs(probes) < 1.25, axis=-1) & (np.linalg.norm(probes, axis=-1) > .50)
    for obstacle in obstacles:
        xy, half = np.asarray(obstacle["xy"]), np.asarray(obstacle["half_size"])
        if xy.shape != (2,) or half.shape != (2,) or not np.isfinite([xy, half]).all() or np.any(half <= 0):
            raise ValueError("invalid static obstacle footprint")
        clearance = np.linalg.norm(np.maximum(np.abs(probes - xy) - half, 0.), axis=-1)
        valid &= clearance > .16
    return valid.mean(axis=1)
