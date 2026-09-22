"""Control-point initialization and redundancy filtering utilities."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np


def uniform_control_points(
    alpha_start: float,
    alpha_end: float = 1.0,
    betas: Iterable[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> list[float]:
    """Map normalized beta positions to the sample-specific alpha interval."""

    if not 0.0 <= alpha_start < alpha_end <= 1.0:
        raise ValueError("expected 0 <= alpha_start < alpha_end <= 1")
    beta_values = [float(beta) for beta in betas]
    if not beta_values or any(not 0.0 <= beta <= 1.0 for beta in beta_values):
        raise ValueError("betas must be non-empty and lie in [0, 1]")
    if any(right <= left for left, right in zip(beta_values[:-1], beta_values[1:])):
        raise ValueError("betas must be strictly increasing")
    return [alpha_start + beta * (alpha_end - alpha_start) for beta in beta_values]


def filter_redundant_control_points(
    points: Iterable[float],
    adjacent_distance: Callable[[float, float], float],
    min_adjacent_distance: float,
) -> tuple[list[float], list[float]]:
    """Remove interior points that are perceptually indistinguishable from a neighbour.

    The first and last points are always retained.  If an interior point has a
    gap below ``min_adjacent_distance`` on either side, it is removed and the
    new neighbouring gap is measured again.  Repeating this makes the filter
    independent of how many redundant points an upstream search inserted.
    """

    if not np.isfinite(min_adjacent_distance) or min_adjacent_distance < 0:
        raise ValueError("min_adjacent_distance must be finite and non-negative")
    current = [float(point) for point in points]
    if len(current) < 2:
        raise ValueError("at least two control points are required")
    if any(not np.isfinite(point) for point in current):
        raise ValueError("control points must be finite")
    if any(right <= left for left, right in zip(current[:-1], current[1:])):
        raise ValueError("control points must be strictly increasing")

    removed: list[float] = []
    changed = True
    while changed and len(current) > 2:
        changed = False
        for index in range(1, len(current) - 1):
            left_gap = float(adjacent_distance(current[index - 1], current[index]))
            right_gap = float(adjacent_distance(current[index], current[index + 1]))
            if not np.isfinite(left_gap) or not np.isfinite(right_gap):
                raise ValueError("adjacent distance must be finite")
            if min(left_gap, right_gap) < min_adjacent_distance:
                removed.append(current.pop(index))
                changed = True
                break
    return current, removed
