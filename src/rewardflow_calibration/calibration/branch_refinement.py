"""Reusable midpoint-branch insertion for activation-boundary refinement.

This module deliberately knows nothing about images or DreamSim.  Callers
provide an evaluator ``alpha -> distance`` so the same refinement routine can
be reused by later calibration stages and other rollout backends.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class BranchRefinementResult:
    """The bracket and all branches inserted during midpoint refinement."""

    alpha_start: float
    invalid_alpha: float
    valid_alpha: float
    invalid_distance: float
    valid_distance: float
    adjacent_distance_gap: float
    inserted_branches: tuple[dict[str, float], ...]
    recursion_depth: int
    final_interval: float
    refined: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def refine_activation_bracket(
    *,
    invalid_alpha: float,
    invalid_distance: float,
    valid_alpha: float,
    valid_distance: float,
    evaluate: Callable[[float], float],
    activation_distance_threshold: float,
    recursion_depth: int = 2,
    adjacent_distance_gap_threshold: float | None = None,
) -> BranchRefinementResult:
    """Insert midpoint branches around an invalid/valid activation bracket.

    The interval always keeps this invariant:

    ``distance(invalid_alpha) <= threshold < distance(valid_alpha)``.

    The midpoint is evaluated exactly ``recursion_depth`` times by default.
    Each result becomes the new invalid or valid endpoint, so if both inserted
    points are still inactive the original valid endpoint remains the final
    ``alpha_start``.  A positive ``adjacent_distance_gap_threshold`` can be
    supplied to skip refinement when the adjacent jump is not large enough.
    """

    values = (invalid_alpha, valid_alpha, invalid_distance, valid_distance)
    if any(not np.isfinite(value) for value in values):
        raise ValueError("bracket values must be finite")
    if not 0.0 <= invalid_alpha < valid_alpha <= 1.0:
        raise ValueError("bracket must satisfy 0 <= invalid_alpha < valid_alpha <= 1")
    if invalid_distance > activation_distance_threshold:
        raise ValueError("invalid endpoint already exceeds activation threshold")
    if valid_distance <= activation_distance_threshold:
        raise ValueError("valid endpoint does not exceed activation threshold")
    if not np.isfinite(activation_distance_threshold) or activation_distance_threshold < 0:
        raise ValueError("activation_distance_threshold must be finite and non-negative")
    if recursion_depth < 0:
        raise ValueError("recursion_depth must be non-negative")
    if adjacent_distance_gap_threshold is not None:
        if not np.isfinite(adjacent_distance_gap_threshold) or adjacent_distance_gap_threshold < 0:
            raise ValueError("adjacent_distance_gap_threshold must be non-negative")

    gap = float(valid_distance - invalid_distance)
    if adjacent_distance_gap_threshold is not None and gap < adjacent_distance_gap_threshold:
        return BranchRefinementResult(
            alpha_start=float(valid_alpha),
            invalid_alpha=float(invalid_alpha),
            valid_alpha=float(valid_alpha),
            invalid_distance=float(invalid_distance),
            valid_distance=float(valid_distance),
            adjacent_distance_gap=gap,
            inserted_branches=(),
            recursion_depth=0,
            final_interval=float(valid_alpha - invalid_alpha),
            refined=False,
        )

    low_alpha = float(invalid_alpha)
    low_distance = float(invalid_distance)
    high_alpha = float(valid_alpha)
    high_distance = float(valid_distance)
    inserted: list[dict[str, float]] = []

    for _ in range(recursion_depth):
        midpoint = (low_alpha + high_alpha) / 2.0
        midpoint_distance = float(evaluate(midpoint))
        if not np.isfinite(midpoint_distance):
            raise ValueError(f"distance is not finite at alpha={midpoint}")
        inserted.append({"alpha": midpoint, "distance": midpoint_distance})
        if midpoint_distance > activation_distance_threshold:
            high_alpha = midpoint
            high_distance = midpoint_distance
        else:
            low_alpha = midpoint
            low_distance = midpoint_distance

    return BranchRefinementResult(
        alpha_start=high_alpha,
        invalid_alpha=low_alpha,
        valid_alpha=high_alpha,
        invalid_distance=low_distance,
        valid_distance=high_distance,
        adjacent_distance_gap=gap,
        inserted_branches=tuple(inserted),
        recursion_depth=recursion_depth,
        final_interval=high_alpha - low_alpha,
        refined=bool(inserted),
    )
