"""Model- and metric-agnostic elastic-band control-point search."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np

from .control_points import filter_redundant_control_points


@dataclass(frozen=True)
class ElasticBandConfig:
    """Hyperparameters for perceptual-gap equalization."""

    target_gap: float = 0.05
    max_points: int = 10
    max_iterations: int = 25
    expand_threshold: float = 0.05
    lam: float = 1.0
    min_alpha_spacing: float = 0.01
    move_fraction: float = 1.0
    base_step_fraction: float = 0.02
    min_meaningful_move: float = 0.001
    min_normalized_gap_for_move: float = 0.08
    min_gap_imbalance_for_move: float = 0.01
    filter_min_adjacent_gap: float = 0.001

    def validate(self) -> None:
        if not np.isfinite(self.target_gap) or self.target_gap <= 0:
            raise ValueError("target_gap must be finite and positive")
        if self.max_points < 2:
            raise ValueError("max_points must be at least 2")
        if self.max_iterations < 0:
            raise ValueError("max_iterations must be non-negative")
        if not np.isfinite(self.expand_threshold) or self.expand_threshold < 0:
            raise ValueError("expand_threshold must be non-negative")
        if not np.isfinite(self.lam) or self.lam < 0:
            raise ValueError("lam must be non-negative")
        if not 0 < self.min_alpha_spacing <= 1:
            raise ValueError("min_alpha_spacing must be in (0, 1]")
        if not 0 <= self.move_fraction <= 1:
            raise ValueError("move_fraction must be in [0, 1]")
        if not 0 < self.base_step_fraction <= 1:
            raise ValueError("base_step_fraction must be in (0, 1]")
        if not np.isfinite(self.min_meaningful_move) or self.min_meaningful_move < 0:
            raise ValueError("min_meaningful_move must be non-negative")
        if not 0 <= self.min_normalized_gap_for_move:
            raise ValueError("min_normalized_gap_for_move must be non-negative")
        if not 0 <= self.min_gap_imbalance_for_move:
            raise ValueError("min_gap_imbalance_for_move must be non-negative")
        if not np.isfinite(self.filter_min_adjacent_gap) or self.filter_min_adjacent_gap < 0:
            raise ValueError("filter_min_adjacent_gap must be non-negative")


@dataclass(frozen=True)
class ElasticBandResult:
    initial_control_points: tuple[float, ...]
    control_points: tuple[float, ...]
    adjacent_gaps: tuple[float, ...]
    evaluated_alphas: tuple[float, ...]
    removed_points: tuple[float, ...]
    iterations: int
    expansions: int
    moves: int
    termination_reason: str
    history: tuple[dict[str, Any], ...]
    config: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for key in ("initial_control_points", "control_points", "adjacent_gaps", "evaluated_alphas", "removed_points", "history"):
            if isinstance(result[key], tuple):
                result[key] = list(result[key])
        return result


def _canonical(value: float) -> float:
    return round(float(value), 10)


def _cosine_decay(iteration: int, max_iterations: int) -> float:
    if max_iterations <= 1:
        return 1.0
    progress = min(1.0, max(0.0, iteration / float(max_iterations - 1)))
    return 0.5 * (1.0 + np.cos(np.pi * progress))


def elastic_band_search(
    *,
    initial_control_points: list[float] | tuple[float, ...],
    evaluate_image: Callable[[float], Any],
    distance: Callable[[Any, Any], float],
    config: ElasticBandConfig | None = None,
) -> ElasticBandResult:
    """Optimize alpha control points using DreamSim-like adjacent distances.

    ``evaluate_image`` and ``distance`` are injected so this routine can be
    reused with any rollout backend and any perceptual metric.  The first and
    last control points are fixed throughout the search.
    """

    config = config or ElasticBandConfig()
    config.validate()
    points = [_canonical(point) for point in initial_control_points]
    if len(points) < 2 or len(points) > config.max_points:
        raise ValueError("initial control-point count is outside the configured bounds")
    if any(not 0.0 <= point <= 1.0 for point in points):
        raise ValueError("control points must be in [0, 1]")
    if any(right <= left for left, right in zip(points[:-1], points[1:])):
        raise ValueError("control points must be strictly increasing")
    if any(right - left < config.min_alpha_spacing for left, right in zip(points[:-1], points[1:])):
        raise ValueError("initial control points violate min_alpha_spacing")

    initial_points = tuple(points)
    image_cache: dict[float, Any] = {}
    pair_cache: dict[tuple[float, float], float] = {}

    def image(alpha: float) -> Any:
        key = _canonical(alpha)
        if key not in image_cache:
            image_cache[key] = evaluate_image(key)
        return image_cache[key]

    def gap(left: float, right: float) -> float:
        left_key, right_key = _canonical(left), _canonical(right)
        key = (left_key, right_key)
        if key not in pair_cache:
            value = float(distance(image(left_key), image(right_key)))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"invalid perceptual distance for ({left_key}, {right_key})")
            pair_cache[key] = value
        return pair_cache[key]

    history: list[dict[str, Any]] = []
    expansions = 0
    moves = 0
    termination_reason = "max_iterations"
    iterations = 0

    for iteration in range(config.max_iterations):
        iterations = iteration + 1
        gaps = [gap(left, right) for left, right in zip(points[:-1], points[1:])]
        largest_index = int(np.argmax(gaps))
        largest_gap = gaps[largest_index]
        if largest_gap > config.target_gap * (1.0 + config.expand_threshold) and len(points) < config.max_points:
            left = points[largest_index]
            right = points[largest_index + 1]
            midpoint = _canonical((left + right) / 2.0)
            if midpoint - left >= config.min_alpha_spacing and right - midpoint >= config.min_alpha_spacing:
                points.insert(largest_index + 1, midpoint)
                expansions += 1
                history.append({
                    "iteration": iteration,
                    "operation": "EXPAND",
                    "left": left,
                    "right": right,
                    "alpha": midpoint,
                    "gap": largest_gap,
                })
                continue

        moved_this_iteration = False
        decay = _cosine_decay(iteration, config.max_iterations)
        interval_width = points[-1] - points[0]
        for index in range(1, len(points) - 1):
            left_gap = gaps[index - 1] / config.target_gap
            right_gap = gaps[index] / config.target_gap
            largest_local_gap = max(left_gap, right_gap)
            imbalance = abs(left_gap - right_gap)
            if largest_local_gap < config.min_normalized_gap_for_move:
                continue
            if imbalance < config.min_gap_imbalance_for_move:
                continue
            direction = -1.0 if left_gap > right_gap else 1.0
            base_step = config.base_step_fraction * interval_width * decay
            step = base_step * (1.0 + config.lam * imbalance)
            current = points[index]
            lower = points[index - 1] + config.min_alpha_spacing
            upper = points[index + 1] - config.min_alpha_spacing
            new_alpha = min(upper, max(lower, current + direction * step))
            move = abs(new_alpha - current)
            move_threshold = max(config.min_meaningful_move, config.move_fraction * config.min_alpha_spacing)
            if move >= move_threshold:
                points[index] = _canonical(new_alpha)
                moves += 1
                moved_this_iteration = True
                history.append({
                    "iteration": iteration,
                    "operation": "MOVE",
                    "index": index,
                    "old_alpha": current,
                    "new_alpha": points[index],
                    "left_gap": left_gap * config.target_gap,
                    "right_gap": right_gap * config.target_gap,
                })
        if not moved_this_iteration:
            termination_reason = "converged"
            break

    def final_gap(left: float, right: float) -> float:
        return gap(left, right)

    filtered_points, removed = filter_redundant_control_points(
        points,
        final_gap,
        config.filter_min_adjacent_gap,
    )
    final_gaps = tuple(final_gap(left, right) for left, right in zip(filtered_points[:-1], filtered_points[1:]))
    return ElasticBandResult(
        initial_control_points=initial_points,
        control_points=tuple(filtered_points),
        adjacent_gaps=final_gaps,
        evaluated_alphas=tuple(sorted(image_cache)),
        removed_points=tuple(removed),
        iterations=iterations,
        expansions=expansions,
        moves=moves,
        termination_reason=termination_reason,
        history=tuple(history),
        config=asdict(config),
    )
