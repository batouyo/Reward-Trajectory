"""Small dependency-free metrics used by endpoint-comparator bake-offs."""

from __future__ import annotations

import math
from typing import Sequence


SMALL_GRADIENT_STEPS = (1e-5, 3e-5, 1e-4)


def average_ranks(values: Sequence[float]) -> list[float]:
    """Return zero-based average ranks, assigning tied values their mean rank."""

    numbers = [float(value) for value in values]
    if not numbers or not all(math.isfinite(value) for value in numbers):
        raise ValueError("Rank values must be a non-empty finite sequence.")
    order = sorted(range(len(numbers)), key=lambda index: (numbers[index], index))
    ranks = [0.0] * len(numbers)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and numbers[order[end]] == numbers[order[start]]:
            end += 1
        average = 0.5 * (start + end - 1)
        for position in range(start, end):
            ranks[order[position]] = average
        start = end
    return ranks


def spearman_correlation(values: Sequence[float], expected: Sequence[float] | None = None) -> float:
    """Compute Spearman correlation using average ranks for ties."""

    if expected is None:
        expected = list(range(len(values)))
    if len(values) != len(expected) or len(values) < 2:
        raise ValueError("Spearman inputs must have the same length of at least two.")
    left = average_ranks(values)
    right = average_ranks(expected)
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    covariance = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_scale = math.sqrt(sum((x - left_mean) ** 2 for x in left))
    right_scale = math.sqrt(sum((y - right_mean) ** 2 for y in right))
    if left_scale == 0 or right_scale == 0:
        return float("nan")
    return covariance / (left_scale * right_scale)


def ordering_diagnostics(values: Sequence[float]) -> dict[str, object]:
    """Separate strict-order success, descending inversions, and exact ties."""

    numbers = [float(value) for value in values]
    if not numbers or not all(math.isfinite(value) for value in numbers):
        raise ValueError("Ordering values must be a non-empty finite sequence.")
    descending = []
    ties = []
    non_strict = []
    for left in range(len(numbers)):
        for right in range(left + 1, len(numbers)):
            pair = [left, right]
            if numbers[left] > numbers[right]:
                descending.append(pair)
                non_strict.append(pair)
            elif numbers[left] == numbers[right]:
                ties.append(pair)
                non_strict.append(pair)
    adjacent_gaps = [right - left for left, right in zip(numbers, numbers[1:])]
    return {
        "strict_order_pass": all(gap > 0 for gap in adjacent_gaps),
        "adjacent_gaps": adjacent_gaps,
        "descending_inversions": descending,
        "descending_inversion_count": len(descending),
        "ties": ties,
        "tie_count": len(ties),
        "non_strict_violations": non_strict,
        "non_strict_violation_count": len(non_strict),
    }


def gradient_direction_gate(trials: Sequence[dict[str, object]]) -> dict[str, object]:
    """Require at least two of the three preregistered small steps to improve."""

    by_step = {float(trial["step_rms"]): bool(trial["direction_correct"]) for trial in trials}
    missing = [step for step in SMALL_GRADIENT_STEPS if step not in by_step]
    if missing:
        raise ValueError(f"Missing preregistered small gradient steps: {missing}.")
    correct_count = sum(by_step[step] for step in SMALL_GRADIENT_STEPS)
    return {
        "small_steps": list(SMALL_GRADIENT_STEPS),
        "correct_small_step_count": correct_count,
        "required_correct_count": 2,
        "passed": correct_count >= 2,
        "larger_steps_diagnostic_only": [
            trial for trial in trials if float(trial["step_rms"]) not in SMALL_GRADIENT_STEPS
        ],
    }
