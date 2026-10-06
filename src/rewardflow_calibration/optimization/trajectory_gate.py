"""Non-differentiable acceptance checks for backward trajectory candidates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import torch


@dataclass(frozen=True)
class TrajectoryGateConfig:
    semantic_tolerance: float = 0.005
    semantic_floor: float = 0.0
    min_visible_dreamsim: float = 0.01
    sourceward_tolerance: float = 1e-5
    max_second_order_deficit: float = 0.25
    keep_l1_tolerance: float = 0.01
    max_total_residual_ratio: float = 0.10
    max_active_residual_ratio: float = 0.5
    distance_epsilon: float = 1e-8

    def validate(self) -> None:
        if self.semantic_tolerance < 0 or self.sourceward_tolerance < 0:
            raise ValueError("semantic and sourceward tolerances must be non-negative")
        if self.min_visible_dreamsim < 0 or self.max_second_order_deficit < 0:
            raise ValueError("DreamSim thresholds must be non-negative")
        if self.keep_l1_tolerance < 0 or self.max_total_residual_ratio <= 0 or self.max_active_residual_ratio <= 0:
            raise ValueError("keep tolerance must be non-negative and residual cap positive")
        if self.distance_epsilon <= 0:
            raise ValueError("distance_epsilon must be positive")


@dataclass
class TrajectoryState:
    name: str
    image: torch.Tensor
    semantic_score: float
    dreamsim_to_source: float
    keep_l1: float
    residual_global_ratio: float = 0.0
    residual_ratio_per_step: list[float] = field(default_factory=list)
    active_residual_native_ratio: float | None = None
    active_residual_ratio_per_step: list[float | None] = field(default_factory=list)
    line_search_ratio: float | None = None
    cumulative_dreamsim: float = 0.0
    metrics: dict[str, object] = field(default_factory=dict)


@dataclass
class GateDecision:
    accepted: bool
    reasons: list[str]
    metrics: dict[str, object]


def _scalar(value: torch.Tensor | float) -> float:
    return float(torch.as_tensor(value).detach().float().mean().cpu())


def assess_candidate(
    previous: TrajectoryState,
    candidate: TrajectoryState,
    accepted_states: Sequence[TrajectoryState],
    dreamsim_distance: Callable[[torch.Tensor, torch.Tensor], torch.Tensor | float],
    config: TrajectoryGateConfig,
) -> GateDecision:
    """Evaluate semantic, perceptual, continuity, keep, and trust-region gates."""
    config.validate()
    adjacent = _scalar(dreamsim_distance(previous.image, candidate.image))
    semantic_delta = candidate.semantic_score - previous.semantic_score
    source_delta = candidate.dreamsim_to_source - previous.dreamsim_to_source
    keep_delta = candidate.keep_l1 - previous.keep_l1
    second_order: float | None = None
    adjacent_gap_ratio: float | None = None
    if len(accepted_states) >= 2:
        older = accepted_states[-2]
        d_ab = _scalar(dreamsim_distance(older.image, previous.image))
        d_ac = _scalar(dreamsim_distance(older.image, candidate.image))
        second_order = (d_ab + adjacent - d_ac) / max(d_ac, config.distance_epsilon)
        adjacent_gap_ratio = adjacent / max(d_ab, config.distance_epsilon)

    metrics: dict[str, object] = {
        "semantic_score": candidate.semantic_score,
        "semantic_delta_from_previous": semantic_delta,
        "semantic_margin_to_floor": candidate.semantic_score - config.semantic_floor,
        "dreamsim_to_source": candidate.dreamsim_to_source,
        "dreamsim_delta_to_source": source_delta,
        "dreamsim_to_previous": adjacent,
        "second_order_deficit": second_order,
        "adjacent_gap_ratio": adjacent_gap_ratio,
        "keep_l1": candidate.keep_l1,
        "keep_l1_delta": keep_delta,
        "total_residual_global_ratio": candidate.residual_global_ratio,
        "total_residual_ratio_per_step": list(candidate.residual_ratio_per_step),
        "active_total_residual_native_ratio": candidate.active_residual_native_ratio,
        "active_residual_ratio_per_step": list(candidate.active_residual_ratio_per_step),
        "active_increment_rms": candidate.metrics.get("active_increment_rms"),
        "active_native_velocity_rms": candidate.metrics.get("active_native_velocity_rms"),
        "active_increment_native_ratio": candidate.metrics.get("active_increment_native_ratio"),
        "active_total_residual_rms": candidate.metrics.get("active_total_residual_rms"),
        "requested_increment_ratio": candidate.line_search_ratio,
        "actual_increment_global_ratio": candidate.metrics.get("actual_increment_global_ratio"),
        "actual_increment_ratio_per_step": candidate.metrics.get("actual_increment_ratio_per_step"),
    }
    reasons: list[str] = []
    if semantic_delta > config.semantic_tolerance:
        reasons.append("semantic_wrong_direction")
    if candidate.semantic_score < config.semantic_floor:
        reasons.append("semantic_floor_violation")
    if adjacent < config.min_visible_dreamsim:
        reasons.append("perceptual_stall")
    if source_delta >= -config.sourceward_tolerance:
        reasons.append("not_sourceward")
    if second_order is not None and second_order > config.max_second_order_deficit:
        reasons.append("second_order_jump")
    if keep_delta > config.keep_l1_tolerance:
        reasons.append("keep_region_drift")
    if candidate.residual_global_ratio > config.max_total_residual_ratio:
        reasons.append("trust_region_violation")
    if candidate.active_residual_native_ratio is not None and candidate.active_residual_native_ratio > config.max_active_residual_ratio:
        reasons.append("active_trust_region_violation")
    return GateDecision(not reasons, reasons, metrics)


def rejection_counts(decisions: Sequence[GateDecision]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for decision in decisions:
        for reason in decision.reasons:
            counts[reason] = counts.get(reason, 0) + 1
    return counts


def stop_reason_for_rejections(decisions: Sequence[GateDecision]) -> str:
    """Map line-search rejections by reasons shared across every trial."""
    if not decisions:
        return "line_search_exhausted_mixed"
    all_have = lambda reason: all(reason in decision.reasons for decision in decisions)
    if all_have("semantic_floor_violation"):
        return "semantic_boundary_reached"
    for reason, stop in (
        ("trust_region_violation", "trust_region_exhausted"),
        ("active_trust_region_violation", "active_trust_region_exhausted"),
        ("not_sourceward", "not_sourceward"),
        ("perceptual_stall", "perceptual_stall"),
        ("keep_region_drift", "preservation_violation"),
        ("second_order_jump", "trajectory_jump"),
        ("semantic_wrong_direction", "trajectory_jump"),
    ):
        if all_have(reason):
            return stop
    return "line_search_exhausted_mixed"
