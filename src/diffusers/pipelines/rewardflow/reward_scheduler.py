"""Deterministic state-dependent reward scheduling for RewardSlider.

The scheduler observes detached semantic trajectory diagnostics and changes
only the effective scalar weights used for the current optimization step. It
never changes the independent high-dimensional controls or any raw loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch


IMAGE_REWARD_NAMES = (
    "semantic_order",
    "semantic_coverage",
    "semantic_pairwise_order",
    "fine_jump",
    "coarse_gap",
    "second",
    "preserve",
)
CONTROL_REWARD_NAMES = ("ctrl", "band", "spatial", "energy")
ALL_REWARD_NAMES = IMAGE_REWARD_NAMES + CONTROL_REWARD_NAMES
VALID_PHASES = ("semantic_recovery", "transition", "trajectory_refinement")


@dataclass(frozen=True)
class TrajectoryRewardSchedulerConfig:
    """Fixed, explicit state-transition parameters."""

    order_tol: float = 0.0
    scheduler_collapse_tol: float = 0.005
    semantic_patience: int = 2
    transition_iters: int = 2
    recovery_preserve_weight: float = 0.05

    def validate(self) -> None:
        if self.order_tol < 0:
            raise ValueError("order_tol must be non-negative.")
        if self.scheduler_collapse_tol < 0:
            raise ValueError("scheduler_collapse_tol must be non-negative.")
        if self.semantic_patience < 1:
            raise ValueError("semantic_patience must be positive.")
        if self.transition_iters < 1:
            raise ValueError("transition_iters must be positive.")
        if self.recovery_preserve_weight < 0:
            raise ValueError("recovery_preserve_weight must be non-negative.")


@dataclass(frozen=True)
class TrajectoryRewardSchedule:
    """One deterministic scheduling decision and its complete audit state."""

    phase: str
    effective_weights: dict[str, float]
    semantic_order_ok: bool
    coarse_not_collapsed: bool
    near_zero_coarse_intervals: int
    consecutive_semantic_ok: int
    min_adjacent_semantic_gap: float
    min_coarse_semantic_gap: float
    max_coarse_semantic_gap: float
    transition_iteration: int

    def as_log_dict(self) -> dict[str, object]:
        return {
            "scheduler_phase": self.phase,
            "effective_weights": self.effective_weights,
            "semantic_order_ok": self.semantic_order_ok,
            "collapse_ok": self.coarse_not_collapsed,
            "near_zero_coarse_intervals": self.near_zero_coarse_intervals,
            "consecutive_semantic_ok": self.consecutive_semantic_ok,
            "min_adjacent_semantic_gap": self.min_adjacent_semantic_gap,
            "min_coarse_semantic_gap": self.min_coarse_semantic_gap,
            "max_coarse_semantic_gap": self.max_coarse_semantic_gap,
            "transition_iteration": self.transition_iteration,
        }


class TrajectoryRewardScheduler:
    """Three-phase, reversible scheduling of RewardSlider reward weights."""

    def __init__(self, config: TrajectoryRewardSchedulerConfig = TrajectoryRewardSchedulerConfig()):
        config.validate()
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.phase = "semantic_recovery"
        self.consecutive_semantic_ok = 0
        self.transition_iteration = 0

    @staticmethod
    def _scalar_min(value: torch.Tensor, name: str) -> float:
        if value.ndim != 1 or value.numel() < 1:
            raise ValueError(f"{name} must be a non-empty one-dimensional tensor.")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must be finite.")
        return float(value.detach().amin())

    @staticmethod
    def _scalar_max(value: torch.Tensor, name: str) -> float:
        if value.ndim != 1 or value.numel() < 1:
            raise ValueError(f"{name} must be a non-empty one-dimensional tensor.")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must be finite.")
        return float(value.detach().amax())

    def _semantic_state(
        self, adjacent_semantic_gaps: torch.Tensor, coarse_semantic_gaps: torch.Tensor
    ) -> tuple[bool, bool, int, float, float, float]:
        min_adjacent = self._scalar_min(adjacent_semantic_gaps, "adjacent_semantic_gaps")
        min_coarse = self._scalar_min(coarse_semantic_gaps, "coarse_semantic_gaps")
        max_coarse = self._scalar_max(coarse_semantic_gaps, "coarse_semantic_gaps")
        order_ok = min_adjacent >= -self.config.order_tol
        # CLIP endpoint-axis values provide ordering, not calibrated distance.
        # A large Source->Weak gap is healthy; only nearly identical anchors
        # prevent progression out of semantic recovery.
        near_zero = int((coarse_semantic_gaps.detach().abs() <= self.config.scheduler_collapse_tol).sum())
        coarse_not_collapsed = min_coarse >= -self.config.order_tol and near_zero == 0
        return order_ok, coarse_not_collapsed, near_zero, min_adjacent, min_coarse, max_coarse

    @staticmethod
    def _base_weights(base_weights: Mapping[str, float]) -> dict[str, float]:
        missing = [name for name in ALL_REWARD_NAMES if name not in base_weights]
        if missing:
            raise ValueError(f"Base weights are missing: {', '.join(missing)}.")
        return {name: float(base_weights[name]) for name in ALL_REWARD_NAMES}

    def _effective_weights(self, base: dict[str, float]) -> dict[str, float]:
        effective = dict(base)
        if self.phase == "semantic_recovery":
            for name in ("fine_jump", "coarse_gap", "second"):
                effective[name] = 0.0
            effective["preserve"] = min(base["preserve"], self.config.recovery_preserve_weight)
            return effective
        if self.phase == "transition":
            fraction = self.transition_iteration / self.config.transition_iters
            for name in ("fine_jump", "coarse_gap", "second"):
                effective[name] = fraction * base[name]
            recovery = min(base["preserve"], self.config.recovery_preserve_weight)
            effective["preserve"] = recovery + fraction * (base["preserve"] - recovery)
            return effective
        if self.phase == "trajectory_refinement":
            return effective
        raise RuntimeError(f"Unknown scheduler phase: {self.phase}.")

    def step(
        self,
        *,
        base_weights: Mapping[str, float],
        adjacent_semantic_gaps: torch.Tensor,
        coarse_semantic_gaps: torch.Tensor,
    ) -> TrajectoryRewardSchedule:
        """Advance state from detached semantic diagnostics and return this step's weights."""

        base = self._base_weights(base_weights)
        order_ok, coarse_not_collapsed, near_zero, min_adjacent, min_coarse, max_coarse = self._semantic_state(
            adjacent_semantic_gaps, coarse_semantic_gaps
        )
        healthy = order_ok and coarse_not_collapsed
        if not healthy:
            self.phase = "semantic_recovery"
            self.consecutive_semantic_ok = 0
            self.transition_iteration = 0
        elif self.phase == "semantic_recovery":
            self.consecutive_semantic_ok += 1
            if self.consecutive_semantic_ok >= self.config.semantic_patience:
                self.phase = "transition"
                self.transition_iteration = 1
        elif self.phase == "transition":
            self.consecutive_semantic_ok += 1
            if self.transition_iteration >= self.config.transition_iters:
                self.phase = "trajectory_refinement"
            else:
                self.transition_iteration += 1
        elif self.phase == "trajectory_refinement":
            self.consecutive_semantic_ok += 1
        else:
            raise RuntimeError(f"Unknown scheduler phase: {self.phase}.")
        return TrajectoryRewardSchedule(
            phase=self.phase,
            effective_weights=self._effective_weights(base),
            semantic_order_ok=order_ok,
            coarse_not_collapsed=coarse_not_collapsed,
            near_zero_coarse_intervals=near_zero,
            consecutive_semantic_ok=self.consecutive_semantic_ok,
            min_adjacent_semantic_gap=min_adjacent,
            min_coarse_semantic_gap=min_coarse,
            max_coarse_semantic_gap=max_coarse,
            transition_iteration=self.transition_iteration,
        )
