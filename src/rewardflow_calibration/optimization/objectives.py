"""Loss assembly kept independent from reward-model implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

RewardFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class GoalLossConfig:
    edit_weight: float = 1.0
    preservation_weight: float = 1.0
    regularization_weight: float = 1e-2
    edit_score_tolerance: float = 0.02

    def validate(self) -> None:
        if min(self.edit_weight, self.preservation_weight, self.regularization_weight) < 0:
            raise ValueError("loss weights must be non-negative")
        if self.edit_score_tolerance < 0:
            raise ValueError("edit_score_tolerance must be non-negative")


@dataclass
class GoalLossValues:
    total: torch.Tensor
    edit: torch.Tensor
    preservation: torch.Tensor
    regularization: torch.Tensor
    edit_score: torch.Tensor


def _scalar(value: torch.Tensor, name: str) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} reward must return a torch.Tensor")
    if value.numel() != 1:
        raise ValueError(f"{name} reward must return one scalar for the single-branch run")
    return value.reshape(())


def goal_residual_loss(
    generated: torch.Tensor,
    source: torch.Tensor,
    baseline_edit_score: torch.Tensor,
    edit_reward: RewardFn,
    preservation_loss: RewardFn,
    residual: torch.Tensor,
    config: GoalLossConfig,
    preservation_reference: torch.Tensor | None = None,
) -> GoalLossValues:
    """Keep edit score above the baseline tolerance while preserving invariants.

    ``edit_reward`` is maximized. ``preservation_loss`` is minimized and should
    encode only attributes that ought to stay invariant (for example, a face
    identity embedding or a background/layout feature loss).
    """
    config.validate()
    edit_score = _scalar(edit_reward(generated, source), "edit")
    target_floor = baseline_edit_score.detach().to(edit_score) - config.edit_score_tolerance
    edit_term = torch.relu(target_floor - edit_score).square()
    reference = source if preservation_reference is None else preservation_reference
    preserve_term = _scalar(preservation_loss(generated, reference), "preservation")
    regularization = residual.float().square().mean()
    total = (
        config.edit_weight * edit_term
        + config.preservation_weight * preserve_term
        + config.regularization_weight * regularization
    )
    return GoalLossValues(total, edit_term, preserve_term, regularization, edit_score)
