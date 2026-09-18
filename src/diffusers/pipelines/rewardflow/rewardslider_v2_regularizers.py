"""Soft V_goal regularizers for RewardSlider V2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class VGoalRegularizerOutput:
    residual: torch.Tensor
    parallel: torch.Tensor
    spatial: torch.Tensor
    diagnostics: tuple[dict[str, torch.Tensor], ...]


def initialize_v_goal_parameters(
    latent_shape: torch.Size | tuple[int, ...],
    *,
    num_branches: int,
    control_steps: int = 4,
    device: torch.device | str | None = None,
) -> nn.ParameterList:
    """Create independent FP32 zero residuals for each controlled timestep."""

    shape = tuple(int(size) for size in latent_shape)
    if len(shape) != 3:
        raise ValueError("`latent_shape` must be [batch, tokens, channels].")
    if shape[0] != 1:
        raise ValueError("The captured native latent must have batch size one.")
    if num_branches < 1 or control_steps < 1 or control_steps > 4:
        raise ValueError("`num_branches` must be positive and `control_steps` must be in [1, 4].")
    parameter_shape = (num_branches, *shape[1:])
    return nn.ParameterList(
        [nn.Parameter(torch.zeros(parameter_shape, device=device, dtype=torch.float32)) for _ in range(control_steps)]
    )


def _expand_direction(direction: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    value = direction.detach().to(device=goal.device, dtype=goal.dtype)
    if value.shape == goal.shape:
        return value
    if value.ndim == goal.ndim and value.shape[0] == 1 and value.shape[1:] == goal.shape[1:]:
        return value.expand_as(goal)
    raise ValueError("Each detached native direction must match V_goal or have one broadcast branch.")


def _expand_relevance(relevance: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    value = relevance.detach().to(device=goal.device, dtype=goal.dtype)
    if value.shape == goal.shape[:2]:
        return value
    if value.ndim == 2 and value.shape[0] == 1 and value.shape[1] == goal.shape[1]:
        return value.expand(goal.shape[0], -1)
    raise ValueError("Each relevance map must have shape [branches, tokens] or [1, tokens].")


def v_goal_regularizers(
    v_goals: Sequence[torch.Tensor],
    native_directions: Sequence[torch.Tensor],
    relevance: Sequence[torch.Tensor],
    *,
    eps: float = 1e-8,
) -> VGoalRegularizerOutput:
    """Return normalized residual, parallel, and soft spatial penalties.

    Native directions and relevance are detached priors.  No hard mask or
    projection is applied to the learnable residuals.
    """

    if not v_goals or not (len(v_goals) == len(native_directions) == len(relevance)):
        raise ValueError("V_goal, native direction, and relevance sequences must be non-empty and aligned.")
    if eps <= 0:
        raise ValueError("`eps` must be positive.")
    residual_values, parallel_values, spatial_values, diagnostics = [], [], [], []
    for goal, direction, relevance_map in zip(v_goals, native_directions, relevance):
        if goal.ndim != 3 or goal.dtype != torch.float32:
            raise ValueError("V_goal must be an FP32 [branches, tokens, channels] tensor.")
        direction = _expand_direction(direction, goal)
        relevance_map = _expand_relevance(relevance_map, goal).clamp(0, 1)
        goal_float = goal.float()
        direction_float = direction.float()
        goal_norm_sq = goal_float.square().sum(dim=(1, 2))
        direction_norm_sq = direction_float.square().sum(dim=(1, 2))
        denominator = direction_norm_sq + eps
        coefficient = (goal_float * direction_float).sum(dim=(1, 2)) / denominator
        projection = coefficient[:, None, None] * direction_float
        projection_norm_sq = projection.square().sum(dim=(1, 2))
        residual_values.append((goal_norm_sq / denominator).mean())
        parallel_values.append((projection_norm_sq / denominator).mean())
        outside = 1 - relevance_map
        low_energy = (goal_float.square() * outside[:, :, None]).sum(dim=(1, 2))
        spatial_values.append(torch.where(goal_norm_sq > eps, low_energy / goal_norm_sq.clamp_min(eps), torch.zeros_like(low_energy)).mean())
        goal_norm = goal_norm_sq.sqrt()
        direction_norm = direction_norm_sq.sqrt()
        cosine = F.cosine_similarity(goal_float.flatten(1), direction_float.flatten(1), dim=1, eps=eps)
        diagnostics.append(
            {
                "goal_norm": goal_norm.mean().detach(),
                "goal_to_direction_norm_ratio": (goal_norm / direction_norm.clamp_min(eps)).mean().detach(),
                "cosine_to_direction": cosine.mean().detach(),
                "projection_ratio": (projection_norm_sq.sqrt() / goal_norm.clamp_min(eps)).mean().detach(),
                "off_region_energy_ratio": (low_energy / goal_norm_sq.clamp_min(eps)).mean().detach(),
            }
        )
    return VGoalRegularizerOutput(
        residual=torch.stack(residual_values).mean(),
        parallel=torch.stack(parallel_values).mean(),
        spatial=torch.stack(spatial_values).mean(),
        diagnostics=tuple(diagnostics),
    )
