"""Explicit, plateau-gated RewardSlider V2 topology changes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from torch import nn

from .rewardslider_v2_lpips import lpips_uniform_kl


@dataclass(frozen=True)
class TopologyEvent:
    operation: str
    old_nodes: int
    new_nodes: int
    affected_interval: int | None
    alpha_before: tuple[float, ...]
    alpha_after: tuple[float, ...]
    reason: str
    kl_before: float | None = None
    kl_after: float | None = None


def _validate_alphas(alphas: torch.Tensor) -> None:
    if alphas.ndim != 1 or alphas.numel() < 3 or not torch.isfinite(alphas).all():
        raise ValueError("Alphas must contain endpoints and at least one finite interior node.")
    if alphas[0] != 0 or alphas[-1] != 1 or torch.any(alphas[1:] <= alphas[:-1]):
        raise ValueError("Alphas must be strictly increasing with endpoints 0 and 1.")


def _validate_goals(goals: Sequence[torch.Tensor], interior: int) -> None:
    if len(goals) < 1 or any(goal.ndim != 3 or goal.shape[0] != interior for goal in goals):
        raise ValueError("Each V_goal tensor must have the current interior branch dimension.")


class TopologyManager:
    def __init__(self, *, max_nodes: int = 10, min_nodes: int = 3):
        if min_nodes < 3 or max_nodes < min_nodes:
            raise ValueError("Topology bounds must satisfy 3 <= min_nodes <= max_nodes.")
        self.max_nodes = int(max_nodes)
        self.min_nodes = int(min_nodes)

    @staticmethod
    def trajectory_kl(images: torch.Tensor, distance: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> torch.Tensor:
        if images.ndim < 2 or images.shape[0] < 2:
            raise ValueError("Trajectory images must contain at least two nodes.")
        distances = []
        for left, right in zip(images[:-1], images[1:]):
            value = distance(left.unsqueeze(0), right.unsqueeze(0))
            distances.append(value.reshape(-1).mean())
        return lpips_uniform_kl(torch.stack(distances).float())

    def virtual_removal_kl(
        self,
        images: torch.Tensor,
        index: int,
        distance: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        if not 0 < index < images.shape[0] - 1:
            raise ValueError("Only interior nodes can be virtually removed.")
        remaining = torch.cat((images[:index], images[index + 1 :]), dim=0)
        return self.trajectory_kl(remaining, distance)

    def insert_node(
        self,
        alphas: torch.Tensor,
        v_goals: Sequence[torch.Tensor],
        normalized_distances: torch.Tensor,
        *,
        reason: str = "worst normalized LPIPS interval",
    ) -> tuple[torch.Tensor, tuple[nn.Parameter, ...], TopologyEvent]:
        _validate_alphas(alphas)
        _validate_goals(v_goals, alphas.numel() - 2)
        if alphas.numel() >= self.max_nodes:
            raise ValueError("Maximum topology node count reached.")
        if normalized_distances.shape != (alphas.numel() - 1,) or not torch.isfinite(normalized_distances).all():
            raise ValueError("Normalized adjacent distances must match the alpha interval count.")
        interval = int(normalized_distances.argmax().item())
        inserted = (alphas[interval] + alphas[interval + 1]) / 2
        updated_alphas = torch.cat((alphas[: interval + 1], inserted.reshape(1), alphas[interval + 1 :])).detach()
        updated_goals = []
        branch_index = interval if interval < alphas.numel() - 2 else alphas.numel() - 2
        for goal in v_goals:
            zero = torch.zeros_like(goal[:1])
            updated_goals.append(nn.Parameter(torch.cat((goal[:branch_index], zero, goal[branch_index:]), dim=0).detach().clone()))
        event = TopologyEvent(
            operation="insert", old_nodes=alphas.numel(), new_nodes=updated_alphas.numel(),
            affected_interval=interval, alpha_before=tuple(alphas.tolist()), alpha_after=tuple(updated_alphas.tolist()), reason=reason,
        )
        return updated_alphas, tuple(updated_goals), event

    def prune_node(
        self,
        alphas: torch.Tensor,
        v_goals: Sequence[torch.Tensor],
        index: int,
        *,
        images: torch.Tensor | None = None,
        distance: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        threshold: float = 0.15,
        tolerance: float = 0.0,
        reason: str = "virtual removal satisfies KL threshold and tolerance",
    ) -> tuple[torch.Tensor, tuple[nn.Parameter, ...], TopologyEvent]:
        _validate_alphas(alphas)
        _validate_goals(v_goals, alphas.numel() - 2)
        if not 0 < index < alphas.numel() - 1:
            raise ValueError("Only interior nodes can be pruned.")
        if alphas.numel() <= self.min_nodes:
            raise ValueError("Minimum topology node count reached.")
        if images is None or distance is None:
            raise ValueError("Virtual pruning requires trajectory images and a differentiable distance.")
        if threshold < 0 or tolerance < 0:
            raise ValueError("Threshold and tolerance must be non-negative.")
        before = self.trajectory_kl(images, distance)
        after = self.virtual_removal_kl(images, index, distance)
        if after > threshold or after > before + tolerance:
            raise ValueError("Virtual removal is not safe under the configured KL criteria.")
        updated_alphas = torch.cat((alphas[:index], alphas[index + 1 :])).detach()
        updated_goals = tuple(nn.Parameter(torch.cat((goal[: index - 1], goal[index:]), dim=0).detach().clone()) for goal in v_goals)
        event = TopologyEvent(
            operation="prune", old_nodes=alphas.numel(), new_nodes=updated_alphas.numel(),
            affected_interval=index - 1, alpha_before=tuple(alphas.tolist()), alpha_after=tuple(updated_alphas.tolist()), reason=reason,
            kl_before=float(before.item()), kl_after=float(after.item()),
        )
        return updated_alphas, updated_goals, event


def rebuild_topology_optimizer(
    alpha_parameters: Sequence[nn.Parameter],
    v_goals: Sequence[nn.Parameter],
    *,
    alpha_lr: float = 1e-3,
    vgoal_lr: float = 1e-3,
) -> torch.optim.Optimizer:
    """Rebuild Adam from current parameters, dropping stale topology state."""

    if alpha_lr <= 0 or vgoal_lr <= 0:
        raise ValueError("Learning rates must be positive.")
    alpha_parameters = tuple(alpha_parameters)
    v_goals = tuple(v_goals)
    if not alpha_parameters or not v_goals:
        raise ValueError("Both alpha and V_goal parameters are required.")
    return torch.optim.Adam(
        [
            {"params": list(alpha_parameters), "lr": alpha_lr},
            {"params": list(v_goals), "lr": vgoal_lr},
        ]
    )
