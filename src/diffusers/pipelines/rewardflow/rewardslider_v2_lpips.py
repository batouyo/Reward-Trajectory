"""Tensor-native LPIPS trajectory smoothness for RewardSlider V2.

Kontinuous Kontext's ``kl-filter-simple`` normalizes adjacent LPIPS distances
and computes KL divergence to a uniform interval distribution.  This module
keeps that criterion differentiable and deliberately does not use CLIP,
SigLIP, or DreamSim for the formal trajectory objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import nn


class TensorImageDistance(Protocol):
    def distance(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor: ...


def lpips_uniform_kl(distances: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """Return ``KL(p || uniform)`` for a one-dimensional distance vector."""

    if distances.ndim != 1 or distances.numel() < 1:
        raise ValueError("LPIPS distances must be a non-empty one-dimensional tensor.")
    if eps <= 0:
        raise ValueError("`eps` must be positive.")
    if not torch.isfinite(distances).all():
        raise ValueError("LPIPS distances must be finite.")
    if torch.any(distances < 0):
        raise ValueError("LPIPS distances must be non-negative.")
    total = distances.sum()
    probabilities = distances / total.clamp_min(eps)
    uniform = torch.full_like(probabilities, 1.0 / probabilities.numel())
    positive = probabilities > 0
    terms = torch.where(
        positive,
        probabilities * (probabilities.clamp_min(eps).log() - uniform.log()),
        torch.zeros_like(probabilities),
    )
    return terms.sum()


@dataclass(frozen=True)
class LPIPSTrajectoryStats:
    distances: torch.Tensor
    normalized_distances: torch.Tensor
    kl_uniform: torch.Tensor
    max_normalized_gap: torch.Tensor
    worst_interval: torch.Tensor


def lpips_trajectory_stats(distances: torch.Tensor, *, eps: float = 1e-8) -> LPIPSTrajectoryStats:
    """Compute formal KL smoothness and auditable gap diagnostics."""

    if distances.ndim != 1 or distances.numel() < 1:
        raise ValueError("LPIPS distances must be a non-empty one-dimensional tensor.")
    total = distances.sum()
    normalized = distances / total.clamp_min(eps)
    return LPIPSTrajectoryStats(
        distances=distances,
        normalized_distances=normalized,
        kl_uniform=lpips_uniform_kl(distances, eps=eps),
        max_normalized_gap=normalized.max(),
        worst_interval=normalized.argmax(),
    )


class LPIPSDistance(nn.Module):
    """Official ``lpips`` PyTorch model behind a tensor-only distance seam.

    Inputs must be ``[B,3,H,W]`` tensors in LPIPS's native ``[-1, 1]`` range.
    A model can be injected for tests or deployments with a preloaded model;
    no PIL, NumPy, or detach operation is used in ``distance``.
    """

    def __init__(self, *, net: str = "vgg", model: nn.Module | None = None):
        super().__init__()
        if model is None:
            try:
                import lpips
            except ImportError as exc:  # pragma: no cover - environment-dependent
                raise RuntimeError("LPIPS is unavailable; install the optional `lpips` dependency.") from exc
            model = lpips.LPIPS(net=net)
        self.model = model
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def distance(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        if first.ndim != 4 or second.shape != first.shape:
            raise ValueError("LPIPS inputs must have matching [B,3,H,W] shapes.")
        value = self.model(first, second)
        if not torch.is_tensor(value):
            raise TypeError("The LPIPS model must return a tensor.")
        return value.reshape(value.shape[0], -1).mean(dim=1)

    def trajectory(self, images: torch.Tensor, *, eps: float = 1e-8) -> LPIPSTrajectoryStats:
        if images.ndim != 4 or images.shape[0] < 2:
            raise ValueError("A trajectory needs at least two [B,3,H,W] image nodes.")
        distances = self.distance(images[:-1], images[1:])
        return lpips_trajectory_stats(distances, eps=eps)
