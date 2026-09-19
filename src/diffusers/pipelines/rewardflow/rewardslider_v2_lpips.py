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


def _normalize_distances(distances: torch.Tensor, eps: float) -> torch.Tensor:
    total = distances.sum()
    uniform = torch.full_like(distances, 1.0 / distances.numel())
    return torch.where(total > eps, distances / total.clamp_min(eps), uniform)

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
    probabilities = _normalize_distances(distances, eps)
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
    path_length: torch.Tensor
    endpoint_distance: torch.Tensor
    collapsed: bool


def lpips_trajectory_stats(distances: torch.Tensor, *, endpoint_distance: torch.Tensor | None = None, eps: float = 1e-8) -> LPIPSTrajectoryStats:
    """Compute formal KL smoothness and auditable gap diagnostics."""

    if distances.ndim != 1 or distances.numel() < 1:
        raise ValueError("LPIPS distances must be a non-empty one-dimensional tensor.")
    total = distances.sum()
    normalized = _normalize_distances(distances, eps)
    path_length = total
    if endpoint_distance is None:
        endpoint_distance = torch.full_like(path_length, float("nan"))
    endpoint_distance = endpoint_distance.reshape(())
    collapsed = bool(path_length.detach() <= eps) or (
        bool(torch.isfinite(endpoint_distance).item()) and bool(endpoint_distance.detach() <= eps)
    )
    return LPIPSTrajectoryStats(
        distances=distances,
        normalized_distances=normalized,
        kl_uniform=lpips_uniform_kl(distances, eps=eps),
        max_normalized_gap=normalized.max(),
        worst_interval=normalized.argmax(),
        path_length=path_length,
        endpoint_distance=endpoint_distance,
        collapsed=collapsed,
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
        # LPIPS checkpoints are normally stored in FP32. Keep the adapter
        # tensor-only and differentiable while avoiding BF16/FP32 convolution
        # mismatches in real FLUX runs.
        value = self.model(first.float(), second.float())
        if not torch.is_tensor(value):
            raise TypeError("The LPIPS model must return a tensor.")
        return value.reshape(value.shape[0], -1).mean(dim=1)

    def trajectory(self, images: torch.Tensor, *, eps: float = 1e-8) -> LPIPSTrajectoryStats:
        if images.ndim != 4 or images.shape[0] < 2:
            raise ValueError("A trajectory needs at least two [B,3,H,W] image nodes.")
        distances = self.distance(images[:-1], images[1:])
        endpoint_distance = self.distance(images[:1], images[-1:]).reshape(-1).mean()
        return lpips_trajectory_stats(distances, endpoint_distance=endpoint_distance, eps=eps)
