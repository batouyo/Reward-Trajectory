"""LPIPS adjacent-gap and KL-to-uniform trajectory metrics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def _normalize(distances: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    total = distances.sum()
    uniform = torch.full_like(distances, 1.0 / distances.numel())
    return torch.where(total > eps, distances / total.clamp_min(eps), uniform)


def kl_to_uniform(distances: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    probabilities = _normalize(distances, eps)
    uniform = torch.full_like(probabilities, 1.0 / probabilities.numel())
    positive = probabilities > 0
    return torch.where(
        positive,
        probabilities * (probabilities.clamp_min(eps).log() - uniform.log()),
        torch.zeros_like(probabilities),
    ).sum()


@dataclass(frozen=True)
class LPIPSTrajectoryStats:
    distances: torch.Tensor
    normalized_distances: torch.Tensor
    kl_uniform: torch.Tensor
    path_length: torch.Tensor
    endpoint_distance: torch.Tensor


class LPIPSDistance(nn.Module):
    def __init__(self, net: str = "vgg"):
        super().__init__()
        try:
            import lpips
        except ImportError as exc:
            raise ImportError("LPIPS is required for trajectory evaluation") from exc
        self.model = lpips.LPIPS(net=net)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def distance(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        if first.ndim != 4 or second.shape != first.shape:
            raise ValueError("LPIPS inputs must have matching [B, 3, H, W] shapes")
        return self.model(first.float(), second.float()).reshape(first.shape[0], -1).mean(dim=1)

    def trajectory(self, images: torch.Tensor) -> LPIPSTrajectoryStats:
        distances = self.distance(images[:-1], images[1:])
        endpoint = self.distance(images[:1], images[-1:]).reshape(()).detach()
        return LPIPSTrajectoryStats(
            distances=distances,
            normalized_distances=_normalize(distances),
            kl_uniform=kl_to_uniform(distances),
            path_length=distances.sum(),
            endpoint_distance=endpoint,
        )
