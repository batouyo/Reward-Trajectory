"""Pluggable differentiable quality reward and input-gradient audit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn


class DifferentiableQualityReward(nn.Module):
    """Wrap a tensor-native quality scorer with a scalar reward contract."""

    def __init__(self, scorer: nn.Module | Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()
        if isinstance(scorer, nn.Module):
            self.scorer = scorer
        else:
            self.scorer = scorer
        if isinstance(self.scorer, nn.Module):
            self.scorer.eval()
            for parameter in self.scorer.parameters():
                parameter.requires_grad_(False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError("Quality reward input must have shape [batch, channels, height, width].")
        value = self.scorer(image)
        if not torch.is_tensor(value):
            raise TypeError("Quality scorer must return a tensor.")
        if value.numel() < 1:
            raise ValueError("Quality scorer returned an empty tensor.")
        value = value.float()
        return value if value.ndim == 0 else value.reshape(-1).mean()


@dataclass(frozen=True)
class QualityGradientAudit:
    passed: bool
    reward_requires_grad: bool
    image_gradient_norm: float
    finite: bool
    reason: str | None = None


def audit_quality_reward(
    reward: DifferentiableQualityReward,
    image: torch.Tensor,
    *,
    require_pass: bool = False,
    eps: float = 1e-12,
) -> QualityGradientAudit:
    """Verify that reward-to-image autograd is finite and non-zero."""

    if eps <= 0:
        raise ValueError("`eps` must be positive.")
    candidate = image.detach().clone().requires_grad_(True)
    value = reward(candidate)
    reward_requires_grad = bool(value.requires_grad)
    gradient = None
    if reward_requires_grad:
        gradient = torch.autograd.grad(value, candidate, allow_unused=True, retain_graph=False)[0]
    finite = bool(gradient is not None and torch.isfinite(gradient).all().item())
    norm = float(gradient.detach().float().norm().item()) if gradient is not None else 0.0
    passed = reward_requires_grad and finite and norm > eps
    reason = None if passed else "gradient audit failed: reward must retain a finite non-zero image gradient"
    result = QualityGradientAudit(passed, reward_requires_grad, norm, finite, reason)
    if require_pass and not result.passed:
        raise RuntimeError(result.reason)
    return result
