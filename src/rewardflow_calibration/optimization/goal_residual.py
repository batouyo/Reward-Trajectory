"""Learnable, sample-specific velocity residuals for early rollout steps."""

from __future__ import annotations

import torch
from torch import nn


class GoalVelocityResidual(nn.Module):
    """Directly optimize one zero-initialized residual tensor per selected step."""

    def __init__(
        self,
        *,
        steps: int,
        latent_shape: tuple[int, int],
        device: torch.device | str,
    ) -> None:
        super().__init__()
        if steps < 1:
            raise ValueError("steps must be positive")
        if len(latent_shape) != 2 or min(latent_shape) < 1:
            raise ValueError("latent_shape must be (latent_tokens, latent_channels)")
        self.velocity = nn.Parameter(
            torch.zeros((steps, *latent_shape), device=device, dtype=torch.float32)
        )

    @property
    def steps(self) -> int:
        return self.velocity.shape[0]

    def regularization(self) -> torch.Tensor:
        """Mean squared residual magnitude; scale-independent of tensor size."""
        return self.velocity.float().square().mean()
