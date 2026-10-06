"""Target-level progress objective independent of legacy goal losses."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ProgressLossConfig:
    progress_weight: float = 1.0
    drift_weight: float = 0.1
    regularization_weight: float = 0.01

    def validate(self) -> None:
        if min(self.progress_weight, self.drift_weight, self.regularization_weight) < 0:
            raise ValueError("progress loss weights must be non-negative")


@dataclass
class ProgressLossValues:
    total: torch.Tensor
    progress: torch.Tensor
    drift: torch.Tensor
    regularization: torch.Tensor
    target_error: torch.Tensor


def progress_control_loss(
    raw_progress: torch.Tensor,
    target_strength: float | torch.Tensor,
    drift: torch.Tensor,
    residual: torch.Tensor,
    config: ProgressLossConfig | None = None,
) -> ProgressLossValues:
    """Minimize squared distance to requested progress; do not maximize progress."""
    config = config or ProgressLossConfig()
    config.validate()
    progress = raw_progress.float().mean()
    target = torch.as_tensor(target_strength, device=progress.device, dtype=progress.dtype)
    if target.numel() != 1 or not torch.isfinite(target) or not 0.0 <= float(target) <= 1.0:
        raise ValueError("target_strength must be a finite scalar in [0, 1]")
    target_error = progress - target.reshape(())
    progress_term = target_error.square()
    drift_term = drift.float().mean()
    regularization = residual.float().square().mean()
    total = (
        config.progress_weight * progress_term
        + config.drift_weight * drift_term
        + config.regularization_weight * regularization
    )
    return ProgressLossValues(total, progress_term, drift_term, regularization, target_error)
