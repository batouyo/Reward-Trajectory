"""RewardFlow-style dynamic weighting for explicit V2 deficits."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .rewards import RewardGuidance


@dataclass(frozen=True)
class DynamicDeficitOutput:
    total_loss: torch.Tensor
    raw_deficits: torch.Tensor
    normalized_deficits: torch.Tensor
    weights: torch.Tensor
    direction: str = "higher_deficit_is_worse"
    mode: str = "dynamic"


class DynamicDeficitCoordinator:
    """Coordinate scalar losses without applying latent updates.

    This reuses ``RewardGuidance``'s EMA/variance/clipping/softmax machinery,
    but treats every input as a non-negative deficit where higher is worse.
    """

    def __init__(
        self,
        *,
        temperature: float = 1.0,
        ema_decay: float = 0.99,
        clip_value: float = 5.0,
        weight_floor: float = 0.0,
        warmup_steps: int = 1,
        dynamic: bool = True,
        eps: float = 1e-6,
    ):
        self._guidance = RewardGuidance(
            [lambda image, prompt: image],
            temperature=temperature,
            ema_decay=ema_decay,
            clip_value=clip_value,
            weight_floor=weight_floor,
            eps=eps,
        )
        if warmup_steps < 0:
            raise ValueError("`warmup_steps` must be non-negative.")
        self.warmup_steps = int(warmup_steps)
        self.dynamic = bool(dynamic)
        self._steps = 0

    @property
    def steps(self) -> int:
        return self._steps

    @property
    def ema_mean(self) -> torch.Tensor | None:
        return self._guidance._ema_mean

    def coordinate(self, deficits: torch.Tensor) -> DynamicDeficitOutput:
        if deficits.ndim != 1 or deficits.numel() < 1:
            raise ValueError("Deficits must be a non-empty one-dimensional tensor.")
        self._steps += 1
        if not torch.isfinite(deficits).all() or torch.any(deficits < 0):
            raise ValueError("Deficits must be finite and non-negative.")
        raw = deficits.float()
        self._guidance._update_ema(raw)
        if not self.dynamic or self._steps <= self.warmup_steps:
            weights = torch.full_like(raw, 1.0 / raw.numel())
            return DynamicDeficitOutput(
                total_loss=(weights * raw).sum(), raw_deficits=raw,
                normalized_deficits=torch.zeros_like(raw), weights=weights,
                mode="warmup" if self._steps <= self.warmup_steps else "uniform",
            )
        mean = self._guidance._ema_mean.to(raw)
        variance = self._guidance._ema_var.to(raw)
        normalized = torch.where(
            variance <= self._guidance.eps,
            raw - raw.mean(),
            (raw - mean) / torch.sqrt(variance + self._guidance.eps),
        )
        if self._guidance.clip_value > 0:
            normalized = torch.clamp(normalized, -self._guidance.clip_value, self._guidance.clip_value)
        weights = torch.softmax(normalized / max(self._guidance.temperature, 1e-6), dim=0)
        if self._guidance.weight_floor > 0:
            weights = weights + self._guidance.weight_floor
            weights = weights / weights.sum()
        return DynamicDeficitOutput(
            total_loss=(weights * raw).sum(),
            raw_deficits=raw,
            normalized_deficits=normalized,
            weights=weights,
        )
