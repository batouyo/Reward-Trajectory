"""Strictly ordered, learnable RewardSlider V2 strength coordinates."""

from __future__ import annotations

import torch
from torch import nn


class OrderedAlphaParameterization(nn.Module):
    """Represent endpoint-inclusive alphas with softmax interval logits."""

    def __init__(self, interval_logits: torch.Tensor):
        super().__init__()
        if interval_logits.ndim != 1 or interval_logits.numel() < 2:
            raise ValueError("At least one interior alpha requires two interval logits.")
        if not torch.isfinite(interval_logits).all():
            raise ValueError("Interval logits must be finite.")
        self.interval_logits = nn.Parameter(interval_logits.detach().float().clone())

    @classmethod
    def random(
        cls,
        *,
        num_interior: int = 3,
        seed: int | None = None,
        device: torch.device | str | None = None,
    ) -> "OrderedAlphaParameterization":
        if num_interior < 1:
            raise ValueError("`num_interior` must be positive.")
        generator = torch.Generator(device="cpu")
        if seed is not None:
            generator.manual_seed(seed)
        # Sorting an initialization sample is allowed; optimization acts only
        # on interval logits thereafter and never sorts the learnable values.
        samples = torch.rand(num_interior, generator=generator, dtype=torch.float64)
        samples, _ = samples.sort()
        gaps = torch.cat((samples[:1], samples[1:] - samples[:-1], 1 - samples[-1:]))
        logits = gaps.log().to(device=device, dtype=torch.float32)
        return cls(logits)

    @classmethod
    def from_alphas(cls, alphas: torch.Tensor, *, eps: float = 1e-7) -> "OrderedAlphaParameterization":
        if alphas.ndim != 1 or alphas.numel() < 3:
            raise ValueError("Explicit alphas must include endpoints and at least one interior node.")
        if eps <= 0:
            raise ValueError("`eps` must be positive.")
        values = alphas.detach().double()
        if not torch.isfinite(values).all() or values[0] != 0 or values[-1] != 1:
            raise ValueError("Explicit alphas must have endpoints exactly 0 and 1.")
        gaps = values[1:] - values[:-1]
        if torch.any(gaps <= 0):
            raise ValueError("Explicit interior alphas must be strictly increasing.")
        return cls(gaps.clamp_min(eps).log().to(dtype=torch.float32))

    @property
    def alphas(self) -> torch.Tensor:
        # Compute the simplex in FP64 so finite extreme logits do not collapse
        # adjacent nodes after FP32 cumulative-sum rounding.
        gaps = torch.softmax(self.interval_logits.double(), dim=0)
        interior = gaps.cumsum(0)[:-1].to(dtype=self.interval_logits.dtype)
        zero = self.interval_logits.new_zeros(1)
        one = self.interval_logits.new_ones(1)
        return torch.cat((zero, interior, one))

    @property
    def num_interior(self) -> int:
        return self.interval_logits.numel() - 1

    def extra_repr(self) -> str:
        return f"num_interior={self.num_interior}"
