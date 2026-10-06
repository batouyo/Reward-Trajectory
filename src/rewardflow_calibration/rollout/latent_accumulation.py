"""Euler latent-state update with opt-in FP32 accumulation."""

from __future__ import annotations

import torch


def euler_latent_update(
    latents: torch.Tensor,
    velocity: torch.Tensor,
    dt: torch.Tensor | float,
    *,
    accumulate_fp32: bool = False,
) -> torch.Tensor:
    """Advance one Euler step, preserving legacy cast-back unless opted in."""
    updated = latents.float() + torch.as_tensor(dt, device=latents.device).float() * velocity.float()
    if accumulate_fp32:
        return updated
    return updated.to(dtype=latents.dtype)
