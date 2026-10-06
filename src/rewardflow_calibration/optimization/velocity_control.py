"""Small tensor-only update helpers for progress control and capacity sweeps."""

from __future__ import annotations

import math

import torch


def normalized_feedback_update(
    progress_gradient: torch.Tensor,
    signed_error: float | torch.Tensor,
    step_size: float,
    *,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return ``-eta * (p-s) * grad(p) / RMS(grad(p))`` and gradient stats."""
    if step_size < 0 or eps <= 0:
        raise ValueError("step_size must be non-negative and eps must be positive")
    gradient = progress_gradient.detach().float()
    if gradient.numel() == 0 or not torch.isfinite(gradient).all():
        raise FloatingPointError("progress gradient is empty or non-finite")
    grad_rms_tensor = gradient.square().mean().sqrt()
    grad_rms = float(grad_rms_tensor.cpu())
    if grad_rms <= eps:
        raise FloatingPointError("progress gradient RMS is zero or too small for feedback update")
    error = torch.as_tensor(signed_error, device=gradient.device, dtype=torch.float32)
    if error.numel() != 1 or not torch.isfinite(error):
        raise ValueError("signed_error must be one finite scalar")
    update = -float(step_size) * error.reshape(()) * gradient / (grad_rms_tensor + eps)
    if not torch.isfinite(update).all():
        raise FloatingPointError("feedback update is non-finite")
    return update, {
        "progress_gradient_rms": grad_rms,
        "progress_gradient_norm": float(gradient.norm().cpu()),
    }


def scale_negative_gradient_to_native_ratio(
    negative_gradient: torch.Tensor,
    native_velocity_rms_per_step: torch.Tensor | list[float] | tuple[float, ...],
    requested_ratio: float,
    *,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale each step independently to a requested residual/native RMS ratio."""
    if negative_gradient.ndim != 3:
        raise ValueError("negative_gradient must have shape [steps, tokens, channels]")
    if not math.isfinite(requested_ratio) or requested_ratio < 0:
        raise ValueError("requested_ratio must be finite and non-negative")
    if eps <= 0:
        raise ValueError("eps must be positive")
    direction = negative_gradient.detach().float()
    native_rms = torch.as_tensor(
        native_velocity_rms_per_step, device=direction.device, dtype=torch.float32
    ).flatten()
    if native_rms.numel() != direction.shape[0]:
        raise ValueError("native_velocity_rms_per_step must have one value per residual step")
    if not torch.isfinite(direction).all() or not torch.isfinite(native_rms).all():
        raise FloatingPointError("gradient direction and native RMS must be finite")
    if torch.any(native_rms <= eps):
        raise FloatingPointError("native velocity RMS is zero or too small for ratio scaling")
    direction_rms = direction.square().mean(dim=(1, 2)).sqrt()
    if torch.any(direction_rms <= eps):
        raise FloatingPointError("negative-gradient direction has a zero or near-zero step")
    residual = (
        float(requested_ratio)
        * native_rms[:, None, None]
        * direction
        / direction_rms[:, None, None]
    )
    actual_ratio = residual.square().mean(dim=(1, 2)).sqrt() / native_rms
    return residual, actual_ratio
