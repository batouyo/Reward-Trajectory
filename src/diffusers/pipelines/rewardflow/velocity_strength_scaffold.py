"""V2 current-state velocity-strength scaffold.

The keep-velocity formula follows the mathematical behavior of VeloEdit's
``compute_reference_velocity``: ``(z_t - z_0) / (sigma + eps)``.  The local
VeloEdit checkout has no visible license file, so this module reimplements the
formula rather than copying source code.

Unlike RewardSlider V1's ``native + arbitrary control`` path, V2 explicitly
constructs ``(1-alpha) * V_keep + alpha * V_edit + V_goal`` for the first four
sampling steps.  At and after the boundary it returns the native edit velocity
without touching the scaffold inputs.
"""

from __future__ import annotations

from typing import Optional

import torch


def compute_keep_velocity(
    current_latent: torch.Tensor,
    source_clean_latent: torch.Tensor,
    sigma: torch.Tensor | float,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute the source-restoring velocity from the current branch state."""

    if current_latent.shape != source_clean_latent.shape:
        raise ValueError("`current_latent` and `source_clean_latent` must have identical shapes.")
    if eps <= 0:
        raise ValueError("`eps` must be positive.")
    sigma_tensor = torch.as_tensor(sigma, device=current_latent.device, dtype=current_latent.dtype)
    return (current_latent - source_clean_latent) / (sigma_tensor + eps)


def _broadcast_alpha(alpha: torch.Tensor | float, reference: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(alpha, device=reference.device, dtype=reference.dtype)
    if value.ndim == 0:
        return value
    if value.ndim == 1 and value.shape[0] == reference.shape[0]:
        return value.view(reference.shape[0], *([1] * (reference.ndim - 1)))
    if value.shape == reference.shape:
        return value
    raise ValueError("`alpha` must be scalar, one value per branch, or match the velocity shape.")


def interpolate_velocity(
    keep_velocity: torch.Tensor,
    edit_velocity: torch.Tensor,
    alpha: torch.Tensor | float,
) -> torch.Tensor:
    """Interpolate keep and edit velocities with a branch-broadcast alpha."""

    if keep_velocity.shape != edit_velocity.shape:
        raise ValueError("`keep_velocity` and `edit_velocity` must have identical shapes.")
    alpha_value = _broadcast_alpha(alpha, edit_velocity)
    return (1 - alpha_value) * keep_velocity + alpha_value * edit_velocity


def build_branch_velocity(
    current_latent: torch.Tensor,
    source_clean_latent: torch.Tensor,
    edit_velocity: torch.Tensor,
    sigma: torch.Tensor | float,
    alpha: torch.Tensor | float,
    v_goal: Optional[torch.Tensor] = None,
    *,
    step_index: int = 0,
    controlled_steps: int = 4,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Build one V2 velocity, intervening only on the controlled prefix."""

    if step_index < 0:
        raise ValueError("`step_index` must be non-negative.")
    if controlled_steps < 1:
        raise ValueError("`controlled_steps` must be positive.")
    if edit_velocity.shape != current_latent.shape:
        raise ValueError("`edit_velocity` must match the current latent shape.")
    if v_goal is not None and v_goal.shape != edit_velocity.shape:
        raise ValueError("`v_goal` must match the edit velocity shape.")
    if step_index >= controlled_steps:
        return edit_velocity
    keep_velocity = compute_keep_velocity(current_latent, source_clean_latent, sigma, eps=eps)
    base_velocity = interpolate_velocity(keep_velocity, edit_velocity, alpha)
    if v_goal is None:
        return base_velocity
    # V_goal is stored in FP32, but the native sampler's velocity/update
    # dtype must remain unchanged for alpha=1, V_goal=0 native parity.
    return base_velocity + v_goal.to(dtype=base_velocity.dtype)
