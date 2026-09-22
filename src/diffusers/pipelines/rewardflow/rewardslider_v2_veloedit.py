"""VeloEdit-compatible FLUX-Kontext velocity intervention."""

from __future__ import annotations

from typing import Callable, Sequence

import torch
from torch.utils.checkpoint import checkpoint

from .paper_components import paper_euler_update
from .rewardslider_v2_unroll import RewardSliderV2UnrollOutput
from .strength_trajectory import expand_shared_initial_latents


def _branch_alpha(alphas: torch.Tensor | float, reference: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(alphas, device=reference.device, dtype=torch.float32)
    if value.ndim == 0:
        value = value.expand(reference.shape[0])
    if value.ndim != 1 or value.shape[0] != reference.shape[0]:
        raise ValueError("VeloEdit rollout requires one alpha value per branch.")
    if not torch.isfinite(value).all() or torch.any(value < 0) or torch.any(value > 1):
        raise ValueError("VeloEdit alpha values must be finite and in [0, 1].")
    return value.view(reference.shape[0], *([1] * (reference.ndim - 1)))


def unroll_veloedit(
    initial_latent: torch.Tensor,
    timesteps: Sequence[torch.Tensor],
    sigmas: torch.Tensor | Sequence[float],
    velocity_fn: Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor],
    alphas: torch.Tensor | float,
    reference_latent: torch.Tensor,
    *,
    v_goals: Sequence[torch.Tensor] = (),
    preserve_steps: int = 4,
    edit_steps: int = 4,
    similarity_threshold: float = 0.8,
    use_checkpointing: bool = False,
    eps: float = 1e-8,
) -> RewardSliderV2UnrollOutput:
    """Run deterministic VeloEdit intervention for one or more branches."""
    if initial_latent.ndim != 3 or initial_latent.shape[0] != 1:
        raise ValueError("initial_latent must have shape [1, tokens, channels].")
    if reference_latent.shape != initial_latent.shape:
        raise ValueError("reference_latent must match initial_latent.")
    if not timesteps:
        raise ValueError("At least one timestep is required.")
    if preserve_steps < 0 or edit_steps < 0:
        raise ValueError("Intervention step counts must be non-negative.")
    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be in [0, 1].")
    if len(v_goals) < max(preserve_steps, edit_steps):
        raise ValueError("One V_goal tensor is required for every controlled timestep.")

    sigmas = torch.as_tensor(sigmas, device=initial_latent.device, dtype=torch.float32)
    if sigmas.ndim != 1 or sigmas.numel() != len(timesteps) + 1:
        raise ValueError("sigmas must contain one value more than timesteps.")

    alpha_tensor = torch.as_tensor(alphas)
    branches = int(alpha_tensor.numel()) if alpha_tensor.ndim else 1
    latent = expand_shared_initial_latents(initial_latent, branches).float()
    reference = reference_latent.expand_as(latent).float()
    alpha = _branch_alpha(alphas, latent)
    states = [latent]
    native_velocities = []
    actual_velocities = []
    effective_alphas = []
    controlled_steps = max(preserve_steps, edit_steps)

    for step_index, timestep in enumerate(timesteps):
        timestep = torch.as_tensor(timestep, device=latent.device)

        def predict(current: torch.Tensor) -> torch.Tensor:
            return velocity_fn(current, timestep, step_index)

        native = checkpoint(predict, latent, use_reentrant=False) if use_checkpointing and latent.requires_grad else predict(latent)
        native = native.float()
        if native.shape != latent.shape:
            raise ValueError("Native velocity must match the current branch latent shape.")
        native_velocities.append(native)

        actual = native
        if step_index < controlled_steps:
            sigma = sigmas[step_index]
            reference_velocity = (latent - reference) / (sigma + eps)
            reference_abs = reference_velocity.abs() + eps
            similarity = reference_abs / (reference_abs + (native - reference_velocity).abs())
            high_similarity = similarity >= similarity_threshold
            low_similarity = ~high_similarity
            if step_index < preserve_steps:
                actual = torch.where(high_similarity, reference_velocity, actual)
            if step_index < edit_steps:
                blend_weight = 1.0 - alpha
                blended = blend_weight * reference_velocity + alpha * native
                actual = torch.where(low_similarity, blended, actual)
            if v_goals:
                actual = actual + v_goals[step_index].to(dtype=actual.dtype)
            effective_alphas.append(alpha.detach().clone())

        actual_velocities.append(actual)
        latent = paper_euler_update(latent, actual, sigmas[step_index], sigmas[step_index + 1]).float()
        states.append(latent)

    return RewardSliderV2UnrollOutput(
        final_latent=latent,
        states=tuple(states),
        native_edit_velocities=tuple(native_velocities),
        actual_velocities=tuple(actual_velocities),
        controlled_step_indices=tuple(range(controlled_steps)),
        effective_alphas=tuple(effective_alphas),
    )
