"""Differentiable RewardSlider V2 velocity unroll."""

from __future__ import annotations



def validate_v2_control_steps(control_steps: int) -> int:
    if control_steps != 4:
        raise ValueError("RewardSlider V2 requires exactly four controlled timesteps.")
    return control_steps
from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from torch.utils.checkpoint import checkpoint

from .paper_components import paper_euler_update
from .strength_trajectory import expand_shared_initial_latents
from .velocity_strength_scaffold import build_branch_velocity


V2VelocityFn = Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]


@dataclass
class RewardSliderV2UnrollOutput:
    final_latent: torch.Tensor
    states: tuple[torch.Tensor, ...]
    native_edit_velocities: tuple[torch.Tensor, ...]
    actual_velocities: tuple[torch.Tensor, ...]
    controlled_step_indices: tuple[int, ...]


def _branch_value(value: torch.Tensor | float, reference: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    if value.ndim == 0:
        return value
    if value.ndim == 1 and value.shape[0] == reference.shape[0]:
        return value
    if value.shape == reference.shape:
        return value
    raise ValueError("Expected a scalar, one value per branch, or a velocity-shaped value.")


def unroll_rewardslider_v2(
    initial_latent: torch.Tensor,
    timesteps: Sequence[torch.Tensor],
    sigmas: torch.Tensor | Sequence[float],
    velocity_fn: V2VelocityFn,
    alphas: torch.Tensor | float,
    v_goals: Sequence[torch.Tensor],
    *,
    source_clean_latent: torch.Tensor | None = None,
    control_steps: int = 4,
    use_checkpointing: bool = True,
) -> RewardSliderV2UnrollOutput:
    """Unroll native FLUX-Kontext dynamics with the V2 prefix scaffold.

    The callback is evaluated on the current batched branch state at every
    step.  Only the first ``control_steps`` (at most four) use
    ``(1-alpha)*V_keep + alpha*V_edit + V_goal``; all later steps use the
    native edit velocity unchanged.  Frozen model parameters remain capable of
    passing input Jacobians because no ``no_grad`` context is used here.
    """

    if initial_latent.ndim != 3 or initial_latent.shape[0] != 1:
        raise ValueError("`initial_latent` must have shape [1, tokens, channels].")
    if not timesteps:
        raise ValueError("At least one timestep is required.")
    if not 0 <= control_steps <= min(4, len(timesteps)):
        raise ValueError("`control_steps` must be between zero and four and fit the schedule.")
    sigmas = torch.as_tensor(sigmas, device=initial_latent.device)
    if sigmas.ndim != 1 or sigmas.numel() != len(timesteps) + 1:
        raise ValueError("`sigmas` must contain one value more than `timesteps`.")
    if source_clean_latent is None:
        source_clean_latent = torch.zeros_like(initial_latent)
    if source_clean_latent.shape != initial_latent.shape:
        raise ValueError("`source_clean_latent` must match `initial_latent`.")
    if len(v_goals) < control_steps:
        raise ValueError("One V_goal tensor is required for every controlled timestep.")

    latent = expand_shared_initial_latents(initial_latent, _infer_num_branches(alphas, v_goals))
    source = source_clean_latent.expand_as(latent)
    alpha_value = _branch_value(alphas, latent)
    expected_shape = latent.shape
    for goal in v_goals[:control_steps]:
        if goal.shape != expected_shape:
            raise ValueError("Each V_goal must have shape [branches, tokens, channels].")

    states = [latent]
    native_velocities = []
    actual_velocities = []
    for step_index, timestep in enumerate(timesteps):
        timestep = torch.as_tensor(timestep, device=latent.device)

        def predict(current: torch.Tensor) -> torch.Tensor:
            return velocity_fn(current, timestep, step_index)

        native = checkpoint(predict, latent, use_reentrant=False) if use_checkpointing and latent.requires_grad else predict(latent)
        if native.shape != latent.shape:
            raise ValueError("Native velocity must match the current branch latent shape.")
        native_velocities.append(native)
        if step_index < control_steps:
            actual = build_branch_velocity(
                latent, source, native, sigmas[step_index], alpha_value,
                v_goals[step_index], step_index=step_index, controlled_steps=control_steps,
            )
        else:
            actual = native
        actual_velocities.append(actual)
        latent = paper_euler_update(latent, actual, sigmas[step_index], sigmas[step_index + 1])
        states.append(latent)
    return RewardSliderV2UnrollOutput(
        final_latent=latent,
        states=tuple(states),
        native_edit_velocities=tuple(native_velocities),
        actual_velocities=tuple(actual_velocities),
        controlled_step_indices=tuple(range(control_steps)),
    )


def _infer_num_branches(alphas: torch.Tensor | float, v_goals: Sequence[torch.Tensor]) -> int:
    if torch.is_tensor(alphas) and alphas.ndim == 1:
        return int(alphas.shape[0])
    if v_goals:
        return int(v_goals[0].shape[0])
    return 1
