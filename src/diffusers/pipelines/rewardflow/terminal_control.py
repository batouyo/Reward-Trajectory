"""Long-horizon terminal control utilities for FLUX-Kontext research.

This module is independent from RewardFlow's paper and local per-step reward
paths.  Controls modify predicted model velocity directly at a fixed prefix of
the deterministic trajectory; a terminal loss then backpropagates through all
remaining native dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from torch.utils.checkpoint import checkpoint

from .paper_components import freeze_module_parameters, paper_euler_update


VelocityFn = Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]


@dataclass
class TerminalControlUnrollOutput:
    """Result of one differentiable deterministic trajectory unroll."""

    final_latent: torch.Tensor
    states: tuple[torch.Tensor, ...]
    # Detached snapshots are diagnostics only and never feed the loss.
    native_control_velocities: tuple[torch.Tensor, ...]
    controlled_step_indices: tuple[int, ...]


@dataclass
class TerminalObjectiveOutput:
    """Scalar terminal objective and auditable endpoint scores."""

    objective_name: str
    loss: torch.Tensor
    objective_error: torch.Tensor
    source_score: torch.Tensor
    full_score: torch.Tensor
    target_score: torch.Tensor
    achieved_score: torch.Tensor


@dataclass(frozen=True)
class BestTerminalControlCheckpoint:
    """Detached best-objective control snapshot that never mutates final controls."""

    iteration: int
    objective_error: float
    controls: tuple[torch.Tensor, ...]


def update_best_control_checkpoint(
    current: BestTerminalControlCheckpoint | None,
    *,
    iteration: int,
    objective_error: torch.Tensor,
    controls: Sequence[torch.Tensor],
) -> BestTerminalControlCheckpoint:
    """Return an updated detached snapshot without restoring or changing controls."""

    if objective_error.numel() != 1 or not torch.isfinite(objective_error):
        raise ValueError("Best-checkpoint objective error must be one finite scalar.")
    value = objective_error.detach().item()
    if current is not None and value >= current.objective_error:
        return current
    return BestTerminalControlCheckpoint(
        iteration=iteration,
        objective_error=value,
        controls=tuple(control.detach().clone() for control in controls),
    )


def initialize_velocity_controls(
    reference: torch.Tensor,
    control_steps: int,
) -> torch.nn.ParameterList:
    """Create FP32 additive velocity controls matching one latent trajectory."""

    if reference.ndim != 3:
        raise ValueError("Terminal controls currently require latent shape [B, image_tokens, channels].")
    if reference.shape[0] != 1:
        raise ValueError("Terminal controls currently support only B=1.")
    if not isinstance(control_steps, int) or isinstance(control_steps, bool) or control_steps < 1:
        raise ValueError("`control_steps` must be a positive integer.")
    return torch.nn.ParameterList(
        [
            torch.nn.Parameter(torch.zeros(reference.shape, device=reference.device, dtype=torch.float32))
            for _ in range(control_steps)
        ]
    )


def normalized_control_energy(controls: Sequence[torch.Tensor]) -> torch.Tensor:
    """Return ``mean_t mean(delta_velocity_t ** 2)`` in FP32."""

    if not controls:
        raise ValueError("At least one terminal velocity control is required.")
    reference_shape = controls[0].shape
    if any(control.shape != reference_shape for control in controls):
        raise ValueError("All terminal velocity controls must have the same shape.")
    return torch.stack([control.float().square().mean() for control in controls]).mean()


def freeze_terminal_control_modules(*modules: object | None) -> int:
    """Freeze inference parameters and clear stale gradients."""

    return sum(freeze_module_parameters(module) for module in modules)


def unroll_terminal_velocity_controls(
    initial_latent: torch.Tensor,
    timesteps: Sequence[torch.Tensor],
    sigmas: torch.Tensor | Sequence[float],
    velocity_fn: VelocityFn,
    controls: Sequence[torch.Tensor] = (),
    *,
    use_checkpointing: bool = True,
) -> TerminalControlUnrollOutput:
    """Unroll native dynamics while adding controls only to early velocities.

    No trajectory state or velocity is detached.  Checkpointing recomputes each
    future velocity forward with ``use_reentrant=False``; detached copies are
    retained solely for norm/cosine logging.
    """

    if initial_latent.ndim != 3:
        raise ValueError("`initial_latent` must have shape [B, image_tokens, channels].")
    if initial_latent.shape[0] != 1:
        raise ValueError("Terminal control currently supports only B=1.")
    if not timesteps:
        raise ValueError("At least one timestep is required.")
    sigmas = torch.as_tensor(sigmas, device=initial_latent.device)
    if sigmas.ndim != 1 or sigmas.numel() != len(timesteps) + 1:
        raise ValueError("`sigmas` must be one-dimensional with len(timesteps) + 1 entries.")
    if len(controls) > len(timesteps):
        raise ValueError("The number of controls cannot exceed the number of denoising steps.")
    for control in controls:
        if control.shape != initial_latent.shape:
            raise ValueError("Every control must exactly match the initial latent shape.")
        if control.dtype != torch.float32:
            raise ValueError("Terminal velocity controls must be stored in FP32.")
        if control.device != initial_latent.device:
            raise ValueError("Controls and initial latent must be on the same device.")

    latent = initial_latent
    states = [latent]
    native_control_velocities = []
    for step_index, timestep in enumerate(timesteps):
        timestep = torch.as_tensor(timestep, device=latent.device)

        def predict(current_latent: torch.Tensor) -> torch.Tensor:
            return velocity_fn(current_latent, timestep, step_index)

        # At step zero no input requires gradients yet.  From the first
        # controlled update onward, this checkpoint preserves d(v_t)/d(z_t)
        # while discarding Transformer activations until backward recomputation.
        if use_checkpointing and latent.requires_grad:
            native_velocity = checkpoint(predict, latent, use_reentrant=False)
        else:
            native_velocity = predict(latent)
        if native_velocity.shape != latent.shape:
            raise ValueError("Native velocity must have the same shape as the current latent.")
        if native_velocity.device != latent.device:
            raise ValueError("Native velocity and latent must be on the same device.")

        if step_index < len(controls):
            native_control_velocities.append(native_velocity.detach())
            # Controls use FP32 master parameters, while their forward value is
            # cast to the native model dtype. Autograd still accumulates FP32
            # gradients on the master parameters.
            effective_velocity = native_velocity + controls[step_index].to(native_velocity.dtype)
        else:
            effective_velocity = native_velocity

        latent = paper_euler_update(
            latent,
            effective_velocity,
            sigmas[step_index],
            sigmas[step_index + 1],
        )
        states.append(latent)

    return TerminalControlUnrollOutput(
        final_latent=latent,
        states=tuple(states),
        native_control_velocities=tuple(native_control_velocities),
        controlled_step_indices=tuple(range(len(controls))),
    )


def endpoint_soft_mask(
    source_image: torch.Tensor,
    full_image: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Build a fixed mean-one soft mask from absolute endpoint difference."""

    _validate_endpoint_pair(source_image, full_image)
    difference = (full_image.detach().float() - source_image.detach().float()).abs().mean(dim=1, keepdim=True)
    mean_difference = difference.mean(dim=(2, 3), keepdim=True)
    if bool((mean_difference <= eps).any().item()):
        raise ValueError("Endpoint soft mask is undefined when source and full endpoints are identical.")
    return (difference / mean_difference.clamp_min(eps)).detach()


def blue_direction_score(image: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    """Return ``mean(B) - 0.5 * (mean(R) + mean(G))`` per image."""

    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("Blue score requires image shape [B, 3, H, W].")
    image = image.float()
    if weight is None:
        channel_means = image.mean(dim=(2, 3))
    else:
        if weight.ndim != 4 or weight.shape[1] != 1 or weight.shape[0] not in (1, image.shape[0]):
            raise ValueError("Blue-score weight must have shape [1 or B, 1, H, W].")
        if weight.shape[-2:] != image.shape[-2:]:
            raise ValueError("Blue-score weight and image spatial shapes must match.")
        weight = weight.to(device=image.device, dtype=image.dtype)
        denominator = weight.sum(dim=(2, 3)).clamp_min(torch.finfo(image.dtype).eps)
        channel_means = (image * weight).sum(dim=(2, 3)) / denominator
    return channel_means[:, 2] - 0.5 * (channel_means[:, 0] + channel_means[:, 1])


class BlueEndpointTargetLoss:
    """Diagnostic endpoint-relative target in one interpretable color score."""

    def __init__(
        self,
        source_image: torch.Tensor,
        full_image: torch.Tensor,
        *,
        weight: torch.Tensor | None = None,
    ):
        _validate_endpoint_pair(source_image, full_image)
        self.weight = None if weight is None else weight.detach().clone()
        self.source_score = blue_direction_score(source_image.detach(), self.weight).detach().clone()
        self.full_score = blue_direction_score(full_image.detach(), self.weight).detach().clone()

    def __call__(self, image: torch.Tensor, strength: float) -> TerminalObjectiveOutput:
        strength = _validate_strength(strength)
        achieved = blue_direction_score(image, self.weight)
        target = (1 - strength) * self.source_score + strength * self.full_score
        return TerminalObjectiveOutput(
            objective_name="blue",
            loss=(achieved - target).square().mean(),
            objective_error=(achieved - target).abs().mean(),
            source_score=self.source_score,
            full_score=self.full_score,
            target_score=target,
            achieved_score=achieved,
        )


class EndpointPixelTargetLoss:
    """Diagnostic pixel-interpolation upper bound, not semantic interpolation."""

    def __init__(
        self,
        source_image: torch.Tensor,
        full_image: torch.Tensor,
        *,
        weight: torch.Tensor | None = None,
    ):
        _validate_endpoint_pair(source_image, full_image)
        self.source_image = source_image.detach().float().clone()
        self.full_image = full_image.detach().float().clone()
        self.weight = None if weight is None else weight.detach().float().clone()
        self.source_score = blue_direction_score(self.source_image, self.weight).detach().clone()
        self.full_score = blue_direction_score(self.full_image, self.weight).detach().clone()

    def __call__(self, image: torch.Tensor, strength: float) -> TerminalObjectiveOutput:
        strength = _validate_strength(strength)
        target_image = (1 - strength) * self.source_image + strength * self.full_image
        squared_error = (image.float() - target_image).square()
        if self.weight is not None:
            squared_error = squared_error * self.weight.to(squared_error.device)
        target_score = (1 - strength) * self.source_score + strength * self.full_score
        return TerminalObjectiveOutput(
            objective_name="pixel",
            loss=squared_error.mean(),
            objective_error=squared_error.mean(),
            source_score=self.source_score,
            full_score=self.full_score,
            target_score=target_score,
            achieved_score=blue_direction_score(image, self.weight),
        )


def _validate_endpoint_pair(source_image: torch.Tensor, full_image: torch.Tensor) -> None:
    if source_image.shape != full_image.shape:
        raise ValueError("Source and full endpoint images must have identical shapes.")
    if source_image.ndim != 4 or source_image.shape[1] != 3:
        raise ValueError("Endpoint images must have shape [B, 3, H, W].")


def _validate_strength(strength: float) -> float:
    strength = float(strength)
    if not 0 <= strength <= 1:
        raise ValueError("Terminal target strength must lie in [0, 1].")
    return strength
