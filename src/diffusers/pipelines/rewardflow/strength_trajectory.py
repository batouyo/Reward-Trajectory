"""Coupled-strength trajectory infrastructure for RewardFlow research.

This module is intentionally separate from both legacy RewardFlow and the
paper-faithful reproduction path. Strength branches are B-major/K-minor and
share their initial latent and every stochastic increment exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, Sequence

import torch

from ...utils.torch_utils import randn_tensor


@dataclass
class StrengthTrajectoryConfig:
    """Configuration for the opt-in coupled-strength research path."""

    enabled: bool = False
    strengths: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8)
    lambda_strength_reward: float = 0.0

    use_shared_sde_noise: bool = True
    gamma_min: float | None = None
    gamma_max: float | None = None
    gamma_rho: float | None = None

    collect_trace: bool = False

    def validate(self) -> None:
        if not self.enabled:
            return
        if not self.strengths:
            raise ValueError("Trajectory `strengths` must be non-empty.")
        strengths = tuple(float(strength) for strength in self.strengths)
        if any(not math.isfinite(strength) or not 0 < strength < 1 for strength in strengths):
            raise ValueError("Every trajectory strength must be finite and strictly inside (0, 1).")
        if any(current >= following for current, following in zip(strengths, strengths[1:])):
            raise ValueError("Trajectory strengths must be strictly increasing and contain no duplicates.")
        if not math.isfinite(self.lambda_strength_reward) or self.lambda_strength_reward < 0:
            raise ValueError("`lambda_strength_reward` must be finite and non-negative.")
        if self.use_shared_sde_noise:
            missing = [
                name
                for name, value in (
                    ("gamma_min", self.gamma_min),
                    ("gamma_max", self.gamma_max),
                    ("gamma_rho", self.gamma_rho),
                )
                if value is None
            ]
            if missing:
                raise ValueError(
                    "Shared trajectory SDE noise requires explicit values for "
                    + ", ".join(f"`{name}`" for name in missing)
                    + "."
                )
            if not math.isfinite(self.gamma_min) or not math.isfinite(self.gamma_max):
                raise ValueError("`gamma_min` and `gamma_max` must be finite.")
            if self.gamma_min <= 0 or self.gamma_max <= 0:
                raise ValueError("`gamma_min` and `gamma_max` must be positive.")
            if self.gamma_max < self.gamma_min:
                raise ValueError("`gamma_max` must be greater than or equal to `gamma_min`.")
            if not math.isfinite(self.gamma_rho) or self.gamma_rho <= 0:
                raise ValueError("`gamma_rho` must be finite and positive.")


@dataclass
class StrengthRewardContext:
    """Optional information available to a per-branch strength reward."""

    prompt: str | list[str] | None = None
    source_endpoint: torch.Tensor | None = None
    target_endpoint: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class StrengthRewardFn(Protocol):
    """A differentiable reward returning one scalar for every flat branch."""

    def __call__(
        self,
        *,
        image: torch.Tensor,
        target_strength: torch.Tensor,
        context: StrengthRewardContext,
    ) -> torch.Tensor:
        """Return rewards with shape ``[B*K]`` for images ``[B*K, 3, H, W]``."""


# A future whole-trajectory objective should use a separate TrajectoryRewardFn
# over [B, K, ...]. It must not be folded into this per-branch protocol.


def expand_for_strengths(tensor: torch.Tensor | None, num_strengths: int) -> torch.Tensor | None:
    """Expand ``[B, ...]`` to B-major/K-minor ``[B*K, ...]``."""

    if tensor is None:
        return None
    if tensor.ndim < 1:
        raise ValueError("Strength expansion requires a tensor with a batch dimension.")
    if num_strengths <= 0:
        raise ValueError("`num_strengths` must be positive.")
    return tensor.repeat_interleave(num_strengths, dim=0)


def flatten_strength_branches(branches: torch.Tensor) -> torch.Tensor:
    """Flatten B-major/K-minor branches from ``[B, K, ...]`` to ``[B*K, ...]``."""

    if branches.ndim < 2:
        raise ValueError("Strength branches must have explicit batch and strength dimensions.")
    return branches.reshape(branches.shape[0] * branches.shape[1], *branches.shape[2:])


def unflatten_strength_branches(
    flat_branches: torch.Tensor,
    base_batch_size: int,
    num_strengths: int,
) -> torch.Tensor:
    """Restore B-major/K-minor branches from ``[B*K, ...]`` to ``[B, K, ...]``."""

    if flat_branches.ndim < 1:
        raise ValueError("Flat strength branches must have a batch dimension.")
    if base_batch_size <= 0 or num_strengths <= 0:
        raise ValueError("`base_batch_size` and `num_strengths` must be positive.")
    expected = base_batch_size * num_strengths
    if flat_branches.shape[0] != expected:
        raise ValueError(f"Expected {expected} flat strength branches, got {flat_branches.shape[0]}.")
    return flat_branches.reshape(base_batch_size, num_strengths, *flat_branches.shape[1:])


def expand_shared_initial_latents(base_latents: torch.Tensor, num_strengths: int) -> torch.Tensor:
    """Copy each of B already-sampled latents exactly across its K branches."""

    return expand_for_strengths(base_latents, num_strengths)


def make_flat_strength_tensor(
    strengths: Sequence[float],
    base_batch_size: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return ``[s0..sK-1, s0..sK-1, ...]`` in B-major/K-minor order."""

    if base_batch_size <= 0:
        raise ValueError("`base_batch_size` must be positive.")
    if not strengths:
        raise ValueError("`strengths` must be non-empty.")
    strength_tensor = torch.tensor(tuple(strengths), device=device, dtype=dtype)
    return strength_tensor.repeat(base_batch_size)


def _broadcast_base_value(
    value: float | torch.Tensor,
    base_reference: torch.Tensor,
    name: str,
) -> torch.Tensor:
    value = torch.as_tensor(value, device=base_reference.device, dtype=base_reference.dtype)
    if value.numel() == 1:
        value = value.reshape(())
    elif value.ndim == 1 and value.shape[0] == base_reference.shape[0]:
        pass
    elif value.shape[0] == base_reference.shape[0] and all(size == 1 for size in value.shape[1:]):
        pass
    else:
        raise ValueError(f"`{name}` must be scalar or contain one value per base sample.")
    while value.ndim < base_reference.ndim:
        value = value.unsqueeze(-1)
    return value


def sample_shared_langevin_noise(
    sample: torch.Tensor,
    gamma: float | torch.Tensor,
    eta: float | torch.Tensor,
    *,
    base_batch_size: int,
    num_strengths: int,
    generator: torch.Generator | list[torch.Generator] | None = None,
) -> torch.Tensor:
    """Sample B Gaussian increments and copy each bitwise across K branches."""

    if base_batch_size <= 0 or num_strengths <= 0:
        raise ValueError("`base_batch_size` and `num_strengths` must be positive.")
    if sample.shape[0] != base_batch_size * num_strengths:
        raise ValueError(
            "Shared Langevin noise requires sample batch B*K; "
            f"got {sample.shape[0]} for B={base_batch_size}, K={num_strengths}."
        )
    if isinstance(generator, list) and len(generator) != base_batch_size:
        raise ValueError(
            "Trajectory generator lists must contain exactly one generator per base sample "
            f"(length B={base_batch_size}), not one per strength branch (B*K={base_batch_size * num_strengths})."
        )

    base_shape = (base_batch_size, *sample.shape[1:])
    base_reference = torch.empty(base_shape, device=sample.device, dtype=sample.dtype)
    gamma = _broadcast_base_value(gamma, base_reference, "gamma")
    eta = _broadcast_base_value(eta, base_reference, "eta")
    if bool(torch.any(gamma < 0).item()) or bool(torch.any(eta < 0).item()):
        raise ValueError("`gamma` and `eta` must be non-negative.")

    base_standard_normal = randn_tensor(
        base_shape,
        generator=generator,
        device=sample.device,
        dtype=sample.dtype,
    )
    base_noise = torch.sqrt(2 * gamma * eta) * base_standard_normal
    return expand_for_strengths(base_noise, num_strengths)


def max_shared_noise_difference(noise: torch.Tensor, base_batch_size: int, num_strengths: int) -> torch.Tensor:
    """Return the maximum within-base difference from the first strength branch."""

    grouped = unflatten_strength_branches(noise, base_batch_size, num_strengths)
    return (grouped - grouped[:, :1]).abs().max()


def group_trajectory_images(images: Any, base_batch_size: int, num_strengths: int) -> list[list[Any]]:
    """Group flat B-major/K-minor outputs as ``[base sample][strength]``."""

    expected = base_batch_size * num_strengths
    if len(images) != expected:
        raise ValueError(f"Expected {expected} flat trajectory images, got {len(images)}.")
    return [
        [images[base_index * num_strengths + strength_index] for strength_index in range(num_strengths)]
        for base_index in range(base_batch_size)
    ]


def _strength_reward_device(reward: object, fallback: torch.device) -> torch.device:
    model = getattr(reward, "model", None)
    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        try:
            return next(parameters()).device
        except StopIteration:
            pass
    explicit_device = getattr(reward, "device", None)
    if explicit_device is not None:
        try:
            return torch.device(explicit_device)
        except (TypeError, RuntimeError):
            pass
    return fallback


class StrengthRewardGuidance:
    """Route and validate one differentiable per-branch strength reward."""

    def __init__(self, reward: StrengthRewardFn):
        if reward is None:
            raise ValueError("StrengthRewardGuidance requires a reward callable.")
        self.reward = reward

    def compute(
        self,
        *,
        image: torch.Tensor,
        target_strength: torch.Tensor,
        context: StrengthRewardContext,
    ) -> torch.Tensor:
        if image.ndim < 1:
            raise ValueError("Strength reward images must have a batch dimension.")
        if target_strength.shape != (image.shape[0],):
            raise ValueError(f"`target_strength` must have shape [{image.shape[0]}].")
        if not isinstance(context, StrengthRewardContext):
            raise TypeError("`context` must be a StrengthRewardContext.")

        maybe_onload = getattr(self.reward, "maybe_onload", None)
        if callable(maybe_onload):
            maybe_onload()
        base_device = image.device
        target_device = _strength_reward_device(self.reward, base_device)
        reward_image = image if image.device == target_device else image.to(target_device)
        reward_strength = target_strength.to(target_device)
        routed_context = replace(
            context,
            source_endpoint=(
                context.source_endpoint.to(target_device) if torch.is_tensor(context.source_endpoint) else None
            ),
            target_endpoint=(
                context.target_endpoint.to(target_device) if torch.is_tensor(context.target_endpoint) else None
            ),
        )
        branch_rewards = self.reward(
            image=reward_image,
            target_strength=reward_strength,
            context=routed_context,
        )
        if not torch.is_tensor(branch_rewards):
            raise TypeError(f"Strength reward must return a torch.Tensor, got {type(branch_rewards)}.")
        expected_shape = (image.shape[0],)
        if branch_rewards.shape != expected_shape:
            raise ValueError(
                "Strength reward must return one scalar per flat branch with shape "
                f"{list(expected_shape)}, got {list(branch_rewards.shape)}."
            )
        return branch_rewards if branch_rewards.device == base_device else branch_rewards.to(base_device)

    def maybe_offload(self) -> None:
        maybe_offload = getattr(self.reward, "maybe_offload", None)
        if callable(maybe_offload):
            maybe_offload()


def validate_trajectory_mode(
    config: StrengthTrajectoryConfig,
    *,
    paper_enabled: bool,
    legacy_reward_guidance: bool,
    strength_reward: StrengthRewardFn | None,
) -> None:
    """Validate separation between legacy, paper, and trajectory modes."""

    if not config.enabled:
        if strength_reward is not None:
            raise ValueError("`strength_reward` requires `trajectory_config.enabled=True`.")
        return
    if paper_enabled:
        raise ValueError("Trajectory mode and paper-faithful RewardFlow mode are mutually exclusive.")
    if legacy_reward_guidance:
        raise ValueError("Trajectory mode and legacy reward guidance are mutually exclusive.")
    config.validate()
    if config.lambda_strength_reward > 0 and strength_reward is None:
        raise ValueError("Positive `lambda_strength_reward` requires a `strength_reward`.")
