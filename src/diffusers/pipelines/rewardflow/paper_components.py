"""Auditable mathematical components for the paper-faithful RewardFlow path.

These utilities are intentionally independent from the legacy RewardFlow
implementation. Unknown experimental hyperparameters have no implicit values.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ...utils.torch_utils import randn_tensor


@dataclass
class PaperRewardFlowConfig:
    """Configuration for the opt-in paper-faithful sampler.

    ``None`` values denote quantities that the paper does not specify well
    enough to reproduce. They must be supplied before the corresponding
    feature is enabled.
    """

    enabled: bool = False

    # ASSUMPTION: The paper defines lambda_R but does not disclose a uniquely
    # recoverable experimental value, so there is no numeric default.
    lambda_reward: float | None = None

    use_kl: bool = True
    lambda_kl: float = 1.5

    use_sde_noise: bool = False
    # ASSUMPTION: The functional schedule is specified, but these three
    # experimental values are not uniquely recoverable from the paper.
    gamma_min: float | None = None
    gamma_max: float | None = None
    gamma_rho: float | None = None

    collect_trace: bool = False
    static_reward_weights: dict[str, float] | None = None

    # ASSUMPTION: The paper defines one source z0 but does not define how a
    # multi-reference pipeline selects it. ``None`` therefore means "reject
    # ambiguous multi-reference KL" rather than silently selecting an image.
    source_image_index: int | None = None

    def validate(self, *, reward_enabled: bool, has_source_image: bool) -> None:
        if not self.enabled:
            return
        if reward_enabled:
            if self.lambda_reward is None:
                raise ValueError("Paper reward guidance requires an explicit `lambda_reward`.")
            if self.lambda_reward < 0:
                raise ValueError("`lambda_reward` must be non-negative.")
        if self.use_kl:
            if not has_source_image:
                raise ValueError("Paper KL guidance requires a source image; set `use_kl=False` for text-to-image.")
            if self.lambda_kl < 0:
                raise ValueError("`lambda_kl` must be non-negative.")
        if self.use_sde_noise:
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
                    "Paper SDE noise requires explicit values for " + ", ".join(f"`{name}`" for name in missing) + "."
                )
            _validate_gamma_parameters(self.gamma_min, self.gamma_max, self.gamma_rho)


def _broadcast_scalar_or_batch(value: float | torch.Tensor, reference: torch.Tensor, name: str) -> torch.Tensor:
    value = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    if value.ndim > 1:
        raise ValueError(f"`{name}` must be a scalar or one value per batch element.")
    if value.ndim == 1 and value.shape[0] not in (1, reference.shape[0]):
        raise ValueError(f"Batch `{name}` has length {value.shape[0]}, expected 1 or {reference.shape[0]}.")
    while value.ndim < reference.ndim:
        value = value.unsqueeze(-1)
    return value


def predict_clean_latent(
    sample: torch.Tensor,
    model_output: torch.Tensor,
    sigma: float | torch.Tensor,
) -> torch.Tensor:
    """Return the flow-matching clean prediction ``x0 = sample - sigma * velocity``."""

    if sample.shape != model_output.shape:
        raise ValueError("`sample` and `model_output` must have identical shapes.")
    if sample.device != model_output.device or sample.dtype != model_output.dtype:
        raise ValueError("`sample` and `model_output` must have the same device and dtype.")
    sigma = _broadcast_scalar_or_batch(sigma, sample, "sigma")
    return sample - sigma * model_output


def reverse_flow_drift(model_output: torch.Tensor) -> torch.Tensor:
    """Map scheduler velocity to reverse-time drift under decreasing sigma."""

    return -model_output


def flow_step_size(
    sigma: float | torch.Tensor,
    sigma_next: float | torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Return ``eta = sigma - sigma_next`` with scheduler-order validation.

    ASSUMPTION: The paper's adaptive algorithm-time step is not recoverable.
    Mapping it to the scheduler sigma interval preserves deterministic Euler
    equivalence when the added paper drifts and noise are zero.
    """

    sigma = _broadcast_scalar_or_batch(sigma, reference, "sigma")
    sigma_next = _broadcast_scalar_or_batch(sigma_next, reference, "sigma_next")
    eta = sigma - sigma_next
    if bool(torch.any(eta < 0).item()):
        raise ValueError("Paper sampling requires non-increasing scheduler sigmas.")
    return eta


def paper_euler_update(
    sample: torch.Tensor,
    model_output: torch.Tensor,
    sigma: float | torch.Tensor,
    sigma_next: float | torch.Tensor,
    reward_drift: torch.Tensor | None = None,
    kl_drift: torch.Tensor | None = None,
    langevin_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply one explicit RewardFlow Euler--Maruyama update.

    Half-precision inputs follow ``FlowMatchEulerDiscreteScheduler.step``:
    arithmetic is performed in float32 and the result is cast back to the
    model-output dtype. This preserves deterministic scheduler equivalence.
    """

    if sample.shape != model_output.shape:
        raise ValueError("`sample` and `model_output` must have identical shapes.")
    if sample.device != model_output.device:
        raise ValueError("`sample` and `model_output` must be on the same device.")

    compute_dtype = torch.float32 if sample.dtype in (torch.float16, torch.bfloat16) else sample.dtype
    sample_compute = sample.to(compute_dtype)
    sigma_tensor = torch.as_tensor(sigma, device=sample.device)
    sigma_next_tensor = torch.as_tensor(sigma_next, device=sample.device, dtype=sigma_tensor.dtype)
    eta = flow_step_size(sigma_tensor, sigma_next_tensor, sample_compute)

    if sigma_tensor.ndim == 0 and sigma_next_tensor.ndim == 0:
        # Preserve the scheduler's exact scalar-tensor promotion order. In
        # particular, float32 scalar * bfloat16 tensor rounds before it is
        # added to the float32-upcast sample.
        updated = sample_compute + (-eta.reshape(())) * model_output
    else:
        updated = sample_compute + eta * reverse_flow_drift(model_output.to(compute_dtype))

    added_drift = None
    if reward_drift is not None:
        added_drift = reward_drift.to(device=sample.device, dtype=compute_dtype)
    if kl_drift is not None:
        kl_drift = kl_drift.to(device=sample.device, dtype=compute_dtype)
        added_drift = kl_drift if added_drift is None else added_drift + kl_drift
    if added_drift is not None:
        updated = updated + eta * added_drift
    if langevin_noise is not None:
        updated = updated + langevin_noise.to(device=sample.device, dtype=compute_dtype)
    return updated.to(model_output.dtype)


def clean_latent_kl_energy(clean_pred: torch.Tensor, source_clean_latent: torch.Tensor) -> torch.Tensor:
    """Return ``0.5 * ||clean_pred - source||^2``, summed per sample and batch-averaged.

    ASSUMPTION: The paper specifies the batch-one gradient but not a batch
    reduction. We sum non-batch dimensions and average samples explicitly.
    """

    if clean_pred.shape != source_clean_latent.shape:
        raise ValueError(
            "Clean prediction and source clean latent must have identical shapes, "
            f"got {tuple(clean_pred.shape)} and {tuple(source_clean_latent.shape)}."
        )
    non_batch_dims = tuple(range(1, clean_pred.ndim))
    per_sample = 0.5 * (clean_pred - source_clean_latent).float().pow(2).sum(dim=non_batch_dims)
    return per_sample.mean()


def _validate_gamma_parameters(gamma_min: float, gamma_max: float, rho: float) -> None:
    if gamma_min <= 0 or gamma_max <= 0:
        raise ValueError("`gamma_min` and `gamma_max` must be positive.")
    if gamma_max < gamma_min:
        raise ValueError("`gamma_max` must be greater than or equal to `gamma_min`.")
    if rho <= 0:
        raise ValueError("`rho` must be positive.")


def paper_gamma_schedule(
    sigma: float | torch.Tensor,
    sigma_start: float | torch.Tensor,
    *,
    gamma_min: float,
    gamma_max: float,
    rho: float,
) -> torch.Tensor:
    """Evaluate the paper's monotonically decreasing diffusion-strength schedule.

    ASSUMPTION: The paper defines the schedule in diffusion time ``t``. This
    flow-matching implementation uses ``sigma / sigma_start`` as ``t / t_bar``.
    The mapping is explicit here because the repository exposes sigma as its
    natural flow-time coordinate.
    """

    _validate_gamma_parameters(gamma_min, gamma_max, rho)
    sigma = torch.as_tensor(sigma)
    sigma_start = torch.as_tensor(sigma_start, device=sigma.device, dtype=sigma.dtype)
    if bool(torch.any(sigma_start <= 0).item()):
        raise ValueError("`sigma_start` must be positive.")
    ratio = sigma / sigma_start
    if bool(torch.any((ratio < 0) | (ratio > 1)).item()):
        raise ValueError("`sigma / sigma_start` must lie in [0, 1].")
    return gamma_min + (gamma_max - gamma_min) * ratio.pow(rho)


def sample_langevin_noise(
    sample: torch.Tensor,
    gamma: float | torch.Tensor,
    eta: float | torch.Tensor,
    *,
    generator: torch.Generator | list[torch.Generator] | None = None,
) -> torch.Tensor:
    """Sample ``sqrt(2 * gamma * eta) * N(0, I)`` reproducibly."""

    gamma = _broadcast_scalar_or_batch(gamma, sample, "gamma")
    eta = _broadcast_scalar_or_batch(eta, sample, "eta")
    if bool(torch.any(gamma < 0).item()) or bool(torch.any(eta < 0).item()):
        raise ValueError("`gamma` and `eta` must be non-negative.")
    standard_normal = randn_tensor(sample.shape, generator=generator, device=sample.device, dtype=sample.dtype)
    return torch.sqrt(2 * gamma * eta) * standard_normal
