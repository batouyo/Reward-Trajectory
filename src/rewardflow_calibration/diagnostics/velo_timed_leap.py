"""Single-control-step LeapAlign estimates using only VeloEdit's timing prior."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

from .leapalign_proxy import (
    _error_metrics,
    flow_matching_clean_prediction,
    jump_to_step,
    nested_gradient_state,
    stop_gradient_connector,
)


def _as_float(value: torch.Tensor | float) -> float:
    return float(torch.as_tensor(value).detach().float().cpu())


def extract_control_step_gradient(
    full_gradient: torch.Tensor, *, control_step_index: int
) -> torch.Tensor:
    """Select one zero-based residual slot from a full-rollout gradient tensor."""
    if full_gradient.ndim != 3:
        raise ValueError("full_gradient must have shape [control_steps, tokens, channels]")
    if not 0 <= control_step_index < full_gradient.shape[0]:
        raise IndexError("control_step_index is out of range")
    return full_gradient[control_step_index]


def leap_gradient_for_control_step(
    residual: torch.Tensor,
    control_step_index: int,
    bridge_step_index: int,
    true_states: Sequence[torch.Tensor],
    sigmas: Sequence[torch.Tensor | float],
    true_final: torch.Tensor,
    *,
    velocity_fn: Callable[[torch.Tensor, torch.Tensor | float], torch.Tensor],
    objective_fn: Callable[[torch.Tensor], torch.Tensor],
    nested_grad_coe: float = 0.3,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Estimate a reward gradient for exactly one residual/control step.

    Indices are zero-based pre-update states: state index ``N - 1`` is the
    latent immediately before velocity update N. Thus bridge step 8 uses
    ``true_states[7]`` and ``sigmas[7]``. The control and bridge indices are
    independent; the only ordering requirement is bridge >= control.

    When control and bridge name the same pre-update state, there is no
    control-to-bridge interval. To reproduce the prior diagnostic's step-4
    case, this boundary uses the control velocity for a one-step clean proxy.
    """
    if residual.ndim not in (2, 3) or (residual.ndim == 3 and residual.shape[0] != 1):
        raise ValueError("residual must have shape [tokens, channels] or [1, tokens, channels]")
    control = int(control_step_index)
    bridge = int(bridge_step_index)
    if control < 0 or bridge < 0:
        raise ValueError("control_step_index and bridge_step_index must be non-negative")
    if bridge < control:
        raise ValueError("bridge_step_index must be greater than or equal to control_step_index")
    if not 0.0 <= float(nested_grad_coe) <= 1.0:
        raise ValueError("nested_grad_coe must be in [0, 1]")
    required_index = max(control, bridge)
    if len(true_states) <= required_index or len(sigmas) <= required_index:
        raise ValueError("true trajectory and sigmas must include the requested control and bridge states")
    if true_final.requires_grad or any(state.requires_grad for state in true_states):
        raise ValueError("true trajectory states and final latent must be detached")

    state = true_states[control].detach()
    bridge_truth = true_states[bridge].detach()
    final_truth = true_final.detach()
    slot = residual.detach().clone().requires_grad_(True)
    slot_batched = slot.unsqueeze(0) if slot.ndim == 2 else slot
    sigma_control = sigmas[control]
    sigma_bridge = sigmas[bridge]

    # Reward sees only this one residual slot. No other timestep residuals are
    # allocated or introduced by this estimator.
    control_velocity = velocity_fn(state, sigma_control) + slot_batched.to(state.dtype)
    bridge_prediction = jump_to_step(
        state, control_velocity, sigma_control, sigma_bridge
    )
    bridge_connected = stop_gradient_connector(bridge_prediction, bridge_truth)

    if control == bridge:
        final_prediction = flow_matching_clean_prediction(
            state, control_velocity, sigma_control
        )
        prediction_mode = "single_step_clean_proxy"
    else:
        bridge_state = nested_gradient_state(bridge_connected, nested_grad_coe)
        bridge_velocity = velocity_fn(bridge_state, sigma_bridge)
        final_prediction = flow_matching_clean_prediction(
            bridge_connected, bridge_velocity, sigma_bridge
        )
        prediction_mode = "bridge_then_clean_prediction"

    final_connected = stop_gradient_connector(final_prediction, final_truth)
    loss = objective_fn(final_connected)
    if loss.numel() != 1:
        raise ValueError("objective_fn must return a scalar tensor")
    gradient = torch.autograd.grad(loss, slot, allow_unused=True)[0]
    if gradient is None:
        gradient = torch.zeros_like(slot)

    bridge_error = _error_metrics(bridge_prediction, bridge_truth)
    final_error = _error_metrics(final_prediction, final_truth)
    diagnostics: dict[str, object] = {
        "control_step": control + 1,
        "control_state_index": control,
        "bridge_step": bridge + 1,
        "bridge_state_index": bridge,
        "state_index_definition": "step N is the latent immediately before velocity update N; state index N-1",
        "control_sigma": _as_float(sigma_control),
        "bridge_sigma": _as_float(sigma_bridge),
        "control_to_bridge": bridge_error,
        "bridge_to_final": final_error,
        "bridge_connected_dtype": str(bridge_connected.dtype),
        "final_connected_dtype": str(final_connected.dtype),
        "bridge_connected_forward_max_abs_error": float(
            (bridge_connected.detach().float() - bridge_truth.detach().float()).abs().max().cpu()
        ),
        "final_connected_forward_max_abs_error": float(
            (final_connected.detach().float() - final_truth.detach().float()).abs().max().cpu()
        ),
        "bridge_to_final_prediction_mode": prediction_mode,
        "nested_grad_coe": float(nested_grad_coe),
        "gradient_rms": float(gradient.detach().float().square().mean().sqrt().cpu()),
    }
    return gradient.detach(), diagnostics


__all__ = ["extract_control_step_gradient", "leap_gradient_for_control_step"]
