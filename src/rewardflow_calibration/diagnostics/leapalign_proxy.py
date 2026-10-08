"""LeapAlign-style straight-through latent surrogates for proxy diagnostics.

This module contains only diagnostic gradient estimators.  It does not call a
trajectory optimizer, acceptance gate, or line search.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch


def flow_matching_clean_prediction(
    latent: torch.Tensor, velocity: torch.Tensor, sigma: torch.Tensor | float
) -> torch.Tensor:
    """Predict the clean latent under the repository's ``z = x0 + sigma*v`` convention."""
    sigma_value = torch.as_tensor(sigma, device=latent.device, dtype=torch.float32)
    return latent.float() - sigma_value * velocity.float()


def jump_to_step(
    latent: torch.Tensor,
    velocity: torch.Tensor,
    sigma_current: torch.Tensor | float,
    sigma_target: torch.Tensor | float,
) -> torch.Tensor:
    """Euler jump using the same increasing/decreasing sigma convention as rollout."""
    current = torch.as_tensor(sigma_current, device=latent.device, dtype=torch.float32)
    target = torch.as_tensor(sigma_target, device=latent.device, dtype=torch.float32)
    return latent.float() + (target - current) * velocity.float()


def stop_gradient_connector(
    surrogate: torch.Tensor, truth: torch.Tensor
) -> torch.Tensor:
    """Keep ``truth`` as the exact forward value and ``surrogate`` as its Jacobian."""
    if surrogate.shape != truth.shape:
        raise ValueError("surrogate and truth must have matching shapes")
    return surrogate + (truth.detach() - surrogate).detach()


def nested_gradient_state(state: torch.Tensor, coefficient: float) -> torch.Tensor:
    """Scale only gradient flowing through a nested transformer input."""
    if not 0.0 <= float(coefficient) <= 1.0:
        raise ValueError("nested_grad_coe must be in [0, 1]")
    value = float(coefficient)
    return value * state + (1.0 - value) * state.detach()


def detached_true_trajectory(
    states: Sequence[torch.Tensor], final_latent: torch.Tensor
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """Detach all trajectory anchors so they cannot carry the full rollout graph."""
    detached_states = tuple(state.detach() for state in states)
    detached_final = final_latent.detach()
    if any(state.requires_grad for state in detached_states) or detached_final.requires_grad:
        raise AssertionError("true trajectory anchors must be detached")
    return detached_states, detached_final


def _error_metrics(prediction: torch.Tensor, truth: torch.Tensor) -> dict[str, float]:
    difference = prediction.detach().float() - truth.detach().float()
    truth_value = truth.detach().float()
    rms = difference.square().mean().sqrt()
    relative = rms / truth_value.square().mean().sqrt().clamp_min(1e-12)
    return {
        "mean_abs_error": float(difference.abs().mean().cpu()),
        "rms_error": float(rms.cpu()),
        "relative_rms_error": float(relative.cpu()),
    }


def leap_gradient(
    residual: torch.Tensor,
    true_states: Sequence[torch.Tensor],
    sigmas: Sequence[torch.Tensor | float],
    true_final: torch.Tensor,
    *,
    velocity_fn: Callable[[torch.Tensor, torch.Tensor | float], torch.Tensor],
    objective_fn: Callable[[torch.Tensor], torch.Tensor],
    goal_steps: int = 4,
    nested_grad_coe: float = 0.3,
) -> tuple[torch.Tensor, list[dict[str, object]]]:
    """Build independent two-leap gradient estimates for each control slot.

    ``true_states[i]`` is the latent at sigma ``sigmas[i]``.  For slots before
    the last controlled slot, Leap 1 jumps from that step's true state to the
    detached step-``goal_steps`` bridge.  The final slot starts at the bridge,
    as specified for the step-4 boundary diagnostic.  Each loss is
    differentiated only with respect to its own residual slot; the resulting
    gradients are stacked in the input residual's shape.
    """
    if residual.ndim != 3:
        raise ValueError("residual must have shape [steps, tokens, channels]")
    if not 1 <= goal_steps <= residual.shape[0]:
        raise ValueError("goal_steps must be within the residual step count")
    if len(true_states) <= goal_steps or len(sigmas) <= goal_steps:
        raise ValueError("true trajectory must include the step-4 bridge state")
    if not 0.0 <= nested_grad_coe <= 1.0:
        raise ValueError("nested_grad_coe must be one of 0.0, 0.3, or 1.0 (or any value in [0,1])")
    if any(state.requires_grad for state in true_states) or true_final.requires_grad:
        raise ValueError("true trajectory states and final latent must be detached")

    bridge_index = goal_steps
    bridge_truth = true_states[bridge_index].detach()
    final_truth = true_final.detach()
    gradients: list[torch.Tensor] = []
    diagnostics: list[dict[str, object]] = []

    for slot_index in range(goal_steps):
        slot = residual[slot_index].detach().clone().requires_grad_(True)
        if slot_index < goal_steps - 1:
            start = true_states[slot_index].detach()
            sigma_start = sigmas[slot_index]
            velocity = velocity_fn(start, sigma_start) + slot.unsqueeze(0).to(start.dtype)
            bridge_prediction = jump_to_step(
                start, velocity, sigma_start, sigmas[bridge_index]
            )
            bridge_connected = stop_gradient_connector(bridge_prediction, bridge_truth)
            bridge_metrics = _error_metrics(bridge_prediction, bridge_truth)
        else:
            # Step 4 is directly evaluated at the common bridge state.
            bridge_prediction = bridge_truth
            bridge_connected = bridge_truth
            bridge_metrics = {
                "mean_abs_error": 0.0,
                "rms_error": 0.0,
                "relative_rms_error": 0.0,
            }

        nested_state = nested_gradient_state(bridge_connected, nested_grad_coe)
        bridge_velocity = velocity_fn(nested_state, sigmas[bridge_index])
        if slot_index == goal_steps - 1:
            bridge_velocity = bridge_velocity + slot.unsqueeze(0).to(bridge_velocity.dtype)
        final_prediction = flow_matching_clean_prediction(
            bridge_connected, bridge_velocity, sigmas[bridge_index]
        )
        final_connected = stop_gradient_connector(final_prediction, final_truth)
        loss = objective_fn(final_connected)
        if loss.numel() != 1:
            raise ValueError("objective_fn must return a scalar tensor")
        gradient = torch.autograd.grad(loss, slot, allow_unused=True)[0]
        if gradient is None:
            gradient = torch.zeros_like(slot)
        gradients.append(gradient.detach())
        diagnostics.append({
            "step": slot_index + 1,
            "bridge_prediction_error": bridge_metrics,
            "final_prediction_error": _error_metrics(final_prediction, final_truth),
            "trajectory_similarity_factor": (
                bridge_metrics["mean_abs_error"]
                + _error_metrics(final_prediction, final_truth)["mean_abs_error"]
            ),
            "nested_grad_coe": float(nested_grad_coe),
            "true_final_requires_grad": final_truth.requires_grad,
            "connected_forward_max_abs_error": float(
                (final_connected.detach() - final_truth).abs().max().cpu()
            ),
            "gradient_rms": float(gradient.float().square().mean().sqrt().cpu()),
        })

    stacked = torch.stack(gradients, dim=0)
    if stacked.shape != residual.shape:
        raise AssertionError(f"leap gradient shape {stacked.shape} != residual shape {residual.shape}")
    return stacked, diagnostics


__all__ = [
    "detached_true_trajectory",
    "flow_matching_clean_prediction",
    "jump_to_step",
    "leap_gradient",
    "nested_gradient_state",
    "stop_gradient_connector",
]
