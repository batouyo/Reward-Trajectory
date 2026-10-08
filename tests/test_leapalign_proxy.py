import pytest
import torch

from rewardflow_calibration.diagnostics.leapalign_proxy import (
    detached_true_trajectory,
    flow_matching_clean_prediction,
    jump_to_step,
    leap_gradient,
    nested_gradient_state,
    stop_gradient_connector,
)


def test_flow_matching_clean_prediction_uses_repository_sigma_sign():
    z = torch.tensor([3.0])
    velocity = torch.tensor([2.0])
    assert torch.equal(flow_matching_clean_prediction(z, velocity, 0.25), torch.tensor([2.5]))


def test_jump_to_step_uses_continuous_euler_sigma_difference():
    z = torch.tensor([3.0])
    velocity = torch.tensor([2.0])
    assert torch.equal(jump_to_step(z, velocity, 0.8, 0.3), torch.tensor([2.0]))


@pytest.mark.parametrize("connector", [stop_gradient_connector])
def test_connector_forward_is_truth_and_backward_is_surrogate_only(connector):
    pred = torch.tensor([2.0], requires_grad=True)
    truth = torch.tensor([5.0], requires_grad=True)
    connected = connector(pred, truth)
    grad_pred, grad_truth = torch.autograd.grad(connected.sum(), (pred, truth), allow_unused=True)
    assert torch.equal(connected.detach(), truth.detach())
    assert torch.equal(grad_pred, torch.ones_like(pred))
    assert grad_truth is None or torch.equal(grad_truth, torch.zeros_like(truth))


def test_bridge_connector_preserves_forward_anchor_and_surrogate_gradient():
    pred_bridge = torch.tensor([1.0, 2.0], requires_grad=True)
    true_bridge = torch.tensor([4.0, 8.0], requires_grad=True)
    bridge = stop_gradient_connector(pred_bridge, true_bridge)
    gradient, anchor_gradient = torch.autograd.grad(
        bridge.sum(), (pred_bridge, true_bridge), allow_unused=True
    )
    assert torch.equal(bridge.detach(), true_bridge.detach())
    assert torch.equal(gradient, torch.ones_like(pred_bridge))
    assert anchor_gradient is None or torch.equal(anchor_gradient, torch.zeros_like(true_bridge))


def test_true_trajectory_anchors_are_detached():
    state = torch.ones(1, requires_grad=True)
    final = 2 * state
    states, detached_final = detached_true_trajectory([state], final)
    assert not states[0].requires_grad
    assert not detached_final.requires_grad


def _toy_leap_gradient():
    sigmas = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)
    residual = torch.zeros(4, 1, 1)
    state = torch.tensor([[[1.0]]])
    states = [state]
    for index in range(5):
        step_residual = residual[index].unsqueeze(0) if index < 4 else torch.zeros_like(state)
        velocity = 0.1 * state + step_residual
        state = jump_to_step(state, velocity, sigmas[index], sigmas[index + 1])
        states.append(state)
    true_final = state.detach()

    def velocity_fn(latent, sigma):
        return 0.1 * latent

    gradient, details = leap_gradient(
        residual,
        [item.detach() for item in states],
        sigmas,
        true_final,
        velocity_fn=velocity_fn,
        objective_fn=lambda latent: latent.sum(),
        goal_steps=4,
        nested_grad_coe=0.3,
    )
    return gradient, details, residual, sigmas


def test_each_leap_estimate_only_populates_its_matching_control_slot():
    gradient, details, _, _ = _toy_leap_gradient()
    assert gradient.shape == (4, 1, 1)
    assert torch.all(gradient.abs() > 0)
    assert [row["step"] for row in details] == [1, 2, 3, 4]
    assert all(row["true_final_requires_grad"] is False for row in details)
    assert all(row["connected_forward_max_abs_error"] == 0.0 for row in details)


def test_nested_gradient_coefficient_scales_gradient():
    values = []
    for coefficient in (0.0, 0.3, 1.0):
        state = torch.tensor([2.0], requires_grad=True)
        nested = nested_gradient_state(state, coefficient)
        values.append(torch.autograd.grad(nested.sum(), state)[0].item())
    assert values == pytest.approx([0.0, 0.3, 1.0])


def _cosine(first, second):
    a, b = first.flatten(), second.flatten()
    return torch.dot(a, b) / (a.norm() * b.norm())


def test_toy_linear_flow_leap_is_closer_to_full_gradient_than_clean_proxy():
    sigmas = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)
    residual = torch.zeros(4, 1, 1, requires_grad=True)
    state = torch.tensor([[[1.0]]])
    full_states = [state]
    for index in range(5):
        velocity = 0.1 * state
        if index < 4:
            velocity = velocity + residual[index].unsqueeze(0)
        state = jump_to_step(state, velocity, sigmas[index], sigmas[index + 1])
        full_states.append(state)
    full_gradient = torch.autograd.grad(state.sum(), residual)[0].detach()

    # Existing clean proxy predicts x0 from the state at the fourth control step.
    proxy_residual = torch.zeros(4, 1, 1, requires_grad=True)
    proxy_state = torch.tensor([[[1.0]]])
    for index in range(3):
        proxy_velocity = 0.1 * proxy_state + proxy_residual[index].unsqueeze(0)
        proxy_state = jump_to_step(proxy_state, proxy_velocity, sigmas[index], sigmas[index + 1])
    proxy_velocity = 0.1 * proxy_state + proxy_residual[3].unsqueeze(0)
    proxy_clean = flow_matching_clean_prediction(proxy_state, proxy_velocity, sigmas[3])
    current_gradient = torch.autograd.grad(proxy_clean.sum(), proxy_residual)[0].detach()

    def velocity_fn(latent, sigma):
        return 0.1 * latent

    leap, _ = leap_gradient(
        torch.zeros(4, 1, 1), [value.detach() for value in full_states], sigmas,
        full_states[-1].detach(), velocity_fn=velocity_fn,
        objective_fn=lambda latent: latent.sum(), goal_steps=4, nested_grad_coe=0.3,
    )
    assert _cosine(leap, full_gradient) > _cosine(current_gradient, full_gradient)


def test_diagnostic_rejects_attached_true_final_anchor():
    residual = torch.zeros(4, 1, 1)
    states = [torch.ones(1, 1, 1) for _ in range(5)]
    final = torch.ones(1, 1, 1, requires_grad=True)
    with pytest.raises(ValueError, match="must be detached"):
        leap_gradient(
            residual, states, (1.0, 0.8, 0.6, 0.4, 0.2), final,
            velocity_fn=lambda z, s: z * 0,
            objective_fn=lambda z: z.sum(),
        )
