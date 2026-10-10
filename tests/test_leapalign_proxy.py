import pytest
import torch

from rewardflow_calibration.diagnostics.leapalign_proxy import (
    detached_true_trajectory,
    flow_matching_clean_prediction,
    jump_to_step,
    leap_gradient,
    mask_gradient_estimators,
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


@pytest.mark.parametrize("truth_dtype", [torch.bfloat16, torch.float32])
def test_bf16_connector_uses_fp32_forward_and_surrogate_only_gradient(
    truth_dtype
):
    # Both dtype combinations matter: mixed precision is explicitly supported,
    # and BF16/BF16 reproduces the low-precision cancellation from the run.
    surrogate = torch.tensor(
        [1.00390625, -2.1171875], dtype=torch.bfloat16, requires_grad=True
    )
    truth = torch.tensor(
        [1.234567, -2.345678], dtype=truth_dtype, requires_grad=True
    )

    connected = stop_gradient_connector(surrogate, truth)
    assert connected.dtype == torch.float32
    assert torch.max(torch.abs(connected.detach() - truth.detach().float())) <= 1e-7

    surrogate_gradient, truth_gradient = torch.autograd.grad(
        connected.sum(), (surrogate, truth), allow_unused=True
    )
    assert torch.equal(surrogate_gradient, torch.ones_like(surrogate))
    assert truth_gradient is None or torch.equal(truth_gradient, torch.zeros_like(truth))


def test_bf16_bridge_and_final_connectors_use_fp32_forward():
    sigmas = (1.0, 0.8, 0.6, 0.4, 0.2)
    states = [
        torch.full((1, 1, 1), 1.0 + index * 0.125, dtype=torch.bfloat16)
        for index in range(5)
    ]
    true_final = torch.full((1, 1, 1), 1.75, dtype=torch.bfloat16)

    gradient, steps = leap_gradient(
        torch.zeros(4, 1, 1), states, sigmas, true_final,
        velocity_fn=lambda latent, sigma: torch.zeros_like(latent),
        objective_fn=lambda latent: latent.float().sum(),
        goal_steps=4,
        nested_grad_coe=0.3,
    )

    first_step = steps[0]
    assert first_step["bridge_prediction_dtype"] == "torch.float32"
    assert first_step["bridge_truth_dtype"] == "torch.bfloat16"
    assert first_step["bridge_connected_dtype"] == "torch.float32"
    assert first_step["final_prediction_dtype"] == "torch.float32"
    assert first_step["true_final_dtype"] == "torch.bfloat16"
    assert first_step["final_connected_dtype"] == "torch.float32"
    assert torch.isfinite(gradient).all()


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


def test_step4_residual_uses_step4_sigma_and_state():
    sigmas = (1.0, 0.8, 0.6, 0.4, 0.2)
    states = [torch.full((1, 1, 1), float(index + 1)) for index in range(5)]
    calls = []

    def velocity_fn(latent, sigma):
        calls.append((float(sigma), float(latent.detach().flatten()[0])))
        return torch.zeros_like(latent)

    gradient, _ = leap_gradient(
        torch.zeros(4, 1, 1), states, sigmas, torch.full_like(states[-1], 9.0),
        velocity_fn=velocity_fn,
        objective_fn=lambda latent: latent.sum(),
        goal_steps=4,
        nested_grad_coe=0.3,
    )

    # Slots 1-3 each evaluate their own start and then the common bridge.
    # The final call belongs to slot 4 and must use true_states[3]/sigmas[3].
    assert calls[-1] == pytest.approx((sigmas[3], float(states[3].item())))
    assert gradient[3].item() == pytest.approx(-sigmas[3])
    assert [calls[index][0] for index in (1, 3, 5)] == pytest.approx([sigmas[3]] * 3)
    assert [calls[index][1] for index in (1, 3, 5)] == pytest.approx([states[3].item()] * 3)


def test_nested_gradient_coefficient_scales_gradient():
    values = []
    for coefficient in (0.0, 0.3, 1.0):
        state = torch.tensor([2.0], requires_grad=True)
        nested = nested_gradient_state(state, coefficient)
        values.append(torch.autograd.grad(nested.sum(), state)[0].item())
    assert values == pytest.approx([0.0, 0.3, 1.0])


def test_toy_linear_flow_leap_gradient_path_remains_finite_after_indexing_fix():
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
    assert torch.isfinite(leap).all()
    assert torch.isfinite(current_gradient).all()
    assert torch.isfinite(full_gradient).all()
    assert torch.count_nonzero(leap) > 0


def test_masked_gradient_estimators_keep_only_frozen_active_energy():
    gradient = torch.tensor(
        [[[1.0], [10.0]], [[2.0], [20.0]], [[3.0], [30.0]], [[4.0], [40.0]]]
    )
    mask = torch.zeros_like(gradient, dtype=torch.bool)
    mask[0, 0] = True
    mask[1, 0] = True

    masked = mask_gradient_estimators({"raw": gradient}, mask)["raw"]
    assert masked.shape == gradient.shape
    assert torch.count_nonzero(masked[~mask]) == 0
    assert torch.equal(masked[mask], gradient[mask])

    from rewardflow_calibration.diagnostics.backward_direction import temporal_energy

    energy = temporal_energy(masked)
    assert energy["energy_fraction_per_step"] == pytest.approx([0.2, 0.8, 0.0, 0.0])


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
