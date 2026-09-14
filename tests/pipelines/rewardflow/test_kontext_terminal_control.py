from __future__ import annotations

import pytest
import torch

from diffusers.pipelines.rewardflow.paper_components import paper_euler_update
from diffusers.pipelines.rewardflow.terminal_control import (
    BlueEndpointTargetLoss,
    freeze_terminal_control_modules,
    initialize_velocity_controls,
    normalized_control_energy,
    unroll_terminal_velocity_controls,
)


def _linear_velocity(scale: float):
    return lambda latent, timestep, step: scale * latent + 0.001 * timestep + 0.01 * step


def test_zero_controls_reproduce_deterministic_euler_trajectory_exactly():
    initial = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4) / 10
    timesteps = [torch.tensor(900.0), torch.tensor(500.0), torch.tensor(100.0)]
    sigmas = torch.tensor([0.9, 0.5, 0.1, 0.0])
    controls = initialize_velocity_controls(initial, control_steps=2)

    expected = initial
    velocity_fn = _linear_velocity(0.125)
    for index, timestep in enumerate(timesteps):
        expected = paper_euler_update(
            expected, velocity_fn(expected, timestep, index), sigmas[index], sigmas[index + 1]
        )
    actual = unroll_terminal_velocity_controls(initial, timesteps, sigmas, velocity_fn, controls).final_latent

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("use_checkpointing", [False, True])
def test_terminal_loss_has_gradient_to_step_zero_through_all_future_steps(use_checkpointing):
    initial = torch.full((1, 2, 3), 0.2)
    controls = initialize_velocity_controls(initial, control_steps=1)
    result = unroll_terminal_velocity_controls(
        initial,
        [torch.tensor(3.0), torch.tensor(2.0), torch.tensor(1.0)],
        torch.tensor([1.0, 0.7, 0.3, 0.0]),
        lambda latent, timestep, step: (0.2 + 0.1 * step) * latent.square() + timestep * 0.001,
        controls,
        use_checkpointing=use_checkpointing,
    )
    loss = (result.final_latent - 0.7).square().mean()
    loss.backward()

    gradient = controls[0].grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_step_zero_control_reaches_later_states_only_through_native_dynamics():
    initial = torch.ones(1, 1, 2)
    controls = initialize_velocity_controls(initial, control_steps=1)
    controls[0].data.fill_(0.25)
    timesteps = [torch.tensor(2.0), torch.tensor(1.0)]
    sigmas = torch.tensor([1.0, 0.5, 0.0])

    def velocity_fn(latent, timestep, step):
        del timestep, step
        return 0.5 * latent

    result = unroll_terminal_velocity_controls(
        initial, timesteps, sigmas, velocity_fn, controls, use_checkpointing=False
    )
    expected_step_1 = paper_euler_update(initial, 0.5 * initial + controls[0], sigmas[0], sigmas[1])
    expected_final = paper_euler_update(expected_step_1, 0.5 * expected_step_1, sigmas[1], sigmas[2])

    torch.testing.assert_close(result.states[1], expected_step_1, rtol=0, atol=0)
    torch.testing.assert_close(result.final_latent, expected_final, rtol=0, atol=0)


def test_model_parameters_are_frozen_and_receive_no_gradient():
    model = torch.nn.Linear(4, 4, bias=False)
    freeze_terminal_control_modules(model)
    initial = torch.ones(1, 2, 4)
    controls = initialize_velocity_controls(initial, control_steps=1)

    result = unroll_terminal_velocity_controls(
        initial,
        [torch.tensor(2.0), torch.tensor(1.0)],
        torch.tensor([1.0, 0.5, 0.0]),
        lambda latent, timestep, step: model(latent),
        controls,
        use_checkpointing=True,
    )
    result.final_latent.square().mean().backward()

    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert all(parameter.grad is None for parameter in model.parameters())
    assert controls[0].grad is not None and controls[0].grad.abs().sum() > 0


def test_adam_reduces_toy_terminal_target_error():
    torch.manual_seed(0)
    initial = torch.full((1, 2, 3), 0.1)
    controls = initialize_velocity_controls(initial, control_steps=2)
    optimizer = torch.optim.Adam(controls, lr=0.2)
    timesteps = [torch.tensor(3.0), torch.tensor(2.0), torch.tensor(1.0)]
    sigmas = torch.tensor([1.0, 0.7, 0.3, 0.0])

    def terminal_error():
        result = unroll_terminal_velocity_controls(
            initial,
            timesteps,
            sigmas,
            _linear_velocity(0.2),
            controls,
            use_checkpointing=True,
        )
        return (result.final_latent - 0.8).square().mean()

    initial_error = terminal_error().detach()
    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)
        error = terminal_error()
        loss = error + 1e-4 * normalized_control_energy(controls)
        loss.backward()
        optimizer.step()
    final_error = terminal_error().detach()

    assert final_error < initial_error * 0.1


def test_blue_endpoint_target_uses_endpoint_interpolation():
    source = torch.zeros(1, 3, 2, 2)
    full = torch.zeros_like(source)
    full[:, 2] = 1
    objective = BlueEndpointTargetLoss(source, full)
    output = objective(full * 0.25, strength=0.25)

    torch.testing.assert_close(output.source_score, torch.tensor([0.0]))
    torch.testing.assert_close(output.full_score, torch.tensor([1.0]))
    torch.testing.assert_close(output.target_score, torch.tensor([0.25]))
    torch.testing.assert_close(output.loss, torch.tensor(0.0))
