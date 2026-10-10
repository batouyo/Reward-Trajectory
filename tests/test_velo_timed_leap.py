import pytest
import torch

from rewardflow_calibration.diagnostics.velo_timed_leap import (
    CONNECTOR_ATOL,
    extract_control_step_gradient,
    leap_gradient_for_control_step,
    validate_connector_error,
)
from rewardflow_calibration.diagnostics.leapalign_proxy import (
    flow_matching_clean_prediction,
    jump_to_step,
)


def _distinct_states(count=15):
    return [
        torch.full((1, 2, 1), float(index + 1), dtype=torch.bfloat16)
        for index in range(count)
    ]


def test_control_step_one_bridge_eight_uses_state_zero_and_state_seven():
    states = _distinct_states()
    sigmas = tuple(1.0 - 0.05 * index for index in range(16))
    calls = []

    def velocity_fn(latent, sigma):
        calls.append((float(sigma), latent.detach().float().clone()))
        return torch.zeros_like(latent, dtype=torch.float32)

    gradient, diagnostics = leap_gradient_for_control_step(
        torch.zeros(2, 1),
        control_step_index=0,
        bridge_step_index=7,
        true_states=states,
        sigmas=sigmas,
        true_final=torch.full_like(states[-1], 20.0),
        velocity_fn=velocity_fn,
        objective_fn=lambda latent: latent.float().sum(),
    )

    assert calls[0][0] == pytest.approx(sigmas[0])
    assert torch.equal(calls[0][1], states[0].float())
    assert calls[1][0] == pytest.approx(sigmas[7])
    assert torch.equal(calls[1][1], states[7].float())
    assert diagnostics["control_state_index"] == 0
    assert diagnostics["bridge_state_index"] == 7
    assert diagnostics["bridge_connected_forward_max_abs_error"] <= 1e-7
    assert torch.isfinite(gradient).all()


def test_control_step_four_bridge_twelve_uses_state_three_and_state_eleven():
    states = _distinct_states()
    sigmas = tuple(1.0 - 0.05 * index for index in range(16))
    calls = []

    def velocity_fn(latent, sigma):
        calls.append((float(sigma), latent.detach().float().clone()))
        return torch.zeros_like(latent, dtype=torch.float32)

    gradient, diagnostics = leap_gradient_for_control_step(
        torch.zeros(2, 1),
        control_step_index=3,
        bridge_step_index=11,
        true_states=states,
        sigmas=sigmas,
        true_final=torch.full_like(states[-1], 20.0),
        velocity_fn=velocity_fn,
        objective_fn=lambda latent: latent.float().sum(),
    )

    assert calls[0][0] == pytest.approx(sigmas[3])
    assert torch.equal(calls[0][1], states[3].float())
    assert calls[1][0] == pytest.approx(sigmas[11])
    assert torch.equal(calls[1][1], states[11].float())
    assert diagnostics["control_state_index"] == 3
    assert diagnostics["bridge_state_index"] == 11
    assert torch.isfinite(gradient).all()


def test_bridge_before_control_is_rejected():
    with pytest.raises(ValueError, match="bridge_step_index.*control_step_index"):
        leap_gradient_for_control_step(
            torch.zeros(2, 1),
            control_step_index=3,
            bridge_step_index=2,
            true_states=_distinct_states(),
            sigmas=tuple(1.0 - 0.05 * index for index in range(16)),
            true_final=torch.ones(1, 2, 1),
            velocity_fn=lambda latent, sigma: torch.zeros_like(latent),
            objective_fn=lambda latent: latent.sum(),
        )


def test_bridge_and_final_connectors_are_fp32_truth_forward():
    states = _distinct_states()
    sigmas = tuple(1.0 - 0.05 * index for index in range(16))
    true_final = torch.full_like(states[-1], 25.0)
    _, diagnostics = leap_gradient_for_control_step(
        torch.zeros(2, 1, dtype=torch.float32),
        control_step_index=0,
        bridge_step_index=7,
        true_states=states,
        sigmas=sigmas,
        true_final=true_final,
        velocity_fn=lambda latent, sigma: torch.zeros_like(latent, dtype=torch.float32),
        objective_fn=lambda latent: latent.sum(),
    )
    assert diagnostics["bridge_connected_dtype"] == "torch.float32"
    assert diagnostics["final_connected_dtype"] == "torch.float32"
    assert diagnostics["bridge_connected_forward_max_abs_error"] <= 1e-7
    assert diagnostics["final_connected_forward_max_abs_error"] <= 1e-7


def test_estimator_differentiates_only_the_supplied_control_residual():
    states = _distinct_states()
    sigmas = tuple(1.0 - 0.05 * index for index in range(16))
    residual = torch.zeros(2, 1, requires_grad=True)
    gradient, _ = leap_gradient_for_control_step(
        residual,
        control_step_index=1,
        bridge_step_index=7,
        true_states=states,
        sigmas=sigmas,
        true_final=torch.full_like(states[-1], 25.0),
        velocity_fn=lambda latent, sigma: torch.zeros_like(latent, dtype=torch.float32),
        objective_fn=lambda latent: latent.sum(),
    )
    assert gradient.shape == residual.shape
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


@pytest.mark.parametrize("bridge_index", [3, 7, 11])
def test_toy_linear_flow_has_finite_gradient_for_each_bridge(bridge_index):
    sigmas = tuple(1.0 - 0.05 * index for index in range(16))
    state = torch.ones(1, 2, 1)
    states = [state]
    for index in range(15):
        velocity = 0.1 * state
        state = jump_to_step(state, velocity, sigmas[index], sigmas[index + 1])
        states.append(state)

    gradient, _ = leap_gradient_for_control_step(
        torch.zeros(2, 1),
        control_step_index=0,
        bridge_step_index=bridge_index,
        true_states=states,
        sigmas=sigmas,
        true_final=states[-1].detach(),
        velocity_fn=lambda latent, sigma: 0.1 * latent,
        objective_fn=lambda latent: latent.sum(),
    )
    assert torch.isfinite(gradient).all()


def test_control_step_four_bridge_four_uses_one_step_clean_proxy():
    states = _distinct_states()
    sigmas = tuple(1.0 - 0.05 * index for index in range(16))
    calls = []

    def velocity_fn(latent, sigma):
        calls.append((float(sigma), latent.detach().float().clone()))
        return torch.zeros_like(latent, dtype=torch.float32)

    gradient, diagnostics = leap_gradient_for_control_step(
        torch.zeros(2, 1),
        control_step_index=3,
        bridge_step_index=3,
        true_states=states,
        sigmas=sigmas,
        true_final=torch.full_like(states[-1], 20.0),
        velocity_fn=velocity_fn,
        objective_fn=lambda latent: latent.sum(),
    )
    assert len(calls) == 1
    assert calls[0][0] == pytest.approx(sigmas[3])
    assert torch.equal(calls[0][1], states[3].float())
    assert diagnostics["bridge_to_final_prediction_mode"] == "single_step_clean_proxy"
    assert torch.count_nonzero(gradient) > 0


def test_bf16_state_keeps_fp32_residual_when_injected_into_control_velocity(monkeypatch):
    from rewardflow_calibration.diagnostics import velo_timed_leap as timed_leap

    states = _distinct_states()
    sigmas = tuple(1.0 - 0.05 * index for index in range(16))
    residual = torch.tensor([[0.1234567], [-0.7654321]], dtype=torch.float32)
    seen = {}
    original_jump = timed_leap.jump_to_step

    def inspect_jump(latent, velocity, sigma_current, sigma_target):
        seen["velocity"] = velocity.detach().clone()
        return original_jump(latent, velocity, sigma_current, sigma_target)

    monkeypatch.setattr(timed_leap, "jump_to_step", inspect_jump)
    _, diagnostics = leap_gradient_for_control_step(
        residual,
        control_step_index=0,
        bridge_step_index=1,
        true_states=states,
        sigmas=sigmas,
        true_final=torch.full_like(states[-1], 20.0),
        velocity_fn=lambda latent, sigma: torch.zeros_like(latent),
        objective_fn=lambda latent: latent.float().sum(),
    )

    assert states[0].dtype == torch.bfloat16
    assert diagnostics["native_control_dtype"] == "torch.bfloat16"
    assert diagnostics["residual_dtype"] == "torch.float32"
    assert diagnostics["control_velocity_dtype"] == "torch.float32"
    assert seen["velocity"].dtype == torch.float32
    assert torch.equal(seen["velocity"].squeeze(0), residual)


def test_connector_error_validator_enforces_fixed_tolerance():
    validate_connector_error(0.5e-6, "bridge")
    validate_connector_error(CONNECTOR_ATOL, "final")
    with pytest.raises(RuntimeError, match="connector forward error"):
        validate_connector_error(CONNECTOR_ATOL + 1e-7, "bridge")


def test_full_gradient_extraction_selects_exactly_one_control_slot():
    residual = torch.zeros(4, 2, 1, requires_grad=True)
    state = torch.ones(1, 2, 1)
    for index in range(4):
        velocity = 0.1 * state + residual[index].unsqueeze(0)
        state = state + 0.1 * velocity
    full_gradient = torch.autograd.grad(state.sum(), residual)[0]
    selected = extract_control_step_gradient(full_gradient, control_step_index=2)
    assert torch.allclose(selected, torch.full((2, 1), 0.101))
    assert selected.shape == (2, 1)
