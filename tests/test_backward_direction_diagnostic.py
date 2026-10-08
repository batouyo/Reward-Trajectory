import pytest
import torch

from rewardflow_calibration.diagnostics.backward_direction import (
    capture_final_gradient,
    compare_directions,
    component_gradients,
    evaluate_proxy_final_pair,
    mask_direction,
    parallel_orthogonal_energy,
    scale_global_direction,
    scale_isolated_timestep,
    temporal_energy,
    veloedit_backward_direction,
)


def _tensor(values):
    return torch.tensor(values, dtype=torch.float32).reshape(4, 2, 1)


def test_backward_sign_is_negative_gradient():
    native = torch.full((4, 2, 1), 3.0)
    reference = torch.full_like(native, 1.0)
    edit_mask = torch.zeros_like(native, dtype=torch.bool)
    edit_mask[:, 0] = True
    edit_direction = (native - reference) * edit_mask
    direction = veloedit_backward_direction(edit_direction)
    beta = 0.25
    actual = native + beta * direction
    expected = torch.where(edit_mask, (1 - beta) * native + beta * reference, native)
    assert torch.equal(actual, expected)


def test_global_scaling_uses_one_scalar_and_hits_requested_ratio():
    direction = _tensor([1, 2, 3, 4, 1, 2, 3, 4])
    native = torch.ones_like(direction) * 2
    scaled, stats = scale_global_direction(direction, native, 0.05)
    assert stats["actual_global_ratio"] == pytest.approx(0.05)
    assert torch.allclose(scaled / direction, torch.full_like(direction, stats["scalar"]))


def test_isolated_timestep_only_changes_selected_step():
    direction = torch.ones(4, 2, 1)
    native = torch.ones_like(direction)
    residual, stats = scale_isolated_timestep(direction, native, 2)
    assert stats["actual_step_ratio"] == pytest.approx(0.02)
    assert torch.count_nonzero(residual[:2]) == 0
    assert torch.count_nonzero(residual[2]) == 2
    assert torch.count_nonzero(residual[3]) == 0


def test_masked_and_unmasked_directions_have_expected_support():
    direction = torch.ones(4, 2, 1)
    mask = torch.zeros_like(direction, dtype=torch.bool)
    mask[:, 0] = True
    masked = mask_direction(direction, mask)
    unmasked = direction.clone()
    assert torch.count_nonzero(masked[:, 1]) == 0
    assert torch.count_nonzero(unmasked[:, 1]) == 4


def test_cosine_and_projection_report_negative_alignment():
    reference = torch.ones(4, 2, 1)
    backward = veloedit_backward_direction(reference)
    assert compare_directions(backward, reference)["global_cosine"] == pytest.approx(-1.0)
    stats = parallel_orthogonal_energy(backward, reference)
    assert stats["parallel_energy_fraction"] == pytest.approx(1.0)
    assert stats["orthogonal_energy_fraction"] == pytest.approx(0.0)


def test_temporal_energy_fractions_sum_to_one():
    value = torch.zeros(4, 2, 1)
    value[0] = 1
    value[3] = 2
    stats = temporal_energy(value)
    assert sum(stats["energy_fraction_per_step"]) == pytest.approx(1.0)
    assert stats["energy_fraction_per_step"][3] == pytest.approx(0.8)


def test_same_residual_is_used_for_proxy_and_full_rollout():
    class Rollout:
        def __init__(self):
            self.residuals = []

        def rollout_native(self, prepared, *, config, goal_residual, early_stop_steps):
            self.residuals.append((goal_residual, early_stop_steps))
            return torch.tensor(float(early_stop_steps or 99))

    residual = torch.zeros(4, 2, 1)
    rollout = Rollout()
    proxy, final = evaluate_proxy_final_pair(rollout, None, None, residual, 4)
    assert proxy.item() == 4
    assert final.item() == 99
    assert rollout.residuals[0][0] is residual and rollout.residuals[1][0] is residual


def test_diagnostic_rollout_pair_needs_no_acceptance_gate_api():
    class Rollout:
        def rollout_native(self, prepared, *, config, goal_residual, early_stop_steps):
            return torch.zeros(1)

    evaluate_proxy_final_pair(Rollout(), None, None, torch.zeros(4, 2, 1), 4)


def test_negative_cosine_is_preserved():
    a, b = torch.ones(4, 2, 1), -torch.ones(4, 2, 1)
    assert compare_directions(a, b)["global_cosine"] == pytest.approx(-1.0)


def test_component_gradients_are_separated():
    residual = torch.ones(4, requires_grad=True)
    losses = {"a": residual.sum(), "b": (2 * residual).sum()}
    grads = component_gradients(losses, residual)
    assert torch.allclose(grads["a"], torch.ones_like(residual))
    assert torch.allclose(grads["b"], torch.ones_like(residual) * 2)


def test_raw_semantic_gradient_survives_zero_hinge_component():
    residual = torch.tensor([0.0], requires_grad=True)
    semantic_score = residual * 2 + 0.5
    hinge_keep = torch.relu(torch.tensor(0.0))
    grads = component_gradients(
        {"keep": hinge_keep, "semantic": semantic_score}, residual
    )
    assert grads["keep"].item() == 0
    assert grads["semantic"].item() == pytest.approx(2.0)


def test_final_gradient_oom_is_captured_without_losing_proxy_results():
    prior_proxy = {"proxy": torch.tensor(1.0)}

    def oom():
        raise RuntimeError("CUDA out of memory while computing final gradient")

    result = capture_final_gradient(oom)
    assert prior_proxy["proxy"].item() == 1.0
    assert result["final_gradient_status"] == "oom"
    assert result["gradients"] is None
    assert "out of memory" in result["error"].lower()
