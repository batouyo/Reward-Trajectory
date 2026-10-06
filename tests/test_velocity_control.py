import pytest
import torch

from rewardflow_calibration.optimization.velocity_control import (
    normalized_feedback_update,
    scale_gradient_to_global_native_ratio,
    scale_negative_gradient_to_native_ratio,
)


def _linear_progress_gradient():
    residual = torch.zeros(8, requires_grad=True)
    progress = residual.mean() + 1.0
    return residual, progress, torch.autograd.grad(progress, residual)[0]


def test_feedback_update_magnitude_tracks_target_error():
    updates = []
    for target in (0.25, 0.50, 0.75):
        _, progress, gradient = _linear_progress_gradient()
        update, _ = normalized_feedback_update(
            gradient, float(progress.detach()) - target, 0.01
        )
        updates.append(float(update.square().mean().sqrt()))
    assert updates[0] > updates[1] > updates[2]
    assert updates[0] / updates[1] == pytest.approx(1.5, rel=1e-5)
    assert updates[1] / updates[2] == pytest.approx(2.0, rel=1e-5)


@pytest.mark.parametrize("initial,target,expected_direction", [(1.2, 0.5, -1), (0.2, 0.5, 1)])
def test_feedback_update_moves_progress_toward_target(initial, target, expected_direction):
    residual = torch.zeros(16)
    gradient = torch.ones_like(residual) / residual.numel()
    update, _ = normalized_feedback_update(gradient, initial - target, 0.01)
    updated_progress = initial + update.mean().item()
    assert (updated_progress - initial) * expected_direction > 0
    assert abs(updated_progress - target) < abs(initial - target)


def test_capacity_scaling_matches_requested_single_step_ratio():
    gradient = torch.tensor([[[1.0, -2.0], [3.0, -4.0]]])
    residual, actual = scale_negative_gradient_to_native_ratio(gradient, [2.0], 0.01)
    assert actual.item() == pytest.approx(0.01, abs=1e-7)
    assert residual.square().mean().sqrt().item() / 2.0 == pytest.approx(0.01, abs=1e-7)


def test_capacity_scaling_matches_ratio_independently_per_step():
    gradient = torch.arange(1, 25, dtype=torch.float32).reshape(4, 3, 2)
    native_rms = torch.tensor([0.5, 2.0, 5.0, 10.0])
    _, actual = scale_negative_gradient_to_native_ratio(gradient, native_rms, 0.02)
    torch.testing.assert_close(actual, torch.full((4,), 0.02), atol=1e-6, rtol=1e-6)


def test_feedback_update_rejects_zero_gradient():
    with pytest.raises(FloatingPointError, match="progress gradient RMS"):
        normalized_feedback_update(torch.zeros(4), 0.5, 1e-3)


def test_capacity_scaling_rejects_zero_per_step_gradient():
    gradient = torch.ones(3, 2, 2)
    gradient[1].zero_()
    with pytest.raises(FloatingPointError, match="zero or near-zero step"):
        scale_negative_gradient_to_native_ratio(gradient, [1.0, 2.0, 3.0], 0.01)


def test_global_capacity_scaling_preserves_direction_and_single_scalar():
    torch.manual_seed(7)
    direction = torch.randn(4, 3, 2)
    direction[0] *= 0.1
    direction[1] *= 0.5
    direction[2] *= 2.0
    direction[3] *= 5.0
    native_rms = torch.tensor([0.5, 1.0, 3.0, 8.0])
    requested = 0.025

    residual, stats = scale_gradient_to_global_native_ratio(
        direction, native_rms, requested
    )

    cosine = torch.nn.functional.cosine_similarity(
        residual.flatten(), direction.flatten(), dim=0
    )
    assert cosine.item() == pytest.approx(1.0, abs=1e-6)
    ratios = residual[direction != 0] / direction[direction != 0]
    torch.testing.assert_close(ratios, torch.full_like(ratios, ratios[0]), atol=1e-6, rtol=1e-6)
    global_native_rms = native_rms.square().mean().sqrt()
    global_residual_rms = residual.square().mean().sqrt()
    assert global_residual_rms.item() / global_native_rms.item() == pytest.approx(requested, rel=1e-6)
    assert stats["requested_global_ratio"] == requested
    assert stats["actual_global_ratio"] == pytest.approx(requested, rel=1e-6)
    assert stats["scale_factor"] > 0
    per_step = stats["actual_ratio_per_step"]
    assert max(per_step) - min(per_step) > 1e-3


def test_global_capacity_scaling_rejects_zero_direction():
    with pytest.raises(FloatingPointError, match="zero or near-zero global RMS"):
        scale_gradient_to_global_native_ratio(torch.zeros(2, 3, 4), [1.0, 2.0], 0.01)
