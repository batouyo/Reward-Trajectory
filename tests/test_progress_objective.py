import pytest
import torch

from rewardflow_calibration.optimization.progress_objective import (
    ProgressLossConfig,
    progress_control_loss,
)


@pytest.mark.parametrize(
    "initial,target,expected_gradient_sign",
    [(0.8, 0.4, 1.0), (0.2, 0.4, -1.0)],
)
def test_progress_gradient_moves_toward_requested_target(initial, target, expected_gradient_sign):
    progress = torch.tensor(initial, requires_grad=True)
    values = progress_control_loss(
        progress, target, torch.zeros(()), torch.zeros((1, 2, 2)),
        ProgressLossConfig(progress_weight=1.0, drift_weight=0.0, regularization_weight=0.0),
    )
    gradient = torch.autograd.grad(values.total, progress)[0]
    # Gradient descent takes -gradient: above target it must decrease progress,
    # and below target it must increase progress.
    assert gradient.item() * expected_gradient_sign > 0


def test_progress_loss_is_target_matching_not_progress_maximization():
    progress = torch.tensor(0.8, requires_grad=True)
    values = progress_control_loss(
        progress, 0.4, torch.zeros(()), torch.zeros((1, 1, 1)),
        ProgressLossConfig(drift_weight=0.0, regularization_weight=0.0),
    )
    assert values.total.item() == pytest.approx(0.16)
