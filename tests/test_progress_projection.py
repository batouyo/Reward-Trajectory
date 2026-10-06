import pytest
import torch

from rewardflow_calibration.optimization.progress_reward import project_progress_features


def test_source_target_and_midpoint_progress():
    source = torch.tensor([[0.0, 0.0]])
    target = torch.tensor([[2.0, 0.0]])
    assert project_progress_features(source, target, source).raw_progress.item() == pytest.approx(0.0)
    assert project_progress_features(source, target, target).raw_progress.item() == pytest.approx(1.0)
    middle = (source + target) / 2
    assert project_progress_features(source, target, middle).raw_progress.item() == pytest.approx(0.5)


def test_orthogonal_change_preserves_progress_and_increases_drift():
    source = torch.tensor([[0.0, 0.0]])
    target = torch.tensor([[1.0, 0.0]])
    candidate = torch.tensor([[0.5, 2.0]])
    result = project_progress_features(source, target, candidate)
    assert result.raw_progress.item() == pytest.approx(0.5)
    assert result.drift.item() > 0


def test_degenerate_native_target_anchor_raises():
    source = torch.ones((1, 4))
    target = source + 1e-9
    with pytest.raises(ValueError, match="degenerate"):
        project_progress_features(source, target, source)
