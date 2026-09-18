import torch

from diffusers.pipelines.rewardflow.rewardslider_v2_coordination import DynamicDeficitCoordinator


def test_dynamic_deficit_weighting_favors_worst_deficit_after_ema_warmup():
    coordinator = DynamicDeficitCoordinator(temperature=0.5, ema_decay=0.5)
    coordinator.coordinate(torch.tensor([1.0, 1.0, 1.0]))
    result = coordinator.coordinate(torch.tensor([0.1, 0.2, 2.0]))
    assert result.weights[2] > result.weights[0]
    assert result.weights[2] > result.weights[1]
    assert torch.isfinite(result.total_loss)


def test_deficit_protocol_keeps_direction_explicit_and_logs_normalized_values():
    coordinator = DynamicDeficitCoordinator()
    result = coordinator.coordinate(torch.tensor([0.5, 1.5]))
    assert result.direction == "higher_deficit_is_worse"
    assert result.raw_deficits.shape == (2,)
    assert result.normalized_deficits.shape == (2,)
    assert torch.isfinite(result.normalized_deficits).all()
    assert torch.isfinite(result.weights).all()
    torch.testing.assert_close(result.weights.sum(), torch.tensor(1.0))

