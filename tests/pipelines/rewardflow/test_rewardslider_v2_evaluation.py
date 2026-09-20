import torch

from diffusers.pipelines.rewardflow.rewardslider_v2_evaluation import interpolate_strengths, fixed_grid_metrics


def test_fixed_grid_strength_interpolation_is_deterministic_and_ordered():
    learned = torch.tensor([0.0, 0.2, 0.7, 1.0])
    requested = torch.linspace(0, 1, 11)
    values = interpolate_strengths(learned, requested)
    assert values[0] == 0 and values[-1] == 1
    assert torch.all(values[1:] >= values[:-1])
    torch.testing.assert_close(interpolate_strengths(learned, torch.tensor([1 / 6])), torch.tensor([0.1]))


def test_fixed_grid_metrics_reports_fair_density_statistics():
    images = torch.arange(6, dtype=torch.float32).reshape(6, 1)
    distance = lambda first, second: (first - second).abs().mean(dim=tuple(range(1, first.ndim)))
    result = fixed_grid_metrics(images, distance)
    assert set(("kl", "max_gap", "min_gap", "max_min_ratio", "path_length", "endpoint_distance")) <= set(result)
    torch.testing.assert_close(result["normalized_lpips"], torch.full((5,), 0.2))
