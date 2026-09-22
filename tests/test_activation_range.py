import torch

from rewardflow_calibration.calibration.activation_range import (
    ActivationRangeConfig,
    ActivationRangeDetector,
    normalize_alpha,
)


class AbsoluteDistance:
    def distance(self, first, second):
        return (first - second).abs().mean(dim=(1, 2, 3))


def test_source_referenced_activation_search(tmp_path):
    source = torch.zeros(1, 3, 2, 2)
    detector = ActivationRangeDetector(
        AbsoluteDistance(),
        ActivationRangeConfig(activation_distance_threshold=0.4, alpha_resolution=0.05),
    )
    result = detector.detect(
        source_image=source,
        rollout=lambda alpha: torch.full_like(source, alpha),
        output_dir=tmp_path,
    )
    assert result["activation_found"]
    assert abs(result["alpha_start"] - 0.4) <= 0.05
    assert result["reference_image_path"].endswith("source.png")


def test_alpha_normalization():
    torch.testing.assert_close(normalize_alpha(torch.tensor([0.0, 0.5, 1.0]), 0.5), torch.tensor([0.5, 0.75, 1.0]))
