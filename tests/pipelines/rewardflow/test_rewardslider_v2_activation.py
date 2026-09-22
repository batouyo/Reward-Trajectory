import torch

from diffusers.pipelines.rewardflow.rewardslider_v2_activation import (
    ActivationRangeConfig,
    ActivationRangeDetector,
    normalize_alpha,
)


class _AbsoluteDistance:
    def distance(self, first, second):
        return (first - second).abs().mean(dim=(1, 2, 3))


def test_normalize_alpha_maps_beta_to_active_interval():
    beta = torch.tensor([0.0, 0.5, 1.0])
    torch.testing.assert_close(normalize_alpha(beta, 0.2), torch.tensor([0.2, 0.6, 1.0]))


def test_detector_coarse_to_fine_finds_first_threshold_crossing(tmp_path):
    source = torch.ones(1, 3, 2, 2)
    detector = ActivationRangeDetector(
        _AbsoluteDistance(),
        ActivationRangeConfig(activation_distance_threshold=0.4, alpha_resolution=0.05),
    )
    result = detector.detect(
        source_image=source,
        rollout=lambda alpha: torch.full_like(source, alpha),
        output_dir=tmp_path,
        source_latent=torch.zeros(1, 4, 2),
        prompt_embeds=torch.zeros(1, 2, 3),
        inference_config={"steps": 4},
    )
    assert result["activation_found"]
    assert abs(result["alpha_start"] - 0.4) <= 0.05
    assert result["probe_results"][0]["distance"] == 0.0
    assert result["reference_image_path"].endswith("alpha_0.000.png")
    assert any(row["alpha"] == 0.5 and row["stage"] == "coarse" for row in result["probe_results"])
    assert (tmp_path / "alpha_0.500.png").exists()


def test_detector_reports_no_activation_and_keeps_endpoint_semantics(tmp_path):
    source = torch.zeros(1, 3, 2, 2)
    detector = ActivationRangeDetector(
        _AbsoluteDistance(),
        ActivationRangeConfig(activation_distance_threshold=2.0, alpha_resolution=0.05),
    )
    result = detector.detect(source_image=source, rollout=lambda alpha: source, output_dir=tmp_path)
    assert not result["activation_found"]
    assert result["alpha_start"] == 1.0
