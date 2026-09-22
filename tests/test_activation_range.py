import torch

from rewardflow_calibration.calibration.activation_range import (
    ActivationRangeConfig,
    ActivationRangeDetector,
    normalize_alpha,
)
from rewardflow_calibration.calibration.branch_refinement import refine_activation_bracket
from rewardflow_calibration.calibration.control_points import (
    filter_redundant_control_points,
    uniform_control_points,
)
from rewardflow_calibration.calibration.elastic_band import ElasticBandConfig, elastic_band_search


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


def test_branch_refinement_runs_two_midpoints_before_accepting_upper_endpoint():
    calls = []

    def evaluate(alpha):
        calls.append(alpha)
        return alpha

    result = refine_activation_bracket(
        invalid_alpha=0.25,
        invalid_distance=0.25,
        valid_alpha=0.5,
        valid_distance=0.5,
        evaluate=evaluate,
        activation_distance_threshold=0.45,
        recursion_depth=2,
    )

    assert calls == [0.375, 0.4375]
    assert result.alpha_start == 0.5
    assert [branch["alpha"] for branch in result.inserted_branches] == [0.375, 0.4375]


def test_branch_refinement_moves_the_valid_endpoint_when_midpoint_is_active():
    result = refine_activation_bracket(
        invalid_alpha=0.25,
        invalid_distance=0.25,
        valid_alpha=0.5,
        valid_distance=0.5,
        evaluate=lambda alpha: alpha,
        activation_distance_threshold=0.3,
        recursion_depth=2,
    )

    assert result.alpha_start == 0.3125
    assert result.valid_alpha == 0.3125
    assert result.invalid_alpha == 0.25


def test_uniform_control_points_are_sample_specific():
    assert uniform_control_points(0.5) == [0.5, 0.625, 0.75, 0.875, 1.0]
    assert uniform_control_points(0.25) == [0.25, 0.4375, 0.625, 0.8125, 1.0]


def test_elastic_band_expands_and_moves_injected_images():
    def evaluate(alpha):
        return torch.tensor([alpha * alpha])

    result = elastic_band_search(
        initial_control_points=[0.0, 0.4, 1.0],
        evaluate_image=evaluate,
        distance=lambda left, right: float((left - right).abs().item()),
        config=ElasticBandConfig(
            target_gap=0.3,
            max_points=6,
            max_iterations=10,
            min_alpha_spacing=0.01,
            filter_min_adjacent_gap=0.0,
        ),
    )

    assert result.expansions > 0
    assert result.control_points[0] == 0.0
    assert result.control_points[-1] == 1.0
    assert result.evaluated_alphas


def test_elastic_band_moves_an_interior_point_toward_the_larger_gap():
    def evaluate(alpha):
        return torch.tensor([alpha * alpha])

    result = elastic_band_search(
        initial_control_points=[0.0, 0.4, 1.0],
        evaluate_image=evaluate,
        distance=lambda left, right: float((left - right).abs().item()),
        config=ElasticBandConfig(
            target_gap=1.0,
            max_points=3,
            max_iterations=5,
            min_alpha_spacing=0.01,
            filter_min_adjacent_gap=0.0,
        ),
    )

    assert result.expansions == 0
    assert result.moves > 0
    assert result.control_points[1] > 0.4


def test_filter_redundant_control_points_keeps_endpoints():
    points, removed = filter_redundant_control_points(
        [0.0, 0.5, 0.5001, 1.0],
        adjacent_distance=lambda left, right: abs(right - left),
        min_adjacent_distance=0.001,
    )
    assert points == [0.0, 0.5001, 1.0]
    assert removed == [0.5]
