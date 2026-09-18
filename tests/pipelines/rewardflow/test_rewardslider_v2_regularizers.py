import torch

from diffusers.pipelines.rewardflow.rewardslider_v2_regularizers import (
    initialize_v_goal_parameters,
    v_goal_regularizers,
)


def test_zero_v_goal_regularizers_are_finite_and_zero():
    goals = tuple(torch.zeros(2, 3, 4) for _ in range(4))
    directions = tuple(torch.ones(2, 3, 4) for _ in range(4))
    relevance = tuple(torch.ones(2, 3) for _ in range(4))
    values = v_goal_regularizers(goals, directions, relevance)
    for value in (values.residual, values.parallel, values.spatial):
        assert torch.isfinite(value)
        torch.testing.assert_close(value, torch.zeros_like(value))


def test_parallel_v_goal_is_penalized_more_than_orthogonal_v_goal():
    direction = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    relevance = (torch.ones(1, 2),)
    parallel = (direction.clone(),)
    orthogonal = (torch.tensor([[[0.0, 1.0], [0.0, 1.0]]]),)
    parallel_values = v_goal_regularizers(parallel, (direction,), relevance)
    orthogonal_values = v_goal_regularizers(orthogonal, (direction,), relevance)
    assert parallel_values.parallel > orthogonal_values.parallel


def test_low_relevance_energy_has_higher_soft_spatial_penalty():
    direction = torch.ones(1, 2, 1)
    relevance = (torch.tensor([[1.0, 0.0]]),)
    high = (torch.tensor([[[2.0], [0.0]]]),)
    low = (torch.tensor([[[0.0], [2.0]]]),)
    high_values = v_goal_regularizers(high, (direction,), relevance)
    low_values = v_goal_regularizers(low, (direction,), relevance)
    assert low_values.spatial > high_values.spatial
    assert low[0].abs().sum() > 0


def test_regularizer_diagnostics_include_each_goal_control_metric():
    goals = tuple(torch.full((2, 2, 3), 0.1 * (step + 1)) for step in range(4))
    directions = tuple(torch.ones(1, 2, 3) for _ in range(4))
    relevance = tuple(torch.tensor([[1.0, 0.25]]) for _ in range(4))
    values = v_goal_regularizers(goals, directions, relevance)
    assert len(values.diagnostics) == 4
    for diagnostic in values.diagnostics:
        assert set(diagnostic) == {
            "goal_norm", "goal_to_direction_norm_ratio", "cosine_to_direction",
            "projection_ratio", "off_region_energy_ratio",
        }
        assert all(torch.isfinite(item) for item in diagnostic.values())


def test_goal_parameters_are_independent_fp32_and_zero_initialized():
    parameters = initialize_v_goal_parameters((1, 2, 3), num_branches=3, control_steps=4)
    assert len(parameters) == 4
    assert all(parameter.dtype == torch.float32 for parameter in parameters)
    assert all(parameter.requires_grad for parameter in parameters)
    assert all(torch.equal(parameter, torch.zeros_like(parameter)) for parameter in parameters)
    assert parameters[0].data_ptr() != parameters[1].data_ptr()
    with torch.no_grad():
        parameters[0].add_(1)
    assert not torch.equal(parameters[0], parameters[1])
