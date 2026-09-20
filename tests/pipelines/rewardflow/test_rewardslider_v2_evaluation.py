import torch

from diffusers.pipelines.rewardflow.rewardslider_v2_evaluation import (
    fixed_grid_metrics, generate_model_fixed_grid, interpolate_rewardslider_controls, interpolate_strengths,
)


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



def test_rewardslider_control_interpolation_has_ordered_alpha_and_zero_endpoints():
    learned = torch.tensor([0.0, 0.2, 0.7, 1.0])
    goals = [torch.tensor([[[1.0]], [[3.0]]]) for _ in range(4)]
    requested = torch.linspace(0, 1, 11)
    alpha, interpolated = interpolate_rewardslider_controls(learned, goals, requested)
    torch.testing.assert_close(alpha[[0, -1]], torch.tensor([0.0, 1.0]))
    assert torch.all(alpha[1:] >= alpha[:-1])
    for goal in interpolated:
        assert goal.shape == (11, 1, 1)
        torch.testing.assert_close(goal[0], torch.zeros_like(goal[0]))
        torch.testing.assert_close(goal[-1], torch.zeros_like(goal[-1]))
        torch.testing.assert_close(goal[4], torch.tensor([[1.4]]))


def test_rewardslider_control_interpolation_midpoint_and_learned_node():
    learned = torch.tensor([0.0, 0.2, 0.7, 1.0])
    goals = [torch.tensor([[[2.0]], [[6.0]]]) for _ in range(4)]
    alpha, interpolated = interpolate_rewardslider_controls(
        learned, goals, torch.tensor([0.5, 2 / 3])
    )
    torch.testing.assert_close(alpha, torch.tensor([0.45, 0.7]))
    torch.testing.assert_close(interpolated[0][0], torch.tensor([[4.0]]))
    torch.testing.assert_close(interpolated[0][1], torch.tensor([[6.0]]))


def test_model_fixed_grid_calls_generation_not_image_interpolation():
    calls = {}
    source = torch.zeros(1, 1, 2, 2)
    native = torch.ones(1, 1, 2, 2)
    learned = torch.tensor([0.0, 0.25, 0.7, 1.0])
    goals = [torch.tensor([[[1.0]], [[3.0]]]) for _ in range(4)]
    requested = torch.linspace(0, 1, 11)

    def rematerialize_inputs(*, num_branches):
        calls["branches"] = num_branches
        return {"branches": num_branches}

    def unroll(inputs, alpha, v_goals, **kwargs):
        calls["alpha"] = alpha.detach().clone()
        calls["goals"] = [goal.detach().clone() for goal in v_goals]
        calls["kwargs"] = kwargs
        return {"latent": torch.arange(9, dtype=torch.float32).reshape(9, 1, 1, 1)}

    def decode(unroll_result, inputs):
        calls["decoded"] = True
        return torch.zeros(9, 1, 2, 2)

    result = generate_model_fixed_grid(
        learned, goals, requested, source_image=source, native_image=native,
        rematerialize_inputs=rematerialize_inputs, unroll_callback=unroll,
        decode_callback=decode, control_steps=4, use_checkpointing=True,
    )
    assert calls["branches"] == 9
    assert calls["decoded"]
    assert calls["kwargs"] == {"control_steps": 4, "use_checkpointing": True}
    assert calls["alpha"].shape == (9,)
    assert calls["goals"][0].shape == (9, 1, 1)
    assert result["images"].shape == (11, 1, 2, 2)
    assert result["provenance"] == "MODEL_GENERATED_REWARDSLIDER_OUTPUT"
    assert result["pixel_blend_used"] is False
    assert result["interior_model_forward_count"] == 9
