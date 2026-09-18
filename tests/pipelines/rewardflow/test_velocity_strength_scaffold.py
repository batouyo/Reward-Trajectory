import pytest
import torch

from diffusers.pipelines.rewardflow.velocity_strength_scaffold import (
    build_branch_velocity,
    compute_keep_velocity,
    interpolate_velocity,
)


def test_keep_velocity_is_current_state_reference_velocity():
    current = torch.tensor([[4.0, 6.0]])
    source = torch.tensor([[1.0, 2.0]])
    torch.testing.assert_close(compute_keep_velocity(current, source, 2.0), torch.tensor([[1.5, 2.0]]))


def test_interpolation_hits_both_endpoints_and_midpoint():
    keep = torch.tensor([[1.0, 3.0]])
    edit = torch.tensor([[5.0, 7.0]])
    torch.testing.assert_close(interpolate_velocity(keep, edit, 0.0), keep)
    torch.testing.assert_close(interpolate_velocity(keep, edit, 1.0), edit)
    torch.testing.assert_close(interpolate_velocity(keep, edit, 0.5), torch.tensor([[3.0, 5.0]]))


def test_build_velocity_adds_branch_goal_only_inside_four_step_prefix():
    current = torch.tensor([[3.0, 3.0]])
    source = torch.zeros_like(current)
    edit = torch.tensor([[4.0, 8.0]])
    goal = torch.tensor([[0.5, -0.5]])
    inside = build_branch_velocity(current, source, edit, 2.0, 1.0, goal, step_index=3)
    outside = build_branch_velocity(current, source, edit, 2.0, 0.0, goal, step_index=4)
    torch.testing.assert_close(inside, edit + goal)
    torch.testing.assert_close(outside, edit)


def test_keep_velocity_changes_when_branch_latent_changes():
    source = torch.zeros(1, 2)
    first = compute_keep_velocity(torch.ones(1, 2), source, 2.0)
    second = compute_keep_velocity(torch.full((1, 2), 3.0), source, 2.0)
    assert not torch.equal(first, second)


def test_branch_alpha_broadcasts_per_branch():
    keep = torch.zeros(2, 1, 2)
    edit = torch.ones_like(keep)
    actual = interpolate_velocity(keep, edit, torch.tensor([0.0, 0.5]))
    torch.testing.assert_close(actual[0], torch.zeros_like(actual[0]))
    torch.testing.assert_close(actual[1], torch.full_like(actual[1], 0.5))


@pytest.mark.parametrize("bad_step", [-1])
def test_invalid_step_is_rejected(bad_step):
    with pytest.raises(ValueError, match="step_index"):
        build_branch_velocity(torch.zeros(1, 2), torch.zeros(1, 2), torch.zeros(1, 2), 1.0, 0.5, step_index=bad_step)
