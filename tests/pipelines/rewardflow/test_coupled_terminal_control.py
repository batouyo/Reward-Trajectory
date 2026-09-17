from __future__ import annotations

import torch

from diffusers.pipelines.rewardflow.coupled_terminal_control import (
    adjacent_ranking_loss,
    control_band_loss,
    control_smoothness_loss,
    gap_bound_loss,
    initialize_independent_coupled_controls,
    make_coupled_prior,
    spatial_prior_loss,
    triangle_deficit_loss,
    unroll_coupled_velocity_controls,
)
from diffusers.pipelines.rewardflow.terminal_control import (
    freeze_terminal_control_modules,
    unroll_terminal_velocity_controls,
)
from diffusers.pipelines.rewardflow.trajectory_objectives import CoupledTrajectoryObjective


def _velocity(latent, timestep, step):
    return (0.1 + 0.05 * step) * latent.square() + timestep * 0.001


def _directions(steps=2):
    return tuple(torch.ones(1, 4, 3) * (index + 1) for index in range(steps))


def test_batched_zero_controls_match_three_sequential_unrolls_exactly():
    initial = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3) / 10
    timesteps, sigmas = [torch.tensor(3.0), torch.tensor(1.0)], torch.tensor([1.0, 0.4, 0.0])
    controls = initialize_independent_coupled_controls(_directions(), num_branches=3, betas=(0, 0, 0))
    batched = unroll_coupled_velocity_controls(initial, timesteps, sigmas, _velocity, controls, use_checkpointing=False)
    expected = torch.cat(
        [unroll_terminal_velocity_controls(initial, timesteps, sigmas, _velocity, [control[index : index + 1] for control in controls], use_checkpointing=False).final_latent for index in range(3)]
    )
    torch.testing.assert_close(batched.final_latent, expected, rtol=0, atol=0)


def test_branch_parameters_are_independent_and_do_not_share_storage():
    controls = initialize_independent_coupled_controls(_directions(), num_branches=3)
    assert controls[0][0].data_ptr() != controls[0][1].data_ptr()
    before = controls[0][1].detach().clone()
    with torch.no_grad():
        controls[0][0].add_(1)
    torch.testing.assert_close(controls[0][1], before, rtol=0, atol=0)


def test_long_horizon_gradient_reaches_earliest_independent_control():
    controls = initialize_independent_coupled_controls(_directions(1), num_branches=3, betas=(0, 0, 0))
    result = unroll_coupled_velocity_controls(
        torch.full((1, 4, 3), 0.2),
        [torch.tensor(3.0), torch.tensor(2.0), torch.tensor(1.0)],
        torch.tensor([1.0, 0.7, 0.3, 0.0]),
        _velocity,
        controls,
        use_checkpointing=True,
    )
    result.final_latent.square().mean().backward()
    assert controls[0].grad is not None and torch.isfinite(controls[0].grad).all() and controls[0].grad.abs().sum() > 0


def test_frozen_model_has_no_grad_while_controls_receive_gradient():
    model = torch.nn.Linear(3, 3, bias=False)
    freeze_terminal_control_modules(model)
    controls = initialize_independent_coupled_controls(_directions(1), num_branches=3, betas=(0, 0, 0))
    result = unroll_coupled_velocity_controls(
        torch.ones(1, 4, 3), [torch.tensor(1.0), torch.tensor(0.0)], torch.tensor([1.0, 0.5, 0.0]),
        lambda latent, timestep, step: model(latent), controls, use_checkpointing=True,
    )
    result.final_latent.square().mean().backward()
    assert controls[0].grad is not None and controls[0].grad.abs().sum() > 0
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in model.parameters())


def test_soft_spatial_prior_penalizes_low_relevance_without_hard_zero():
    relevance = (torch.tensor([[1.0, 0.0, 0.2, 0.8]]),)
    high = (torch.tensor([[[2.0], [0.0], [0.0], [0.0]]]),)
    low = (torch.tensor([[[0.0], [2.0], [0.0], [0.0]]]),)
    assert spatial_prior_loss(low, relevance) > spatial_prior_loss(high, relevance)
    assert high[0][0, 0].item() != 0 and low[0][0, 1].item() != 0


def test_control_band_prefers_in_corridor_and_penalizes_wrong_or_orthogonal():
    direction = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    relevance = (torch.ones(1, 2),)
    good = (0.5 * direction,)
    wrong = (-0.5 * direction,)
    orth = (0.5 * direction + torch.tensor([[[0.0, 2.0], [0.0, 2.0]]]),)
    good_loss, _ = control_band_loss(good, (direction,), relevance)
    wrong_loss, _ = control_band_loss(wrong, (direction,), relevance)
    orth_loss, _ = control_band_loss(orth, (direction,), relevance)
    assert good_loss < wrong_loss and good_loss < orth_loss


def test_second_difference_is_zero_for_linear_control_nodes_and_large_for_zigzag():
    direction = torch.ones(1, 2, 2)
    linear = (torch.stack([torch.full((2, 2), 0.75), torch.full((2, 2), 0.5), torch.full((2, 2), 0.25)]),)
    zigzag = (linear[0] + torch.tensor([[[0.5, 0.5], [0.5, 0.5]], [[-0.5, -0.5], [-0.5, -0.5]], [[0.5, 0.5], [0.5, 0.5]]]),)
    assert control_smoothness_loss(linear, (direction,)) < 1e-7
    assert control_smoothness_loss(zigzag, (direction,)) > 0.1


def test_ranking_gap_and_triangle_objectives_have_expected_direction():
    assert adjacent_ranking_loss(torch.tensor([0.0, 0.2, 0.5, 0.8, 1.0]), margin=0.01) == 0
    assert adjacent_ranking_loss(torch.tensor([0.0, 0.5, 0.3]), margin=0.01) > 0
    assert adjacent_ranking_loss(torch.tensor([0.0, 0.005]), margin=0.01) > 0
    normal, _, _ = gap_bound_loss(torch.tensor([0.25, 0.3, 0.2, 0.25]), torch.tensor(1.0), min_ratio=0.25, max_ratio=2.0)
    collapse, _, _ = gap_bound_loss(torch.tensor([0.01, 0.25, 0.25, 0.25]), torch.tensor(1.0), min_ratio=0.25, max_ratio=2.0)
    jump, _, _ = gap_bound_loss(torch.tensor([0.25, 0.25, 0.25, 0.8]), torch.tensor(1.0), min_ratio=0.25, max_ratio=2.0)
    assert normal == 0 and collapse > 0 and jump > 0
    straight, _ = triangle_deficit_loss(torch.tensor([1.0, 1.0]), torch.tensor([2.0]), torch.tensor(2.0))
    detour, _ = triangle_deficit_loss(torch.tensor([1.0, 1.0]), torch.tensor([0.5]), torch.tensor(2.0))
    assert straight == 0 and detour > 0


class _FeatureEncoder:
    def encode_image(self, image):
        return self.encode_images(image)[0]

    def encode_images(self, images):
        mean = images.mean(dim=(1, 2, 3))
        return torch.stack((1 - mean, mean), dim=1)


class _Distance:
    def distance(self, first, second):
        return (first - second).square().mean(dim=(1, 2, 3)).sqrt()


def test_coupled_trajectory_objective_caches_fixed_endpoints_and_keeps_candidate_gradient():
    source = torch.zeros(1, 3, 4, 4, requires_grad=True)
    full = torch.ones_like(source, requires_grad=True)
    prior = make_coupled_prior((torch.ones(1, 4, 2),), (torch.ones(1, 4),))
    objective = CoupledTrajectoryObjective(
        feature_encoder=_FeatureEncoder(),
        dreamsim=_Distance(),
        source_image=source,
        native_full_image=full,
        prior=prior,
        token_height=2,
        token_width=2,
    )
    controls = initialize_independent_coupled_controls(prior.directions, num_branches=3)
    candidates = torch.rand(3, 3, 4, 4, requires_grad=True)
    result = objective(candidates, controls)
    result.total.backward()
    assert source.grad is None and full.grad is None
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in controls)
    assert candidates.grad is not None and torch.isfinite(candidates.grad).all() and candidates.grad.abs().sum() > 0
