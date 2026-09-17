from __future__ import annotations

import torch

from diffusers.pipelines.rewardflow.coupled_terminal_control import (
    adjacent_ranking_loss,
    coarse_anchor_indices,
    control_band_loss,
    control_no_jump_loss,
    fine_jump_loss,
    initialize_independent_coupled_controls,
    make_coupled_prior,
    relative_gap_loss,
    scalar_direction_residual_diagnostics,
    semantic_coverage_loss,
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


def test_spatial_prior_is_scale_invariant_for_nonzero_controls():
    relevance = (torch.tensor([[1.0, 0.0, 0.5, 0.5]]),)
    control = (torch.tensor([[[2.0], [1.0], [3.0], [4.0]]]),)
    baseline = spatial_prior_loss(control, relevance)
    torch.testing.assert_close(spatial_prior_loss((2 * control[0],), relevance), baseline)
    torch.testing.assert_close(spatial_prior_loss((0.5 * control[0],), relevance), baseline)
    assert torch.isfinite(spatial_prior_loss((torch.zeros_like(control[0]),), relevance))


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


def test_optional_control_no_jump_uses_no_keep_edit_anchor_or_equal_spacing_target():
    smooth = (torch.stack([torch.full((2, 2), 0.8), torch.full((2, 2), 0.6), torch.full((2, 2), 0.2)]),)
    outlier = (torch.stack([torch.full((2, 2), 0.8), torch.full((2, 2), -5.0), torch.full((2, 2), 0.2)]),)
    assert control_no_jump_loss(smooth, max_jump_ratio=3.0) == 0
    assert control_no_jump_loss(outlier, max_jump_ratio=1.5) > 0


def test_ranking_gap_and_triangle_objectives_have_expected_direction():
    assert adjacent_ranking_loss(torch.tensor([0.0, 0.2, 0.5, 0.8, 1.0]), margin=0.01) == 0
    assert adjacent_ranking_loss(torch.tensor([0.0, 0.5, 0.3]), margin=0.01) > 0
    assert adjacent_ranking_loss(torch.tensor([0.0, 0.005]), margin=0.01) > 0
    normal, _, _, fractions = relative_gap_loss(torch.tensor([0.2, 0.2, 0.25, 0.35]))
    collapse, _, _, _ = relative_gap_loss(torch.tensor([0.01, 0.02, 0.02, 0.95]))
    jump, _, _, _ = relative_gap_loss(torch.tensor([0.01, 0.02, 0.02, 0.95]))
    assert normal == 0 and collapse > 0 and jump > 0
    torch.testing.assert_close(fractions, torch.tensor([0.2, 0.2, 0.25, 0.35]))
    straight, _, _ = triangle_deficit_loss(torch.tensor([1.0, 1.0]), torch.tensor([2.0]))
    detour, _, _ = triangle_deficit_loss(torch.tensor([1.0, 1.0]), torch.tensor([0.5]))
    assert straight == 0 and detour > 0


def test_semantic_order_coverage_and_dense_nodes_are_hierarchical():
    ordered = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    reversed_ = torch.tensor([0.0, 0.8, 0.5, 0.75, 1.0])
    assert adjacent_ranking_loss(ordered, margin=0) == 0
    assert adjacent_ranking_loss(reversed_, margin=0) > 0
    coverage, collapse, jump, gaps = semantic_coverage_loss(ordered)
    assert coverage == 0 and collapse == 0 and jump == 0
    concentrated = torch.tensor([0.0, 0.8, 0.9, 0.95, 1.0])
    assert semantic_coverage_loss(concentrated)[0] > 0
    dense = torch.linspace(0, 1, 10)
    assert adjacent_ranking_loss(dense, margin=0) == 0
    assert semantic_coverage_loss(dense)[0] == 0
    assert gaps.numel() == 4


def test_anchor_selection_is_unique_ordered_and_endpoint_inclusive():
    assert coarse_anchor_indices(5) == (0, 1, 2, 3, 4)
    for nodes in (7, 10):
        anchors = coarse_anchor_indices(nodes)
        assert anchors[0] == 0 and anchors[-1] == nodes - 1
        assert len(anchors) == len(set(anchors)) == 5
        assert tuple(sorted(anchors)) == anchors


def test_fine_dreamsim_only_penalizes_worst_jump_not_small_dense_gaps():
    small_dense = torch.tensor([0.01] * 9)
    assert fine_jump_loss(small_dense)[0] == 0
    jump, fractions, worst = fine_jump_loss(torch.tensor([0.01, 0.01, 0.01, 0.97]))
    assert jump > 0 and worst > 0.55 and fractions.argmax().item() == 3


def test_coarse_triangle_detects_detour_without_fine_node_assumption():
    smooth, _, _ = triangle_deficit_loss(torch.tensor([1.0, 1.0]), torch.tensor([2.0]))
    detour, _, _ = triangle_deficit_loss(torch.tensor([1.0, 1.0]), torch.tensor([0.5]))
    assert smooth == 0 and detour > 0


def test_triangle_deficit_uses_local_skip_distance_for_normalization():
    loss_small_skip, _, normalized_small = triangle_deficit_loss(torch.tensor([1.0, 1.0]), torch.tensor([1.0]))
    loss_large_skip, _, normalized_large = triangle_deficit_loss(torch.tensor([2.0, 2.0]), torch.tensor([3.0]))
    assert normalized_small.item() == 1.0
    torch.testing.assert_close(normalized_large, torch.tensor([1 / 3]))
    assert loss_small_skip > loss_large_skip


def test_scalar_direction_residual_detects_departure_from_scalar_d_family():
    direction = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    scalar = (0.5 * direction.expand(3, -1, -1),)
    orthogonal = (scalar[0] + torch.tensor([[[0.0, 1.0], [0.0, 1.0]]] * 3),)
    scalar_diagnostics = scalar_direction_residual_diagnostics(scalar, (direction,))[0]
    orthogonal_diagnostics = scalar_direction_residual_diagnostics(orthogonal, (direction,))[0]
    assert scalar_diagnostics["residual_ratio"].max() < 1e-7
    assert orthogonal_diagnostics["residual_ratio"].min() > 0.5


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


def test_joint_objective_updates_independent_branch_controls_differently():
    source = torch.zeros(1, 3, 4, 4)
    full = torch.ones_like(source)
    prior = make_coupled_prior((torch.ones(1, 4, 2),), (torch.ones(1, 4),))
    objective = CoupledTrajectoryObjective(
        feature_encoder=_FeatureEncoder(), dreamsim=_Distance(), source_image=source,
        native_full_image=full, prior=prior, token_height=2, token_width=2,
    )
    controls = initialize_independent_coupled_controls(prior.directions, num_branches=3, betas=(0.1, 0.4, 0.8))
    optimizer = torch.optim.Adam(controls, lr=0.1)
    # Deliberately reverse semantic coordinates so the semantic-order loss,
    # rather than an L2-to-zero regularizer, supplies the branch gradients.
    candidates = torch.stack([-control.mean().expand(3, 4, 4) for control in controls[0]], dim=0)
    optimizer.zero_grad(set_to_none=True)
    objective(candidates, controls).total.backward()
    before = controls[0].detach().clone()
    optimizer.step()
    assert not torch.equal(controls[0][0], controls[0][1])
    assert not torch.equal(controls[0], before)
    fine_jump_loss,
