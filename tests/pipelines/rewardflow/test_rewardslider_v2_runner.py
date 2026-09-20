import copy
import pytest
import json

import torch

from diffusers.pipelines.rewardflow.rewardslider_v2_runner import (
    RewardSliderV2Runner,
    build_quality_reward,
    coordinate_image_deficits,
    build_rewardslider_v2_parser,
    routed_optimization_step,
    summarize_native_parity,
    TrajectoryPlateauTracker,
)
from diffusers.pipelines.rewardflow.rewardslider_v2_alpha import OrderedAlphaParameterization
from diffusers.pipelines.rewardflow.rewardslider_v2_scheduler import RewardSliderV2Scheduler

from diffusers.pipelines.rewardflow.rewardslider_v2_optimization import BestQualityState, BestTrajectoryState, OptimizationTransaction, v_goal_off_axis_diagnostics

def test_runner_parser_contains_v2_controls():
    args = build_rewardslider_v2_parser().parse_args([])
    assert args.initial_nodes == 5
    assert args.max_nodes == 10
    assert args.control_steps == 4
    assert args.trajectory_kl_threshold == 0.15

def test_batched_native_parity_summary_is_exact_for_identical_branches():
    latent = torch.ones(2, 2, 3)
    image = torch.ones(2, 3, 2, 2)
    result = summarize_native_parity(latent, latent[:1], image, image[:1])
    torch.testing.assert_close(result["per_branch_latent_mae"], torch.zeros(2))
    torch.testing.assert_close(result["per_branch_latent_cosine"], torch.ones(2))
    torch.testing.assert_close(result["per_branch_image_mae"], torch.zeros(2))
    torch.testing.assert_close(result["within_batch_max_latent_difference"], torch.tensor(0.0))


def test_routed_optimization_step_respects_scheduler_phase():
    alpha = OrderedAlphaParameterization.random(num_interior=3, seed=11)
    goals = [torch.nn.Parameter(torch.ones(2, 2, 1)) for _ in range(4)]
    scheduler = RewardSliderV2Scheduler(alpha, goals, joint_alpha_lr_scale=0.1)
    alpha_optimizer = torch.optim.SGD([alpha.interval_logits], lr=0.1)
    vgoal_optimizer = torch.optim.SGD(goals, lr=0.1)
    alpha_before = alpha.interval_logits.detach().clone()
    goals_before = [goal.detach().clone() for goal in goals]
    routed_optimization_step(
        scheduler,
        alpha_optimizer,
        vgoal_optimizer,
        trajectory_loss=alpha.interval_logits.square().sum(),
        quality_loss=sum(goal.square().sum() for goal in goals),
        trajectory_kl=0.2,
    )
    assert not torch.equal(alpha.interval_logits, alpha_before)
    assert all(torch.equal(goal, before) for goal, before in zip(goals, goals_before))
    assert all(goal.grad is None for goal in goals)
    scheduler.force_phase("quality_repair")
    routed_optimization_step(
        scheduler,
        alpha_optimizer,
        vgoal_optimizer,
        trajectory_loss=alpha.interval_logits.square().sum(),
        quality_loss=sum(goal.square().sum() for goal in goals),
        trajectory_kl=0.2,
    )
    assert all(not torch.equal(goal, before) for goal, before in zip(goals, goals_before))
    assert alpha.interval_logits.grad is None


def test_runner_step_writes_auditable_jsonl_record(tmp_path):
    alpha = torch.nn.Parameter(torch.zeros(4))
    goals = [torch.nn.Parameter(torch.ones(2, 2, 1)) for _ in range(4)]
    output = tmp_path / "run.jsonl"
    runner = RewardSliderV2Runner(alpha, goals, output)
    runner.step(alpha.square().sum(), sum(goal.square().sum() for goal in goals), trajectory_kl=0.1)
    record = json.loads(output.read_text().splitlines()[0])
    assert {"trajectory", "reward", "gradient", "control", "topology", "system"} <= set(record)
    assert "alpha_gradient_norm" in record["gradient"]
    assert record["trajectory"]["current_number_of_nodes"] == 5


def test_routed_optimization_step_routes_trajectory_guard_to_v_goal():
    alpha = OrderedAlphaParameterization.random(num_interior=3, seed=13)
    goals = [torch.nn.Parameter(torch.ones(2, 2, 1)) for _ in range(4)]
    scheduler = RewardSliderV2Scheduler(alpha, goals)
    scheduler.force_phase("quality_repair")
    alpha_optimizer = torch.optim.SGD([alpha.interval_logits], lr=0.1)
    vgoal_optimizer = torch.optim.SGD(goals, lr=0.1)
    routed_optimization_step(
        scheduler,
        alpha_optimizer,
        vgoal_optimizer,
        trajectory_loss=alpha.interval_logits.square().sum(),
        quality_loss=None,
        trajectory_guard_loss=sum(goal.square().sum() for goal in goals),
        trajectory_kl=0.2,
        trajectory_collapsed=True,

    )
    assert alpha.interval_logits.grad is None
    assert all(goal.grad is not None and goal.grad.abs().sum() > 0 for goal in goals)


def test_local_refinement_masks_v_goal_gradients_to_relevant_branches():
    alpha = OrderedAlphaParameterization.random(num_interior=4, seed=17)
    goals = [torch.nn.Parameter(torch.ones(4, 2, 1)) for _ in range(4)]
    scheduler = RewardSliderV2Scheduler(alpha, goals)
    scheduler.force_phase("quality_repair")
    alpha_optimizer = torch.optim.SGD([alpha.interval_logits], lr=0.1)
    vgoal_optimizer = torch.optim.SGD(goals, lr=0.1)
    routed_optimization_step(
        scheduler, alpha_optimizer, vgoal_optimizer,
        trajectory_loss=None,
        quality_loss=sum(goal.square().sum() for goal in goals),
        trajectory_kl=0.2,
        vgoal_branch_indices=(1, 2),
        optimize_alpha=False,
    )
    assert all(goal.grad is not None for goal in goals)
    for goal in goals:
        assert goal.grad[0].abs().sum() == 0
        assert goal.grad[3].abs().sum() == 0
        assert goal.grad[1].abs().sum() > 0
        assert goal.grad[2].abs().sum() > 0
    assert alpha.interval_logits.grad is None


def test_trajectory_plateau_tracker_requires_stale_calibration_steps():
    tracker = TrajectoryPlateauTracker(delta=0.01, patience=2)
    assert not tracker.update(1.0, phase="trajectory_calibration")
    assert not tracker.update(0.995, phase="trajectory_calibration")
    assert tracker.update(0.996, phase="trajectory_calibration")
    assert not tracker.update(0.9, phase="quality_repair")


def test_best_trajectory_state_restores_best_logits_after_later_degradation():
    parameterization = OrderedAlphaParameterization.from_alphas(
        torch.tensor([0.0, 0.2, 0.6, 1.0])
    )
    best = BestTrajectoryState()
    assert best.update(parameterization, kl=0.4, iteration=0)
    with torch.no_grad():
        parameterization.interval_logits.add_(0.7)
    assert best.update(parameterization, kl=0.2, iteration=1)
    expected_logits = parameterization.interval_logits.detach().clone()
    expected_alphas = parameterization.alphas.detach().clone()
    with torch.no_grad():
        parameterization.interval_logits.sub_(1.3)
    assert not best.update(parameterization, kl=0.3, iteration=2)
    assert best.restore(parameterization)
    torch.testing.assert_close(parameterization.interval_logits, expected_logits)
    torch.testing.assert_close(parameterization.alphas, expected_alphas)
    assert best.best_kl == 0.2
    assert best.best_iteration == 1


def test_best_trajectory_state_restores_alpha_and_all_v_goals():
    parameterization = OrderedAlphaParameterization.from_alphas(
        torch.tensor([0.0, 0.2, 0.6, 1.0])
    )
    goals = [torch.nn.Parameter(torch.ones(3, 2, 1) * index) for index in range(4)]
    best = BestTrajectoryState()
    assert best.update(parameterization, kl=0.2, iteration=0, v_goals=goals)
    saved_alpha = parameterization.interval_logits.detach().clone()
    saved_goals = [goal.detach().clone() for goal in goals]
    with torch.no_grad():
        parameterization.interval_logits.add_(0.5)
        for goal in goals:
            goal.add_(3.0)
    assert best.restore(parameterization, goals)
    torch.testing.assert_close(parameterization.interval_logits, saved_alpha)
    for goal, saved in zip(goals, saved_goals):
        torch.testing.assert_close(goal, saved)


def test_optimization_transaction_restores_joint_state_and_scheduler():
    alpha = OrderedAlphaParameterization.random(num_interior=2, seed=29)
    goals = [torch.nn.Parameter(torch.ones(2, 2, 1)) for _ in range(4)]
    scheduler = RewardSliderV2Scheduler(alpha, goals)
    scheduler.force_phase("joint_refinement")
    alpha_optimizer = torch.optim.Adam([alpha.interval_logits], lr=0.1)
    vgoal_optimizer = torch.optim.Adam(goals, lr=0.1)
    loss = alpha.interval_logits.square().sum() + sum(goal.square().sum() for goal in goals)
    loss.backward()
    alpha_optimizer.step()
    vgoal_optimizer.step()
    before_alpha = alpha.interval_logits.detach().clone()
    before_goals = [goal.detach().clone() for goal in goals]
    before_alpha_state = copy.deepcopy(alpha_optimizer.state_dict())
    before_vgoal_state = copy.deepcopy(vgoal_optimizer.state_dict())
    scheduler.advance(0.2)
    before_scheduler = (scheduler.phase, scheduler.phase_iterations, scheduler._bad_streak)
    transaction = OptimizationTransaction.capture(
        alpha_parameters=(alpha.interval_logits,),
        v_goal_parameters=tuple(goals),
        alpha_optimizer=alpha_optimizer,
        vgoal_optimizer=vgoal_optimizer,
        scheduler=scheduler,
    )
    with torch.no_grad():
        alpha.interval_logits.add_(1.0)
        for goal in goals:
            goal.add_(1.0)
    scheduler.advance(0.5)
    transaction.restore(alpha_optimizer, vgoal_optimizer)
    torch.testing.assert_close(alpha.interval_logits, before_alpha)
    for goal, before in zip(goals, before_goals):
        torch.testing.assert_close(goal, before)
    after_alpha_state = alpha_optimizer.state_dict()
    after_vgoal_state = vgoal_optimizer.state_dict()
    assert after_alpha_state["param_groups"] == before_alpha_state["param_groups"]
    assert after_vgoal_state["param_groups"] == before_vgoal_state["param_groups"]
    for after, before in zip(after_alpha_state["state"].values(), before_alpha_state["state"].values()):
        for key in after:
            torch.testing.assert_close(after[key], before[key])
    for after, before in zip(after_vgoal_state["state"].values(), before_vgoal_state["state"].values()):
        for key in after:
            torch.testing.assert_close(after[key], before[key])
    assert (scheduler.phase, scheduler.phase_iterations, scheduler._bad_streak) == before_scheduler
def test_quality_coordination_uses_one_preservation_weight_without_quality():
    from diffusers.pipelines.rewardflow.rewardslider_v2_coordination import DynamicDeficitCoordinator

    coordinator = DynamicDeficitCoordinator()
    preservation = torch.tensor(0.7, requires_grad=True)
    result = coordinate_image_deficits(coordinator, preservation)
    torch.testing.assert_close(result.weights, torch.ones(1))
    torch.testing.assert_close(result.total_loss, preservation)
    result.total_loss.backward()
    assert preservation.grad is not None
    assert preservation.grad != 0


def test_quality_reward_builder_rejects_unavailable_and_audits_mock():
    with pytest.raises(ValueError, match="unavailable"):
        build_quality_reward("missing", torch.device("cpu"))
    reward = build_quality_reward("mock", torch.device("cpu"))
    assert reward is not None
    from diffusers.pipelines.rewardflow.rewardslider_v2_quality import audit_quality_reward

    audit = audit_quality_reward(reward, torch.ones(1, 3, 2, 2), require_pass=True)
    assert audit.passed
    assert audit.image_gradient_norm > 0


def test_best_quality_state_reset_accepts_first_checkpoint_after_topology_change():
    parameterization = OrderedAlphaParameterization.from_alphas(torch.tensor([0.0, 0.2, 0.6, 1.0]))
    old = BestQualityState(best_deficit=0.1)
    new = BestQualityState()
    goals = [torch.nn.Parameter(torch.zeros(2, 1, 1)) for _ in range(4)]
    assert not old.update(parameterization, kl=0.2, deficit=0.2, preservation=0.2, quality=None, iteration=1, v_goals=goals)
    assert new.update(parameterization, kl=0.2, deficit=0.2, preservation=0.2, quality=None, iteration=1, v_goals=goals)
    assert new.best_deficit == 0.2

def test_v_goal_off_axis_diagnostics_uses_null_for_zero_goal():
    goals = [torch.zeros(2, 1, 1) for _ in range(4)]
    directions = [torch.ones(2, 1, 1) for _ in range(4)]
    result = v_goal_off_axis_diagnostics(goals, directions)
    assert result["mean_nonzero_off_axis_ratio"] is None
    assert all(row["off_axis_ratio"] is None and row["zero_vgoal"] for rows in result["per_timestep"] for row in rows)
