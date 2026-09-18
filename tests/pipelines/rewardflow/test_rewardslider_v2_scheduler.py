import torch

from diffusers.pipelines.rewardflow.rewardslider_v2_alpha import OrderedAlphaParameterization
from diffusers.pipelines.rewardflow.rewardslider_v2_scheduler import RewardSliderV2Scheduler


def _make_scheduler(**kwargs):
    alpha = OrderedAlphaParameterization.random(num_interior=3, seed=7)
    goals = torch.nn.ParameterList([torch.nn.Parameter(torch.ones(2, 2, 2)) for _ in range(4)])
    return RewardSliderV2Scheduler(alpha, goals, **kwargs), alpha, goals


def test_phase_one_routes_only_trajectory_gradient_to_alpha():
    scheduler, alpha, goals = _make_scheduler()
    trajectory = alpha.interval_logits.square().sum()
    quality = sum(goal.square().sum() for goal in goals)
    scheduler.backward(trajectory_loss=trajectory, quality_loss=quality)
    assert alpha.interval_logits.grad is not None and alpha.interval_logits.grad.abs().sum() > 0
    assert all(goal.grad is None for goal in goals)


def test_phase_two_routes_quality_control_and_guard_only_to_v_goal():
    scheduler, alpha, goals = _make_scheduler()
    scheduler.force_phase("quality_repair")
    loss = sum(goal.square().sum() for goal in goals)
    scheduler.backward(trajectory_loss=alpha.interval_logits.square().sum(), quality_loss=loss)
    assert alpha.interval_logits.grad is None
    assert all(goal.grad is not None and goal.grad.abs().sum() > 0 for goal in goals)


def test_phase_three_routes_trajectory_to_alpha_and_other_losses_to_v_goal():
    scheduler, alpha, goals = _make_scheduler()
    scheduler.force_phase("joint_refinement")
    trajectory = alpha.interval_logits.square().sum()
    repair = sum(goal.square().sum() for goal in goals)
    scheduler.backward(trajectory_loss=trajectory, quality_loss=repair)
    assert alpha.interval_logits.grad is not None and alpha.interval_logits.grad.abs().sum() > 0
    assert all(goal.grad is not None and goal.grad.abs().sum() > 0 for goal in goals)


def test_phase_transition_is_deterministic_and_does_not_use_semantic_order():
    first, _, _ = _make_scheduler(trajectory_patience=2, min_repair_iterations=2)
    second, _, _ = _make_scheduler(trajectory_patience=2, min_repair_iterations=2)
    sequence = [0.2, 0.1, 0.1, 0.1]
    for kl in sequence:
        first.advance(kl)
        second.advance(kl)
    assert first.phase == second.phase == "quality_repair"
    first.advance(0.1)
    first.advance(0.1)
    assert first.phase == "joint_refinement"


def test_bad_trajectory_can_roll_back_from_quality_phase():
    scheduler, _, _ = _make_scheduler(trajectory_patience=1, rollback_patience=2, trajectory_tolerance=0.01)
    scheduler.advance(0.1)
    assert scheduler.phase == "quality_repair"
    scheduler.advance(0.2)
    scheduler.advance(0.2)
    assert scheduler.phase == "trajectory_calibration"
