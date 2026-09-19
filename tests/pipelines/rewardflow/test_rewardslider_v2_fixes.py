import torch

from diffusers.pipelines.rewardflow.rewardslider_v2_coordination import DynamicDeficitCoordinator
from diffusers.pipelines.rewardflow.rewardslider_v2_lpips import lpips_trajectory_stats
from diffusers.pipelines.rewardflow.rewardslider_v2_alpha import OrderedAlphaParameterization
from diffusers.pipelines.rewardflow.rewardslider_v2_scheduler import RewardSliderV2Scheduler
from diffusers.pipelines.rewardflow.rewardslider_v2_runner import routed_optimization_step


def test_near_zero_lpips_distances_use_a_finite_uniform_distribution():
    stats = lpips_trajectory_stats(torch.tensor([0.0, 1e-12, 0.0]))
    torch.testing.assert_close(stats.normalized_distances, torch.full((3,), 1 / 3))
    torch.testing.assert_close(stats.kl_uniform, torch.tensor(0.0))
    assert stats.collapsed


def test_dynamic_coordinator_has_explicit_uniform_warmup_then_dynamic_mode():
    coordinator = DynamicDeficitCoordinator(temperature=0.5, ema_decay=0.5, warmup_steps=1)
    warmup = coordinator.coordinate(torch.tensor([0.1, 3.0]))
    torch.testing.assert_close(warmup.weights, torch.full((2,), 0.5))
    assert warmup.mode == "warmup"
    dynamic = coordinator.coordinate(torch.tensor([0.1, 3.0]))
    assert dynamic.mode == "dynamic"
    assert dynamic.weights[1] > dynamic.weights[0]


def test_scheduler_records_phase_reference_kl_when_entering_quality_repair():
    alpha = OrderedAlphaParameterization.random(num_interior=3, seed=3)
    goals = [torch.nn.Parameter(torch.zeros(3, 2, 1)) for _ in range(4)]
    scheduler = RewardSliderV2Scheduler(alpha, goals, trajectory_patience=1)
    scheduler.advance(0.12)
    assert scheduler.phase == "quality_repair"
    assert scheduler.phase_reference_kl == 0.12


def test_joint_refinement_performs_both_optimizer_updates():
    alpha = OrderedAlphaParameterization.random(num_interior=3, seed=4)
    goals = [torch.nn.Parameter(torch.ones(3, 2, 1)) for _ in range(4)]
    scheduler = RewardSliderV2Scheduler(alpha, goals)
    scheduler.force_phase("joint_refinement")
    alpha_optimizer = torch.optim.SGD([alpha.interval_logits], lr=0.1)
    goal_optimizer = torch.optim.SGD(goals, lr=0.1)
    alpha_before = alpha.interval_logits.detach().clone()
    goals_before = [goal.detach().clone() for goal in goals]
    routed_optimization_step(
        scheduler, alpha_optimizer, goal_optimizer,
        trajectory_loss=alpha.interval_logits.square().sum(),
        quality_loss=sum(goal.square().sum() for goal in goals),
        trajectory_kl=0.2,
    )
    assert not torch.equal(alpha.interval_logits, alpha_before)
    assert any(not torch.equal(goal, before) for goal, before in zip(goals, goals_before))
