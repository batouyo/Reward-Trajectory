import torch

from diffusers.pipelines.rewardflow.coupled_terminal_control import (
    active_adjacent_ranking_loss,
    coarse_pairwise_ranking_loss,
)
from diffusers.pipelines.rewardflow.reward_scheduler import (
    TrajectoryRewardScheduler,
    TrajectoryRewardSchedulerConfig,
)
from diffusers.pipelines.rewardflow.trajectory_objectives import (
    RewardSliderV1LossWeights,
    dreamsim_first_order_loss,
)


BASE_WEIGHTS = {
    "semantic_order": 1.0,
    "semantic_coverage": 0.0,
    "semantic_pairwise_order": 1.0,
    "first_order_smoothness": 1.0,
    "coarse_gap": 0.0,
    "second": 0.25,
    "preserve": 1.0,
    "ctrl": 0.0,
    "band": 0.01,
    "spatial": 0.05,
    "energy": 0.0,
}


def _schedule(scheduler, q):
    return scheduler.step(
        base_weights=BASE_WEIGHTS,
        adjacent_semantic_gaps=q[1:] - q[:-1],
    )


def test_semantic_order_only_penalizes_adjacent_reversals():
    reversed_q = torch.tensor([0.0, 0.9, 0.8, 0.6, 1.0], requires_grad=True)
    loss = active_adjacent_ranking_loss(reversed_q)
    assert loss > 0
    loss.backward()
    assert torch.isfinite(reversed_q.grad).all()
    assert reversed_q.grad[1] > 0 and reversed_q.grad[3] < 0


def test_active_adjacent_order_only_updates_the_single_reversed_pair():
    q = torch.tensor([0.0, 0.3, 0.6, 0.55, 1.0], requires_grad=True)
    loss = active_adjacent_ranking_loss(q)
    torch.testing.assert_close(loss, torch.tensor(0.05))
    loss.backward()
    assert torch.isfinite(q.grad).all()
    torch.testing.assert_close(q.grad[[0, 1, 4]], torch.zeros(3))
    assert q.grad[[2, 3]].abs().sum() > 0


def test_tiny_ties_and_dense_nondecreasing_q_are_semantically_healthy():
    for q in (
        torch.tensor([0.0, 0.84, 0.841, 0.842, 1.0]),
        torch.tensor([0.0, 0.84, 0.84, 0.84, 1.0]),
        torch.tensor([0.0, 0.8, 0.81, 0.95, 1.0]),
        torch.tensor([0.0, 0.2, 0.2, 0.2, 0.2, 0.5, 0.5, 0.8, 0.8, 1.0]),
    ):
        assert active_adjacent_ranking_loss(q) == 0
        assert coarse_pairwise_ranking_loss(q) == 0
        schedule = _schedule(TrajectoryRewardScheduler(), q)
        assert schedule.semantic_order_ok
        assert schedule.phase == "semantic_recovery"
        scheduler = TrajectoryRewardScheduler(
            TrajectoryRewardSchedulerConfig(semantic_patience=1, transition_iters=1)
        )
        assert _schedule(scheduler, q).phase == "transition"
        assert _schedule(scheduler, q).phase == "trajectory_refinement"


def test_linear_pairwise_order_averages_only_violations_and_is_differentiable_zero():
    q = torch.tensor([0.0, 0.9, 0.8, 0.6, 1.0], requires_grad=True)
    loss = coarse_pairwise_ranking_loss(q)
    torch.testing.assert_close(loss, torch.tensor(0.2))
    loss.backward()
    assert torch.isfinite(q.grad).all()

    ordered_ties = torch.tensor([0.0, 0.84, 0.84, 0.84, 1.0], requires_grad=True)
    zero = coarse_pairwise_ranking_loss(ordered_ties)
    assert zero == 0 and zero.requires_grad
    zero.backward()
    torch.testing.assert_close(ordered_ties.grad, torch.zeros_like(ordered_ties))


def test_semantic_recovery_weights_only_order_and_weak_priors():
    weights = TrajectoryRewardScheduler().step(
        base_weights=BASE_WEIGHTS,
        adjacent_semantic_gaps=torch.tensor([0.8, -0.2, -0.2, 0.6]),
    ).effective_weights
    assert weights["semantic_order"] == 1.0
    assert weights["semantic_pairwise_order"] == 1.0
    assert weights["semantic_coverage"] == 0.0
    assert weights["first_order_smoothness"] == 0.0
    assert weights["coarse_gap"] == 0.0
    assert weights["second"] == 0.0
    assert weights["preserve"] == 0.05
    assert weights["ctrl"] == weights["energy"] == 0.0


def test_order_scheduler_transitions_on_ties_and_returns_on_reversal():
    scheduler = TrajectoryRewardScheduler(TrajectoryRewardSchedulerConfig(semantic_patience=2, transition_iters=1))
    tied_q = torch.tensor([0.0, 0.84, 0.84, 0.84, 1.0])
    assert _schedule(scheduler, tied_q).phase == "semantic_recovery"
    assert _schedule(scheduler, tied_q).phase == "transition"
    assert _schedule(scheduler, tied_q).phase == "trajectory_refinement"
    recovered = _schedule(scheduler, torch.tensor([0.0, 0.9, 0.8, 0.6, 1.0]))
    assert recovered.phase == "semantic_recovery"
    assert not hasattr(recovered, "coarse_not_collapsed")


def test_first_order_dreamsim_loss_detects_single_visual_jump():
    uniform, path, worst = dreamsim_first_order_loss(torch.tensor([0.25, 0.25, 0.25, 0.25]))
    single_jump, _, _ = dreamsim_first_order_loss(torch.tensor([0.7, 0.1, 0.1, 0.1]))
    two_jumps, _, _ = dreamsim_first_order_loss(torch.tensor([0.5, 0.0, 0.0, 0.5]))
    dense, dense_path, dense_worst = dreamsim_first_order_loss(
        torch.tensor([0.18, 0.17, 0.16, 0.15, 0.14, 0.10, 0.06, 0.03, 0.01])
    )
    torch.testing.assert_close(uniform, torch.tensor(0.25))
    torch.testing.assert_close(path, torch.tensor(1.0))
    torch.testing.assert_close(worst, torch.tensor(0.25))
    torch.testing.assert_close(single_jump, torch.tensor(0.7))
    torch.testing.assert_close(two_jumps, torch.tensor(0.5))
    assert single_jump > uniform and two_jumps > uniform
    torch.testing.assert_close(dense_path, torch.tensor(1.0))
    torch.testing.assert_close(dense_worst, torch.tensor(0.18))
    torch.testing.assert_close(dense, torch.tensor(0.18))


def test_default_objective_weights_separate_semantic_order_from_visual_spacing():
    weights = RewardSliderV1LossWeights()
    assert weights.semantic_order == weights.semantic_pairwise_order == 1.0
    assert weights.semantic_coverage == 0.0
    assert weights.first_order_smoothness == 1.0
    assert weights.coarse_gap == 0.0
    assert weights.second == 0.25
