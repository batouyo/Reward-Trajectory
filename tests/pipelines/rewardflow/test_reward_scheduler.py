import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch

from diffusers.pipelines.rewardflow.coupled_terminal_control import (
    adjacent_ranking_loss,
    coarse_pairwise_ranking_loss,
    coarse_anchor_indices,
    per_interval_semantic_coverage_loss,
)
from diffusers.pipelines.rewardflow.reward_scheduler import (
    TrajectoryRewardScheduler,
    TrajectoryRewardSchedulerConfig,
)


BASE_WEIGHTS = {
    "semantic_order": 1.0,
    "semantic_coverage": 1.0,
    "semantic_pairwise_order": 1.0,
    "fine_jump": 1.0,
    "coarse_gap": 1.0,
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
        coarse_semantic_gaps=q[1:] - q[:-1],
    )


def test_per_interval_coverage_penalizes_all_collapsed_middle_intervals():
    q = torch.tensor([0.0, 0.8, 0.8, 0.8, 1.0], requires_grad=True)
    loss, collapse, jump, gaps = per_interval_semantic_coverage_loss(q)
    assert collapse > 0 and jump > 0
    assert (gaps < 0.05).sum().item() == 2
    loss.backward()
    assert torch.isfinite(q.grad).all()
    # Both collapsed intervals are in the mean, and their outside endpoints
    # receive corrective gradients. The shared middle node may cancel under the
    # requested symmetric summed formula, so it is not used as a false gate.
    assert q.grad[1].abs() > 0
    assert q.grad[3].abs() > 0


def test_reversed_semantic_order_has_finite_corrective_gradients():
    q = torch.tensor([0.0, 0.8, 0.6, 0.4, 1.0], requires_grad=True)
    order = adjacent_ranking_loss(q, margin=0.0)
    assert order > 0
    order.backward()
    assert torch.isfinite(q.grad).all()
    assert q.grad[1].abs() > 0 and q.grad[3].abs() > 0


def test_healthy_nonuniform_trajectory_needs_no_semantic_penalty():
    q = torch.tensor([0.0, 0.15, 0.42, 0.78, 1.0])
    coverage, collapse, jump, _ = per_interval_semantic_coverage_loss(q)
    assert adjacent_ranking_loss(q, margin=0.0) == 0
    assert coverage == 0 and collapse == 0 and jump == 0


def test_dense_trajectory_allows_small_fine_gaps_when_coarse_anchors_are_healthy():
    q = torch.tensor([0.0, 0.01, 0.08, 0.09, 0.30, 0.31, 0.32, 0.65, 0.66, 1.0])
    anchors = coarse_anchor_indices(q.numel())
    coverage, _, _, coarse_gaps = per_interval_semantic_coverage_loss(q, anchor_indices=anchors)
    assert adjacent_ranking_loss(q, margin=0.0) == 0
    assert coverage == 0
    assert torch.all(coarse_gaps >= 0.05)


def test_scheduler_recovery_disables_geometry_and_keeps_semantics_active():
    scheduler = TrajectoryRewardScheduler()
    schedule = _schedule(scheduler, torch.tensor([0.0, 0.8, 0.6, 0.4, 1.0]))
    assert schedule.phase == "semantic_recovery"
    assert schedule.effective_weights["semantic_order"] == 1.0
    assert schedule.effective_weights["semantic_coverage"] == 1.0
    assert schedule.effective_weights["fine_jump"] == 0.0
    assert schedule.effective_weights["coarse_gap"] == 0.0
    assert schedule.effective_weights["second"] == 0.0
    assert schedule.effective_weights["preserve"] == 0.05


def test_scheduler_transitions_then_refines_after_healthy_patience():
    scheduler = TrajectoryRewardScheduler(TrajectoryRewardSchedulerConfig(semantic_patience=2, transition_iters=2))
    healthy = torch.tensor([0.0, 0.15, 0.42, 0.78, 1.0])
    assert _schedule(scheduler, healthy).phase == "semantic_recovery"
    first_transition = _schedule(scheduler, healthy)
    assert first_transition.phase == "transition"
    assert first_transition.effective_weights["second"] == 0.125
    second_transition = _schedule(scheduler, healthy)
    assert second_transition.phase == "transition"
    assert second_transition.effective_weights["second"] == 0.25
    refined = _schedule(scheduler, healthy)
    assert refined.phase == "trajectory_refinement"
    assert refined.effective_weights == BASE_WEIGHTS


def test_scheduler_falls_back_from_refinement_on_semantic_reversal():
    scheduler = TrajectoryRewardScheduler(TrajectoryRewardSchedulerConfig(semantic_patience=1, transition_iters=1))
    healthy = torch.tensor([0.0, 0.15, 0.42, 0.78, 1.0])
    _schedule(scheduler, healthy)
    _schedule(scheduler, healthy)
    assert scheduler.phase == "trajectory_refinement"
    recovered = _schedule(scheduler, torch.tensor([0.0, 0.8, 0.6, 0.4, 1.0]))
    assert recovered.phase == "semantic_recovery"
    assert recovered.consecutive_semantic_ok == 0


def test_scheduler_allows_large_first_clip_gap_to_transition():
    scheduler = TrajectoryRewardScheduler(TrajectoryRewardSchedulerConfig(semantic_patience=2))
    q = torch.tensor([0.0, 0.84, 0.87, 0.92, 1.0])
    first = _schedule(scheduler, q)
    second = _schedule(scheduler, q)
    assert first.semantic_order_ok and first.coarse_not_collapsed
    assert second.phase == "transition"
    assert second.max_coarse_semantic_gap > 0.55


def test_scheduler_keeps_genuine_collapse_in_recovery():
    schedule = _schedule(TrajectoryRewardScheduler(), torch.tensor([0.0, 0.8, 0.8, 0.8, 1.0]))
    assert schedule.phase == "semantic_recovery"
    assert not schedule.coarse_not_collapsed
    assert schedule.near_zero_coarse_intervals == 2


def test_pairwise_order_penalizes_reversed_anchors_with_endpoint_gradients():
    q = torch.tensor([0.0, 0.9, 0.8, 0.6, 1.0], requires_grad=True)
    loss = coarse_pairwise_ranking_loss(q)
    assert loss > 0
    loss.backward()
    assert torch.isfinite(q.grad).all()
    assert q.grad[1] > 0 and q.grad[3] < 0


def test_pairwise_order_is_zero_for_healthy_uneven_trajectory():
    assert coarse_pairwise_ranking_loss(torch.tensor([0.0, 0.8, 0.85, 0.92, 1.0])) == 0


def test_dense_pairwise_order_does_not_impose_equal_spacing():
    q = torch.tensor([0.0, 0.01, 0.08, 0.09, 0.30, 0.31, 0.32, 0.65, 0.66, 1.0])
    assert adjacent_ranking_loss(q, margin=0.0) == 0
    assert coarse_pairwise_ranking_loss(q, anchor_indices=coarse_anchor_indices(q.numel())) == 0


def test_scheduler_log_contains_effective_weights_and_collapse_health():
    row = _schedule(TrajectoryRewardScheduler(), torch.tensor([0.0, 0.84, 0.87, 0.92, 1.0])).as_log_dict()
    assert {"effective_weights", "collapse_ok", "scheduler_phase"} <= row.keys()


def test_gradient_audit_logs_raw_effective_and_weighted_norms():
    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location("reward_runner", root / "scripts/run_coupled_reward_trajectory_v1.py")
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    images = torch.ones((1, 3, 2, 2), requires_grad=True)
    names = (
        "semantic_order",
        "semantic_pairwise_order",
        "semantic_coverage",
        "fine_jump",
        "coarse_gap",
        "second",
        "preserve",
    )
    result = SimpleNamespace(components={name: (index + 1) * images.square().mean() for index, name in enumerate(names)})
    weights = {name: float(index) / 10 for index, name in enumerate(names)}
    rows = runner._reward_gradient_diagnostics(
        images, result, iteration=1, reason="unit_test", schedule=None, effective_weights=weights
    )["image_reward_gradients"]
    for name in names:
        assert {"raw_gradient_l2", "effective_weight", "weighted_gradient_l2"} <= rows[name].keys()
        assert rows[name]["weighted_gradient_l2"] == abs(weights[name]) * rows[name]["raw_gradient_l2"]
