import pytest
import torch

from rewardflow_calibration.optimization.trajectory_gate import (
    GateDecision,
    TrajectoryGateConfig,
    TrajectoryState,
    assess_candidate,
    stop_reason_for_rejections,
)


def _state(name, image, sem, srcdist, keep=0.0, residual=0.0):
    return TrajectoryState(name, torch.tensor([[[[image]]]]), sem, srcdist, keep, residual)


def _distance(a, b):
    return (a - b).abs().mean()


def _config(**kwargs):
    values = dict(
        semantic_tolerance=0.01,
        semantic_floor=0.5,
        min_visible_dreamsim=0.05,
        sourceward_tolerance=0.001,
        max_second_order_deficit=0.25,
        keep_l1_tolerance=0.02,
        max_total_residual_ratio=0.1,
    )
    values.update(kwargs)
    return TrajectoryGateConfig(**values)


def test_semantic_monotonicity_allows_small_jitter_but_rejects_large_reversal():
    previous = _state("p", 0.8, 0.8, 0.5)
    accepted = [previous]
    small = _state("small", 0.7, 0.805, 0.4)
    large = _state("large", 0.6, 0.83, 0.3)
    assert assess_candidate(previous, small, accepted, _distance, _config()).accepted
    decision = assess_candidate(previous, large, accepted, _distance, _config())
    assert "semantic_wrong_direction" in decision.reasons


def test_semantic_floor_violation_is_identified():
    previous = _state("p", 0.8, 0.8, 0.5)
    candidate = _state("c", 0.6, 0.49, 0.3)
    decision = assess_candidate(previous, candidate, [previous], _distance, _config())
    assert "semantic_floor_violation" in decision.reasons


def test_visible_change_gate_identifies_perceptual_stall():
    previous = _state("p", 0.8, 0.8, 0.5)
    candidate = _state("c", 0.799, 0.79, 0.4)
    decision = assess_candidate(previous, candidate, [previous], _distance, _config())
    assert "perceptual_stall" in decision.reasons


def test_source_distance_must_decrease():
    previous = _state("p", 0.8, 0.8, 0.5)
    candidate = _state("c", 0.6, 0.7, 0.51)
    decision = assess_candidate(previous, candidate, [previous], _distance, _config())
    assert "not_sourceward" in decision.reasons


def test_second_order_deficit_formula_and_gate():
    a = _state("a", 0.0, 0.9, 0.9)
    b = _state("b", 1.0, 0.8, 0.8)
    c = _state("c", 2.0, 0.7, 0.7)
    distances = {(0.0, 1.0): 0.2, (1.0, 2.0): 0.2, (0.0, 2.0): 0.1}
    def custom(x, y):
        key = (float(x.item()), float(y.item()))
        reverse = (key[1], key[0])
        return torch.tensor(distances.get(key, distances.get(reverse, 0.2)))
    decision = assess_candidate(b, c, [a, b], custom, _config(min_visible_dreamsim=0.01))
    assert decision.metrics["second_order_deficit"] == pytest.approx(3.0)
    assert "second_order_jump" in decision.reasons


def test_keep_region_drift_is_rejected():
    previous = _state("p", 0.8, 0.8, 0.5, keep=0.1)
    candidate = _state("c", 0.6, 0.7, 0.4, keep=0.2)
    decision = assess_candidate(previous, candidate, [previous], _distance, _config())
    assert "keep_region_drift" in decision.reasons


def test_all_floor_only_trials_mean_semantic_boundary_not_failure():
    decisions = [
        GateDecision(False, ["semantic_floor_violation"], {}),
        GateDecision(False, ["semantic_floor_violation"], {}),
    ]
    assert stop_reason_for_rejections(decisions) == "semantic_boundary_reached"
