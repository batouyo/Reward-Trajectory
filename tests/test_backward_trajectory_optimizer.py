import json
from types import SimpleNamespace

import pytest
import torch

from rewardflow_calibration.optimization.backward_reward import BackwardReward, BackwardRewardConfig
from rewardflow_calibration.optimization.backward_trajectory_optimizer import (
    BackwardTrajectoryConfig,
    BackwardTrajectoryOptimizer,
    build_image_space_masks,
    freeze_native_edit_mask,
    freeze_native_control_context,
)
from rewardflow_calibration.optimization.trajectory_gate import TrajectoryGateConfig
from rewardflow_calibration.optimization.velocity_control import scale_gradient_to_global_native_ratio
from rewardflow_calibration.rollout.veloedit import VeloEditRolloutConfig


def test_global_increment_scaling_preserves_timestep_relative_weights():
    direction = torch.tensor([[[1.0, 2.0]], [[10.0, 20.0]], [[100.0, 200.0]], [[1000.0, 2000.0]]])
    increment, stats = scale_gradient_to_global_native_ratio(direction, [1.0] * 4, 0.05)
    ratios = increment / direction
    torch.testing.assert_close(ratios, torch.full_like(ratios, ratios.flatten()[0]))
    assert stats["actual_global_ratio"] == pytest.approx(0.05, rel=1e-6)
    step_rms = increment.square().mean(dim=(1, 2)).sqrt()
    assert step_rms[0] < step_rms[1] < step_rms[2] < step_rms[3]


def test_image_mask_recovery_checks_flux_grid_and_token_order():
    masks = torch.zeros(4, 4, 2, dtype=torch.bool)
    masks[:, 0, :] = True
    coords = torch.stack(torch.meshgrid(torch.arange(2), torch.arange(2), indexing="ij"), dim=-1).reshape(-1, 2)
    one_block = torch.cat([torch.zeros(4, 1), coords], dim=1)
    latent_ids = torch.cat([one_block, one_block], dim=0)
    edit, keep = build_image_space_masks(
        masks, height=32, width=32, vae_scale_factor=8, latent_ids=latent_ids
    )
    assert edit.shape == keep.shape == (1, 1, 32, 32)
    torch.testing.assert_close(edit + keep, torch.ones_like(edit))
    with pytest.raises(ValueError, match="row-major"):
        bad_ids = latent_ids.clone()
        bad_ids[0], bad_ids[1] = latent_ids[1].clone(), latent_ids[0].clone()
        build_image_space_masks(masks, height=32, width=32, vae_scale_factor=8, latent_ids=bad_ids)
    with pytest.raises(ValueError, match="token count mismatch"):
        build_image_space_masks(torch.zeros(4, 5, 2), height=32, width=32, vae_scale_factor=8)


def test_native_mask_is_detached_and_frozen_once():
    trace = []
    for i in range(4):
        trace.append({
            "hard_edit_mask": torch.full((2, 3), i % 2 == 0, requires_grad=False),
            "native_velocity_rms": torch.tensor(float(i + 1)),
        })
    mask, keep, rms = freeze_native_edit_mask(trace, 4)
    trace[0]["hard_edit_mask"].zero_()
    assert mask[0].all()
    assert torch.equal(keep, ~mask)
    assert rms == [1.0, 2.0, 3.0, 4.0]
    assert not mask.requires_grad



def test_freeze_native_control_context_preserves_exact_velocity_tensor():
    trace = [
        {"hard_edit_mask": torch.tensor([[True, False], [False, True]]),
         "native_velocity_rms": torch.tensor(2.0),
         "native_velocity": torch.tensor([[[10.0, 20.0], [30.0, 40.0]]])},
        {"hard_edit_mask": torch.zeros(2, 2, dtype=torch.bool),
         "native_velocity_rms": torch.tensor(3.0),
         "native_velocity": torch.full((1, 2, 2), 5.0)},
    ]
    mask, keep, rms, native = freeze_native_control_context(trace, 2)
    assert torch.equal(native[0], torch.tensor([[10., 20.], [30., 40.]]))
    assert torch.equal(mask[1], torch.zeros_like(mask[1]))
    assert rms == [2.0, 3.0]


def test_active_rms_uses_true_boolean_mask_and_zero_mask_has_none():
    optimizer = object.__new__(BackwardTrajectoryOptimizer)
    optimizer.config = SimpleNamespace(goal_steps=2)
    optimizer.hard_edit_mask = torch.tensor([[[True, False], [False, False]],
                                             [[False, False], [False, False]]])
    optimizer.native_velocity = torch.tensor([[[3., 100.], [200., 300.]],
                                               [[4., 4.], [4., 4.]]])
    residual = torch.tensor([[[1., 999.], [888., 777.]],
                             [[5., 5.], [5., 5.]]])
    stats = optimizer._active_residual_stats(residual)
    assert stats["active_residual_rms"] == pytest.approx(1.0)
    assert stats["active_native_velocity_rms"] == pytest.approx(3.0)
    assert stats["active_residual_native_ratio"] == pytest.approx(1/3)
    assert stats["active_residual_ratio_per_step"][1] is None
    optimizer.hard_edit_mask.zero_()
    empty = optimizer._active_residual_stats(residual)
    assert empty["active_residual_native_ratio"] is None
    assert empty["active_residual_ratio_per_step"] == [None, None]


def test_verify_final_detects_both_proxy_final_order_mismatch_directions():
    class Reward:
        def semantic_score(self, image):
            return image.mean()
        def dreamsim(self, source, image):
            return float(image.mean())
    class Rollout:
        def __init__(self, value):
            self.value = value
        def rollout_native(self, *args, **kwargs):
            return torch.full((1, 3, 2, 2), self.value)
    def make(value):
        obj = object.__new__(BackwardTrajectoryOptimizer)
        obj.native_proxy = torch.full((1, 3, 2, 2), 0.8)
        obj.native_full = torch.full((1, 3, 2, 2), 0.8)
        obj.source = torch.zeros_like(obj.native_full)
        obj.reward, obj.rollout, obj.prepared, obj.rollout_config = Reward(), Rollout(value), None, None
        return obj
    state_down = SimpleNamespace(image=torch.full((1, 3, 2, 2), 0.7),
                                 semantic_score=0.7, dreamsim_to_source=0.7)
    assert make(0.9)._verify_final([SimpleNamespace(), state_down], [torch.zeros(1)])["proxy_final_mismatch"]
    state_up = SimpleNamespace(image=torch.full((1, 3, 2, 2), 0.9),
                               semantic_score=0.9, dreamsim_to_source=0.9)
    assert make(0.7)._verify_final([SimpleNamespace(), state_up], [torch.zeros(1)])["proxy_final_mismatch"]


def test_verify_final_checks_native_pair_with_no_accepted_residuals():
    class Reward:
        def semantic_score(self, image): return image.mean()
        def dreamsim(self, source, image): return float(image.mean())
    obj = object.__new__(BackwardTrajectoryOptimizer)
    obj.native_proxy = torch.full((1, 3, 2, 2), 0.4)
    obj.native_full = torch.full((1, 3, 2, 2), 0.8)
    obj.source = torch.zeros_like(obj.native_full)
    obj.reward, obj.rollout, obj.prepared, obj.rollout_config = Reward(), None, None, None
    result = obj._verify_final([SimpleNamespace(image=obj.native_proxy)], [])
    assert result["accepted_count"] == 0
    assert result["proxy_semantic_scores"] == [pytest.approx(0.4)]
    assert result["final_semantic_scores"] == [pytest.approx(0.8)]
    assert result["adjacent_semantic_order_agreement"] is None
    assert result["adjacent_source_distance_order_agreement"] is None
    assert result["proxy_final_mismatch"] is False


def _features(image):
    mean = image.mean(dim=(1, 2, 3)).clamp(0.0, 1.0)
    return torch.stack([mean, torch.sqrt((1.0 - mean.square()).clamp_min(1e-8))], dim=-1)


def _toy_reward(source, full):
    return BackwardReward(
        "target",
        device="cpu",
        config=BackwardRewardConfig(
            semantic_floor_fraction=0.75,
            semantic_anchor_min_gap=0.1,
            source_weight=1.0,
            semantic_weight=1.0,
            keep_weight=1.0,
        ),
        semantic_image_encoder=_features,
        semantic_text_encoder=lambda _: torch.tensor([[1.0, 0.0]]),
        dino_image_encoder=lambda image: torch.stack([
            image.mean(dim=(1, 2, 3)), 1.0 - image.mean(dim=(1, 2, 3))
        ], dim=-1),
        dreamsim_distance=lambda a, b: (a - b).abs().mean().reshape(1),
    )


class FakeRollout:
    device = torch.device("cpu")

    def rollout_native(self, prepared, *, config, goal_residual=None,
                       early_stop_steps=None, velocity_trace=None):
        delta = torch.zeros(()) if goal_residual is None else goal_residual.mean()
        return torch.ones(1, 3, 8, 8) * (0.8 + 5.0 * delta)


def test_mock_controller_rejects_large_step_then_accepts_smaller_steps(tmp_path):
    source = torch.full((1, 3, 8, 8), 0.2)
    full = torch.full_like(source, 0.8)
    proxy = full.clone()
    reward = _toy_reward(source, full)
    reward.set_anchors(source, full, proxy)
    prepared = SimpleNamespace(
        latents=torch.zeros(1, 2, 4),
        sigma_schedule=torch.linspace(1.0, 0.0, 16),
        height=8,
        width=8,
    )
    config = VeloEditRolloutConfig(
        steps=15, first_step_align_steps=0, preserve_steps=0, edit_steps=0
    )
    hard_mask = torch.zeros(4, 2, 4, dtype=torch.bool)
    hard_mask[:, 0, :] = True
    optimizer = BackwardTrajectoryOptimizer(
        FakeRollout(),
        prepared=prepared,
        source_image=source,
        native_full_image=full,
        native_proxy_image=proxy,
        reward=reward,
        hard_edit_mask=hard_mask,
        image_keep_mask=torch.zeros(1, 1, 8, 8),
        native_velocity_rms_per_step=[1.0] * 4,
        rollout_config=config,
        config=BackwardTrajectoryConfig(
            goal_steps=4,
            max_iterations=2,
            line_search_ratios=(0.08, 0.03, 0.01),
            max_total_residual_ratio=0.1,
        ),
        gate_config=TrajectoryGateConfig(
            semantic_tolerance=0.01,
            semantic_floor=float(reward.semantic_floor),
            min_visible_dreamsim=0.02,
            sourceward_tolerance=1e-5,
            max_second_order_deficit=0.5,
            keep_l1_tolerance=0.01,
            max_total_residual_ratio=0.1,
        ),
    )
    reward_evaluate = reward.evaluate
    calls = {"count": 0}
    def counted_evaluate(*args, **kwargs):
        calls["count"] += 1
        return reward_evaluate(*args, **kwargs)
    reward.evaluate = counted_evaluate
    summary = optimizer.run(tmp_path)
    event_rows = [json.loads(line) for line in (tmp_path / "trajectory.jsonl").read_text().splitlines()]
    trial_count = sum(row.get("event") == "trial" for row in event_rows)
    gradient_count = sum(row.get("event") == "gradient" for row in event_rows)
    assert calls["count"] == 1 + trial_count + gradient_count
    assert summary["accepted_count"] >= 2
    states = summary["accepted_states"]
    assert states[0]["dreamsim_to_source"] > states[1]["dreamsim_to_source"]
    residual_files = sorted(tmp_path.glob("accepted_residual_*.pt"))
    assert len(residual_files) >= 2
    residual = torch.load(residual_files[-1], weights_only=True)
    assert torch.count_nonzero(residual[:, 1, :]) == 0
    events = [json.loads(line) for line in (tmp_path / "trajectory.jsonl").read_text().splitlines()]
    large = next(row for row in events if row.get("event") == "trial" and row.get("requested_increment_ratio") == 0.08)
    small_accept = [row for row in events if row.get("event") == "accepted"]
    assert "semantic_floor_violation" in large["rejection_reasons"]
    assert len(small_accept) >= 2
