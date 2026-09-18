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
)
from diffusers.pipelines.rewardflow.rewardslider_v2_alpha import OrderedAlphaParameterization
from diffusers.pipelines.rewardflow.rewardslider_v2_scheduler import RewardSliderV2Scheduler


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

    )
    assert alpha.interval_logits.grad is None
    assert all(goal.grad is not None and goal.grad.abs().sum() > 0 for goal in goals)
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
