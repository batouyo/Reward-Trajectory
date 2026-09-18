import json

import torch

from diffusers.pipelines.rewardflow.rewardslider_v2_runner import (
    RewardSliderV2Runner,
    build_rewardslider_v2_parser,
    summarize_native_parity,
)


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

