"""THIS REWARD IS FOR INFRASTRUCTURE TESTING ONLY.
IT IS NOT A MEANINGFUL IMAGE-EDIT STRENGTH REWARD.
"""

import argparse
from pathlib import Path

import torch
from PIL import Image

from diffusers import FluxRewardFlowPipeline
from diffusers.pipelines.rewardflow import StrengthRewardContext, StrengthTrajectoryConfig


class ToyMeanIntensityStrengthReward:
    """Test only: align mean image intensity with the requested scalar."""

    def __call__(self, *, image, target_strength, context):
        progress = image.flatten(1).mean(dim=1)
        return -(progress - target_strength).square()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="RewardFlow model path or Hub id")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("strength_trajectory_output"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gamma-min", type=float, required=True)
    parser.add_argument("--gamma-max", type=float, required=True)
    parser.add_argument("--gamma-rho", type=float, required=True)
    args = parser.parse_args()

    pipe = FluxRewardFlowPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16).to("cuda")
    source = Image.open(args.source).convert("RGB")
    config = StrengthTrajectoryConfig(
        enabled=True,
        strengths=(0.2, 0.4, 0.6, 0.8),
        lambda_strength_reward=0.05,  # Arbitrary toy-example value; not a method or paper default.
        use_shared_sde_noise=True,
        gamma_min=args.gamma_min,
        gamma_max=args.gamma_max,
        gamma_rho=args.gamma_rho,
        collect_trace=True,
    )
    result = pipe(
        image=source,
        prompt=args.prompt,
        generator=torch.Generator(device="cuda").manual_seed(args.seed),
        trajectory_config=config,
        strength_reward=ToyMeanIntensityStrengthReward(),
        strength_reward_context=StrengthRewardContext(prompt=args.prompt),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for strength, image in zip(result.strengths, result.grouped_images[0]):
        image.save(args.output_dir / f"strength_{strength:.2f}.png")
    print(pipe.last_strength_trajectory_trace)


if __name__ == "__main__":
    main()
