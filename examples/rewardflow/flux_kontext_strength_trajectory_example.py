"""Minimal FLUX.1-Kontext coupled-strength infrastructure example."""

import argparse
from pathlib import Path

import torch

from diffusers import FluxKontextStrengthTrajectoryPipeline
from diffusers.pipelines.rewardflow import StrengthRewardContext, StrengthTrajectoryConfig
from diffusers.utils import load_image


class ToyMeanIntensityReward:
    """Infrastructure-only differentiable reward; not a semantic edit-strength reward."""

    def __call__(self, *, image, target_strength, context):
        del context
        return -(image.float().mean(dim=(1, 2, 3)) - target_strength).square()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Local FLUX.1-Kontext checkpoint or Hub model id.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--strengths", nargs="+", type=float, default=[0.2, 0.4, 0.6, 0.8])
    parser.add_argument("--gamma-min", type=float, required=True)
    parser.add_argument("--gamma-max", type=float, required=True)
    parser.add_argument("--gamma-rho", type=float, required=True)
    parser.add_argument("--branch-chunk-size", type=int)
    parser.add_argument("--no-strength-reward", action="store_true")
    parser.add_argument("--no-shared-sde", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="kontext_strength_outputs")
    return parser.parse_args()


def main():
    args = parse_args()
    print("THIS REWARD IS FOR INFRASTRUCTURE TESTING ONLY. IT IS NOT A SEMANTIC EDIT-STRENGTH REWARD.")

    pipe = FluxKontextStrengthTrajectoryPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    source = load_image(args.source).convert("RGB")
    use_reward = not args.no_strength_reward
    config = StrengthTrajectoryConfig(
        enabled=True,
        strengths=tuple(args.strengths),
        lambda_strength_reward=0.05 if use_reward else 0.0,
        use_shared_sde_noise=not args.no_shared_sde,
        gamma_min=args.gamma_min,
        gamma_max=args.gamma_max,
        gamma_rho=args.gamma_rho,
        branch_chunk_size=args.branch_chunk_size,
        collect_trace=True,
    )
    result = pipe(
        image=source,
        prompt=args.prompt,
        generator=torch.Generator(device="cuda").manual_seed(args.seed),
        trajectory_config=config,
        strength_reward=ToyMeanIntensityReward() if use_reward else None,
        strength_reward_context=StrengthRewardContext(prompt=args.prompt),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for strength, image in zip(result.strengths, result.grouped_images[0]):
        image.save(output_dir / f"strength_{strength:.3f}.png")
    print(f"saved {len(result.images)} images to {output_dir}")


if __name__ == "__main__":
    main()
