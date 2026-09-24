#!/usr/bin/env python3
"""Test script for reward-guided VeloEdit.

Tests the implementation on the "make him old" failure sample.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rewardflow_calibration.rollout.veloedit import (
    VeloEditCompatibleRollout,
    VeloEditRolloutConfig,
)
from rewardflow_calibration.rollout.reward_guided_veloedit import (
    RewardGuidedVeloEditRollout,
    RewardGuidanceConfig,
)
from rewardflow_calibration.rewards import (
    CompositeReward,
    CompositeRewardConfig,
    RewardWeights,
    FaceIdentityReward,
    SpatialLayoutReward,
    SemanticReward,
)
from rewardflow_calibration.utils.images import image_tensor, save_tensor_image


def parse_args():
    parser = argparse.ArgumentParser(description="Test reward-guided VeloEdit")
    parser.add_argument(
        "--model",
        default="/data15/hyp/weight/FLUX.1-Kontext-dev",
        help="Path to FLUX-Kontext model",
    )
    parser.add_argument(
        "--source",
        default="/home/hyp/Code/RewardFlow-VeloEdit-Calibration/outputs/image_7/7/baseline/alpha_0.00.png",
        help="Source image path",
    )
    parser.add_argument(
        "--prompt",
        default="Make him old.",
        help="Edit prompt",
    )
    parser.add_argument(
        "--target-prompt",
        default="an old man",
        help="Target description for semantic reward",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/test_reward_guided"),
        help="Output directory",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device to use",
    )
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[0.0, 0.25, 0.50, 0.75, 1.0],
        help="Alpha values to test",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=15,
        help="Number of denoising steps",
    )
    parser.add_argument(
        "--no-reward-guidance",
        action="store_true",
        help="Disable reward guidance (baseline comparison)",
    )
    parser.add_argument(
        "--identity-weight",
        type=float,
        default=1.0,
        help="Weight for identity reward",
    )
    parser.add_argument(
        "--layout-weight",
        type=float,
        default=1.0,
        help="Weight for layout reward",
    )
    parser.add_argument(
        "--semantic-weight",
        type=float,
        default=0.5,
        help="Weight for semantic reward",
    )
    parser.add_argument(
        "--reward-lr",
        type=float,
        default=0.005,
        help="Learning rate for reward gradient updates",
    )
    parser.add_argument(
        "--reward-alpha-threshold",
        type=float,
        default=0.4,
        help="Alpha threshold for reward activation",
    )
    parser.add_argument(
        "--reward-threshold",
        type=float,
        default=0.95,
        help="Threshold for applying corrections (0-1, higher = more strict)",
    )
    parser.add_argument(
        "--max-correction",
        type=float,
        default=0.01,
        help="Maximum correction magnitude per step",
    )
    return parser.parse_args()


def create_rewards(device: str, reward_threshold: float = 0.5) -> tuple[CompositeReward | None, list[str]]:
    """Create reward functions, returning None and warnings if some are unavailable.

    Args:
        device: Device for models
        reward_threshold: Threshold for applying corrections

    Returns:
        Tuple of (composite_reward, list of warnings)
    """
    warnings = []
    identity_reward = None
    layout_reward = None
    semantic_reward = None

    # Try to create identity reward FIRST (most important for face preservation)
    try:
        identity_reward = FaceIdentityReward(
            device=device,
            cache_dir="/data15/hyp/weight",
        )
        print("✓ FaceIdentityReward loaded (InsightFace buffalo_l)")
    except Exception as e:
        warnings.append(f"FaceIdentityReward failed: {e}")
        print(f"✗ FaceIdentityReward failed: {e}")

    # Try to create layout reward (DINOv2) - detects spatial shifts
    try:
        local_path = "/data15/hyp/weight/dinov2-large"
        import os
        if os.path.exists(local_path):
            layout_reward = SpatialLayoutReward(
                device=device,
                model_name=local_path,
                cache_dir=None,
            )
        else:
            layout_reward = SpatialLayoutReward(
                device=device,
                model_name="facebook/dinov2-vitb14",
                cache_dir="/data15/hyp/weight",
            )
        print("✓ SpatialLayoutReward loaded (DINOv2)")
    except Exception as e:
        warnings.append(f"SpatialLayoutReward failed: {e}")
        print(f"✗ SpatialLayoutReward failed: {e}")

    # Try to create semantic reward (SigLIP) - ensures text alignment
    try:
        siglip_path = "/data15/hyp/weight/reward_models/siglip-so400m-patch14-384"
        import os
        if os.path.exists(siglip_path):
            semantic_reward = SemanticReward(
                device=device,
                model_name=siglip_path,
                cache_dir=None,
            )
        else:
            semantic_reward = SemanticReward(
                device=device,
                model_name="google/siglip-so400m-patch14-384",
                cache_dir="/data15/hyp/weight",
            )
        print("✓ SemanticReward loaded (SigLIP)")
    except Exception as e:
        warnings.append(f"SemanticReward failed: {e}")
        print(f"✗ SemanticReward failed: {e}")

    # Create composite reward
    if identity_reward is None and layout_reward is None and semantic_reward is None:
        print("⚠ No reward models available, running without rewards")
        return None, warnings

    # KEY: Use IDENTITY weight as the primary correction signal
    # Semantic reward is just to ensure edit direction, not to detect failures
    weights = RewardWeights(
        identity=3.0 if identity_reward else 0.0,  # Higher weight for identity
        layout=1.0 if layout_reward else 0.0,
        semantic=0.3 if semantic_reward else 0.0,  # Lower weight - not a failure detector
        kl=0.1,
    )

    composite = CompositeReward(
        semantic_reward=semantic_reward,
        identity_reward=identity_reward,
        layout_reward=layout_reward,
        weights=weights,
        config=CompositeRewardConfig(
            use_gradient=True,
            reward_lr=0.005,
            reward_alpha_threshold=0.3,
            reward_threshold=reward_threshold,  # Use command-line threshold
            cache_source=True,
        ),
        device=device,
    )

    return composite, warnings


def run_test(args):
    """Run the test."""
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Testing Reward-Guided VeloEdit")
    print(f"{'='*60}")
    print(f"Source: {args.source}")
    print(f"Target prompt: {args.prompt}")
    print(f"Alphas: {args.alphas}")
    print(f"Reward guidance: {'disabled' if args.no_reward_guidance else 'enabled'}")
    print(f"{'='*60}\n")

    # Load source image
    source_pil = Image.open(args.source).convert("RGB")
    source = image_tensor(source_pil, device=device)

    # Create reward functions (pass threshold for identity detection)
    composite_reward, reward_warnings = create_rewards(device, reward_threshold=args.reward_threshold)
    if composite_reward is not None and not args.no_reward_guidance:
        # Register source image
        composite_reward.register_source(
            source_image=source,
            source_prompt=args.prompt,
        )
        print(f"✓ Source registered for reward computation")

    # Initialize rollout
    print(f"\nInitializing VeloEdit rollout...")
    config = VeloEditRolloutConfig(
        steps=args.steps,
        seed=args.seed,
        guidance_scale=2.5,
        first_step_align_steps=4,
        preserve_steps=4,
        edit_steps=4,
    )

    # Create rollout class
    if args.no_reward_guidance:
        rollout = VeloEditCompatibleRollout(args.model, device=device)
        prepared = rollout.prepare(source_pil, args.prompt, config=config, seed=args.seed)

        # Run baseline
        print(f"\nRunning baseline VeloEdit (no reward guidance)...")
        baseline_images = rollout.rollout(prepared, args.alphas, config=config)
        baseline_dir = args.output_dir / "baseline"
        baseline_dir.mkdir(exist_ok=True)
        for i, (alpha, img) in enumerate(zip(args.alphas, baseline_images)):
            save_tensor_image(img, baseline_dir / f"alpha_{alpha:.2f}.png")
            print(f"  Saved alpha={alpha:.2f}")

        results = {
            "mode": "baseline",
            "alphas": args.alphas,
            "source": str(args.source),
            "prompt": args.prompt,
            "warnings": reward_warnings,
        }

    else:
        rollout = RewardGuidedVeloEditRollout(args.model, device=device)
        prepared = rollout.prepare(source_pil, args.prompt, config=config, seed=args.seed)

        # Configure reward guidance
        reward_config = RewardGuidanceConfig(
            enabled=True,
            identity_weight=args.identity_weight,
            layout_weight=args.layout_weight,
            semantic_weight=args.semantic_weight,
            reward_alpha_threshold=args.reward_alpha_threshold,
            reward_lr=args.reward_lr,
            reward_step_start=4,
            reward_update_interval=2,
            reward_threshold=args.reward_threshold,
            max_correction=args.max_correction,
        )

        # Run with reward guidance
        print(f"\nRunning Reward-Guided VeloEdit...")
        if composite_reward:
            result = rollout.rollout_with_rewards(
                prepared,
                args.alphas,
                target_prompt=args.target_prompt,
                composite_reward=composite_reward,
                reward_config=reward_config,
                config=config,
            )
        else:
            # Fall back to baseline if no rewards
            print("⚠ No rewards available, falling back to baseline")
            baseline_images = rollout.rollout(prepared, args.alphas, config=config)
            result = type('Result', (), {'images': baseline_images, 'alpha': torch.tensor(args.alphas), 'reward_diagnostics': None})()

        # Save results
        reward_dir = args.output_dir / "reward_guided"
        reward_dir.mkdir(exist_ok=True)
        for i, (alpha, img) in enumerate(zip(result.alpha.tolist(), result.images)):
            save_tensor_image(img, reward_dir / f"alpha_{alpha:.2f}.png")
            print(f"  Saved alpha={alpha:.2f}")

        # Save diagnostics
        if result.reward_diagnostics:
            diagnostics_path = reward_dir / "diagnostics.json"
            with open(diagnostics_path, "w") as f:
                json.dump(result.reward_diagnostics, f, indent=2, default=str)
            print(f"\n✓ Diagnostics saved to {diagnostics_path}")

            # Print summary
            corrections = result.reward_diagnostics.get("corrections_applied", [])
            if corrections:
                print(f"\nReward corrections summary:")
                print(f"  Total corrections applied: {len(corrections)}")
                for c in corrections[:5]:  # Show first 5
                    id_sim = c.get('identity_similarity', 'N/A')
                    id_str = f"{id_sim:.4f}" if isinstance(id_sim, float) else str(id_sim)
                    print(f"    Step {c['step']}, branch {c['branch']}, "
                          f"alpha={c['alpha']:.2f}, identity_sim={id_str}")

            # Print identity similarity summary
            identity_sims = [r.get("identity_similarity", -1) for r in result.reward_diagnostics.get("reward_values", []) if r.get("identity_similarity", -1) >= 0]
            if identity_sims:
                print(f"\nIdentity similarity summary:")
                print(f"  Min: {min(identity_sims):.4f}")
                print(f"  Max: {max(identity_sims):.4f}")
                print(f"  Mean: {sum(identity_sims)/len(identity_sims):.4f}")
                print(f"  Threshold: {args.reward_threshold:.2f}")
                print(f"  Corrections triggered: {len(corrections)}")

        results = {
            "mode": "reward_guided",
            "alphas": result.alpha.tolist() if hasattr(result.alpha, 'tolist') else list(result.alpha),
            "source": str(args.source),
            "prompt": args.prompt,
            "target_prompt": args.target_prompt,
            "reward_config": {
                "identity_weight": args.identity_weight,
                "layout_weight": args.layout_weight,
                "semantic_weight": args.semantic_weight,
                "reward_lr": args.reward_lr,
                "reward_alpha_threshold": args.reward_alpha_threshold,
                "reward_threshold": args.reward_threshold,
            },
            "warnings": reward_warnings,
            "corrections_count": len(result.reward_diagnostics.get("corrections_applied", [])) if result.reward_diagnostics else 0,
            "reward_values_summary": {
                "total": len(result.reward_diagnostics.get("reward_values", [])) if result.reward_diagnostics else 0,
                "identity_similarities": [
                    r.get("identity_similarity", -1)
                    for r in result.reward_diagnostics.get("reward_values", []) if r.get("identity_similarity", -1) >= 0
                ] if result.reward_diagnostics else [],
                "min_identity_sim": min([
                    r.get("identity_similarity", -1)
                    for r in result.reward_diagnostics.get("reward_values", []) if r.get("identity_similarity", -1) >= 0
                ], default=0) if result.reward_diagnostics else 0,
            },
        }

    # Save final results
    results_path = args.output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✓ Results saved to {results_path}")
    print(f"\n{'='*60}")
    print("Test complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    args = parse_args()
    run_test(args)
