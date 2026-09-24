#!/usr/bin/env python3
"""Fixed-alpha VeloEdit comparison with a per-sample optimized V_goal."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rewardflow_calibration.optimization import GoalLossConfig, GoalResidualOptimizer  # noqa: E402
from rewardflow_calibration.optimization.optimizer import (  # noqa: E402
    MAX_GOAL_RESIDUAL_ITERATIONS,
)
from rewardflow_calibration.rollout.veloedit import (  # noqa: E402
    VeloEditCompatibleRollout,
    VeloEditRolloutConfig,
)
from rewardflow_calibration.optimization.reward_adapters import (  # noqa: E402
    build_default_goal_rewards,
    face_identity_similarity,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/data15/hyp/weight/FLUX.1-Kontext-dev")
    parser.add_argument(
        "--source",
        default="/home/hyp/Code/VeloEdit/testdata/7.jpg",
    )
    parser.add_argument("--prompt", default="make him old")
    parser.add_argument("--target-prompt", default="an old man")
    parser.add_argument("--alpha", type=float, default=0.843)
    parser.add_argument(
        "--preservation-reference", type=Path, default=None,
        help="safe-alpha edited image to anchor DreamSim/DINO preservation; defaults to the source image",
    )
    parser.add_argument("--preservation-anchor-alpha", type=float, default=None)
    parser.add_argument(
        "--local-continuity-reference", type=Path, default=None,
        help="previous accepted optimized alpha image used as the DreamSim continuity anchor",
    )
    parser.add_argument("--local-continuity-alpha", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--max-area", type=int, default=1024 * 1024,
                        help="maximum rollout image area; lower values reduce gradient memory/time")
    parser.add_argument("--goal-steps", type=int, default=4)
    parser.add_argument(
        "--proxy-steps", type=int, default=None,
        help="optimize against an early clean-latent prediction after N steps; final report still uses the full rollout",
    )
    parser.add_argument(
        "--restrict-negative-edit-projection", action="store_true",
        help="project out negative V_goal components along the baseline V_edit-V_keep direction",
    )
    parser.add_argument("--iterations", type=int, default=MAX_GOAL_RESIDUAL_ITERATIONS)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--edit-weight", type=float, default=1.0)
    parser.add_argument("--preservation-weight", type=float, default=1.0)
    parser.add_argument("--structure-weight", type=float, default=1.0,
                        help="weight for dense DINOv2 structure preservation inside the preservation loss")
    parser.add_argument("--regularization-weight", type=float, default=1e-2)
    parser.add_argument("--edit-score-tolerance", type=float, default=0.02)
    parser.add_argument("--cache-dir", default="/data15/hyp/weight")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/image_7_goal_residual"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--float32", action="store_true")
    return parser.parse_args()


def save_tensor_image(image: torch.Tensor, path: Path) -> None:
    pixels = image[0].detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    Image.fromarray(np.rint(pixels * 255).astype(np.uint8), mode="RGB").save(path)


def main() -> None:
    args = parse_args()
    if args.iterations > MAX_GOAL_RESIDUAL_ITERATIONS:
        raise SystemExit(
            f"--iterations cannot exceed {MAX_GOAL_RESIDUAL_ITERATIONS}"
        )
    device = torch.device(args.device)
    dtype = torch.float32 if args.float32 else torch.bfloat16
    rollout_config = VeloEditRolloutConfig(steps=args.steps, seed=args.seed, max_area=args.max_area)
    source_image = Image.open(args.source).convert("RGB")

    rollout = VeloEditCompatibleRollout(
        args.model,
        device=device,
        dtype=dtype,
        local_files_only=True,
    )
    prepared = rollout.prepare(source_image, args.prompt, config=rollout_config, seed=args.seed)
    preservation_reference = None
    if args.preservation_reference is not None:
        anchor_image = Image.open(args.preservation_reference).convert("RGB")
        anchor_image = rollout.pipeline.image_processor.resize(
            anchor_image, prepared.height, prepared.width
        )
        anchor_pixels = rollout.pipeline.image_processor.preprocess(
            anchor_image, prepared.height, prepared.width
        ).to(device=device, dtype=torch.float32)
        preservation_reference = ((anchor_pixels + 1.0) / 2.0).clamp(0, 1)
    local_continuity_reference = None
    if args.local_continuity_reference is not None:
        local_image = Image.open(args.local_continuity_reference).convert("RGB")
        local_image = rollout.pipeline.image_processor.resize(
            local_image, prepared.height, prepared.width
        )
        local_pixels = rollout.pipeline.image_processor.preprocess(
            local_image, prepared.height, prepared.width
        ).to(device=device, dtype=torch.float32)
        local_continuity_reference = ((local_pixels + 1.0) / 2.0).clamp(0, 1)
    rewards, face_metric = build_default_goal_rewards(
        device=device,
        target_prompt=args.target_prompt,
        cache_dir=args.cache_dir,
        structure_weight=args.structure_weight,
    )
    preservation_objective = rewards.preservation_loss
    if local_continuity_reference is not None:
        if preservation_reference is None:
            raise ValueError("--local-continuity-reference requires --preservation-reference as the global anchor")

        def preservation_objective(image: torch.Tensor, _source: torch.Tensor) -> torch.Tensor:
            return (
                rewards.preservation_loss.dreamsim_loss(image, local_continuity_reference)
                + args.structure_weight
                * rewards.preservation_loss.dino_loss(image, preservation_reference)
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    def save_progress(record: dict[str, float]) -> None:
        progress_path = args.output_dir / "optimization_progress.json"
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            progress = []
        progress.append(record)
        progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
        print(json.dumps(record), flush=True)

    optimizer = GoalResidualOptimizer(
        rollout,
        prepared=prepared,
        alpha=args.alpha,
        rollout_config=rollout_config,
        edit_reward=rewards.edit_score,
        preservation_loss=preservation_objective,
        loss_config=GoalLossConfig(
            edit_weight=args.edit_weight,
            preservation_weight=args.preservation_weight,
            regularization_weight=args.regularization_weight,
            edit_score_tolerance=args.edit_score_tolerance,
        ),
        goal_steps=args.goal_steps,
        learning_rate=args.learning_rate,
        iterations=args.iterations,
        proxy_steps=args.proxy_steps,
        preservation_reference=preservation_reference,
        restrict_negative_edit_projection=args.restrict_negative_edit_projection,
        progress_callback=save_progress,
    )
    result = optimizer.run()
    source_tensor = optimizer.prepared_image_tensor()
    preservation_tensor = optimizer.preservation_reference_tensor()
    face_baseline = face_identity_similarity(face_metric, preservation_tensor, result.baseline_image)
    face_optimized = face_identity_similarity(face_metric, preservation_tensor, result.optimized_image)
    preservation = rewards.preservation_loss
    dreamsim_reference = (
        preservation_tensor
        if local_continuity_reference is None
        else local_continuity_reference
    )
    with torch.no_grad():
        baseline_dreamsim = float(preservation.dreamsim_loss(result.baseline_image, dreamsim_reference))
        optimized_dreamsim = float(preservation.dreamsim_loss(result.optimized_image, dreamsim_reference))
        baseline_dino = float(preservation.dino_loss(result.baseline_image, preservation_tensor))
        optimized_dino = float(preservation.dino_loss(result.optimized_image, preservation_tensor))

    save_tensor_image(result.baseline_image, args.output_dir / "baseline_vgoal_zero.png")
    save_tensor_image(result.optimized_image, args.output_dir / "optimized_vgoal.png")
    torch.save(result.residual, args.output_dir / "v_goal.pt")
    report = {
        "source": str(Path(args.source).resolve()),
        "prompt": args.prompt,
        "target_prompt": args.target_prompt,
        "alpha_fixed": args.alpha,
        "preservation_reference": (
            None if args.preservation_reference is None else str(args.preservation_reference.resolve())
        ),
        "preservation_anchor_alpha": args.preservation_anchor_alpha,
        "local_continuity_reference": (
            None if args.local_continuity_reference is None
            else str(args.local_continuity_reference.resolve())
        ),
        "local_continuity_alpha": args.local_continuity_alpha,
        "dreamsim_reference_role": (
            "local continuity" if local_continuity_reference is not None else "preservation anchor"
        ),
        "dinov2_reference_role": "global safe anchor",
        "seed": args.seed,
        "steps": args.steps,
        "goal_steps": args.goal_steps,
        "proxy_steps": args.proxy_steps,
        "restrict_negative_edit_projection": args.restrict_negative_edit_projection,
        "iterations": args.iterations,
        "learning_rate": args.learning_rate,
        "loss_weights": {
            "edit": args.edit_weight,
            "preservation": args.preservation_weight,
            "dino_structure": args.structure_weight,
            "regularization": args.regularization_weight,
        },
        "edit_score_tolerance": args.edit_score_tolerance,
        "baseline_edit_score": result.baseline_edit_score,
        "optimized_edit_score": result.optimized_edit_score,
        "baseline_proxy_edit_score": result.baseline_proxy_edit_score,
        "optimized_proxy_edit_score": result.optimized_proxy_edit_score,
        "baseline_proxy_preservation_loss": result.baseline_proxy_preservation_loss,
        "optimized_proxy_preservation_loss": result.optimized_proxy_preservation_loss,
        "baseline_preservation_loss": result.baseline_preservation_loss,
        "optimized_preservation_loss": result.optimized_preservation_loss,
        "baseline_dreamsim_drift": baseline_dreamsim,
        "optimized_dreamsim_drift": optimized_dreamsim,
        "baseline_dinov2_drift": baseline_dino,
        "optimized_dinov2_drift": optimized_dino,
        "baseline_face_identity_similarity": face_baseline,
        "optimized_face_identity_similarity": face_optimized,
        "residual_rms": float(result.residual.square().mean().sqrt()),
        "projection_diagnostics": result.projection_diagnostics,
        "history": result.history,
    }
    (args.output_dir / "goal_residual_result.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
