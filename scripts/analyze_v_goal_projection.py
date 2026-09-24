#!/usr/bin/env python3
"""Measure a saved V_goal's signed projection onto V_edit - V_keep."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rewardflow_calibration.optimization.optimizer import projection_statistics  # noqa: E402
from rewardflow_calibration.rollout.veloedit import (  # noqa: E402
    VeloEditCompatibleRollout,
    VeloEditRolloutConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/data15/hyp/weight/FLUX.1-Kontext-dev")
    parser.add_argument("--source", default="/home/hyp/Code/VeloEdit/testdata/7.jpg")
    parser.add_argument("--prompt", default="Make him old.")
    parser.add_argument("--alpha", type=float, default=0.561)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--goal-steps", type=int, default=4)
    parser.add_argument("--max-area", type=int, default=1024 * 1024)
    parser.add_argument(
        "--residual",
        default="outputs/image_7_safe_anchor_alpha_0561_iter16/v_goal.pt",
    )
    parser.add_argument(
        "--output",
        default="outputs/image_7_safe_anchor_alpha_0561_iter16/projection_diagnostics.json",
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    config = VeloEditRolloutConfig(
        steps=args.steps,
        seed=args.seed,
        max_area=args.max_area,
    )
    rollout = VeloEditCompatibleRollout(args.model, device=device)
    image = Image.open(args.source).convert("RGB")
    prepared = rollout.prepare(image, args.prompt, config=config, seed=args.seed)
    residual = torch.load(args.residual, map_location="cpu", weights_only=True)
    if residual.shape[0] < args.goal_steps:
        raise ValueError("saved residual has fewer steps than --goal-steps")
    residual = residual[: args.goal_steps].to(device=device, dtype=rollout.pipeline.transformer.dtype)

    baseline_trace: list[dict[str, torch.Tensor]] = []
    optimized_trace: list[dict[str, torch.Tensor]] = []
    with torch.no_grad():
        rollout.rollout(
            prepared, [args.alpha], config=config, velocity_trace=baseline_trace
        )
        rollout.rollout(
            prepared,
            [args.alpha],
            config=config,
            goal_residual=residual,
            velocity_trace=optimized_trace,
        )

    baseline_directions = [entry["edit_direction"] for entry in baseline_trace[: args.goal_steps]]
    optimized_directions = [entry["edit_direction"] for entry in optimized_trace[: args.goal_steps]]
    report = {
        "alpha": args.alpha,
        "goal_steps": args.goal_steps,
        "residual_path": str(Path(args.residual).resolve()),
        "projection_direction": "masked V_edit - V_keep on low-similarity interpolation elements",
        "against_zero_residual_trajectory": projection_statistics(residual, baseline_directions),
        "against_optimized_trajectory": projection_statistics(residual, optimized_directions),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
