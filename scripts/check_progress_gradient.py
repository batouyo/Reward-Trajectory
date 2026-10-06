#!/usr/bin/env python3
"""Check that the native progress-control loss has the expected residual gradient."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rewardflow_calibration.optimization.progress_objective import ProgressLossConfig, progress_control_loss
from rewardflow_calibration.optimization.progress_reward import ProgressEstimator
from rewardflow_calibration.rollout.veloedit import VeloEditCompatibleRollout, VeloEditRolloutConfig


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/data15/hyp/weight/FLUX.1-Kontext-dev")
    parser.add_argument("--source", default="/home/hyp/Code/VeloEdit/testdata/7.jpg")
    parser.add_argument("--prompt", default="make him old")
    parser.add_argument("--target-strength", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--goal-steps", type=int, default=4)
    parser.add_argument("--proxy-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--epsilon", type=float, default=1e-2)
    parser.add_argument("--progress-backbone", choices=["siglip", "dino"], default="siglip")
    parser.add_argument("--cache-dir", default="/data15/hyp/weight")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=Path("outputs/reward_level_control/gradient_sanity.json"))
    parser.add_argument("--float32", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = VeloEditRolloutConfig(
        steps=args.steps, seed=args.seed,
        first_step_align_steps=0, preserve_steps=0, edit_steps=0,
    )
    device = torch.device(args.device)
    dtype = torch.float32 if args.float32 else torch.bfloat16
    rollout = VeloEditCompatibleRollout(args.model, device=device, dtype=dtype)
    prepared = rollout.prepare(
        Image.open(args.source).convert("RGB"), args.prompt, config=config, seed=args.seed
    )
    source_pixels = rollout.pipeline.image_processor.preprocess(
        prepared.working_image, prepared.height, prepared.width
    ).to(device=device, dtype=torch.float32)
    source = ((source_pixels + 1) / 2).clamp(0, 1)
    with torch.no_grad():
        native_full = rollout.rollout_native(prepared, config=config).detach()
    estimator = ProgressEstimator(
        args.progress_backbone, device=device, cache_dir=args.cache_dir
    )
    anchors = estimator.set_anchors(source, native_full)
    with torch.no_grad():
        initial_image = rollout.rollout_native(
            prepared, config=config, early_stop_steps=args.proxy_steps
        )
    residual = torch.zeros(
        (args.goal_steps, *prepared.latents.shape[1:]),
        device=device, dtype=torch.float32, requires_grad=True,
    )
    proxy = rollout.rollout_native(
        prepared, config=config, goal_residual=residual,
        early_stop_steps=args.proxy_steps,
    )
    projected = estimator(proxy)
    values = progress_control_loss(
        projected.raw_progress, args.target_strength, projected.drift,
        residual, ProgressLossConfig(),
    )
    gradient = torch.autograd.grad(values.total, residual)[0]
    if gradient is None:
        raise RuntimeError("gradient sanity precondition failed: gradient is missing")
    gradient_norm = float(gradient.norm().detach().cpu())
    if not torch.isfinite(gradient).all() or gradient_norm <= 0:
        raise RuntimeError("gradient sanity precondition failed: gradient non-finite or zero")
    direction = gradient / gradient.norm().clamp_min(1e-12)
    per_step_norm = gradient.float().flatten(1).norm(dim=1).detach().cpu().tolist()

    def evaluate(delta):
        with torch.no_grad():
            image = rollout.rollout_native(
                prepared, config=config, goal_residual=delta,
                early_stop_steps=args.proxy_steps,
            )
            p = estimator(image)
            loss = progress_control_loss(
                p.raw_progress, args.target_strength, p.drift,
                delta, ProgressLossConfig(),
            )
        return float(p.raw_progress.mean().cpu()), float(loss.total.cpu())

    initial_progress = float(projected.raw_progress.detach().mean().cpu())
    initial_loss = float(values.total.detach().cpu())
    negative_progress, negative_loss = evaluate((-args.epsilon * direction).detach())
    positive_progress, positive_loss = evaluate((args.epsilon * direction).detach())
    expect_negative_progress = (
        negative_progress < initial_progress if initial_progress > args.target_strength
        else negative_progress > initial_progress if initial_progress < args.target_strength
        else True
    )
    sanity_failed = negative_loss > initial_loss + 1e-7 or not expect_negative_progress
    report = {
        "source": str(Path(args.source).resolve()),
        "prompt": args.prompt,
        "target_strength": args.target_strength,
        "initial_progress": initial_progress,
        "target_progress": args.target_strength,
        "initial_loss": initial_loss,
        "negative_gradient_step_progress": negative_progress,
        "negative_gradient_step_loss": negative_loss,
        "positive_gradient_step_progress": positive_progress,
        "positive_gradient_step_loss": positive_loss,
        "gradient_norm": gradient_norm,
        "per_step_gradient_norm": per_step_norm,
        "epsilon": args.epsilon,
        "expected_negative_step_moves_toward_target": expect_negative_progress,
        "sanity_check_failed": sanity_failed,
        "anchor_diagnostics": anchors,
        "initial_proxy_image_shape": list(initial_image.shape),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
