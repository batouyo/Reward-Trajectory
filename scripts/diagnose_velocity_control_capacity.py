#!/usr/bin/env python3
"""Measure native terminal progress change at fixed residual/native velocity ratios."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rewardflow_calibration.optimization.progress_objective import ProgressLossConfig, progress_control_loss
from rewardflow_calibration.optimization.progress_reward import ProgressEstimator
from rewardflow_calibration.optimization.velocity_control import scale_negative_gradient_to_native_ratio
from rewardflow_calibration.rollout.veloedit import VeloEditCompatibleRollout, VeloEditRolloutConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/data15/hyp/weight/FLUX.1-Kontext-dev")
    parser.add_argument("--source", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--target-strength", type=float, default=0.5)
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--goal-steps", type=int, default=4)
    parser.add_argument("--progress-backbone", choices=["siglip", "dino"], default="siglip")
    parser.add_argument("--progress-weight", type=float, default=1.0)
    parser.add_argument("--drift-weight", type=float, default=0.1)
    parser.add_argument("--regularization-weight", type=float, default=0.01)
    parser.add_argument("--max-area", type=int, default=1024 * 1024)
    parser.add_argument("--cache-dir", default="/data15/hyp/weight")
    parser.add_argument("--progress-model", default=None)
    parser.add_argument("--fp32-latent-accumulation", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/velocity_control_capacity"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--float32", action="store_true")
    return parser.parse_args()


def save_image(tensor: torch.Tensor, path: Path) -> None:
    pixels = tensor[0].detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    Image.fromarray(np.rint(pixels * 255).astype(np.uint8), mode="RGB").save(path)


def save_capacity_grid(paths: list[Path], labels: list[str], output: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in paths]
    width, height = max(image.width for image in images), max(image.height for image in images)
    grid = Image.new("RGB", (width * len(images), height + 30), "white")
    for index, (image, label) in enumerate(zip(images, labels)):
        tile = Image.new("RGB", (width, height + 30), "white")
        tile.paste(image.resize((width, height)), (0, 30))
        ImageDraw.Draw(tile).text((8, 8), label, fill="black")
        grid.paste(tile, (index * width, 0))
    grid.save(output)


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.target_strength <= 1.0:
        raise SystemExit("target-strength must be in [0, 1]")
    if args.steps < 1 or not 1 <= args.goal_steps <= args.steps:
        raise SystemExit("goal-steps must be in [1, steps]")
    if not args.ratios or any(not np.isfinite(ratio) or ratio < 0 for ratio in args.ratios):
        raise SystemExit("ratios must be finite and non-negative")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model_dtype = torch.float32 if args.float32 else torch.bfloat16
    config = VeloEditRolloutConfig(
        steps=args.steps,
        seed=args.seed,
        first_step_align_steps=0,
        preserve_steps=0,
        edit_steps=0,
        max_area=args.max_area,
        accumulate_latents_fp32=args.fp32_latent_accumulation,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rollout = VeloEditCompatibleRollout(args.model, device=device, dtype=model_dtype, local_files_only=True)
    source_image = Image.open(args.source).convert("RGB")
    prepared = rollout.prepare(source_image, args.prompt, config=config, seed=args.seed)
    source_pixels = rollout.pipeline.image_processor.preprocess(
        prepared.working_image, prepared.height, prepared.width
    ).to(device=device, dtype=torch.float32)
    source = ((source_pixels + 1.0) / 2.0).clamp(0, 1)
    save_image(source, args.output_dir / "source.png")

    with torch.no_grad():
        native_full = rollout.rollout_native(prepared, config=config)
    estimator = ProgressEstimator(
        args.progress_backbone, device=device, cache_dir=args.cache_dir, model_path=args.progress_model
    )
    anchors = estimator.set_anchors(source, native_full)
    residual_zero = torch.zeros(
        (args.goal_steps, *prepared.latents.shape[1:]),
        device=device, dtype=torch.float32, requires_grad=True,
    )
    baseline_trace: list[dict[str, torch.Tensor]] = []
    baseline_image = rollout.rollout_native(
        prepared, config=config, goal_residual=residual_zero, velocity_trace=baseline_trace
    )
    baseline_values = estimator(baseline_image)
    loss_config = ProgressLossConfig(
        progress_weight=args.progress_weight,
        drift_weight=args.drift_weight,
        regularization_weight=args.regularization_weight,
    )
    baseline_loss = progress_control_loss(
        baseline_values.raw_progress, args.target_strength, baseline_values.drift,
        residual_zero, loss_config,
    )
    terminal_gradient = torch.autograd.grad(baseline_loss.total, residual_zero)[0]
    if not torch.isfinite(terminal_gradient).all():
        raise FloatingPointError("terminal residual gradient contains NaN or Inf")
    native_rms = torch.stack([
        entry["native_velocity_rms"].detach().float().to(device)
        for entry in baseline_trace[:args.goal_steps]
    ])
    baseline_progress = float(baseline_values.raw_progress.detach().mean().cpu())
    direction = -terminal_gradient.detach()
    save_image(native_full, args.output_dir / "native_full.png")
    grid_paths = [args.output_dir / "native_full.png"]
    grid_labels = ["native"]
    ratio_results = []

    for ratio in args.ratios:
        residual, actual_per_step = scale_negative_gradient_to_native_ratio(
            direction, native_rms, ratio
        )
        actual_ratio = float(actual_per_step.mean().cpu())
        with torch.no_grad():
            candidate = rollout.rollout_native(
                prepared, config=config, goal_residual=residual
            )
            values = estimator(candidate)
        progress = float(values.raw_progress.detach().mean().cpu())
        drift = float(values.drift.detach().mean().cpu())
        tag = ("%g" % ratio).replace(".", "p")
        image_path = args.output_dir / ("ratio_%s.png" % tag)
        save_image(candidate, image_path)
        grid_paths.append(image_path)
        grid_labels.append("%.1f%%" % (ratio * 100))
        ratio_results.append({
            "requested_ratio": float(ratio),
            "actual_ratio": actual_ratio,
            "actual_ratio_per_step": actual_per_step.detach().cpu().tolist(),
            "final_progress": progress,
            "final_target_error": abs(progress - args.target_strength),
            "drift": drift,
            "image": str(image_path.resolve()),
        })

    capacity_grid = args.output_dir / "capacity_grid.png"
    save_capacity_grid(grid_paths, grid_labels, capacity_grid)
    report = {
        "source": str(Path(args.source).resolve()),
        "prompt": args.prompt,
        "target_strength": args.target_strength,
        "seed": args.seed,
        "steps": args.steps,
        "goal_steps": args.goal_steps,
        "progress_backbone": args.progress_backbone,
        "progress_anchors": anchors,
        "baseline_final_progress": baseline_progress,
        "baseline_loss": float(baseline_loss.total.detach().cpu()),
        "baseline_drift": float(baseline_values.drift.detach().mean().cpu()),
        "progress_gradient_rms": float(terminal_gradient.float().square().mean().sqrt().cpu()),
        "native_velocity_rms_per_step": native_rms.detach().cpu().tolist(),
        "optimizer": None,
        "ratios": ratio_results,
        "capacity_grid": str(capacity_grid.resolve()),
    }
    (args.output_dir / "capacity.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
