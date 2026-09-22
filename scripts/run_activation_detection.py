#!/usr/bin/env python3
"""Run VeloEdit-compatible alpha probing and source-referenced calibration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rewardflow_calibration.calibration.activation_range import (  # noqa: E402
    ActivationRangeConfig,
    ActivationRangeDetector,
    normalize_alpha,
)
from rewardflow_calibration.metrics.dreamsim import DreamSimDistance  # noqa: E402
from rewardflow_calibration.metrics.lpips_trajectory import LPIPSDistance  # noqa: E402
from rewardflow_calibration.rollout.veloedit import (  # noqa: E402
    VeloEditCompatibleRollout,
    VeloEditRolloutConfig,
)
from rewardflow_calibration.utils.images import image_tensor, save_tensor_image  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/data15/hyp/weight/FLUX.1-Kontext-dev")
    parser.add_argument("--source", nargs="+", default=["/home/hyp/Code/VeloEdit/testdata/7.jpg"])
    parser.add_argument("--prompt", default="Make him old.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--first-step-align-steps", type=int, default=4)
    parser.add_argument("--preserve-steps", type=int, default=4)
    parser.add_argument("--edit-steps", type=int, default=4)
    parser.add_argument("--similarity-threshold", type=float, default=0.8)
    parser.add_argument("--activation-distance-threshold", type=float, default=0.001)
    parser.add_argument("--alpha-resolution", type=float, default=0.05)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser


def contact_sheet(images: list[Image.Image], labels: list[str], path: Path) -> None:
    tiles = []
    for image, label in zip(images, labels):
        tile = Image.new("RGB", (image.width, image.height + 26), "white")
        tile.paste(image, (0, 26))
        ImageDraw.Draw(tile).text((5, 5), label, fill="black")
        tiles.append(tile)
    columns = min(5, len(tiles))
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tiles[0].width, rows * tiles[0].height), "white")
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % columns) * tile.width, (index // columns) * tile.height))
    sheet.save(path)


def run(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = VeloEditRolloutConfig(
        steps=args.steps,
        seed=args.seed,
        guidance_scale=args.guidance_scale,
        first_step_align_steps=args.first_step_align_steps,
        preserve_steps=args.preserve_steps,
        edit_steps=args.edit_steps,
        similarity_threshold=args.similarity_threshold,
    )
    runner = VeloEditCompatibleRollout(args.model, device=device)
    dreamsim = DreamSimDistance(device)
    lpips = LPIPSDistance(net="vgg").to(device)
    rows = []
    details = []

    for index, source_path in enumerate(args.source):
        source_path = Path(source_path)
        sample_dir = args.output_dir / source_path.stem
        sample_dir.mkdir(parents=True, exist_ok=True)
        source_pil = Image.open(source_path).convert("RGB")
        prepared = runner.prepare(source_pil, args.prompt, config=config, seed=args.seed + index)
        source = image_tensor(prepared.working_image, device=device)

        coarse_alphas = [0.25, 0.5, 0.75, 1.0]
        coarse_images = runner.rollout(prepared, coarse_alphas, config=config)
        cache = {round(alpha, 10): coarse_images[i:i + 1] for i, alpha in enumerate(coarse_alphas)}

        def rollout(alpha: float) -> torch.Tensor:
            key = round(float(alpha), 10)
            if key not in cache:
                cache[key] = runner.rollout(prepared, [key], config=config)
            return cache[key]

        activation = ActivationRangeDetector(
            dreamsim,
            ActivationRangeConfig(
                activation_distance_threshold=args.activation_distance_threshold,
                alpha_resolution=args.alpha_resolution,
                candidate_alphas=tuple(coarse_alphas),
            ),
        ).detect(
            source_image=source,
            rollout=rollout,
            output_dir=sample_dir / "probes",
            inference_config={
                "model": args.model,
                "prompt": args.prompt,
                "seed": args.seed + index,
                "steps": args.steps,
                "height": prepared.height,
                "width": prepared.width,
                "guidance_scale": args.guidance_scale,
                "first_step_align_steps": args.first_step_align_steps,
                "preserve_steps": args.preserve_steps,
                "edit_steps": args.edit_steps,
                "similarity_threshold": args.similarity_threshold,
            },
        )

        baseline_betas = [0.0, 0.25, 0.5, 0.75, 1.0]
        calibrated_alphas = [float(normalize_alpha(beta, activation["alpha_start"])) for beta in baseline_betas]
        baseline_alphas = baseline_betas
        baseline_images = [rollout(alpha) for alpha in baseline_alphas]
        calibrated_images = [rollout(alpha) for alpha in calibrated_alphas]

        baseline_dir = sample_dir / "baseline"
        calibrated_dir = sample_dir / "calibrated"
        baseline_dir.mkdir(exist_ok=True)
        calibrated_dir.mkdir(exist_ok=True)
        for alpha, image in zip(baseline_alphas, baseline_images):
            save_tensor_image(image, baseline_dir / f"alpha_{alpha:.2f}.png")
        for beta, alpha, image in zip(baseline_betas, calibrated_alphas, calibrated_images):
            save_tensor_image(image, calibrated_dir / f"beta_{beta:.2f}_alpha_{alpha:.3f}.png")

        baseline_nodes = torch.cat([source, *baseline_images], dim=0)
        calibrated_nodes = torch.cat([source, *calibrated_images], dim=0)
        baseline_stats = lpips.trajectory(baseline_nodes * 2 - 1)
        calibrated_stats = lpips.trajectory(calibrated_nodes * 2 - 1)
        endpoint_error = (baseline_images[-1] - calibrated_images[-1]).abs()
        contact_sheet(
            [Image.open(sample_dir / "baseline" / f"alpha_{a:.2f}.png") for a in baseline_alphas]
            + [Image.open(calibrated_dir / f"beta_{b:.2f}_alpha_{a:.3f}.png") for b, a in zip(baseline_betas, calibrated_alphas)],
            [f"baseline α={a:.2f}" for a in baseline_alphas]
            + [f"calibrated β={b:.2f} α={a:.3f}" for b, a in zip(baseline_betas, calibrated_alphas)],
            sample_dir / "baseline_vs_calibrated.png",
        )

        detail = {
            "sample": str(source_path),
            "activation": activation,
            "baseline": {
                "alphas": baseline_alphas,
                "adjacent_lpips": baseline_stats.distances.detach().cpu().tolist(),
                "trajectory_kl_to_uniform": float(baseline_stats.kl_uniform.detach().cpu()),
            },
            "calibrated": {
                "betas": baseline_betas,
                "alphas": calibrated_alphas,
                "adjacent_lpips": calibrated_stats.distances.detach().cpu().tolist(),
                "trajectory_kl_to_uniform": float(calibrated_stats.kl_uniform.detach().cpu()),
            },
            "same_final_alpha_1": {
                "mean_absolute_image_difference": float(endpoint_error.mean().detach().cpu()),
                "max_absolute_image_difference": float(endpoint_error.max().detach().cpu()),
                "pass": bool(endpoint_error.max().item() == 0.0),
            },
        }
        details.append(detail)
        rows.append({
            "sample": str(source_path),
            "alpha_start": activation["alpha_start"],
            "distance_curve": activation["distance_curve"],
            "baseline_trajectory_kl": detail["baseline"]["trajectory_kl_to_uniform"],
            "calibrated_trajectory_kl": detail["calibrated"]["trajectory_kl_to_uniform"],
        })
        print(f"{source_path}: alpha_start={activation['alpha_start']:.4f}")

    result = {"config": vars(args), "samples": details, "table": rows}
    result_path = args.output_dir / "activation_result.json"
    result_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"saved {result_path}")
    return result


if __name__ == "__main__":
    run(build_parser().parse_args())
