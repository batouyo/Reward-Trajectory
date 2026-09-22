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
from rewardflow_calibration.calibration.elastic_band import (  # noqa: E402
    ElasticBandConfig,
    elastic_band_search,
)
from rewardflow_calibration.calibration.control_points import uniform_control_points  # noqa: E402
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
    parser.add_argument("--branch-refinement-depth", type=int, default=2)
    parser.add_argument("--adjacent-distance-gap-threshold", type=float, default=None)
    parser.add_argument("--elastic-target-gap", type=float, default=0.05)
    parser.add_argument("--elastic-max-points", type=int, default=10)
    parser.add_argument("--elastic-max-iterations", type=int, default=25)
    parser.add_argument("--elastic-expand-threshold", type=float, default=0.05)
    parser.add_argument("--elastic-min-alpha-spacing", type=float, default=0.01)
    parser.add_argument("--elastic-base-step-fraction", type=float, default=0.02)
    parser.add_argument("--elastic-filter-min-adjacent-gap", type=float, default=0.001)
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
                branch_refinement_depth=args.branch_refinement_depth,
                adjacent_distance_gap_threshold=args.adjacent_distance_gap_threshold,
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
        uniform_calibrated_alphas = uniform_control_points(activation["alpha_start"], betas=baseline_betas)
        uniform_calibrated_images = [rollout(alpha) for alpha in uniform_calibrated_alphas]

        def scalar_dreamsim(left: torch.Tensor, right: torch.Tensor) -> float:
            value = dreamsim.distance(left, right)
            return float(value.reshape(-1).mean().detach().cpu())

        elastic_config = ElasticBandConfig(
            target_gap=args.elastic_target_gap,
            max_points=args.elastic_max_points,
            max_iterations=args.elastic_max_iterations,
            expand_threshold=args.elastic_expand_threshold,
            min_alpha_spacing=args.elastic_min_alpha_spacing,
            base_step_fraction=args.elastic_base_step_fraction,
            filter_min_adjacent_gap=args.elastic_filter_min_adjacent_gap,
        )
        elastic_result = elastic_band_search(
            initial_control_points=uniform_calibrated_alphas,
            evaluate_image=rollout,
            distance=scalar_dreamsim,
            config=elastic_config,
        )
        calibrated_alphas = list(elastic_result.control_points)
        calibrated_images = [rollout(alpha) for alpha in calibrated_alphas]
        baseline_alphas = baseline_betas
        baseline_images = [rollout(alpha) for alpha in baseline_alphas]

        baseline_dir = sample_dir / "baseline"
        uniform_calibrated_dir = sample_dir / "uniform_calibrated"
        calibrated_dir = sample_dir / "calibrated"
        baseline_dir.mkdir(exist_ok=True)
        uniform_calibrated_dir.mkdir(exist_ok=True)
        calibrated_dir.mkdir(exist_ok=True)
        for alpha, image in zip(baseline_alphas, baseline_images):
            save_tensor_image(image, baseline_dir / f"alpha_{alpha:.2f}.png")
        for beta, alpha, image in zip(baseline_betas, uniform_calibrated_alphas, uniform_calibrated_images):
            save_tensor_image(image, uniform_calibrated_dir / f"beta_{beta:.2f}_alpha_{alpha:.3f}.png")
        for alpha, image in zip(calibrated_alphas, calibrated_images):
            beta = (alpha - activation["alpha_start"]) / (1.0 - activation["alpha_start"])
            save_tensor_image(image, calibrated_dir / f"beta_{beta:.2f}_alpha_{alpha:.3f}.png")

        baseline_nodes = torch.cat([source, *baseline_images], dim=0)
        uniform_calibrated_nodes = torch.cat([source, *uniform_calibrated_images], dim=0)
        calibrated_nodes = torch.cat([source, *calibrated_images], dim=0)
        baseline_stats = lpips.trajectory(baseline_nodes * 2 - 1)
        uniform_calibrated_stats = lpips.trajectory(uniform_calibrated_nodes * 2 - 1)
        calibrated_stats = lpips.trajectory(calibrated_nodes * 2 - 1)
        endpoint_error = (baseline_images[-1] - calibrated_images[-1]).abs()
        contact_sheet(
            [Image.open(sample_dir / "baseline" / f"alpha_{a:.2f}.png") for a in baseline_alphas]
            + [Image.open(calibrated_dir / f"beta_{(a - activation['alpha_start']) / (1.0 - activation['alpha_start']):.2f}_alpha_{a:.3f}.png") for a in calibrated_alphas],
            [f"baseline α={a:.2f}" for a in baseline_alphas]
            + [f"elastic β={(a - activation['alpha_start']) / (1.0 - activation['alpha_start']):.2f} α={a:.3f}" for a in calibrated_alphas],
            sample_dir / "baseline_vs_calibrated.png",
        )

        elastic_betas = [
            (alpha - activation["alpha_start"]) / (1.0 - activation["alpha_start"])
            for alpha in calibrated_alphas
        ]
        detail = {
            "sample": str(source_path),
            "activation": activation,
            "baseline": {
                "alphas": baseline_alphas,
                "adjacent_lpips": baseline_stats.distances.detach().cpu().tolist(),
                "trajectory_kl_to_uniform": float(baseline_stats.kl_uniform.detach().cpu()),
            },
            "calibrated": {
                "betas": elastic_betas,
                "alphas": calibrated_alphas,
                "adjacent_lpips": calibrated_stats.distances.detach().cpu().tolist(),
                "trajectory_kl_to_uniform": float(calibrated_stats.kl_uniform.detach().cpu()),
            },
            "uniform_calibrated": {
                "betas": baseline_betas,
                "alphas": uniform_calibrated_alphas,
                "adjacent_lpips": uniform_calibrated_stats.distances.detach().cpu().tolist(),
                "trajectory_kl_to_uniform": float(uniform_calibrated_stats.kl_uniform.detach().cpu()),
            },
            "elastic_band": {
                "initial_alphas": uniform_calibrated_alphas,
                "result": elastic_result.as_dict(),
                "source_dreamsim": [
                    {"alpha": alpha, "distance": scalar_dreamsim(source, image)}
                    for alpha, image in zip(calibrated_alphas, calibrated_images)
                ],
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
            "elastic_control_points": calibrated_alphas,
        })
        print(f"{source_path}: alpha_start={activation['alpha_start']:.4f}")

    result = {"config": vars(args), "samples": details, "table": rows}
    result_path = args.output_dir / "activation_result.json"
    result_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"saved {result_path}")
    return result


if __name__ == "__main__":
    run(build_parser().parse_args())
