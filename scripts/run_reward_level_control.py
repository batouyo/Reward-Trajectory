#!/usr/bin/env python3
"""Run the independent native-Kontext Reward-Level Velocity Control MVP."""

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

from rewardflow_calibration.optimization.progress_objective import ProgressLossConfig
from rewardflow_calibration.optimization.progress_optimizer import ProgressResidualOptimizer
from rewardflow_calibration.optimization.progress_reward import ProgressEstimator
from rewardflow_calibration.rollout.veloedit import VeloEditCompatibleRollout, VeloEditRolloutConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/data15/hyp/weight/FLUX.1-Kontext-dev")
    parser.add_argument("--source", default="/home/hyp/Code/VeloEdit/testdata/7.jpg")
    parser.add_argument("--prompt", default="make him old")
    parser.add_argument("--strengths", nargs="+", type=float, default=[0.25, 0.50, 0.75])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--goal-steps", type=int, default=4)
    parser.add_argument("--proxy-steps", type=int, default=4)
    parser.add_argument("--reward-mode", choices=["proxy", "final"], default="final")
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--progress-backbone", choices=["siglip", "dino"], default="siglip")
    parser.add_argument("--progress-weight", type=float, default=1.0)
    parser.add_argument("--drift-weight", type=float, default=0.1)
    parser.add_argument("--regularization-weight", type=float, default=0.01)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--max-area", type=int, default=1024 * 1024)
    parser.add_argument("--cache-dir", default="/data15/hyp/weight")
    parser.add_argument("--progress-model", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reward_level_control"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--float32", action="store_true")
    return parser.parse_args()


def save_image(tensor: torch.Tensor, path: Path) -> None:
    value = tensor[0].detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    Image.fromarray(np.rint(value * 255).astype(np.uint8), mode="RGB").save(path)


def save_grid(paths: list[Path], labels: list[str], output: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in paths]
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    tiles = []
    for image, label in zip(images, labels):
        image = image.resize((width, height))
        tile = Image.new("RGB", (width, height + 30), "white")
        tile.paste(image, (0, 30))
        ImageDraw.Draw(tile).text((8, 8), label, fill="black")
        tiles.append(tile)
    grid = Image.new("RGB", (width * len(tiles), height + 30), "white")
    for index, tile in enumerate(tiles):
        grid.paste(tile, (index * width, 0))
    grid.save(output)


def infer_failure(summary: dict[str, object]) -> str:
    signal = summary.get("progress_signal_diagnostic", {})
    if isinstance(signal, dict) and signal.get("dynamic_range", 1.0) < 0.05:
        return "progress_estimator"
    sanity = summary.get("gradient_sanity_check", {})
    if isinstance(sanity, dict) and sanity.get("sanity_check_failed"):
        return "gradient_sign_or_graph"
    rows = summary.get("strength_results", [])
    if isinstance(rows, list) and rows:
        gradients = [float(row.get("first_gradient_norm", 0.0)) for row in rows]
        if gradients and max(gradients) < 1e-8:
            return "gradient_path"
        if any(row.get("gradient_sanity_check_failed") for row in rows):
            return "gradient_sign_or_graph"
        if any(row.get("proxy_terminal_mismatch", 0.0) > 0.2 for row in rows):
            return "proxy_terminal_mismatch"
        if any(row.get("final_drift", 0.0) > 4.0 for row in rows):
            return "reward_hacking_or_insufficient_drift_constraint"
        if any(row.get("residual_native_ratio", 0.0) > 10.0 for row in rows):
            return "native_dynamics_too_stiff_or_wrong_control_window"
        if any(row.get("veloedit_parallel_fraction", 0.0) > 0.95 for row in rows):
            return "degenerates_to_veloedit_rescaling"
        progress = [row.get("optimized_final_progress") for row in rows if row.get("optimized_final_progress") is not None]
        targets = [row.get("requested_strength") for row in rows if row.get("optimized_final_progress") is not None]
        if len(progress) > 1 and any(b <= a for a, b in zip(progress, progress[1:])):
            return "level_tracking_failure"
        if len(progress) < len(targets):
            return "control_capacity"
    return "unclear"


def main() -> None:
    args = parse_args()
    if args.steps < 1 or not 1 <= args.goal_steps <= args.steps:
        raise SystemExit("steps must be positive and goal-steps must be in [1, steps]")
    if not args.goal_steps <= args.proxy_steps <= args.steps:
        raise SystemExit("proxy-steps must cover goal-steps and not exceed steps")
    if args.iterations > 16:
        raise SystemExit("iterations cannot exceed 16")
    if any(not 0 <= value <= 1 for value in args.strengths):
        raise SystemExit("strengths must be in [0, 1]")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = torch.float32 if args.float32 else torch.bfloat16
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = VeloEditRolloutConfig(
        steps=args.steps,
        seed=args.seed,
        guidance_scale=args.guidance_scale,
        first_step_align_steps=0,
        preserve_steps=0,
        edit_steps=0,
        max_area=args.max_area,
    )
    source_image = Image.open(args.source).convert("RGB")
    rollout = VeloEditCompatibleRollout(
        args.model, device=device, dtype=dtype, local_files_only=True
    )
    prepared = rollout.prepare(source_image, args.prompt, config=config, seed=args.seed)
    prepared_source_pixels = rollout.pipeline.image_processor.preprocess(
        prepared.working_image, prepared.height, prepared.width
    ).to(device=device, dtype=torch.float32)
    source_tensor = ((prepared_source_pixels + 1.0) / 2.0).clamp(0, 1)
    save_image(source_tensor, args.output_dir / "source.png")

    with torch.no_grad():
        native_full = rollout.rollout_native(prepared, config=config).detach()
        zero_residual = torch.zeros(
            (args.goal_steps, *prepared.latents.shape[1:]),
            device=rollout.device,
            dtype=torch.float32,
        )
        zero_residual_full = rollout.rollout_native(
            prepared, config=config, goal_residual=zero_residual
        ).detach()
    zero_difference = (native_full - zero_residual_full).abs()
    zero_residual_sanity = {
        "max_abs_difference": float(zero_difference.max().cpu()),
        "mean_abs_difference": float(zero_difference.mean().cpu()),
        "pass": bool(torch.equal(native_full, zero_residual_full)),
    }
    save_image(native_full, args.output_dir / "native_full.png")

    estimator = ProgressEstimator(
        args.progress_backbone,
        device=device,
        cache_dir=args.cache_dir,
        model_path=args.progress_model,
    )
    anchor_diagnostics = estimator.set_anchors(source_tensor, native_full)
    if anchor_diagnostics["source_progress_error"] > 1e-4 or anchor_diagnostics["target_progress_error"] > 1e-3:
        raise RuntimeError(f"progress anchor self-check failed: {anchor_diagnostics}")

    result_rows: list[dict[str, object]] = []
    image_paths: dict[float, Path] = {0.0: args.output_dir / "source.png", 1.0: args.output_dir / "native_full.png"}
    for strength in args.strengths:
        strength = float(strength)
        if strength == 0.0 or strength == 1.0:
            continue
        tag = f"{strength:.2f}"
        strength_dir = args.output_dir / f"strength_{tag}"
        strength_dir.mkdir(parents=True, exist_ok=True)

        def save_progress(record: dict[str, object]) -> None:
            progress_path = strength_dir / "optimization_progress.json"
            try:
                rows = json.loads(progress_path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                rows = []
            rows.append(record)
            progress_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

        optimizer = ProgressResidualOptimizer(
            rollout,
            prepared=prepared,
            native_full_image=native_full,
            progress_estimator=estimator,
            target_strength=strength,
            rollout_config=config,
            loss_config=ProgressLossConfig(
                progress_weight=args.progress_weight,
                drift_weight=args.drift_weight,
                regularization_weight=args.regularization_weight,
            ),
            goal_steps=args.goal_steps,
            proxy_steps=args.proxy_steps,
            learning_rate=args.learning_rate,
            iterations=args.iterations,
            clip_grad_norm=args.clip_grad_norm,
            reward_mode=args.reward_mode,
            progress_callback=save_progress,
        )
        result = optimizer.run()
        proxy_path = strength_dir / f"controlled_s_{tag}_proxy.png"
        final_path = args.output_dir / f"controlled_s_{tag}.png"
        save_image(result.initial_proxy_image, strength_dir / "initial_proxy.png")
        save_image(result.optimized_proxy_image, proxy_path)
        save_image(result.optimized_final_image, final_path)
        torch.save(result.residual, strength_dir / "delta_v.pt")
        image_paths[strength] = final_path

        first_grad = (
            result.history[0]["gradient_norm_total"] if result.history else 0.0
        )
        projection = result.projection_diagnostics
        projection_parallel = float(projection.get("parallel_energy_fraction_of_residual", 0.0))
        projection_orthogonal = float(projection.get("orthogonal_energy_fraction_of_residual", 1.0))
        proxy_final_mismatch = abs(
            float(result.optimized_proxy["raw_progress"])
            - float(result.optimized_final["raw_progress"])
        )
        diagnostics = {
            "requested_strength": strength,
            "anchor_norm": anchor_diagnostics["anchor_norm"],
            "source_progress": anchor_diagnostics["source_progress"],
            "target_progress": anchor_diagnostics["target_progress"],
            "initial_proxy_progress": result.initial_proxy["raw_progress"],
            "optimized_proxy_progress": result.optimized_proxy["raw_progress"],
            "optimized_final_progress": result.optimized_final["raw_progress"],
            "proxy_target_error": abs(float(result.optimized_proxy["raw_progress"]) - strength),
            "final_target_error": abs(float(result.optimized_final["raw_progress"]) - strength),
            "initial_drift": result.initial_proxy["drift"],
            "optimized_proxy_drift": result.optimized_proxy["drift"],
            "optimized_final_drift": result.optimized_final["drift"],
            "initial_proxy_diagnostics": result.initial_proxy,
            "optimized_proxy_diagnostics": result.optimized_proxy,
            "optimized_final_diagnostics": result.optimized_final,
            "proxy_final_progress_difference": proxy_final_mismatch,
            "first_gradient_norm": first_grad,
            "optimization_history": result.history,
            "velocity_diagnostics": result.velocity_diagnostics,
            "projection_diagnostics": projection,
            "veloedit_parallel_fraction": projection_parallel,
            "veloedit_orthogonal_fraction": projection_orthogonal,
            "residual_native_ratio": result.velocity_diagnostics["max_residual_native_rms_ratio"],
            "warnings": result.warnings,
            "files": {
                "initial_proxy": str((strength_dir / "initial_proxy.png").resolve()),
                "optimized_proxy": str(proxy_path.resolve()),
                "optimized_final": str(final_path.resolve()),
                "residual": str((strength_dir / "delta_v.pt").resolve()),
            },
        }
        (strength_dir / "diagnostics.json").write_text(
            json.dumps(diagnostics, indent=2), encoding="utf-8"
        )
        result_rows.append(diagnostics)
        print(json.dumps({
            "strength": strength,
            "initial_proxy_progress": diagnostics["initial_proxy_progress"],
            "optimized_proxy_progress": diagnostics["optimized_proxy_progress"],
            "optimized_final_progress": diagnostics["optimized_final_progress"],
            "final_target_error": diagnostics["final_target_error"],
            "residual_native_ratio": diagnostics["residual_native_ratio"],
        }), flush=True)

    ordered = [0.0, *[float(x) for x in args.strengths if 0 < x < 1], 1.0]
    grid_paths = [image_paths[value] for value in ordered]
    grid_labels = ["source" if value == 0 else "native full" if value == 1 else f"progress {value:.2f}" for value in ordered]
    grid_path = args.output_dir / "comparison_grid.png"
    save_grid(grid_paths, grid_labels, grid_path)
    progress_values = [row["optimized_final_progress"] for row in result_rows]
    signal_range = float(max(progress_values) - min(progress_values)) if progress_values else 0.0
    sr = sorted(result_rows, key=lambda q: float(q["requested_strength"]))
    sp = [float(q["optimized_final_progress"]) for q in sr]
    ordered = all(b > a for a,b in zip(sp,sp[1:]))
    gaps = {f"gap_{int(100*a['requested_strength']):02d}_{int(100*b['requested_strength']):02d}":float(b["optimized_final_progress"])-float(a["optimized_final_progress"]) for a,b in zip(sr,sr[1:])}
    summary: dict[str, object] = {
        "source": str(Path(args.source).resolve()),
        "prompt": args.prompt,
        "model": args.model,
        "progress_backbone": args.progress_backbone,
        "progress_representation": (
            "normalized mean of DINOv2 patch tokens; CLS excluded"
            if args.progress_backbone == "dino" else "normalized SigLIP image feature"
        ),
        "seed": args.seed,
        "steps": args.steps,
        "reward_mode": args.reward_mode,
        "final_progress_is_ordered": ordered,
        "level_gaps": gaps,
        "native_rollout_config": {
            "first_step_align_steps": 0,
            "preserve_steps": 0,
            "edit_steps": 0,
            "guidance_scale": args.guidance_scale,
        },
        "zero_residual_native_sanity": zero_residual_sanity,
        "progress_anchors": anchor_diagnostics,
        "progress_signal_diagnostic": {
            "dynamic_range": signal_range,
            "note": "Range is across requested controlled strengths unless an independent VeloEdit signal report is available.",
        },
        "loss_weights": {
            "progress": args.progress_weight,
            "drift": args.drift_weight,
            "regularization": args.regularization_weight,
        },
        "strength_results": result_rows,
        "comparison_grid": str(grid_path.resolve()),
        "probable_failure": "unclear",
        "warnings": [],
    }
    signal_path = args.output_dir / "progress_signal" / "progress_signal.json"
    if signal_path.exists():
        signal_report = json.loads(signal_path.read_text(encoding="utf-8"))
        alpha_rows = [
            row for row in signal_report.get("rows", [])
            if row.get("alpha") is not None
        ]
        values = [float(row[f"{args.progress_backbone}_progress"]) for row in alpha_rows]
        deltas = [right - left for left, right in zip(values, values[1:])]
        summary["progress_signal_diagnostic"] = {
            "source": str(signal_path.resolve()),
            "backbone": args.progress_backbone,
            "dynamic_range": max(values) - min(values) if values else 0.0,
            "near_tie_adjacent_pairs": sum(abs(delta) < 0.02 for delta in deltas),
            "adjacent_order_reversals": sum(delta < -0.02 for delta in deltas),
            "siglip_dino_mean_absolute_progress_difference":
                signal_report.get("siglip_dino_mean_absolute_progress_difference"),
            "note": "VeloEdit alpha images are used only to assess feature dynamic range.",
        }
        summary["progress_signal_report"] = signal_report
    sanity_path = args.output_dir / "gradient_sanity.json"
    if sanity_path.exists():
        summary["gradient_sanity_check"] = json.loads(
            sanity_path.read_text(encoding="utf-8")
        )
    summary["probable_failure"] = infer_failure(summary)
    if not zero_residual_sanity["pass"]:
        summary["warnings"].append("Zero-residual native rollout differs from native rollout.")
    if anchor_diagnostics.get("warning"):
        summary["warnings"].append(anchor_diagnostics["warning"])
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
