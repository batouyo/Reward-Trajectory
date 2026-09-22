"""Run activation-range detection and A/B/C trajectory comparisons on FLUX-Kontext."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/data15/hyp/weight/FLUX.1-Kontext-dev")
    parser.add_argument("--source", nargs="+", default=["/home/hyp/Code/VeloEdit/testdata/7.jpg"])
    parser.add_argument("--prompt", default="make him old")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--first-step-align-steps", type=int, default=4)
    parser.add_argument("--activation-distance-threshold", type=float, default=0.065)
    parser.add_argument("--alpha-resolution", type=float, default=0.05)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/activation_detection"))
    return parser


def _tensor_image(image: torch.Tensor) -> Image.Image:
    value = image.detach().float().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(np.rint(value * 255).astype(np.uint8), mode="RGB")


def _save_contact_sheet(images: list[Image.Image], labels: list[str], path: Path) -> None:
    tiles = []
    for image, label in zip(images, labels):
        tile = Image.new("RGB", (image.width, image.height + 26), "white")
        tile.paste(image, (0, 26))
        ImageDraw.Draw(tile).text((5, 5), label, fill="black")
        tiles.append(tile)
    columns = 5
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tiles[0].width, rows * tiles[0].height), "white")
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % columns) * tile.width, (index // columns) * tile.height))
    sheet.save(path)


def _metric_record(images: list[torch.Tensor], source: torch.Tensor, distance) -> dict:
    nodes = torch.cat([source, *images], dim=0)
    stats = distance.trajectory(nodes * 2.0 - 1.0)
    return {
        "adjacent_lpips": [float(value) for value in stats.distances.detach().cpu()],
        "normalized_lpips": [float(value) for value in stats.normalized_distances.detach().cpu()],
        "trajectory_kl_to_uniform": float(stats.kl_uniform.detach().cpu()),
        "path_length": float(stats.path_length.detach().cpu()),
        "source_to_final_lpips": float(stats.endpoint_distance.detach().cpu()),
    }


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def run(args: argparse.Namespace) -> dict:
    if args.steps < 4:
        raise ValueError("V2 rollout requires at least four inference steps.")
    if args.height % 16 or args.width % 16:
        raise ValueError("height and width must be divisible by 16.")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from diffusers.pipelines.rewardflow.pipeline_flux_kontext_rewardslider_v2 import (
        FluxKontextRewardSliderV2Pipeline,
    )
    from diffusers.pipelines.rewardflow.rewardslider_v2_activation import (
        ActivationRangeConfig,
        ActivationRangeDetector,
        normalize_alpha,
    )
    from diffusers.pipelines.rewardflow.rewardslider_v2_lpips import LPIPSDistance

    pipe = FluxKontextRewardSliderV2Pipeline.from_pretrained(
        args.model, torch_dtype=dtype, local_files_only=True
    )
    if args.cpu_offload:
        pipe.enable_model_cpu_offload(device=device)
    else:
        pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    metric = LPIPSDistance(net="vgg").to(device)
    for parameter in metric.parameters():
        parameter.requires_grad_(False)

    sample_rows = []
    sample_details = []
    for sample_index, source_path in enumerate(args.source):
        source_path = Path(source_path)
        sample_dir = args.output_dir / source_path.stem
        sample_dir.mkdir(parents=True, exist_ok=True)
        source_pil = Image.open(source_path).convert("RGB").resize(
            (args.width, args.height), Image.Resampling.LANCZOS
        )
        source_arr = np.asarray(source_pil).copy()
        source = torch.from_numpy(source_arr).permute(2, 0, 1).unsqueeze(0).to(
            device=device, dtype=torch.float32
        ) / 255.0
        sample_seed = args.seed + sample_index

        print(f"[{sample_index + 1}/{len(args.source)}] Preparing {source_path}")
        inputs = pipe.prepare_rewardslider_v2_inputs(
            num_branches=1,
            image=source_pil,
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=torch.Generator(device=device).manual_seed(sample_seed),
            first_step_align_steps=args.first_step_align_steps,
        )
        zero_goals = [
            torch.zeros(1, *inputs.native.initial_latent.shape[1:], device=device, dtype=torch.float32)
            for _ in range(4)
        ]
        cache: dict[float, torch.Tensor] = {}

        def rollout(alpha: float) -> torch.Tensor:
            key = round(float(alpha), 10)
            if key not in cache:
                with torch.no_grad():
                    result = pipe.unroll_rewardslider_v2_controls(
                        inputs,
                        torch.tensor([key], device=device),
                        zero_goals,
                        control_steps=4,
                        use_checkpointing=False,
                    )
                    cache[key] = pipe.decode_rewardslider_v2_terminal(
                        result.final_latent, inputs
                    ).detach()
            return cache[key]

        detector = ActivationRangeDetector(
            metric,
            ActivationRangeConfig(
                activation_distance_threshold=args.activation_distance_threshold,
                alpha_resolution=args.alpha_resolution,
            ),
        )
        activation = detector.detect(
            source_image=source,
            source_latent=inputs.native.initial_latent,
            prompt_embeds=inputs.forward_kwargs["prompt_embeds"],
            inference_config={
                "model": args.model,
                "prompt": args.prompt,
                "seed": sample_seed,
                "steps": args.steps,
                "height": args.height,
                "width": args.width,
                "guidance_scale": args.guidance_scale,
                "first_step_align_steps": args.first_step_align_steps,
            },
            rollout=rollout,
            output_dir=sample_dir / "probes",
        )

        baseline_alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
        calibrated_alphas = [
            float(normalize_alpha(beta, activation["alpha_start"]))
            for beta in baseline_alphas
        ]
        baseline_images = [rollout(alpha) for alpha in baseline_alphas]
        calibrated_images = [rollout(alpha) for alpha in calibrated_alphas]

        baseline_dir = sample_dir / "baseline"
        calibrated_dir = sample_dir / "calibrated"
        baseline_dir.mkdir(exist_ok=True)
        calibrated_dir.mkdir(exist_ok=True)
        for label, alpha, image in zip(baseline_alphas, baseline_alphas, baseline_images):
            _tensor_image(image).save(baseline_dir / f"alpha_{alpha:.2f}.png")
        for beta, alpha, image in zip(baseline_alphas, calibrated_alphas, calibrated_images):
            _tensor_image(image).save(calibrated_dir / f"beta_{beta:.2f}_alpha_{alpha:.3f}.png")

        baseline_metrics = _metric_record(baseline_images, source, metric)
        calibrated_metrics = _metric_record(calibrated_images, source, metric)
        endpoint_mae = float(
            (baseline_images[-1].float() - calibrated_images[-1].float()).abs().mean().cpu()
        )
        endpoint_max_abs = float(
            (baseline_images[-1].float() - calibrated_images[-1].float()).abs().max().cpu()
        )
        _save_contact_sheet(
            [_tensor_image(image) for image in baseline_images + calibrated_images],
            [f"baseline α={alpha:.2f}" for alpha in baseline_alphas]
            + [f"calibrated β={beta:.2f} α={alpha:.3f}"
               for beta, alpha in zip(baseline_alphas, calibrated_alphas)],
            sample_dir / "baseline_vs_calibrated.png",
        )
        detail = {
            "sample": str(source_path),
            "activation": activation,
            "baseline": {"alphas": baseline_alphas, **baseline_metrics},
            "calibrated": {
                "betas": baseline_alphas,
                "alphas": calibrated_alphas,
                **calibrated_metrics,
            },
            "same_final_alpha_1": {
                "baseline_alpha": baseline_alphas[-1],
                "calibrated_alpha": calibrated_alphas[-1],
                "mean_absolute_image_difference": endpoint_mae,
                "max_absolute_image_difference": endpoint_max_abs,
                "pass": endpoint_max_abs == 0.0,
            },
            "artifacts": {
                "baseline_contact_sheet": str(sample_dir / "baseline_vs_calibrated.png"),
                "probe_dir": str(sample_dir / "probes"),
                "baseline_dir": str(baseline_dir),
                "calibrated_dir": str(calibrated_dir),
            },
        }
        sample_details.append(detail)
        sample_rows.append({
            "sample": str(source_path),
            "alpha_start": activation["alpha_start"],
            "activation_found": activation["activation_found"],
            "coarse_distance_curve": json.dumps(
                [{"alpha": row["alpha"], "distance": row["distance"]}
                 for row in activation["probe_results"] if row["stage"] == "coarse"]
            ),
            "baseline_trajectory_kl": baseline_metrics["trajectory_kl_to_uniform"],
            "calibrated_trajectory_kl": calibrated_metrics["trajectory_kl_to_uniform"],
            "final_endpoint_max_abs": endpoint_max_abs,
        })
        print(
            f"  alpha_start={activation['alpha_start']:.4f} "
            f"coarse_curve={[(round(row['alpha'], 2), round(row['distance'], 4)) for row in activation['probe_results'] if row['stage'] == 'coarse']}"
        )

    csv_path = args.output_dir / "experiment_a_activation_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sample_rows[0]))
        writer.writeheader()
        writer.writerows(sample_rows)
    result = {
        "method": "VeloEdit rollout with valid-range alpha calibration",
        "metric": "LPIPS-VGG",
        "experiments": {
            "A_inactive_region": "coarse alpha probes and per-sample first threshold crossing",
            "B_calibration": "baseline and calibrated adjacent LPIPS plus KL to uniform",
            "C_endpoint_control": "same final alpha=1 rollout and exact endpoint pixel comparison",
        },
        "config": {
            "model": args.model,
            "prompt": args.prompt,
            "seed": args.seed,
            "steps": args.steps,
            "height": args.height,
            "width": args.width,
            "guidance_scale": args.guidance_scale,
            "first_step_align_steps": args.first_step_align_steps,
            "activation_distance_threshold": args.activation_distance_threshold,
            "alpha_resolution": args.alpha_resolution,
        },
        "activation_table_csv": str(csv_path),
        "samples": sample_details,
    }
    result_path = args.output_dir / "activation_result.json"
    result_path.write_text(json.dumps(result, indent=2, default=_json_safe), encoding="utf-8")
    print(f"[Done] Results: {result_path}")
    print(f"[Done] Experiment A table: {csv_path}")
    return result


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
