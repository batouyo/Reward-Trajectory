"""Real FLUX.1-Kontext functional check for the strength-reward gradient path.

This is an opt-in GPU validation script, not a default-CI test.  Its
``BlueDirectionStrengthReward`` is deliberately a toy differentiable probe;
it is not an endpoint-relative or semantic edit-strength reward.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from diffusers import FluxKontextStrengthTrajectoryPipeline
from diffusers.pipelines.rewardflow import StrengthRewardContext, StrengthTrajectoryConfig


DEFAULT_STRENGTHS = (0.2, 0.5, 0.8)
DEFAULT_LAMBDAS = (0.01, 0.03, 0.1, 0.3)


class BlueDirectionStrengthReward:
    """Toy image-space direction reward used only to verify gradient control."""

    def __call__(self, *, image, target_strength, context):
        del context
        channel_means = image.float().mean(dim=(2, 3))
        blue_score = channel_means[:, 2] - 0.5 * (channel_means[:, 0] + channel_means[:, 1])
        return target_strength.float() * blue_score


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.getenv("FLUX_KONTEXT_MODEL_PATH"))
    parser.add_argument("--source", required=True)
    parser.add_argument(
        "--prompt",
        default="Make the weighted training ball blue while preserving its shape, texture, lighting, and background.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--lambdas", type=float, nargs="+", default=DEFAULT_LAMBDAS)
    parser.add_argument("--include-lambda-one", action="store_true")
    args = parser.parse_args()
    if not args.model:
        parser.error("Set FLUX_KONTEXT_MODEL_PATH or pass --model.")
    if args.steps < 1:
        parser.error("--steps must be positive.")
    if any(value <= 0 or not math.isfinite(value) for value in args.lambdas):
        parser.error("Every --lambdas value must be finite and positive.")
    return args


def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    array = image.detach().clamp(0, 1).mul(255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array, mode="RGB")


def _float_tag(value: float) -> str:
    return format(value, "g").replace(".", "p")


def _blue_score(images: torch.Tensor) -> torch.Tensor:
    channel_means = images.float().mean(dim=(2, 3))
    return channel_means[:, 2] - 0.5 * (channel_means[:, 0] + channel_means[:, 1])


def _saturation_diagnostics(images: torch.Tensor) -> list[dict[str, float]]:
    diagnostics = []
    for image in images.float():
        diagnostics.append(
            {
                "all_channel_extreme_ratio": ((image <= 0.01) | (image >= 0.99)).float().mean().item(),
                "blue_channel_high_ratio": (image[2] >= 0.99).float().mean().item(),
                "pixel_min": image.min().item(),
                "pixel_max": image.max().item(),
            }
        )
    return diagnostics


def _rank(values: list[float]) -> list[float]:
    result = [0.0] * len(values)
    order = sorted(range(len(values)), key=values.__getitem__)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = 0.5 * (start + end - 1)
        for index in order[start:end]:
            result[index] = average_rank
        start = end
    return result


def _spearman(first: list[float], second: list[float]) -> float | None:
    x = torch.tensor(_rank(first), dtype=torch.float64)
    y = torch.tensor(_rank(second), dtype=torch.float64)
    x = x - x.mean()
    y = y - y.mean()
    denominator = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if denominator == 0:
        return None
    return float(torch.dot(x, y) / denominator)


def _make_grid(
    source: Image.Image,
    baseline: Image.Image,
    strength_images: list[Image.Image],
    strengths: tuple[float, ...],
    reward_lambda: float,
) -> Image.Image:
    labels = ["Source", "Kontext baseline", *(f"strength={value:g}" for value in strengths)]
    images = [source, baseline, *strength_images]
    tile_width, tile_height = source.size
    label_height = 28
    grid = Image.new("RGB", (tile_width * len(images), tile_height + label_height), "white")
    draw = ImageDraw.Draw(grid)
    for index, (label, image) in enumerate(zip(labels, images)):
        left = index * tile_width
        grid.paste(image.resize((tile_width, tile_height), Image.Resampling.LANCZOS), (left, label_height))
        draw.text((left + 6, 7), label, fill="black")
    draw.text((6, tile_height + label_height - 1), f"lambda={reward_lambda:g}", fill="black")
    return grid


def _trace_for_strength(trace: list[dict], strength_index: int) -> dict:
    reward_values = [entry["branch_rewards"][0][strength_index] for entry in trace if entry["branch_rewards"]]
    gradient_norms = [
        entry["branch_reward_grad_norms"][0][strength_index] for entry in trace if entry["reward_active"]
    ]
    return {
        "reward_values": reward_values,
        "reward_final": reward_values[-1] if reward_values else None,
        "reward_mean": sum(reward_values) / len(reward_values) if reward_values else None,
        "gradient_norms": gradient_norms,
        "gradient_norm_final": gradient_norms[-1] if gradient_norms else 0.0,
        "gradient_norm_mean": sum(gradient_norms) / len(gradient_norms) if gradient_norms else 0.0,
    }


def _run_trajectory(pipe, args, source, reward_lambda: float):
    strengths = DEFAULT_STRENGTHS
    final_latents = {}
    step_divergence = []

    def callback(_pipe, step, _timestep, callback_kwargs):
        latents = callback_kwargs["latents"]
        grouped = latents.reshape(1, len(strengths), *latents.shape[1:]).float()
        reference = grouped[:, :1]
        step_divergence.append(
            {
                "step": step,
                "max_abs_from_strength_0.2": [
                    (grouped[:, index : index + 1] - reference).abs().max().item() for index in range(len(strengths))
                ],
            }
        )
        if step == args.steps - 1:
            final_latents["value"] = latents.detach().float().cpu()
        return callback_kwargs

    config = StrengthTrajectoryConfig(
        enabled=True,
        strengths=strengths,
        lambda_strength_reward=reward_lambda,
        reward_start_step=0,
        reward_every_n_steps=1,
        use_shared_sde_noise=False,
        lazy_branch_materialization=False,
        branch_chunk_size=None,
        collect_trace=True,
    )
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    torch.cuda.synchronize(args.device)
    started = time.perf_counter()
    output = pipe(
        image=source,
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        max_area=args.height * args.width,
        _auto_resize=False,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.steps,
        output_type="pt",
        generator=generator,
        trajectory_config=config,
        strength_reward=BlueDirectionStrengthReward() if reward_lambda > 0 else None,
        strength_reward_context=StrengthRewardContext(prompt=args.prompt),
        callback_on_step_end=callback,
        callback_on_step_end_tensor_inputs=["latents"],
    )
    torch.cuda.synchronize(args.device)
    runtime_seconds = time.perf_counter() - started
    if "value" not in final_latents:
        raise RuntimeError("The final-step callback did not capture trajectory latents.")
    return {
        "latents": final_latents["value"],
        "images": output.images.detach().float().cpu(),
        "trace": list(pipe.last_strength_trajectory_trace),
        "step_divergence": step_divergence,
        "runtime_seconds": runtime_seconds,
    }


def _branch_metrics(run, baseline, reward_lambda: float) -> list[dict]:
    scores = _blue_score(run["images"])
    saturation = _saturation_diagnostics(run["images"])
    rows = []
    for index, strength in enumerate(DEFAULT_STRENGTHS):
        latent_difference = run["latents"][index] - baseline["latents"][index]
        image_difference = run["images"][index] - baseline["images"][index]
        trace = _trace_for_strength(run["trace"], index)
        rows.append(
            {
                "lambda": reward_lambda,
                "strength": strength,
                "blue_score": scores[index].item(),
                "latent_mean_abs_diff": latent_difference.abs().mean().item(),
                "latent_max_abs_diff": latent_difference.abs().max().item(),
                "latent_l2_norm_diff": torch.linalg.vector_norm(latent_difference).item(),
                "image_pixel_mean_abs_diff": image_difference.abs().mean().item(),
                "reward_trace_final": trace["reward_final"],
                "reward_trace_mean": trace["reward_mean"],
                "reward_gradient_norm": trace["gradient_norm_mean"],
                "reward_gradient_norm_final": trace["gradient_norm_final"],
                **saturation[index],
            }
        )
    return rows


def _save_run_images(output_dir: Path, source: Image.Image, baseline, run, reward_lambda: float) -> None:
    run_dir = output_dir / f"lambda_{_float_tag(reward_lambda)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    baseline_image = _tensor_to_pil(baseline["images"][0])
    baseline_image.save(run_dir / "baseline.png")
    strength_images = []
    for index, strength in enumerate(DEFAULT_STRENGTHS):
        image = _tensor_to_pil(run["images"][index])
        image.save(run_dir / f"strength_{_float_tag(strength)}.png")
        strength_images.append(image)
    grid = _make_grid(source, baseline_image, strength_images, DEFAULT_STRENGTHS, reward_lambda)
    grid.save(output_dir / f"grid_lambda_{_float_tag(reward_lambda)}.png")


def _write_summary_csv(path: Path, rows: list[dict]) -> None:
    columns = [
        "lambda",
        "strength",
        "blue_score",
        "latent_mean_abs_diff",
        "latent_max_abs_diff",
        "latent_l2_norm_diff",
        "image_pixel_mean_abs_diff",
        "reward_trace_final",
        "reward_trace_mean",
        "reward_gradient_norm",
        "reward_gradient_norm_final",
        "all_channel_extreme_ratio",
        "blue_channel_high_ratio",
        "pixel_min",
        "pixel_max",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This real-model validation requires CUDA.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = Path(args.source)
    source = Image.open(source_path).convert("RGB").resize((args.width, args.height), Image.Resampling.LANCZOS)
    source.save(output_dir / "source.png")

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    torch.cuda.set_device(device)
    print(f"Loading {args.model} on {device} as {dtype}", flush=True)
    pipe = FluxKontextStrengthTrajectoryPipeline.from_pretrained(
        args.model,
        torch_dtype=dtype,
        local_files_only=True,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    print("Running same-shape lambda=0 baseline", flush=True)
    baseline = _run_trajectory(pipe, args, source, 0.0)
    baseline_pairwise_max = [
        (baseline["latents"][index] - baseline["latents"][0]).abs().max().item()
        for index in range(len(DEFAULT_STRENGTHS))
    ]
    for index, strength in enumerate(DEFAULT_STRENGTHS):
        _tensor_to_pil(baseline["images"][index]).save(output_dir / f"baseline_strength_{_float_tag(strength)}.png")

    lambdas = list(args.lambdas)
    if args.include_lambda_one and 1.0 not in lambdas:
        lambdas.append(1.0)
    runs = {0.0: baseline}
    all_rows = _branch_metrics(baseline, baseline, 0.0)
    _save_run_images(output_dir, source, baseline, baseline, 0.0)
    for reward_lambda in lambdas:
        print(f"Running lambda={reward_lambda:g}", flush=True)
        run = _run_trajectory(pipe, args, source, reward_lambda)
        runs[reward_lambda] = run
        rows = _branch_metrics(run, baseline, reward_lambda)
        all_rows.extend(rows)
        _save_run_images(output_dir, source, baseline, run, reward_lambda)
        print(json.dumps({"lambda": reward_lambda, "rows": rows}, indent=2), flush=True)

    lambda_summaries = {}
    for reward_lambda, run in runs.items():
        scores = _blue_score(run["images"]).tolist()
        rows = [row for row in all_rows if row["lambda"] == reward_lambda]
        lambda_summaries[str(reward_lambda)] = {
            "blue_scores": scores,
            "strictly_increasing": all(left < right for left, right in zip(scores, scores[1:])),
            "spearman_strength_blue_score": _spearman(list(DEFAULT_STRENGTHS), scores),
            "max_image_pixel_mean_abs_diff": max(row["image_pixel_mean_abs_diff"] for row in rows),
            "max_blue_channel_high_ratio": max(row["blue_channel_high_ratio"] for row in rows),
            "runtime_seconds": run["runtime_seconds"],
            "step_divergence": run["step_divergence"],
        }

    baseline_saturation = max(row["blue_channel_high_ratio"] for row in all_rows if row["lambda"] == 0.0)
    # This is an explicit engineering heuristic for triage, not a paper metric or a visual-quality verdict.
    heuristic_candidates = [
        value
        for value in lambdas
        if lambda_summaries[str(value)]["strictly_increasing"]
        and lambda_summaries[str(value)]["max_image_pixel_mean_abs_diff"] > 1 / 255
        and lambda_summaries[str(value)]["max_blue_channel_high_ratio"] < baseline_saturation + 0.1
    ]
    report = {
        "scope": "toy blue-direction gradient-path sanity check; not semantic strength control",
        "model_path": os.path.realpath(args.model),
        "source_path": os.path.realpath(args.source),
        "prompt": args.prompt,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "dtype": str(dtype),
        "height": args.height,
        "width": args.width,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "strengths": list(DEFAULT_STRENGTHS),
        "lambdas": lambdas,
        "trajectory_config": {
            "lazy_branch_materialization": False,
            "use_shared_sde_noise": False,
            "branch_chunk_size": None,
            "reward_start_step": 0,
            "reward_every_n_steps": 1,
            "collect_trace": True,
        },
        "baseline": {
            "latent_shape": list(baseline["latents"].shape),
            "pairwise_max_abs_from_strength_0.2": baseline_pairwise_max,
            "exact_branch_invariant": all(value == 0 for value in baseline_pairwise_max),
            "runtime_seconds": baseline["runtime_seconds"],
        },
        "lambda_summaries": lambda_summaries,
        "heuristic_candidate_lambdas": heuristic_candidates,
        "heuristic_note": "Candidate requires monotonic scores, mean pixel delta > 1/255, and <0.1 added blue clipping.",
        "rows": all_rows,
    }
    _write_summary_csv(output_dir / "summary.csv", all_rows)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print("FINAL_REPORT=" + json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
