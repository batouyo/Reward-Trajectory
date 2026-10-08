#!/usr/bin/env python3
"""Diagnose root causes of masked backward reward directions without a gate."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rewardflow_calibration.diagnostics.backward_direction import (
    build_proxy_directions,
    capture_final_gradient,
    compare_directions,
    component_gradients,
    exact_veloedit_overrides,
    evaluate_proxy_final_pair,
    global_direction_names,
    mask_direction,
    parallel_orthogonal_energy,
    root_cause_evidence,
    scale_global_direction,
    scale_isolated_timestep,
    temporal_energy,
    veloedit_backward_direction,
)
from rewardflow_calibration.metrics.dreamsim import DreamSimDistance
from rewardflow_calibration.optimization.backward_reward import (
    BackwardReward,
    BackwardRewardConfig,
)
from rewardflow_calibration.optimization.backward_trajectory_optimizer import (
    build_image_space_masks,
)
from rewardflow_calibration.rollout.veloedit import (
    VeloEditCompatibleRollout,
    VeloEditRolloutConfig,
)


DEFAULT_IMAGE = "/home/hyp/Code/VeloEdit/testdata/9.jpg"
DEFAULT_MODEL = "/data15/hyp/weight/FLUX.1-Kontext-dev"
DEFAULT_SIGLIP = BackwardReward.DEFAULT_SIGLIP_PATH
DEFAULT_DINO = BackwardReward.DEFAULT_DINO_PATH
DEFAULT_DREAMSIM = "/data15/hyp/weight/dreamsim_ckpts"
DEFAULT_PROMPT = "Change the car to a modern sport car"
GLOBAL_RATIOS = (0.005, 0.02, 0.05)
EXACT_VELOEDIT_ALPHAS = (1.0, 0.75, 0.50, 0.25)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "outputs/backward_direction_root_cause_car_seed42",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--goal-steps", type=int, default=4)
    parser.add_argument("--similarity-threshold", type=float, default=0.8)
    parser.add_argument("--max-area", type=int, default=1024 * 1024)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--siglip-model-path", default=DEFAULT_SIGLIP)
    parser.add_argument("--dino-model-path", default=DEFAULT_DINO)
    parser.add_argument("--dreamsim-cache-dir", default=DEFAULT_DREAMSIM)
    parser.add_argument("--cache-dir", default="/data15/hyp/weight")
    parser.add_argument("--source-weight", type=float, default=1.0)
    parser.add_argument("--semantic-weight", type=float, default=1.0)
    parser.add_argument("--keep-weight", type=float, default=1.0)
    parser.add_argument("--semantic-floor-fraction", type=float, default=0.5)
    parser.add_argument("--semantic-anchor-min-gap", type=float, default=0.02)
    return parser.parse_args()


def _json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_safe(value), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_safe(row), ensure_ascii=False, allow_nan=False) + "\n")


def _save_image(image: torch.Tensor, path: Path) -> None:
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
        raise ValueError("expected a single [1, 3, H, W] image")
    pixels = image[0].detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    Image.fromarray(np.rint(pixels * 255).astype(np.uint8), mode="RGB").save(path)


def _cuda_memory(device: torch.device) -> dict[str, int | None]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {"allocated_bytes": None, "peak_allocated_bytes": None}
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
    }


def _begin_stage(device: torch.device) -> float:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    return time.perf_counter()


def _end_stage(device: torch.device, started: float) -> dict[str, Any]:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    return {"seconds": time.perf_counter() - started, **_cuda_memory(device)}


def _scalar(value: torch.Tensor | float) -> float:
    return float(torch.as_tensor(value).detach().float().mean().cpu())


def _measure_image(
    reward: BackwardReward,
    image: torch.Tensor,
    source: torch.Tensor,
    keep_mask: torch.Tensor,
    comparison_image: torch.Tensor,
) -> dict[str, float]:
    with torch.no_grad():
        values = reward.evaluate(image, source, keep_mask)
        return {
            "dino_source_loss": _scalar(values.source_loss),
            "dreamsim_to_source": reward.dreamsim(source, image),
            "siglip_semantic_score": _scalar(values.semantic_score),
            "keep_region_l1": _scalar(values.keep_loss),
            "dreamsim_to_baseline": reward.dreamsim(comparison_image, image),
        }


def _metric_delta(metrics: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    return {f"delta_{key}": value - baseline[key] for key, value in metrics.items()}


def _masked_active_ratio(
    residual: torch.Tensor, native: torch.Tensor, hard_mask: torch.Tensor
) -> float | None:
    mask = hard_mask.to(device=residual.device, dtype=torch.bool)
    if not bool(mask.any()):
        return None
    active_native = native.to(residual.device).float()[mask]
    denominator = active_native.square().mean().sqrt()
    if float(denominator) <= 1e-12:
        return None
    active_residual = residual.float()[mask]
    return float((active_residual.square().mean().sqrt() / denominator).cpu())


def _directional_derivatives(
    direction: torch.Tensor,
    perturbation: torch.Tensor,
    proxy_gradients: dict[str, torch.Tensor],
    final_gradients: dict[str, torch.Tensor] | None,
) -> dict[str, Any]:
    def dot(gradients: dict[str, torch.Tensor] | None) -> dict[str, float | None]:
        if gradients is None:
            return {name: None for name in ("source", "total", "semantic", "keep")}
        result: dict[str, float | None] = {}
        for name in ("source", "total", "semantic", "keep"):
            gradient = gradients.get(name)
            result[name] = (
                None if gradient is None
                else float((gradient.detach().float() * perturbation.float()).sum().cpu())
            )
        return result

    def raw_dot(gradients: dict[str, torch.Tensor] | None) -> dict[str, float | None]:
        if gradients is None:
            return {name: None for name in ("source", "total", "semantic", "keep")}
        return {
            name: (
                None if gradients.get(name) is None
                else float((gradients[name].detach().float() * direction.float()).sum().cpu())
            )
            for name in ("source", "total", "semantic", "keep")
        }

    return {
        "proxy_dot_applied_residual": dot(proxy_gradients),
        "proxy_dot_raw_direction": raw_dot(proxy_gradients),
        "final_dot_applied_residual": dot(final_gradients),
        "final_dot_raw_direction": raw_dot(final_gradients),
    }


def _make_grid(items: list[tuple[str, Path]], output: Path) -> None:
    if not items:
        return
    columns = 4
    tile_width, tile_height, label_height = 320, 260, 32
    rows = math.ceil(len(items) / columns)
    canvas = Image.new(
        "RGB", (columns * tile_width, rows * (tile_height + label_height)), "white"
    )
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    for index, (label, path) in enumerate(items):
        x = (index % columns) * tile_width
        y = (index // columns) * (tile_height + label_height)
        draw.rectangle((x, y, x + tile_width - 1, y + tile_height + label_height - 1), outline=(180, 180, 180))
        draw.text((x + 6, y + 7), label[:48], fill=(15, 15, 15), font=font)
        image = Image.open(path).convert("RGB")
        image.thumbnail((tile_width - 12, tile_height - 10), Image.Resampling.LANCZOS)
        canvas.paste(
            image,
            (x + (tile_width - image.width) // 2, y + label_height + (tile_height - image.height) // 2),
        )
    canvas.save(output)


def _record_candidate(
    *,
    name: str,
    kind: str,
    ratio: float,
    direction: torch.Tensor,
    residual: torch.Tensor,
    scaling: dict[str, Any],
    rollout,
    prepared,
    rollout_config: VeloEditRolloutConfig,
    goal_steps: int,
    reward: BackwardReward,
    source: torch.Tensor,
    keep_mask: torch.Tensor,
    proxy_baseline_image: torch.Tensor,
    final_baseline_image: torch.Tensor,
    proxy_baseline_metrics: dict[str, float],
    final_baseline_metrics: dict[str, float],
    proxy_gradients: dict[str, torch.Tensor],
    final_gradients: dict[str, torch.Tensor] | None,
    output_dir: Path,
    device: torch.device,
    jsonl_path: Path,
    temporal_step: int | None = None,
) -> dict[str, Any]:
    started = _begin_stage(device)
    proxy, final = evaluate_proxy_final_pair(
        rollout, prepared, rollout_config, residual, goal_steps
    )
    rollout_time = _end_stage(device, started)
    proxy_metrics = _measure_image(reward, proxy, source, keep_mask, proxy_baseline_image)
    final_metrics = _measure_image(reward, final, source, keep_mask, final_baseline_image)

    if kind == "global_direction":
        directory_name = "velo_local" if name == "velo_local_backward" else name
        candidate_dir = output_dir / "directions" / directory_name
        ratio_tag = f"{ratio:.3f}"
        proxy_path = candidate_dir / f"ratio_{ratio_tag}_proxy.png"
        final_path = candidate_dir / f"ratio_{ratio_tag}_final.png"
        candidate_dir.mkdir(parents=True, exist_ok=True)
    else:
        candidate_dir = output_dir / "temporal"
        family = "reward" if name == "proxy_total_masked" else "velo"
        step_tag = f"step{temporal_step + 1}"
        proxy_path = candidate_dir / f"{family}_{step_tag}_proxy.png"
        final_path = candidate_dir / f"{family}_{step_tag}_final.png"
    _save_image(proxy, proxy_path)
    _save_image(final, final_path)
    record = {
        "status": "evaluated",
        "candidate_type": kind,
        "direction": name,
        "requested_ratio": ratio,
        "temporal_step": temporal_step,
        "scaling": scaling,
        "rollout_time": rollout_time,
        "proxy": {
            "metrics": proxy_metrics,
            "deltas": _metric_delta(proxy_metrics, proxy_baseline_metrics),
            "image": str(proxy_path.resolve()),
        },
        "final": {
            "metrics": final_metrics,
            "deltas": _metric_delta(final_metrics, final_baseline_metrics),
            "image": str(final_path.resolve()),
        },
        "directional_derivative": _directional_derivatives(
            direction, residual, proxy_gradients, final_gradients
        ),
    }
    _append_jsonl(jsonl_path, record)
    del proxy, final
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return record


def main() -> None:
    args = parse_args()
    if args.steps < 1 or not 1 <= args.goal_steps <= args.steps:
        raise SystemExit("goal-steps must be in [1, steps]")
    for label, path in (
        ("model", args.model_path), ("input image", args.image),
        ("SigLIP", args.siglip_model_path), ("DINOv2", args.dino_model_path),
    ):
        if not Path(path).exists():
            raise FileNotFoundError(f"{label} path does not exist: {path}")
    if args.dreamsim_cache_dir:
        Path(args.dreamsim_cache_dir).mkdir(parents=True, exist_ok=True)

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    candidates_path = output / "candidates.jsonl"
    candidates_path.write_text("", encoding="utf-8")
    for name in (
        "proxy_total_masked", "proxy_source_masked", "proxy_total_unmasked",
        "proxy_source_unmasked", "proxy_total_reverse", "final_source_masked", "velo_local",
    ):
        (output / "directions" / name).mkdir(parents=True, exist_ok=True)
    (output / "temporal").mkdir(parents=True, exist_ok=True)
    (output / "exact_veloedit_low_only").mkdir(parents=True, exist_ok=True)
    (output / "exact_veloedit_full").mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    dtype = torch.bfloat16
    rollout_config = VeloEditRolloutConfig(
        steps=args.steps,
        seed=args.seed,
        guidance_scale=args.guidance_scale,
        first_step_align_steps=0,
        preserve_steps=0,
        edit_steps=0,
        similarity_threshold=args.similarity_threshold,
        max_area=args.max_area,
        accumulate_latents_fp32=False,
    )
    reward_config = BackwardRewardConfig(
        semantic_floor_fraction=args.semantic_floor_fraction,
        semantic_anchor_min_gap=args.semantic_anchor_min_gap,
        source_weight=args.source_weight,
        semantic_weight=args.semantic_weight,
        keep_weight=args.keep_weight,
    )
    config = {
        "experiment": "Backward Velocity Direction Root-Cause Diagnostic",
        "model_path": args.model_path,
        "image": str(Path(args.image).resolve()),
        "prompt": args.prompt,
        "output_dir": str(output),
        "device": args.device,
        "seed": args.seed,
        "steps": args.steps,
        "goal_steps": args.goal_steps,
        "dtype": str(dtype),
        "native_rollout": asdict(rollout_config),
        "exact_veloedit_low_only": {
            **exact_veloedit_overrides(full=False),
            "similarity_threshold": args.similarity_threshold,
            "alphas": list(EXACT_VELOEDIT_ALPHAS),
        },
        "exact_veloedit_full": {
            **exact_veloedit_overrides(full=True),
            "similarity_threshold": args.similarity_threshold,
            "alphas": list(EXACT_VELOEDIT_ALPHAS),
        },
        "global_residual_native_ratios": list(GLOBAL_RATIOS),
        "temporal_isolated_ratio": 0.02,
        "reward": asdict(reward_config),
        "siglip_model_path": args.siglip_model_path,
        "dino_model_path": args.dino_model_path,
        "dreamsim_cache_dir": args.dreamsim_cache_dir,
        "cache_dir": args.cache_dir,
        "optimizer_or_acceptance_gate_used": False,
    }
    _write_json(output / "config.json", config)

    runtime: dict[str, Any] = {}
    started = _begin_stage(device)
    rollout = VeloEditCompatibleRollout(
        args.model_path, device=device, dtype=dtype, local_files_only=True
    )
    runtime["pipeline_load"] = _end_stage(device, started)
    source_pil = Image.open(args.image).convert("RGB")
    prepared = rollout.prepare(
        source_pil, args.prompt, config=rollout_config, seed=args.seed
    )
    pixels = rollout.pipeline.image_processor.preprocess(
        prepared.working_image, prepared.height, prepared.width
    ).to(device=device, dtype=torch.float32)
    source = ((pixels + 1.0) / 2.0).clamp(0, 1).detach()

    native_trace: list[dict[str, torch.Tensor]] = []
    started = _begin_stage(device)
    with torch.no_grad():
        native_full = rollout.rollout_native(
            prepared, config=rollout_config, velocity_trace=native_trace
        ).detach()
    runtime["native_full"] = _end_stage(device, started)
    started = _begin_stage(device)
    with torch.no_grad():
        native_proxy = rollout.rollout_native(
            prepared, config=rollout_config, early_stop_steps=args.goal_steps
        ).detach()
    runtime["native_proxy"] = _end_stage(device, started)
    _save_image(source, output / "source.png")
    _save_image(native_proxy, output / "native_proxy.png")
    _save_image(native_full, output / "native_full.png")

    native_rows = native_trace[:args.goal_steps]
    hard_mask = torch.stack([row["hard_edit_mask"].detach().bool() for row in native_rows])
    native_velocity = torch.stack([
        row["native_velocity"].detach().float().squeeze(0) for row in native_rows
    ])
    velo_edit_direction = torch.stack([
        row["edit_direction"].detach().float().squeeze(0) for row in native_rows
    ])
    velo_local = veloedit_backward_direction(velo_edit_direction)
    image_edit_mask, image_keep_mask = build_image_space_masks(
        hard_mask,
        height=prepared.height,
        width=prepared.width,
        vae_scale_factor=rollout.pipeline.vae_scale_factor,
        latent_ids=prepared.latent_ids,
    )
    hard_mask = hard_mask.to(device)
    native_velocity = native_velocity.to(device)
    velo_local = velo_local.to(device)
    image_keep_mask = image_keep_mask.to(device)
    mask_metrics = {
        "hard_mask_coverage": float(hard_mask.float().mean().cpu()),
        "hard_mask_coverage_per_step": hard_mask.float().mean(dim=(1, 2)).cpu().tolist(),
        "native_velocity_rms_per_step": native_velocity.square().mean(dim=(1, 2)).sqrt().cpu().tolist(),
        "token_grid": [
            prepared.height // (rollout.pipeline.vae_scale_factor * 2),
            prepared.width // (rollout.pipeline.vae_scale_factor * 2),
        ],
        "token_count": int(prepared.latents.shape[1]),
        "image_edit_mask_coverage": float(image_edit_mask.mean().cpu()),
    }

    dreamsim = DreamSimDistance(
        device, cache_dir=args.dreamsim_cache_dir or None
    )
    reward = BackwardReward(
        args.prompt,
        device=device,
        config=reward_config,
        siglip_model_path=args.siglip_model_path,
        dino_model_path=args.dino_model_path,
        cache_dir=args.cache_dir,
        dreamsim_distance=dreamsim.distance,
    )
    started = _begin_stage(device)
    anchors = reward.set_anchors(source, native_full, native_proxy)
    runtime["reward_and_anchor_load"] = _end_stage(device, started)

    started = _begin_stage(device)
    source_metrics = _measure_image(reward, source, source, image_keep_mask, source)
    proxy_baseline_metrics = _measure_image(
        reward, native_proxy, source, image_keep_mask, native_proxy
    )
    final_baseline_metrics = _measure_image(
        reward, native_full, source, image_keep_mask, native_full
    )
    runtime["baseline_metric_evaluation"] = _end_stage(device, started)
    baseline_metrics = {
        "anchors": anchors,
        "mask_diagnostics": mask_metrics,
        "source": source_metrics,
        "native_proxy": proxy_baseline_metrics,
        "native_full": final_baseline_metrics,
        "runtime": runtime,
        "images": {
            "source": str((output / "source.png").resolve()),
            "native_proxy": str((output / "native_proxy.png").resolve()),
            "native_full": str((output / "native_full.png").resolve()),
        },
    }
    _write_json(output / "baseline_metrics.json", baseline_metrics)

    residual_var = torch.zeros(
        (args.goal_steps, *prepared.latents.shape[1:]),
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    started = _begin_stage(device)
    with torch.enable_grad():
        proxy_grad_image = rollout.rollout_native(
            prepared,
            config=rollout_config,
            goal_residual=residual_var,
            early_stop_steps=args.goal_steps,
        )
        proxy_values = reward.evaluate(proxy_grad_image, source, image_keep_mask)
        proxy_gradients = component_gradients(
            {
                "total": proxy_values.total,
                "source": proxy_values.source_loss,
                "keep": proxy_values.keep_loss,
                "semantic": proxy_values.semantic_score,
            },
            residual_var,
        )
    runtime["proxy_gradients"] = _end_stage(device, started)
    del proxy_grad_image, proxy_values
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    final_gradient_started = _begin_stage(device)
    final_gradient_result = capture_final_gradient(
        lambda: _final_gradient_bundle(
            rollout, prepared, rollout_config, residual_var,
            reward, source, image_keep_mask,
        )
    )
    runtime["final_gradients"] = _end_stage(device, final_gradient_started)
    final_gradient_status = final_gradient_result["final_gradient_status"]
    final_gradients = final_gradient_result.get("gradients")
    if final_gradients is not None:
        final_gradients = {key: value.detach() for key, value in final_gradients.items()}
    else:
        final_gradients = None
    final_gradient_error = final_gradient_result.get("error")
    del residual_var
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    directions = build_proxy_directions(proxy_gradients, hard_mask, velo_local)
    if final_gradients is not None:
        directions["final_source_masked"] = mask_direction(
            final_gradients["source"], hard_mask, sign=-1
        )
        directions["final_total_masked"] = mask_direction(
            final_gradients["total"], hard_mask, sign=-1
        )

    geometry_pairs = (
        ("proxy_total_masked", "proxy_source_masked"),
        ("proxy_total_masked", "proxy_keep_masked"),
        ("proxy_total_masked", "proxy_total_unmasked"),
        ("proxy_total_masked", "proxy_total_reverse"),
        ("proxy_source_masked", "proxy_source_unmasked"),
        ("proxy_source_unmasked", "velo_local_backward"),
        ("proxy_total_masked", "velo_local_backward"),
        ("proxy_source_masked", "velo_local_backward"),
        ("proxy_semantic_down", "velo_local_backward"),
    )
    if final_gradients is not None:
        geometry_pairs += (
            ("proxy_source_masked", "final_source_masked"),
            ("proxy_total_masked", "final_total_masked"),
            ("final_source_masked", "velo_local_backward"),
        )
    direction_geometry = {
        "comparisons": {
            f"{a}_vs_{b}": compare_directions(directions[a], directions[b])
            for a, b in geometry_pairs
        },
        "projection_to_velo_local": {
            name: parallel_orthogonal_energy(value, directions["velo_local_backward"])
            for name, value in directions.items()
            if name in {
                "proxy_total_masked", "proxy_source_masked", "proxy_source_unmasked", "proxy_semantic_down",
                "final_source_masked", "final_total_masked",
            }
        },
        "final_gradient_status": final_gradient_status,
        "final_gradient_error": final_gradient_error,
    }
    _write_json(output / "direction_geometry.json", direction_geometry)

    proxy_temporal: dict[str, Any] = {}
    for name in ("proxy_total_masked", "proxy_source_masked", "proxy_keep_masked"):
        proxy_temporal[name] = temporal_energy(directions[name])
    proxy_temporal["proxy_step4_energy_fraction"] = proxy_temporal[
        "proxy_total_masked"
    ]["energy_fraction_per_step"][args.goal_steps - 1]
    final_temporal: dict[str, Any] | None = None
    if final_gradients is not None:
        final_temporal = {
            "final_source": temporal_energy(directions["final_source_masked"]),
            "final_step4_energy_fraction": temporal_energy(
                directions["final_source_masked"]
            )["energy_fraction_per_step"][args.goal_steps - 1],
        }
        if "final_total_masked" in directions:
            final_temporal["final_total"] = temporal_energy(directions["final_total_masked"])
    gradient_temporal = {
        "proxy_total": temporal_energy(directions["proxy_total_masked"]),
        "proxy_source": temporal_energy(directions["proxy_source_masked"]),
        "proxy_keep": temporal_energy(directions["proxy_keep_masked"]),
        "final_source": None if final_gradients is None else temporal_energy(directions["final_source_masked"]),
        "final_total": None if final_gradients is None else temporal_energy(directions["final_total_masked"]),
        "velo_local": temporal_energy(directions["velo_local_backward"]),
        "proxy_step4_energy_fraction": proxy_temporal["proxy_step4_energy_fraction"],
        "final_step4_energy_fraction": None if final_temporal is None else final_temporal["final_step4_energy_fraction"],
        "final_gradient_status": final_gradient_status,
    }
    _write_json(output / "gradient_temporal.json", gradient_temporal)

    candidate_records: list[dict[str, Any]] = []
    for name in global_direction_names(final_gradient_available=final_gradients is not None):
        direction = directions.get(name)
        if direction is None:
            for ratio in GLOBAL_RATIOS:
                row = {"status": "skipped", "reason": "final_gradient_unavailable",
                       "candidate_type": "global_direction", "direction": name,
                       "requested_ratio": ratio}
                _append_jsonl(candidates_path, row)
                candidate_records.append(row)
            continue
        for ratio in GLOBAL_RATIOS:
            try:
                residual, scaling = scale_global_direction(
                    direction, native_velocity, ratio, active_mask=hard_mask
                )
            except FloatingPointError as exc:
                row = {"status": "skipped", "reason": str(exc),
                       "candidate_type": "global_direction", "direction": name,
                       "requested_ratio": ratio}
                _append_jsonl(candidates_path, row)
                candidate_records.append(row)
                continue
            record = _record_candidate(
                name=name, kind="global_direction", ratio=ratio,
                direction=direction, residual=residual, scaling=scaling,
                rollout=rollout, prepared=prepared, rollout_config=rollout_config,
                goal_steps=args.goal_steps, reward=reward, source=source,
                keep_mask=image_keep_mask, proxy_baseline_image=native_proxy,
                final_baseline_image=native_full,
                proxy_baseline_metrics=proxy_baseline_metrics,
                final_baseline_metrics=final_baseline_metrics,
                proxy_gradients=proxy_gradients, final_gradients=final_gradients,
                output_dir=output, device=device, jsonl_path=candidates_path,
            )
            candidate_records.append(record)

    for name in ("proxy_total_masked", "velo_local_backward"):
        for step in range(args.goal_steps):
            try:
                residual, scaling = scale_isolated_timestep(
                    directions[name], native_velocity, step, requested_ratio=0.02
                )
            except FloatingPointError as exc:
                row = {"status": "skipped", "reason": str(exc),
                       "candidate_type": "temporal_direction", "direction": name,
                       "requested_ratio": 0.02, "temporal_step": step}
                _append_jsonl(candidates_path, row)
                candidate_records.append(row)
                continue
            record = _record_candidate(
                name=name, kind="temporal_direction", ratio=0.02,
                direction=directions[name], residual=residual, scaling=scaling,
                rollout=rollout, prepared=prepared, rollout_config=rollout_config,
                goal_steps=args.goal_steps, reward=reward, source=source,
                keep_mask=image_keep_mask, proxy_baseline_image=native_proxy,
                final_baseline_image=native_full,
                proxy_baseline_metrics=proxy_baseline_metrics,
                final_baseline_metrics=final_baseline_metrics,
                proxy_gradients=proxy_gradients, final_gradients=final_gradients,
                output_dir=output, device=device, jsonl_path=candidates_path,
                temporal_step=step,
            )
            candidate_records.append(record)

    exact_metrics_by_mode: dict[str, list[dict[str, Any]]] = {
        "low_only": [],
        "full": [],
    }
    for mode, full in (("low_only", False), ("full", True)):
        exact_config = VeloEditRolloutConfig(
            **{
                **asdict(rollout_config),
                **exact_veloedit_overrides(full=full),
                "similarity_threshold": args.similarity_threshold,
            }
        )
        started = _begin_stage(device)
        with torch.no_grad():
            exact_images = rollout.rollout(
                prepared,
                torch.tensor(EXACT_VELOEDIT_ALPHAS, device=device),
                config=exact_config,
            ).detach()
        runtime[f"exact_veloedit_{mode}_rollouts"] = _end_stage(device, started)
        directory = output / f"exact_veloedit_{mode}"
        for index, alpha in enumerate(EXACT_VELOEDIT_ALPHAS):
            image = exact_images[index:index + 1]
            image_path = directory / f"alpha_{alpha:.2f}.png"
            _save_image(image, image_path)
            metrics = _measure_image(reward, image, source, image_keep_mask, native_full)
            row = {
                "candidate_type": f"exact_veloedit_{mode}",
                "alpha": alpha,
                "metrics": metrics,
                "deltas": _metric_delta(metrics, final_baseline_metrics),
                "image": str(image_path.resolve()),
                "runtime": runtime[f"exact_veloedit_{mode}_rollouts"],
            }
            exact_metrics_by_mode[mode].append(row)
            _append_jsonl(candidates_path, row)
        del exact_images
    grid_items: list[tuple[str, Path]] = [
        ("source", output / "source.png"),
        ("native full · 15 steps", output / "native_full.png"),
        ("native proxy · 4 steps", output / "native_proxy.png"),
    ]
    for row in candidate_records:
        if row.get("status") == "evaluated":
            grid_items.append((
                f"{row['direction']} · {row['requested_ratio']:.3f} final"
                if row["candidate_type"] == "global_direction"
                else f"{row['direction']} · step {row['temporal_step'] + 1} final",
                Path(row["final"]["image"]),
            ))
    grid_items.extend(
        (f"Exact Velo low-only · alpha {row['alpha']:.2f}", Path(row["image"]))
        for row in exact_metrics_by_mode["low_only"]
    )
    grid_items.extend(
        (f"Exact Velo full · alpha {row['alpha']:.2f}", Path(row["image"]))
        for row in exact_metrics_by_mode["full"]
    )
    _make_grid(grid_items, output / "comparison_grid.png")

    direction_geometry["final_gradient_status"] = final_gradient_status
    direction_geometry["final_gradient_error"] = final_gradient_error
    root_evidence = root_cause_evidence(
        direction_geometry, gradient_temporal,
        candidate_records, exact_metrics_by_mode["low_only"],
        exact_metrics_by_mode["full"], final_gradient_status,
    )
    summary = {
        "experiment": "Backward Velocity Direction Root-Cause Diagnostic",
        "status": "complete",
        "base_commit_expected": "cfb16d6a98abe0d0267c889efde90cc3bc2007bd",
        "sample": {"image": args.image, "prompt": args.prompt, "seed": args.seed},
        "steps": args.steps,
        "goal_steps": args.goal_steps,
        "optimizer_or_acceptance_gate_used": False,
        "final_gradient_status": final_gradient_status,
        "final_gradient_error": final_gradient_error,
        "candidate_count": len(candidate_records),
        "evaluated_candidate_count": sum(row.get("status") == "evaluated" for row in candidate_records),
        "skipped_candidate_count": sum(row.get("status") == "skipped" for row in candidate_records),
        "exact_veloedit_low_only_count": len(exact_metrics_by_mode["low_only"]),
        "exact_veloedit_full_count": len(exact_metrics_by_mode["full"]),
        "root_cause_evidence": root_evidence,
        "runtime": runtime,
        "images": {
            "source": str((output / "source.png").resolve()),
            "native_proxy": str((output / "native_proxy.png").resolve()),
            "native_full": str((output / "native_full.png").resolve()),
            "comparison_grid": str((output / "comparison_grid.png").resolve()),
        },
        "diagnostics": {
            "baseline_metrics": str((output / "baseline_metrics.json").resolve()),
            "direction_geometry": str((output / "direction_geometry.json").resolve()),
            "gradient_temporal": str((output / "gradient_temporal.json").resolve()),
            "candidates": str(candidates_path.resolve()),
        },
    }
    _write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _final_gradient_bundle(
    rollout,
    prepared,
    config,
    residual_var: torch.Tensor,
    reward: BackwardReward,
    source: torch.Tensor,
    keep_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    device = residual_var.device
    with torch.enable_grad():
        final_image = rollout.rollout_native(
            prepared,
            config=config,
            goal_residual=residual_var,
            early_stop_steps=None,
        )
        values = reward.evaluate(final_image, source, keep_mask)
        gradients = component_gradients(
            {
                "total": values.total,
                "source": values.source_loss,
                "semantic": values.semantic_score,
                "keep": values.keep_loss,
            },
            residual_var,
        )
    del final_image, values
    return gradients


if __name__ == "__main__":
    main()
