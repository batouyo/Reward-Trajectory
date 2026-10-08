#!/usr/bin/env python3
"""Compare clean-proxy, final-value connector, LeapAlign and full gradients."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rewardflow_calibration.diagnostics.backward_direction import (  # noqa: E402
    compare_directions,
    temporal_energy,
)
from rewardflow_calibration.diagnostics.leapalign_proxy import (  # noqa: E402
    flow_matching_clean_prediction,
    leap_gradient,
    stop_gradient_connector,
)
from rewardflow_calibration.metrics.dreamsim import DreamSimDistance  # noqa: E402
from rewardflow_calibration.optimization.backward_reward import (  # noqa: E402
    BackwardReward,
    BackwardRewardConfig,
)
from rewardflow_calibration.optimization.backward_trajectory_optimizer import (  # noqa: E402
    build_image_space_masks,
)
from rewardflow_calibration.rollout.veloedit import (  # noqa: E402
    VeloEditCompatibleRollout,
    VeloEditRolloutConfig,
)


DEFAULT_IMAGE = "/home/hyp/Code/VeloEdit/testdata/9.jpg"
DEFAULT_MODEL = "/data15/hyp/weight/FLUX.1-Kontext-dev"
DEFAULT_PROMPT = "Change the car to a modern sport car"
DEFAULT_OUTPUT = ROOT / "outputs/leapalign_proxy_diagnostic_car_seed42"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--goal-steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--max-area", type=int, default=1024 * 1024)
    parser.add_argument("--similarity-threshold", type=float, default=0.8)
    parser.add_argument("--nested-grad-coe", type=float, default=0.3, choices=(0.0, 0.3, 1.0))
    parser.add_argument("--siglip-model-path", default=BackwardReward.DEFAULT_SIGLIP_PATH)
    parser.add_argument("--dino-model-path", default=BackwardReward.DEFAULT_DINO_PATH)
    parser.add_argument("--dreamsim-cache-dir", default="/data15/hyp/weight/dreamsim_ckpts")
    parser.add_argument("--cache-dir", default="/data15/hyp/weight")
    parser.add_argument("--semantic-floor-fraction", type=float, default=0.5)
    parser.add_argument("--semantic-anchor-min-gap", type=float, default=0.02)
    parser.add_argument("--source-weight", type=float, default=1.0)
    parser.add_argument("--semantic-weight", type=float, default=1.0)
    parser.add_argument("--keep-weight", type=float, default=1.0)
    parser.add_argument("--finite-difference-ratios", nargs="+", type=float, default=(0.005, 0.02))
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _save_image(image: torch.Tensor, path: Path) -> None:
    pixels = image[0].detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    import numpy as np
    Image.fromarray(np.rint(pixels * 255).astype("uint8"), mode="RGB").save(path)


def _component_losses(reward, image, source, keep_mask) -> dict[str, torch.Tensor]:
    values = reward.evaluate(image, source, keep_mask)
    return {"source": values.source_loss, "total": values.total}


def _gradients(losses: dict[str, torch.Tensor], variable: torch.Tensor) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    names = list(losses)
    for index, name in enumerate(names):
        grad = torch.autograd.grad(
            losses[name], variable, retain_graph=True,
            allow_unused=True,
        )[0]
        result[name] = torch.zeros_like(variable) if grad is None else grad.detach()
    return result


def _gradient_stats(gradient: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    return {
        **compare_directions(gradient, reference),
        "gradient_rms": float(gradient.float().square().mean().sqrt().cpu()),
        "per_step_rms": gradient.float().square().mean(dim=(1, 2)).sqrt().cpu().tolist(),
        "norm_ratio_to_full": float(
            gradient.float().norm().div(reference.float().norm().clamp_min(1e-12)).cpu()
        ),
    }


def _rollout_objective(reward, rollout, prepared, source, keep_mask, latent):
    image = rollout.decode_latents(prepared, latent)
    return _component_losses(reward, image, source, keep_mask)


def main() -> None:
    args = parse_args()
    if not 1 <= args.goal_steps <= args.steps:
        raise SystemExit("goal-steps must be in [1, steps]")
    for label, path in (
        ("model", args.model_path), ("source image", args.image),
        ("SigLIP", args.siglip_model_path), ("DINOv2", args.dino_model_path),
        ("DreamSim cache", args.dreamsim_cache_dir),
    ):
        if not Path(path).exists():
            raise FileNotFoundError(f"{label} path is unavailable: {path}")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dtype = torch.bfloat16
    rollout_config = VeloEditRolloutConfig(
        steps=args.steps, seed=args.seed, guidance_scale=args.guidance_scale,
        first_step_align_steps=0, preserve_steps=0, edit_steps=0,
        similarity_threshold=args.similarity_threshold, max_area=args.max_area,
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
        "experiment": "LeapAlign-style proxy surrogate diagnostic",
        "model_path": args.model_path, "image": str(Path(args.image).resolve()),
        "prompt": args.prompt, "output_dir": str(output), "device": args.device,
        "seed": args.seed, "steps": args.steps, "goal_steps": args.goal_steps,
        "dtype": str(dtype), "native_rollout": asdict(rollout_config),
        "nested_grad_coe": args.nested_grad_coe,
        "finite_difference_ratios": args.finite_difference_ratios,
        "reward": asdict(reward_config), "optimizer_gate_or_line_search_used": False,
    }
    _write_json(output / "config.json", config)

    rollout = VeloEditCompatibleRollout(
        args.model_path, device=device, dtype=dtype, local_files_only=True
    )
    source_pil = Image.open(args.image).convert("RGB")
    prepared = rollout.prepare(source_pil, args.prompt, config=rollout_config, seed=args.seed)
    pixels = rollout.pipeline.image_processor.preprocess(
        prepared.working_image, prepared.height, prepared.width
    ).to(device=device, dtype=torch.float32)
    source = ((pixels + 1.0) / 2.0).clamp(0, 1).detach()

    with torch.no_grad():
        native_full, true_trace = rollout.rollout_native(
            prepared, config=rollout_config, return_latent_trace=True
        )
        native_proxy, proxy_trace = rollout.rollout_native(
            prepared, config=rollout_config, early_stop_steps=args.goal_steps,
            goal_residual=torch.zeros((args.goal_steps, *prepared.latents.shape[1:]), device=device),
            return_latent_trace=True,
        )
    true_states = true_trace["latent_states"]
    sigmas = true_trace["sigmas"]
    true_final_latent = true_trace["true_final_latent"]
    if not isinstance(true_final_latent, torch.Tensor) or true_final_latent.requires_grad:
        raise AssertionError("full-trajectory final anchor must be detached")
    for image, name in ((source, "source.png"), (native_proxy, "native_proxy.png"), (native_full, "native_full.png")):
        _save_image(image, output / name)

    # Use the same edit-support mask construction as the prior diagnostic.
    trace_rows: list[dict[str, torch.Tensor]] = []
    for index in range(args.goal_steps):
        z = true_states[index]
        velocity = true_trace["native_velocities"][index]
        sigma = sigmas[index]
        reference = prepared.reference_latent
        ref_velocity = (z.float() - reference.float()) / (sigma.float() + 1e-8)
        ref_abs = ref_velocity.abs() + 1e-8
        similarity = ref_abs / (ref_abs + (velocity.float() - ref_velocity).abs())
        trace_rows.append({"hard_edit_mask": ~(similarity >= args.similarity_threshold)})
    hard_mask = torch.stack([row["hard_edit_mask"].bool().squeeze(0) for row in trace_rows])
    _, keep_mask = build_image_space_masks(
        hard_mask, height=prepared.height, width=prepared.width,
        vae_scale_factor=rollout.pipeline.vae_scale_factor,
        latent_ids=prepared.latent_ids,
    )
    keep_mask = keep_mask.to(device)

    dreamsim = DreamSimDistance(device, cache_dir=args.dreamsim_cache_dir)
    reward = BackwardReward(
        args.prompt, device=device, config=reward_config,
        siglip_model_path=args.siglip_model_path, dino_model_path=args.dino_model_path,
        cache_dir=args.cache_dir, dreamsim_distance=dreamsim.distance,
    )
    anchors = reward.set_anchors(source, native_full, native_proxy)
    residual = torch.zeros(
        (args.goal_steps, *prepared.latents.shape[1:]), device=device,
        dtype=torch.float32, requires_grad=True,
    )

    # A. Existing early clean-prediction reward and its backward path.
    proxy_image, proxy_graph_trace = rollout.rollout_native(
        prepared, config=rollout_config, goal_residual=residual,
        early_stop_steps=args.goal_steps,
        return_latent_trace=True,
    )
    current_losses = _component_losses(reward, proxy_image, source, keep_mask)
    current_gradients = _gradients(current_losses, residual)

    # B. Keep the same proxy Jacobian, but connect its forward value to z_final.
    boundary_index = args.goal_steps - 1
    proxy_latent = flow_matching_clean_prediction(
        proxy_graph_trace["latent_states_graph"][boundary_index],
        proxy_graph_trace["actual_velocities_graph"][boundary_index],
        proxy_trace["sigmas"][boundary_index],
    ).to(dtype=rollout.pipeline.transformer.dtype)
    connected_latent = stop_gradient_connector(proxy_latent, true_final_latent)
    connector_image = rollout.decode_latents(prepared, connected_latent)
    connector_losses = _component_losses(reward, connector_image, source, keep_mask)
    connector_gradients = _gradients(connector_losses, residual)
    connector_forward_error = float((connected_latent.detach() - true_final_latent).abs().max().cpu())

    del (
        proxy_image, connector_image, current_losses, connector_losses,
        proxy_graph_trace, proxy_latent, connected_latent,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # D. Ground-truth reference: reward backpropagated through all 15 Euler steps.
    full_image = rollout.rollout_native(
        prepared, config=rollout_config, goal_residual=residual, early_stop_steps=None
    )
    full_losses = _component_losses(reward, full_image, source, keep_mask)
    full_gradients = _gradients(full_losses, residual)
    del full_image, full_losses
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # C. Per-control-slot LeapAlign two-leap gradient.  True anchors are detached.
    def velocity_fn(latent: torch.Tensor, sigma: torch.Tensor | float) -> torch.Tensor:
        return rollout.native_velocity(prepared, latent, sigma)

    leap_gradients: dict[str, torch.Tensor] = {}
    leap_diagnostics: dict[str, list[dict[str, object]]] = {}
    for component in ("source", "total"):
        def objective(latent: torch.Tensor, component_name: str = component) -> torch.Tensor:
            image = rollout.decode_latents(prepared, latent)
            return _component_losses(reward, image, source, keep_mask)[component_name]

        gradient, steps = leap_gradient(
            torch.zeros_like(residual), true_states, sigmas, true_final_latent,
            velocity_fn=velocity_fn, objective_fn=objective,
            goal_steps=args.goal_steps, nested_grad_coe=args.nested_grad_coe,
        )
        leap_gradients[component] = gradient
        leap_diagnostics[component] = steps
        if device.type == "cuda":
            torch.cuda.empty_cache()

    estimators = {
        "current_proxy": current_gradients,
        "connector": connector_gradients,
        "leap": leap_gradients,
        "full": full_gradients,
    }
    gradient_comparison: dict[str, Any] = {}
    temporal: dict[str, Any] = {}
    for component in ("source", "total"):
        full_gradient = full_gradients[component]
        gradient_comparison[component] = {
            f"{name}_vs_full": _gradient_stats(gradients[component], full_gradient)
            for name, gradients in estimators.items() if name != "full"
        }
        temporal[component] = {
            name: temporal_energy(gradients[component]) for name, gradients in estimators.items()
        }
    _write_json(output / "gradient_comparison.json", gradient_comparison)
    _write_json(output / "temporal_energy.json", temporal)
    _write_json(output / "leap_approximation.json", {
        "nested_grad_coe": args.nested_grad_coe,
        "true_trajectory_anchors_detached": not true_final_latent.requires_grad
        and all(not state.requires_grad for state in true_states),
        "connector_forward_max_abs_error": connector_forward_error,
        "per_component": leap_diagnostics,
    })

    # Finite perturbations use matched global residual/native RMS ratios.
    native_rms = torch.stack([
        velocity.float() for velocity in true_trace["native_velocities"][:args.goal_steps]
    ]).square().mean().sqrt()
    baseline_source_loss = float(reward.evaluate(native_full, source, keep_mask).source_loss.detach().cpu())
    baseline_dreamsim = reward.dreamsim(source, native_full)
    finite_rows: list[dict[str, Any]] = []
    for estimator_name in estimators:
        direction = -estimators[estimator_name]["source"].detach().float()
        direction_rms = direction.square().mean().sqrt()
        for ratio in args.finite_difference_ratios:
            if float(direction_rms) <= 1e-12:
                finite_rows.append({"estimator": estimator_name, "ratio": ratio, "status": "zero_gradient"})
                continue
            perturbation = direction * (float(ratio) * native_rms / direction_rms)
            with torch.no_grad():
                candidate = rollout.rollout_native(
                    prepared, config=rollout_config, goal_residual=perturbation,
                )
                values = reward.evaluate(candidate, source, keep_mask)
                final_source_loss = float(values.source_loss.detach().cpu())
                dreamsim_value = reward.dreamsim(source, candidate)
            full_gradient = full_gradients["source"].detach().float()
            predicted = float((full_gradient * perturbation).sum().cpu())
            observed = final_source_loss - baseline_source_loss
            finite_rows.append({
                "estimator": estimator_name, "ratio": float(ratio),
                "actual_global_ratio": float((perturbation.square().mean().sqrt() / native_rms).cpu()),
                "predicted_directional_derivative": predicted,
                "observed_final_source_loss_delta": observed,
                "directional_sign_agreement": bool((predicted == 0.0 and observed == 0.0) or predicted * observed > 0),
                "final_source_loss": final_source_loss,
                "dreamsim_to_source": dreamsim_value,
                "dreamsim_delta": dreamsim_value - baseline_dreamsim,
            })
            del candidate
            if device.type == "cuda":
                torch.cuda.empty_cache()
    _write_json(output / "finite_difference.json", {
        "baseline_final_source_loss": baseline_source_loss,
        "baseline_dreamsim_to_source": baseline_dreamsim,
        "global_residual_native_rms_ratios": args.finite_difference_ratios,
        "rows": finite_rows,
    })

    summary = {
        "status": "complete", "base_reward_anchors": anchors,
        "connector_forward_matches_true_final": connector_forward_error == 0.0,
        "true_final_branch_detached": not true_final_latent.requires_grad,
        "final_gradient_status": "ok", "output_dir": str(output),
        "outputs": {
            name: str((output / name).resolve()) for name in (
                "config.json", "gradient_comparison.json", "temporal_energy.json",
                "leap_approximation.json", "finite_difference.json",
                "source.png", "native_proxy.png", "native_full.png",
            )
        },
    }
    _write_json(output / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
