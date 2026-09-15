"""Validate spatiotemporally masked shared-direction Kontext terminal control.

Pixel endpoint interpolation is an oracle controllability diagnostic, not a
semantic edit-strength definition. Native Kontext velocity is never scaled.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from diffusers.pipelines.flux.pipeline_flux_kontext import FluxKontextPipeline
from diffusers.pipelines.rewardflow.pipeline_flux_kontext_terminal_control import (
    FluxKontextTerminalControlPipeline,
)
from diffusers.pipelines.rewardflow.terminal_control import (
    EndpointPixelTargetLoss,
    MonotonicStrengthCalibration,
    amplitude_scaled_effective_controls,
    endpoint_soft_mask,
    initialize_velocity_controls,
    masked_effective_controls,
    normalized_effective_control_energy,
    update_best_control_checkpoint,
)


DEFAULT_PROMPT = "Make the weighted training ball blue while preserving its shape, texture, lighting, and background."


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.getenv("FLUX_KONTEXT_MODEL_PATH"))
    parser.add_argument("--source", required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--strengths", type=float, nargs="+", default=(0.2, 0.5, 0.8))
    parser.add_argument("--control-steps", type=int, choices=(2, 3, 4), default=2)
    parser.add_argument(
        "--control-mode", choices=("independent", "shared-linear", "shared-calibrated"), default="independent"
    )
    parser.add_argument("--control-mask-mode", choices=("none", "velocity-topk"), default="none")
    parser.add_argument("--control-mask-topk-fraction", type=float, default=0.25)
    parser.add_argument(
        "--outer-iters",
        type=int,
        default=20,
        help="Optimization iterations for independent/shared-linear modes; ignored by shared-calibrated.",
    )
    parser.add_argument(
        "--direction-iters", type=int, default=10, help="Stage-A iterations in shared-calibrated mode."
    )
    parser.add_argument(
        "--calibration-iters", type=int, default=10, help="Stage-B iterations in shared-calibrated mode."
    )
    parser.add_argument("--calibration-lr", type=float, default=0.05, help="Stage-B calibration learning rate.")
    parser.add_argument("--control-lr", type=float, default=0.1)
    parser.add_argument("--lambda-control", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--use-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if not args.model:
        parser.error("Set FLUX_KONTEXT_MODEL_PATH or pass --model.")
    if args.steps < args.control_steps:
        parser.error("--steps must be at least --control-steps.")
    if args.outer_iters < 0:
        parser.error("--outer-iters must be non-negative.")
    if args.direction_iters < 0 or args.calibration_iters < 0:
        parser.error("--direction-iters and --calibration-iters must be non-negative.")
    if args.control_lr <= 0 or not math.isfinite(args.control_lr):
        parser.error("--control-lr must be finite and positive.")
    if args.calibration_lr <= 0 or not math.isfinite(args.calibration_lr):
        parser.error("--calibration-lr must be finite and positive.")
    if args.lambda_control < 0 or not math.isfinite(args.lambda_control):
        parser.error("--lambda-control must be finite and non-negative.")
    if not 0 < args.control_mask_topk_fraction <= 1:
        parser.error("--control-mask-topk-fraction must lie in (0, 1].")
    if args.grad_clip is not None and (args.grad_clip <= 0 or not math.isfinite(args.grad_clip)):
        parser.error("--grad-clip must be finite and positive when provided.")
    if len(set(args.strengths)) != len(args.strengths) or any(not 0 <= value <= 1 for value in args.strengths):
        parser.error("Strengths must be unique values in [0, 1].")
    if args.control_mode == "shared-calibrated":
        if args.control_steps != 4:
            parser.error("shared-calibrated is a fixed T=4 diagnostic; pass --control-steps 4.")
        if args.control_mask_mode != "velocity-topk" or args.control_mask_topk_fraction != 0.25:
            parser.error(
                "shared-calibrated requires --control-mask-mode velocity-topk and --control-mask-topk-fraction 0.25."
            )
        if any(not 0 < value < 1 for value in args.strengths):
            parser.error("shared-calibrated strengths must be interior points in (0, 1).")
        if tuple(sorted(args.strengths)) != tuple(args.strengths):
            parser.error("shared-calibrated strengths must be strictly increasing.")
    return args


def _pil_to_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    array = image.detach().clamp(0, 1).mul(255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array, mode="RGB")


def _float_tag(value: float) -> str:
    return format(value, "g").replace("-", "m").replace(".", "p")


def _latent_parity(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual = actual.detach().float().flatten()
    expected = expected.detach().float().flatten()
    difference = (actual - expected).abs()
    return {
        "max_absolute_error": difference.max().item(),
        "mean_absolute_error": difference.mean().item(),
        "cosine_similarity": F.cosine_similarity(actual[None], expected[None]).item(),
    }


def _model_has_gradient(pipe) -> bool:
    modules = (pipe.transformer, pipe.vae, pipe.text_encoder, pipe.text_encoder_2)
    return any(
        parameter.grad is not None for module in modules if module is not None for parameter in module.parameters()
    )


def _pixel_target(source: torch.Tensor, full: torch.Tensor, strength: float) -> torch.Tensor:
    return (1 - strength) * source + strength * full


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    expanded = weight.to(device=value.device, dtype=value.dtype).expand_as(value)
    return (value * expanded).sum() / expanded.sum().clamp_min(torch.finfo(value.dtype).eps)


def _endpoint_metrics(image, source, full, target, endpoint_weight, velocity_image_mask) -> dict[str, float]:
    differences = {
        "source": image.float() - source.float(),
        "full": image.float() - full.float(),
        "target": image.float() - target.float(),
    }
    metrics = {}
    for name, difference in differences.items():
        metrics[f"mse_to_{name}"] = difference.square().mean().item()
        metrics[f"mad_to_{name}"] = difference.abs().mean().item()
    unit_endpoint = (endpoint_weight / endpoint_weight.amax(dim=(2, 3), keepdim=True).clamp_min(1e-8)).clamp(0, 1)
    background = 1 - unit_endpoint
    metrics["background_mse_to_source"] = _weighted_mean(differences["source"].square(), background).item()
    metrics["background_mad_to_source"] = _weighted_mean(differences["source"].abs(), background).item()
    inside = velocity_image_mask
    outside = 1 - inside
    metrics["inside_velocity_mask_mse_to_source"] = _weighted_mean(differences["source"].square(), inside).item()
    metrics["outside_velocity_mask_mse_to_source"] = _weighted_mean(differences["source"].square(), outside).item()
    return metrics


def _trajectory_distances(unroll, native_states) -> list[dict[str, float | int]]:
    rows = []
    for state_index, (controlled, native) in enumerate(zip(unroll.states, native_states)):
        difference = controlled.detach().float() - native.detach().float()
        rows.append(
            {
                "state_index": state_index,
                "after_step_index": state_index - 1,
                "latent_mad_from_native": difference.abs().mean().item(),
                "latent_l2_from_native": torch.linalg.vector_norm(difference).item(),
            }
        )
    return rows


def _active_energy(controls, masks) -> torch.Tensor:
    values = []
    for control, mask in zip(controls, masks):
        expanded = mask.to(control).expand_as(control)
        values.append((control.float().square() * expanded).sum() / expanded.sum().clamp_min(1))
    return torch.stack(values).mean()


def _control_diagnostics(directions, controls, native_velocities, masks):
    rows = []
    for step_index, (direction, control, native, mask) in enumerate(
        zip(directions, controls, native_velocities, masks)
    ):
        direction_flat = direction.detach().float().flatten()
        control_float = control.detach().float()
        native_float = native.detach().float()
        control_flat = control_float.flatten()
        native_flat = native_float.flatten()
        expanded_mask = mask.to(control_float).expand_as(control_float)
        active_count = expanded_mask.sum().clamp_min(1)
        control_norm = torch.linalg.vector_norm(control_flat).item()
        native_norm = torch.linalg.vector_norm(native_flat).item()
        active_control_rms = ((control_float.square() * expanded_mask).sum() / active_count).sqrt().item()
        active_native_rms = ((native_float.square() * expanded_mask).sum() / active_count).sqrt().item()
        rows.append(
            {
                "step_index": step_index,
                "raw_direction_norm": torch.linalg.vector_norm(direction_flat).item(),
                "effective_control_norm": control_norm,
                "native_velocity_norm": native_norm,
                "active_region_effective_control_rms": active_control_rms,
                "active_region_native_velocity_rms": active_native_rms,
                "active_control_native_ratio": active_control_rms / active_native_rms if active_native_rms else None,
                "global_effective_control_rms": control_float.square().mean().sqrt().item(),
                "global_native_velocity_rms": native_float.square().mean().sqrt().item(),
                "global_control_native_ratio": control_norm / native_norm if native_norm else None,
                "control_native_cosine": (
                    F.cosine_similarity(control_flat[None], native_flat[None]).item()
                    if control_norm and native_norm
                    else None
                ),
                "mask_active_fraction": mask.float().mean().item(),
                "max_control_outside_mask": (control_float * (1 - expanded_mask)).abs().max().item(),
            }
        )
    return rows


def _evaluate(
    pipe,
    inputs,
    directions,
    masks,
    strength,
    shared,
    objective,
    native_states,
    source,
    full,
    target,
    endpoint_weight,
    velocity_image_mask,
    amplitude=None,
):
    if amplitude is None:
        controls = masked_effective_controls(directions, masks, strength=strength if shared else None)
    else:
        controls = amplitude_scaled_effective_controls(directions, masks, amplitude)
    with torch.no_grad():
        unroll = pipe.unroll_terminal_controls(inputs, controls, use_checkpointing=False)
        image = pipe.decode_terminal_latent(unroll.final_latent, inputs).detach()
        output = objective(image, strength)
    return {
        "image": image.cpu(),
        "objective_error": output.objective_error.item(),
        "terminal_objective_loss": output.loss.item(),
        "source_blue_score": output.source_score.mean().item(),
        "full_blue_score": output.full_score.mean().item(),
        "target_blue_score": output.target_score.mean().item(),
        "achieved_blue_score": output.achieved_score.mean().item(),
        "controlled_steps": _control_diagnostics(directions, controls, unroll.native_control_velocities, masks),
        "trajectory_distances": _trajectory_distances(unroll, native_states),
        "endpoint_distances": _endpoint_metrics(image, source, full, target, endpoint_weight, velocity_image_mask),
    }


def _independent_optimize(
    pipe, inputs, masks, objective, args, strength, native_states, source, full, target, endpoint_weight, image_mask
):
    directions = initialize_velocity_controls(inputs.initial_latent, args.control_steps)
    optimizer = torch.optim.Adam(directions, lr=args.control_lr)
    initial = _evaluate(
        pipe,
        inputs,
        directions,
        masks,
        strength,
        False,
        objective,
        native_states,
        source,
        full,
        target,
        endpoint_weight,
        image_mask,
    )
    best = update_best_control_checkpoint(
        None, iteration=0, objective_error=torch.tensor(initial["objective_error"]), controls=directions
    )
    trace = []
    for iteration in range(1, args.outer_iters + 1):
        optimizer.zero_grad(set_to_none=True)
        controls = masked_effective_controls(directions, masks)
        started = time.perf_counter()
        unroll = pipe.unroll_terminal_controls(inputs, controls, use_checkpointing=args.use_checkpointing)
        image = pipe.decode_terminal_latent(unroll.final_latent, inputs)
        output = objective(image, strength)
        best = update_best_control_checkpoint(
            best,
            iteration=iteration - 1,
            objective_error=output.objective_error,
            controls=directions,
        )
        regularization = normalized_effective_control_energy([controls])
        total_loss = output.loss + args.lambda_control * regularization
        total_loss.backward()
        if any(parameter.grad is None or not torch.isfinite(parameter.grad).all() for parameter in directions):
            raise RuntimeError("Independent control gradients are missing or non-finite.")
        if _model_has_gradient(pipe):
            raise RuntimeError("Frozen model parameters unexpectedly received gradients.")
        diagnostics = _control_diagnostics(directions, controls, unroll.native_control_velocities, masks)
        for row, parameter in zip(diagnostics, directions):
            row["direction_gradient_norm"] = torch.linalg.vector_norm(parameter.grad.detach().float()).item()
        if args.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(directions, args.grad_clip)
        optimizer.step()
        trace.append(
            {
                "outer_iter": iteration,
                "strength": strength,
                "objective_error_before_step": output.objective_error.detach().item(),
                "terminal_objective_loss_before_step": output.loss.detach().item(),
                "global_control_energy": regularization.detach().item(),
                "active_control_energy": _active_energy(controls, masks).detach().item(),
                "controlled_steps": diagnostics,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        del image, output, total_loss, unroll
    final = _evaluate(
        pipe,
        inputs,
        directions,
        masks,
        strength,
        False,
        objective,
        native_states,
        source,
        full,
        target,
        endpoint_weight,
        image_mask,
    )
    best = update_best_control_checkpoint(
        best, iteration=args.outer_iters, objective_error=torch.tensor(final["objective_error"]), controls=directions
    )
    best_result = _evaluate(
        pipe,
        inputs,
        best.controls,
        masks,
        strength,
        False,
        objective,
        native_states,
        source,
        full,
        target,
        endpoint_weight,
        image_mask,
    )
    controls = masked_effective_controls(directions, masks)
    global_energy = normalized_effective_control_energy([controls]).detach().item()
    reduction = initial["objective_error"] - final["objective_error"]
    return {
        "initial": initial,
        "best": best_result,
        "final": final,
        "best_iter": best.iteration,
        "best_objective_error": best.objective_error,
        "final_objective_error": final["objective_error"],
        "global_control_energy": global_energy,
        "active_control_energy": _active_energy(controls, masks).detach().item(),
        "error_reduction": reduction,
        "error_reduction_per_global_control_energy": reduction / global_energy if global_energy else None,
        "trace": trace,
    }


def _shared_optimize(
    pipe,
    inputs,
    masks,
    objective,
    args,
    strengths,
    native_states,
    source,
    full,
    targets,
    endpoint_weight,
    image_mask,
    iteration_count=None,
):
    if iteration_count is None:
        iteration_count = args.outer_iters
    directions = initialize_velocity_controls(inputs.initial_latent, args.control_steps)
    optimizer = torch.optim.Adam(directions, lr=args.control_lr)

    def evaluate_all(parameters):
        return {
            strength: _evaluate(
                pipe,
                inputs,
                parameters,
                masks,
                strength,
                True,
                objective,
                native_states,
                source,
                full,
                targets[strength],
                endpoint_weight,
                image_mask,
            )
            for strength in strengths
        }

    initial = evaluate_all(directions)
    initial_mean = sum(item["objective_error"] for item in initial.values()) / len(strengths)
    best = update_best_control_checkpoint(
        None, iteration=0, objective_error=torch.tensor(initial_mean), controls=directions
    )
    trace = []
    for iteration in range(1, iteration_count + 1):
        optimizer.zero_grad(set_to_none=True)
        branch_errors = []
        for strength in strengths:
            controls = masked_effective_controls(directions, masks, strength=strength)
            started = time.perf_counter()
            unroll = pipe.unroll_terminal_controls(inputs, controls, use_checkpointing=args.use_checkpointing)
            image = pipe.decode_terminal_latent(unroll.final_latent, inputs)
            output = objective(image, strength)
            (output.loss / len(strengths)).backward()
            branch_errors.append(output.objective_error.detach())
            trace.append(
                {
                    "outer_iter": iteration,
                    "strength": strength,
                    "objective_error_before_step": output.objective_error.detach().item(),
                    "terminal_objective_loss_before_step": output.loss.detach().item(),
                    "controlled_steps": _control_diagnostics(
                        directions, controls, unroll.native_control_velocities, masks
                    ),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            del image, output, unroll
        regularization_controls = [
            masked_effective_controls(directions, masks, strength=strength) for strength in strengths
        ]
        regularization = normalized_effective_control_energy(regularization_controls)
        (args.lambda_control * regularization).backward()
        if any(parameter.grad is None or not torch.isfinite(parameter.grad).all() for parameter in directions):
            raise RuntimeError("Shared direction gradients are missing or non-finite.")
        if _model_has_gradient(pipe):
            raise RuntimeError("Frozen model parameters unexpectedly received gradients.")
        mean_error = torch.stack(branch_errors).mean()
        best = update_best_control_checkpoint(
            best, iteration=iteration - 1, objective_error=mean_error, controls=directions
        )
        active_energy = torch.stack([_active_energy(controls, masks) for controls in regularization_controls]).mean()
        gradient_norms = [torch.linalg.vector_norm(parameter.grad.detach().float()).item() for parameter in directions]
        for row in trace[-len(strengths) :]:
            row["global_control_energy"] = regularization.detach().item()
            row["active_control_energy"] = active_energy.detach().item()
            row["shared_direction_gradient_norms"] = gradient_norms
        if args.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(directions, args.grad_clip)
        optimizer.step()

    final = evaluate_all(directions)
    final_mean = sum(item["objective_error"] for item in final.values()) / len(strengths)
    best = update_best_control_checkpoint(
        best, iteration=iteration_count, objective_error=torch.tensor(final_mean), controls=directions
    )
    best_results = evaluate_all(best.controls)
    families = [masked_effective_controls(directions, masks, strength=s) for s in strengths]
    global_energy = normalized_effective_control_energy(families).detach().item()
    active_energy = torch.stack([_active_energy(controls, masks) for controls in families]).mean().detach().item()
    results = {}
    for strength in strengths:
        reduction = initial[strength]["objective_error"] - final[strength]["objective_error"]
        results[strength] = {
            "initial": initial[strength],
            "best": best_results[strength],
            "final": final[strength],
            "best_iter": best.iteration,
            "best_family_mean_objective_error": best.objective_error,
            "final_objective_error": final[strength]["objective_error"],
            "global_control_energy": global_energy,
            "active_control_energy": active_energy,
            "error_reduction": reduction,
            "error_reduction_per_global_control_energy": reduction / global_energy if global_energy else None,
            "trace": [row for row in trace if row["strength"] == strength],
        }
    stats = {
        "raw_shared_direction_norms": [
            torch.linalg.vector_norm(direction.detach().float()).item() for direction in directions
        ],
        "strength_scaling": {str(strength): 1 - strength for strength in strengths},
        "number_of_direction_parameter_tensors": len(directions),
        "has_branch_specific_direction_parameters": False,
        "best_family_mean_objective_error": best.objective_error,
        "final_family_mean_objective_error": final_mean,
    }
    frozen_directions = tuple(direction.detach().clone() for direction in directions)
    return results, stats, frozen_directions


def _calibration_state(calibration, strengths):
    return {
        "raw_interval_logits": calibration.raw_interval_logits.detach().float().cpu().tolist(),
        "interval_drops": calibration.interval_drops().detach().float().cpu().tolist(),
        "amplitudes": {
            str(strength): calibration.amplitude(strength).detach().float().cpu().item() for strength in strengths
        },
        "endpoint_amplitudes": {
            "0.0": calibration.amplitude(0.0).detach().float().cpu().item(),
            "1.0": calibration.amplitude(1.0).detach().float().cpu().item(),
        },
    }


def _calibrate_amplitudes(
    pipe,
    inputs,
    directions,
    masks,
    objective,
    args,
    strengths,
    native_states,
    source,
    full,
    targets,
    endpoint_weight,
    image_mask,
):
    directions = tuple(direction.detach().clone().requires_grad_(False) for direction in directions)
    calibration = MonotonicStrengthCalibration(strengths, device=inputs.initial_latent.device)
    optimizer = torch.optim.Adam(calibration.parameters(), lr=args.calibration_lr)

    def evaluate_all(module):
        return {
            strength: _evaluate(
                pipe,
                inputs,
                directions,
                masks,
                strength,
                True,
                objective,
                native_states,
                source,
                full,
                targets[strength],
                endpoint_weight,
                image_mask,
                amplitude=module.amplitude(strength).detach(),
            )
            for strength in strengths
        }

    initial_state = _calibration_state(calibration, strengths)
    initial = evaluate_all(calibration)
    initial_mean = sum(item["objective_error"] for item in initial.values()) / len(strengths)
    best_mean = initial_mean
    best_iteration = 0
    best_logits = calibration.raw_interval_logits.detach().clone()
    trace = []

    for iteration in range(1, args.calibration_iters + 1):
        optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        branch_errors = {}
        branch_losses = {}
        state_before_step = _calibration_state(calibration, strengths)
        for strength in strengths:
            # Each branch gets a newly computed amplitude graph. This makes sequential
            # backward safe without retain_graph while keeping the shared logits trainable.
            amplitude = calibration.amplitude(strength)
            controls = amplitude_scaled_effective_controls(directions, masks, amplitude)
            unroll = pipe.unroll_terminal_controls(inputs, controls, use_checkpointing=args.use_checkpointing)
            image = pipe.decode_terminal_latent(unroll.final_latent, inputs)
            output = objective(image, strength)
            (output.loss / len(strengths)).backward()
            branch_errors[str(strength)] = output.objective_error.detach().item()
            branch_losses[str(strength)] = output.loss.detach().item()
            del controls, unroll, image, output, amplitude

        # Recompute all amplitudes and controls so the regularizer owns a fresh graph.
        regularization_controls = [
            amplitude_scaled_effective_controls(directions, masks, calibration.amplitude(strength))
            for strength in strengths
        ]
        regularization = normalized_effective_control_energy(regularization_controls)
        (args.lambda_control * regularization).backward()
        gradient = calibration.raw_interval_logits.grad
        if gradient is None or not torch.isfinite(gradient).all():
            raise RuntimeError("Calibration-logit gradient is missing or non-finite.")
        if any(direction.grad is not None for direction in directions):
            raise RuntimeError("Frozen shared directions unexpectedly received gradients during calibration.")
        if _model_has_gradient(pipe):
            raise RuntimeError("Frozen model parameters unexpectedly received gradients during calibration.")

        mean_error = sum(branch_errors.values()) / len(strengths)
        if mean_error < best_mean:
            best_mean = mean_error
            best_iteration = iteration - 1
            best_logits = calibration.raw_interval_logits.detach().clone()
        active_energy = torch.stack([_active_energy(controls, masks) for controls in regularization_controls]).mean()
        trace.append(
            {
                "calibration_iter": iteration,
                "state_before_step": state_before_step,
                "objective_error_before_step": branch_errors,
                "terminal_objective_loss_before_step": branch_losses,
                "mean_objective_error_before_step": mean_error,
                "global_control_energy": regularization.detach().item(),
                "active_control_energy": active_energy.detach().item(),
                "calibration_logit_gradient_norm": torch.linalg.vector_norm(gradient.detach().float()).item(),
                "directions_have_gradient": any(direction.grad is not None for direction in directions),
                "model_parameters_have_gradient": _model_has_gradient(pipe),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        optimizer.step()

    final_state = _calibration_state(calibration, strengths)
    final = evaluate_all(calibration)
    final_mean = sum(item["objective_error"] for item in final.values()) / len(strengths)
    final_families = [
        amplitude_scaled_effective_controls(directions, masks, calibration.amplitude(strength).detach())
        for strength in strengths
    ]
    final_global_energy = normalized_effective_control_energy(final_families).detach().item()
    if final_mean < best_mean:
        best_mean = final_mean
        best_iteration = args.calibration_iters
        best_logits = calibration.raw_interval_logits.detach().clone()

    with torch.no_grad():
        calibration.raw_interval_logits.copy_(best_logits)
    best_state = _calibration_state(calibration, strengths)
    best = evaluate_all(calibration)
    best_families = [
        amplitude_scaled_effective_controls(directions, masks, calibration.amplitude(strength).detach())
        for strength in strengths
    ]
    best_global_energy = normalized_effective_control_energy(best_families).detach().item()

    results = {}
    for strength in strengths:
        final_controls = amplitude_scaled_effective_controls(
            directions, masks, torch.tensor(final_state["amplitudes"][str(strength)], device=directions[0].device)
        )
        final_energy = normalized_effective_control_energy([final_controls]).detach().item()
        reduction = initial[strength]["objective_error"] - final[strength]["objective_error"]
        results[strength] = {
            "initial": initial[strength],
            "best": best[strength],
            "final": final[strength],
            "best_iter": best_iteration,
            "best_family_mean_objective_error": best_mean,
            "final_objective_error": final[strength]["objective_error"],
            "global_control_energy": final_energy,
            "active_control_energy": _active_energy(final_controls, masks).detach().item(),
            "error_reduction": reduction,
            "error_reduction_per_global_control_energy": reduction / final_energy if final_energy else None,
            "trace": [],
        }

    calibration_report = {
        "parameterization": "softmax interval drops with exact amplitude endpoints A(0)=1 and A(1)=0",
        "directions_frozen": True,
        "initial": initial_state,
        "best": {"iteration": best_iteration, "mean_objective_error": best_mean, **best_state},
        "best_global_control_energy": best_global_energy,
        "final": {
            "iteration": args.calibration_iters,
            "mean_objective_error": final_mean,
            **final_state,
        },
        "final_global_control_energy": final_global_energy,
        "trace": trace,
    }
    return results, calibration_report


def _score_to_pil(score: torch.Tensor, token_height: int, token_width: int, image_size):
    value = score.reshape(1, 1, token_height, token_width)
    minimum, maximum = value.amin(), value.amax()
    normalized = ((value - minimum) / (maximum - minimum).clamp_min(1e-8))[0, 0]
    rgb = torch.stack([normalized, 1 - (2 * normalized - 1).abs(), 1 - normalized])
    array = rgb.mul(255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array, mode="RGB").resize(image_size, Image.Resampling.NEAREST)


def _mask_to_pil(mask: torch.Tensor, token_height: int, token_width: int, image_size):
    array = mask[0, :, 0].reshape(token_height, token_width).mul(255).to(torch.uint8).cpu().numpy()
    return Image.fromarray(array, mode="L").convert("RGB").resize(image_size, Image.Resampling.NEAREST)


def _save_mask_visualizations(output_dir, scores, masks, token_height, token_width, image_size):
    panels = []
    for step, (score, mask) in enumerate(zip(scores, masks)):
        score_image = _score_to_pil(score, token_height, token_width, image_size)
        mask_image = _mask_to_pil(mask, token_height, token_width, image_size)
        score_image.save(output_dir / f"velocity_score_step{step}.png")
        mask_image.save(output_dir / f"velocity_mask_step{step}.png")
        panels.extend([(score_image, f"Score step {step}"), (mask_image, f"Mask step {step}")])
    width, height = image_size
    grid = Image.new("RGB", (width * len(panels), height + 30), "white")
    draw = ImageDraw.Draw(grid)
    for index, (panel, label) in enumerate(panels):
        draw.text((index * width + 6, 8), label, fill="black")
        grid.paste(panel, (index * width, 30))
    grid.save(output_dir / "velocity_masks_grid.png")


def _make_grid(source, native, targets, results, result_name):
    images = [source]
    labels = ["Source"]
    for strength, target in targets.items():
        images.extend([_tensor_to_pil(target[0]), _tensor_to_pil(results[strength][result_name]["image"][0])])
        labels.extend([f"Target {strength:g}", f"{result_name.title()} {strength:g}"])
    images.append(native)
    labels.append("Native Full")
    width, height = source.size
    grid = Image.new("RGB", (width * len(images), height + 30), "white")
    draw = ImageDraw.Draw(grid)
    for index, (image, label) in enumerate(zip(images, labels)):
        draw.text((index * width + 6, 8), label, fill="black")
        grid.paste(image.resize((width, height), Image.Resampling.LANCZOS), (index * width, 30))
    return grid


def _make_linear_vs_calibrated_grid(source, native, targets, linear_results, calibrated_results, result_name):
    images = [source]
    labels = ["Source"]
    for strength, target in targets.items():
        images.extend(
            [
                _tensor_to_pil(target[0]),
                _tensor_to_pil(linear_results[strength]["final"]["image"][0]),
                _tensor_to_pil(calibrated_results[strength][result_name]["image"][0]),
            ]
        )
        labels.extend([f"Target {strength:g}", f"Linear {strength:g}", f"Calibrated {result_name} {strength:g}"])
    images.append(native)
    labels.append("Native Full")
    width, height = source.size
    grid = Image.new("RGB", (width * len(images), height + 30), "white")
    draw = ImageDraw.Draw(grid)
    for index, (image, label) in enumerate(zip(images, labels)):
        draw.text((index * width + 6, 8), label, fill="black")
        grid.paste(image.resize((width, height), Image.Resampling.LANCZOS), (index * width, 30))
    return grid


def _write_trace(path, rows):
    normalized = [
        {key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in row.items()}
        for row in rows
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        if normalized:
            writer = csv.DictWriter(handle, fieldnames=list(normalized[0]))
            writer.writeheader()
            writer.writerows(normalized)


def _serializable(result):
    output = {}
    for key, value in result.items():
        if key == "trace":
            continue
        if key in {"initial", "best", "final"}:
            output[key] = {nested: item for nested, item in value.items() if nested != "image"}
        else:
            output[key] = value
    return output


def main():
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Real Kontext validation requires CUDA.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_pil = Image.open(args.source).convert("RGB").resize((args.width, args.height), Image.Resampling.LANCZOS)
    source_pil.save(output_dir / "source.png")
    config = vars(args).copy()
    config["model"] = os.path.realpath(args.model)
    config["source"] = os.path.realpath(args.source)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    torch.cuda.set_device(device)
    pipe = FluxKontextTerminalControlPipeline.from_pretrained(args.model, torch_dtype=dtype, local_files_only=True).to(
        device
    )
    pipe.set_progress_bar_config(disable=True)
    inputs = pipe.prepare_terminal_control_inputs(
        image=source_pil,
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=torch.Generator(device=device).manual_seed(args.seed),
    )
    common = {
        "image": source_pil,
        "prompt": args.prompt,
        "height": args.height,
        "width": args.width,
        "max_area": args.height * args.width,
        "_auto_resize": False,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "output_type": "latent",
    }
    official = FluxKontextPipeline.__call__(
        pipe, **common, generator=torch.Generator(device=device).manual_seed(args.seed)
    ).images
    velocity_masks, native_unroll = pipe.prepare_velocity_edit_masks(
        inputs,
        control_steps=args.control_steps,
        mode=args.control_mask_mode,
        topk_fraction=args.control_mask_topk_fraction,
    )
    official_parity = _latent_parity(inputs.native_final_latent, official)
    zero_parity = _latent_parity(native_unroll.final_latent, inputs.native_final_latent)
    if official_parity["max_absolute_error"] != 0 or zero_parity["max_absolute_error"] != 0:
        raise RuntimeError("Zero-control parity failed; optimization was not started.")

    _save_mask_visualizations(
        output_dir,
        velocity_masks.scores,
        velocity_masks.masks,
        inputs.sampling_token_height,
        inputs.sampling_token_width,
        source_pil.size,
    )
    union_mask = (
        torch.stack(velocity_masks.masks)
        .amax(dim=0)
        .permute(0, 2, 1)
        .reshape(1, 1, inputs.sampling_token_height, inputs.sampling_token_width)
    )
    velocity_image_mask = F.interpolate(union_mask, size=(args.height, args.width), mode="nearest")
    source = _pil_to_tensor(source_pil, device)
    with torch.no_grad():
        full = pipe.decode_terminal_latent(inputs.native_final_latent, inputs).detach()
    native_pil = _tensor_to_pil(full[0])
    native_pil.save(output_dir / "native_full.png")
    endpoint_weight = endpoint_soft_mask(source, full)
    objective = EndpointPixelTargetLoss(source, full)
    strengths = tuple(float(value) for value in args.strengths)
    targets = {strength: _pixel_target(source, full, strength).detach() for strength in strengths}
    for strength, target in targets.items():
        _tensor_to_pil(target[0]).save(output_dir / f"target_{_float_tag(strength)}.png")

    calibration_report = None
    linear_results = None
    stage_times = {}
    if args.control_mode == "shared-linear":
        stage_started = time.perf_counter()
        results, shared_stats, _ = _shared_optimize(
            pipe,
            inputs,
            velocity_masks.masks,
            objective,
            args,
            strengths,
            native_unroll.states,
            source,
            full,
            targets,
            endpoint_weight,
            velocity_image_mask,
        )
        stage_times["shared_linear_seconds"] = time.perf_counter() - stage_started
        (output_dir / "shared_direction_stats.json").write_text(json.dumps(shared_stats, indent=2), encoding="utf-8")
    elif args.control_mode == "shared-calibrated":
        stage_started = time.perf_counter()
        linear_results, shared_stats, directions = _shared_optimize(
            pipe,
            inputs,
            velocity_masks.masks,
            objective,
            args,
            strengths,
            native_unroll.states,
            source,
            full,
            targets,
            endpoint_weight,
            velocity_image_mask,
            iteration_count=args.direction_iters,
        )
        stage_times["direction_learning_seconds"] = time.perf_counter() - stage_started
        stage_started = time.perf_counter()
        results, calibration_report = _calibrate_amplitudes(
            pipe,
            inputs,
            directions,
            velocity_masks.masks,
            objective,
            args,
            strengths,
            native_unroll.states,
            source,
            full,
            targets,
            endpoint_weight,
            velocity_image_mask,
        )
        stage_times["amplitude_calibration_seconds"] = time.perf_counter() - stage_started
        (output_dir / "shared_direction_stats.json").write_text(json.dumps(shared_stats, indent=2), encoding="utf-8")
        (output_dir / "strength_calibration.json").write_text(
            json.dumps(calibration_report, indent=2), encoding="utf-8"
        )
        (output_dir / "calibration_evolution.json").write_text(
            json.dumps(calibration_report["trace"], indent=2), encoding="utf-8"
        )
        for strength, result in linear_results.items():
            _tensor_to_pil(result["final"]["image"][0]).save(output_dir / f"linear_final_{_float_tag(strength)}.png")
        _make_grid(source_pil, native_pil, targets, linear_results, "final").save(output_dir / "linear_grid.png")
    else:
        shared_stats = None
        results = {
            strength: _independent_optimize(
                pipe,
                inputs,
                velocity_masks.masks,
                objective,
                args,
                strength,
                native_unroll.states,
                source,
                full,
                targets[strength],
                endpoint_weight,
                velocity_image_mask,
            )
            for strength in strengths
        }

    trace = []
    for strength, result in results.items():
        trace.extend(result["trace"])
        tag = _float_tag(strength)
        for result_name in ("best", "final"):
            _tensor_to_pil(result[result_name]["image"][0]).save(output_dir / f"{result_name}_{tag}.png")
    _make_grid(source_pil, native_pil, targets, results, "best").save(output_dir / "comparison_grid_best.png")
    _make_grid(source_pil, native_pil, targets, results, "final").save(output_dir / "comparison_grid_final.png")
    if linear_results is not None:
        _make_grid(source_pil, native_pil, targets, results, "best").save(output_dir / "calibrated_best_grid.png")
        _make_grid(source_pil, native_pil, targets, results, "final").save(output_dir / "calibrated_final_grid.png")
        _make_linear_vs_calibrated_grid(source_pil, native_pil, targets, linear_results, results, "best").save(
            output_dir / "linear_vs_calibrated_grid.png"
        )
    _write_trace(output_dir / "trace.csv", trace)

    source_order = [results[strength]["final"]["endpoint_distances"]["mse_to_source"] for strength in strengths]
    full_order = [results[strength]["final"]["endpoint_distances"]["mse_to_full"] for strength in strengths]
    report = {
        "scope": "oracle controller validation; pixel interpolation is not semantic strength",
        **config,
        "gpu_name": torch.cuda.get_device_name(device),
        "official_vs_native_parity": official_parity,
        "zero_control_vs_native_parity": zero_parity,
        "source_alignment": {
            "sampling_shape": list(inputs.initial_latent.shape),
            "aligned_source_shape": list(inputs.source_clean_latent.shape),
            "sampling_token_grid": [inputs.sampling_token_height, inputs.sampling_token_width],
        },
        "controlled_step_indices": list(range(args.control_steps)),
        "mask_active_fractions": [mask.float().mean().item() for mask in velocity_masks.masks],
        "max_absolute_effective_control_outside_mask": max(
            row["max_control_outside_mask"]
            for result in results.values()
            for row in result["final"]["controlled_steps"]
        ),
        "strength_ordering": {
            "mse_to_source_non_decreasing": all(a <= b for a, b in zip(source_order, source_order[1:])),
            "mse_to_full_non_increasing": all(a >= b for a, b in zip(full_order, full_order[1:])),
            "mse_to_source": source_order,
            "mse_to_full": full_order,
        },
        "model_parameters_have_gradient": _model_has_gradient(pipe),
        "shared_direction_stats": shared_stats,
        "stage_times": stage_times,
        "strength_calibration": calibration_report,
        "linear_stage_results": (
            {str(strength): _serializable(result) for strength, result in linear_results.items()}
            if linear_results is not None
            else None
        ),
        "results": {str(strength): _serializable(result) for strength, result in results.items()},
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("FINAL_REPORT=" + json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
