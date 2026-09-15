"""Run the frozen v5 FLUX-Kontext terminal controller with the v4 feature reward."""

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
from PIL import Image, ImageDraw

from diffusers.pipelines.rewardflow.endpoint_feature_distance import FeatureEndpointDistanceReward
from diffusers.pipelines.rewardflow.feature_controller_evaluation import (
    DENSE_STRENGTHS,
    FEATURE_CONTROL_PROVENANCE,
    NATIVE_FULL_PROVENANCE,
    SOURCE_INPUT_PROVENANCE,
    TRAINING_STRENGTHS,
    dense_feature_curve_diagnostics,
    format_strength_tag,
)
from diffusers.pipelines.rewardflow.pipeline_flux_kontext_terminal_control import FluxKontextTerminalControlPipeline
from diffusers.pipelines.rewardflow.relative_endpoint_parser import parse_relative_endpoint_semantic_json
from diffusers.pipelines.rewardflow.rewards import Qwen25VQATeacherForcedScorer
from diffusers.pipelines.rewardflow.terminal_control import (
    freeze_terminal_control_modules,
    initialize_velocity_controls,
    masked_effective_controls,
    normalized_control_energy,
    normalized_effective_control_energy,
    update_best_control_checkpoint,
)


DEFAULT_MODEL = "/data15/hyp/weight/FLUX.1-Kontext-dev"
DEFAULT_QWEN = "/data15/hyp/weight/reward_models/Qwen2.5-VL-3B-Instruct"
DEFAULT_SOURCE = "/data15/hyp/dataset/kontinuous_kontext/raw/source_images/source_000000.png"
DEFAULT_SPEC = "examples/rewardflow/relative_endpoint_ball_spec.json"
DEFAULT_OUTPUT = "experiments/feature_reward_controller_v5"
DEFAULT_PROMPT = "Make the weighted training ball blue while preserving its shape, texture, lighting, and background."


def _args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.getenv("FLUX_KONTEXT_MODEL_PATH", DEFAULT_MODEL))
    parser.add_argument("--qwen-model", default=os.getenv("QWEN25_VL_MODEL_PATH", DEFAULT_QWEN))
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--relative-spec", default=DEFAULT_SPEC)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--control-steps", type=int, default=4)
    parser.add_argument("--topk-fraction", type=float, default=0.25)
    parser.add_argument("--outer-iters", type=int, default=20)
    parser.add_argument("--control-lr", type=float, default=0.1)
    parser.add_argument("--lambda-control", type=float, default=1e-4)
    parser.add_argument("--use-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    for name in ("model", "qwen_model", "source", "relative_spec"):
        if not Path(getattr(args, name)).exists():
            parser.error(f"`--{name.replace('_', '-')}` must point to an existing local path.")
    formal = (
        args.prompt == DEFAULT_PROMPT
        and (args.steps, args.seed, args.height, args.width) == (12, 20260914, 256, 256)
        and args.guidance_scale == 2.5
        and args.control_steps == 4
        and args.topk_fraction == 0.25
        and args.outer_iters == 20
        and args.control_lr == 0.1
        and args.lambda_control == 1e-4
        and args.use_checkpointing
    )
    if not formal:
        parser.error("The formal v5 controller configuration is frozen and cannot be changed.")
    return args


def _serialize(value):
    if torch.is_tensor(value):
        value = value.detach().float().cpu()
        return value.item() if value.numel() == 1 else value.tolist()
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_serialize(item) for item in value]
    return value


def _json(path: Path, value) -> None:
    path.write_text(json.dumps(_serialize(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(_serialize(rows))


def _pil(image: torch.Tensor) -> Image.Image:
    array = image.detach().clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array, mode="RGB")


def _tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32).copy() / 255
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def _load_spec(path: str | Path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return parse_relative_endpoint_semantic_json(json.dumps(payload.get("spec", payload)))


def _module_frozen(module) -> bool:
    return all(not parameter.requires_grad for parameter in module.parameters())


def _module_has_grad(module) -> bool:
    return any(parameter.grad is not None for parameter in module.parameters())


def _gradient_norm(parameters) -> torch.Tensor:
    squares = [parameter.grad.float().square().sum() for parameter in parameters if parameter.grad is not None]
    return torch.stack(squares).sum().sqrt() if squares else torch.tensor(float("nan"))


def _weighted_diagnostic(output, spec, key: str) -> torch.Tensor:
    weights = output.achieved_score.new_tensor([primitive.weight for primitive in spec.primitives])
    weights = weights / weights.sum()
    values = torch.stack([output.diagnostics[primitive.id][key] for primitive in spec.primitives])
    return (weights * values).sum()


def _evaluate_strengths(pipe, inputs, directions, masks, objective, strengths, source, full):
    results = {}
    with torch.no_grad():
        for strength in strengths:
            controls = masked_effective_controls(directions, masks, strength=strength)
            unroll = pipe.unroll_terminal_controls(inputs, controls, use_checkpointing=False)
            image = pipe.decode_terminal_latent(unroll.final_latent, inputs)
            output = objective(image, strength)
            results[float(strength)] = {
                "image": image.detach().cpu(),
                "feature_coordinate": output.achieved_score.detach(),
                "target": float(strength),
                "absolute_coordinate_error": output.objective_error.detach(),
                "semantic_loss": output.loss.detach(),
                "control_energy": normalized_control_energy(controls).detach(),
                "mse_to_source": (image.float() - source.float()).square().mean().detach(),
                "mse_to_native_full": (image.float() - full.float()).square().mean().detach(),
                "cosine_distance_source": _weighted_diagnostic(output, objective.spec, "cosine_distance_source"),
                "cosine_distance_full": _weighted_diagnostic(output, objective.spec, "cosine_distance_full"),
                "euclidean_distance_ratio": _weighted_diagnostic(output, objective.spec, "euclidean_distance_ratio"),
                "axis_projection": _weighted_diagnostic(output, objective.spec, "axis_projection"),
                "latent_max_abs_vs_native": (unroll.final_latent - inputs.native_final_latent).abs().max().detach(),
            }
    return results


def _mean_error(results) -> float:
    return sum(float(value["absolute_coordinate_error"]) for value in results.values()) / len(results)


def _optimize(pipe, inputs, masks, objective, source, full, args):
    directions = initialize_velocity_controls(inputs.initial_latent, args.control_steps)
    optimizer = torch.optim.Adam(directions, lr=args.control_lr)
    initial = _evaluate_strengths(pipe, inputs, directions, masks, objective, TRAINING_STRENGTHS, source, full)
    best = update_best_control_checkpoint(
        None,
        iteration=0,
        objective_error=torch.tensor(_mean_error(initial)),
        controls=directions,
    )
    trace = []
    gradients_finite = True
    for optimizer_step in range(1, args.outer_iters + 1):
        optimizer.zero_grad(set_to_none=True)
        pending_rows = []
        feature_errors = []
        for strength in TRAINING_STRENGTHS:
            controls = masked_effective_controls(directions, masks, strength=strength)
            unroll = pipe.unroll_terminal_controls(inputs, controls, use_checkpointing=args.use_checkpointing)
            image = pipe.decode_terminal_latent(unroll.final_latent, inputs)
            output = objective(image, strength)
            (output.loss / len(TRAINING_STRENGTHS)).backward()
            feature_errors.append(output.objective_error.detach())
            pending_rows.append(
                {
                    "optimizer_step": optimizer_step,
                    "evaluated_control_state": optimizer_step - 1,
                    "strength": strength,
                    "feature_coordinate": output.achieved_score.detach(),
                    "target": strength,
                    "absolute_coordinate_error": output.objective_error.detach(),
                    "semantic_loss": output.loss.detach(),
                    "control_energy": normalized_control_energy(controls).detach(),
                }
            )
            del unroll, image, output
        regularization = normalized_effective_control_energy(
            [masked_effective_controls(directions, masks, strength=value) for value in TRAINING_STRENGTHS]
        )
        (args.lambda_control * regularization).backward()
        gradient_norm = _gradient_norm(directions)
        current_gradients_finite = all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in directions
        )
        gradients_finite = gradients_finite and bool(current_gradients_finite)
        if not current_gradients_finite:
            raise RuntimeError("Shared terminal-control direction has missing or non-finite gradients.")
        if any(_module_has_grad(module) for module in (pipe.transformer, pipe.vae, objective.scorer.model)):
            raise RuntimeError("A frozen FLUX, VAE, or Qwen module accumulated parameter gradients.")
        mean_error = torch.stack(feature_errors).mean()
        best = update_best_control_checkpoint(
            best,
            iteration=optimizer_step - 1,
            objective_error=mean_error,
            controls=directions,
        )
        for row in pending_rows:
            row["mean_coordinate_error"] = mean_error
            row["shared_direction_gradient_norm"] = gradient_norm
            row["mean_effective_control_energy"] = regularization.detach()
            trace.append(row)
        optimizer.step()
    final = _evaluate_strengths(pipe, inputs, directions, masks, objective, TRAINING_STRENGTHS, source, full)
    best = update_best_control_checkpoint(
        best,
        iteration=args.outer_iters,
        objective_error=torch.tensor(_mean_error(final)),
        controls=directions,
    )
    best_results = _evaluate_strengths(pipe, inputs, best.controls, masks, objective, TRAINING_STRENGTHS, source, full)
    return directions, initial, best_results, final, best, trace, gradients_finite


def _save_phase_images(output_dir, phase, results):
    for strength, value in results.items():
        _pil(value["image"][0]).save(output_dir / f"{phase}_s{format_strength_tag(strength)}_MODEL_GENERATED.png")


def _grid(panels, path: Path, *, banner: str, label_height: int = 54):
    width, height = panels[0][0].size
    grid = Image.new("RGB", (width * len(panels), height + label_height), "white")
    draw = ImageDraw.Draw(grid)
    draw.text((4, 4), banner, fill="red")
    for index, (image, label) in enumerate(panels):
        draw.text((index * width + 4, 27), label, fill="black")
        grid.paste(image, (index * width, label_height))
    grid.save(path)


def _phase_rows(initial, best, final):
    rows = []
    for strength in TRAINING_STRENGTHS:
        rows.append(
            {
                "strength": strength,
                "initial_coordinate": initial[strength]["feature_coordinate"],
                "best_coordinate": best[strength]["feature_coordinate"],
                "final_coordinate": final[strength]["feature_coordinate"],
                "target": strength,
                "best_error": best[strength]["absolute_coordinate_error"],
                "initial_error": initial[strength]["absolute_coordinate_error"],
                "final_error": final[strength]["absolute_coordinate_error"],
                "best_control_energy": best[strength]["control_energy"],
            }
        )
    return rows


def main():
    args = _args()
    if not torch.cuda.is_available():
        raise RuntimeError("The formal feature-controller experiment requires CUDA.")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.init()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_pil = Image.open(args.source).convert("RGB").resize((args.width, args.height), Image.Resampling.LANCZOS)
    spec = _load_spec(args.relative_spec)
    # Formal-data validation is separate from the N-primitive core loop above.
    if [primitive.id for primitive in spec.primitives] != ["ball_color"] or spec.unresolved_instruction_items:
        raise ValueError("Formal v5 requires the submitted human-audited ball semantic spec.")

    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    pipe = FluxKontextTerminalControlPipeline.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, local_files_only=True
    ).to(device)
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
    masks, native = pipe.prepare_velocity_edit_masks(
        inputs,
        control_steps=args.control_steps,
        mode="velocity-topk",
        topk_fraction=args.topk_fraction,
    )
    zero_control_native_parity = torch.equal(native.final_latent, inputs.native_final_latent)
    if not zero_control_native_parity:
        raise RuntimeError("Zero-control terminal unroll does not exactly match native FLUX output.")
    with torch.no_grad():
        native_full = pipe.decode_terminal_latent(inputs.native_final_latent, inputs)
    source = _tensor(source_pil, device)
    source_pil.save(output_dir / "source_INPUT.png")
    _pil(native_full[0]).save(output_dir / "native_full_MODEL_GENERATED.png")

    scorer = Qwen25VQATeacherForcedScorer(
        args.qwen_model,
        device=device,
        dtype=torch.float32,
        local_files_only=True,
    )
    freeze_terminal_control_modules(
        pipe.transformer,
        pipe.vae,
        pipe.text_encoder,
        pipe.text_encoder_2,
        scorer.model,
    )
    objective = FeatureEndpointDistanceReward(scorer, spec, source, native_full)
    modules_frozen = {
        "transformer": _module_frozen(pipe.transformer),
        "vae": _module_frozen(pipe.vae),
        "qwen": _module_frozen(scorer.model),
    }
    directions, initial, best, final, checkpoint, trace, gradients_finite = _optimize(
        pipe, inputs, masks.masks, objective, source, native_full, args
    )
    _save_phase_images(output_dir, "initial", initial)
    _save_phase_images(output_dir, "best", best)
    _save_phase_images(output_dir, "final", final)
    phase_rows = _phase_rows(initial, best, final)
    _csv(output_dir / "training_strengths_initial_best_final.csv", phase_rows)
    _csv(output_dir / "optimization_trace.csv", trace)

    dense = _evaluate_strengths(
        pipe, inputs, checkpoint.controls, masks.masks, objective, DENSE_STRENGTHS, source, native_full
    )
    dense_rows = []
    dense_panels = []
    for strength, value in dense.items():
        image = _pil(value["image"][0])
        image.save(output_dir / f"dense_s{format_strength_tag(strength)}_MODEL_GENERATED.png")
        dense_panels.append((image, f"s={strength:.1f} coord={float(value['feature_coordinate']):.3f}"))
        dense_rows.append(
            {"requested_strength": strength, **{key: item for key, item in value.items() if key != "image"}}
        )
    _csv(output_dir / "dense_strengths.csv", dense_rows)
    _grid(
        dense_panels,
        output_dir / "MODEL_GENERATED_DENSE_FEATURE_SLIDER.png",
        banner="MODEL-GENERATED FLUX-KONTEXT OUTPUTS — FEATURE COORDINATE, NOT PERCEIVED STRENGTH",
    )
    feature_panels = [
        (source_pil, "Source input"),
        *[
            (
                _pil(best[strength]["image"][0]),
                f"Best s={strength:.1f} coord={float(best[strength]['feature_coordinate']):.3f}",
            )
            for strength in TRAINING_STRENGTHS
        ],
        (_pil(native_full[0]), "Native Full"),
    ]
    _grid(
        feature_panels,
        output_dir / "MODEL_GENERATED_FEATURE_CONTROLLER_GRID.png",
        banner="SOURCE + MODEL-GENERATED FLUX-KONTEXT FEATURE-CONTROL OUTPUTS",
    )

    initial_mean = _mean_error(initial)
    best_mean = _mean_error(best)
    final_mean = _mean_error(final)
    improvement = (initial_mean - best_mean) / initial_mean if initial_mean > 0 else float("nan")
    best_coordinates = [float(best[strength]["feature_coordinate"]) for strength in TRAINING_STRENGTHS]
    best_errors = [float(best[strength]["absolute_coordinate_error"]) for strength in TRAINING_STRENGTHS]
    all_values_finite = all(
        math.isfinite(float(value))
        for collection in (initial, best, final, dense)
        for result in collection.values()
        for key, value in result.items()
        if key != "image"
    )
    gates = {
        "all_shared_direction_gradients_finite": gradients_finite,
        "qwen_parameters_frozen": modules_frozen["qwen"],
        "flux_transformer_parameters_frozen": modules_frozen["transformer"],
        "vae_parameters_frozen": modules_frozen["vae"],
        "best_mean_error_reduced_at_least_50_percent": improvement >= 0.5,
        "every_best_training_error_at_most_0p10": all(error <= 0.10 for error in best_errors),
        "best_training_coordinates_strictly_increasing": all(
            left < right for left, right in zip(best_coordinates, best_coordinates[1:])
        ),
        "all_values_finite": all_values_finite,
    }
    controller_status = "PASS" if all(gates.values()) else "FAIL"
    dense_curve = dense_feature_curve_diagnostics(
        DENSE_STRENGTHS, [float(dense[strength]["feature_coordinate"]) for strength in DENSE_STRENGTHS]
    )
    strength_one_controls = masked_effective_controls(checkpoint.controls, masks.masks, strength=1.0)
    strength_one_control_exactly_zero = all(
        torch.equal(control, torch.zeros_like(control)) for control in strength_one_controls
    )
    report = {
        "scope": "Feature reward terminal controller v5; frozen reward definition and frozen shared controller",
        "config": vars(args),
        "provenance": {
            "source": SOURCE_INPUT_PROVENANCE,
            "native_full": NATIVE_FULL_PROVENANCE,
            "controlled_outputs": FEATURE_CONTROL_PROVENANCE,
            "old_oracle_outputs_read": False,
            "pixel_blends_used": False,
        },
        "feature_definition": {
            "qwen_dtype": "float32",
            "prompt": "unchanged v4 focus-conditioned prompt",
            "layer": "last",
            "token": "final prompt token",
            "normalization": "L2",
            "primary_coordinate": "cosine distance ratio",
            "loss": "squared coordinate residual",
        },
        "controller_definition": {
            "shared_direction": True,
            "effective_control": "(1-strength) * velocity_topk_mask * shared_direction",
            "control_steps": 4,
            "topk_fraction": 0.25,
            "checkpoint_selection": "mean feature-coordinate absolute error only",
        },
        "initial_mean_error": initial_mean,
        "best_mean_error": best_mean,
        "final_mean_error": final_mean,
        "best_error_improvement_fraction": improvement,
        "best_iteration": checkpoint.iteration,
        "best_mean_effective_control_energy": float(
            normalized_effective_control_energy(
                [
                    masked_effective_controls(checkpoint.controls, masks.masks, strength=value)
                    for value in TRAINING_STRENGTHS
                ]
            )
        ),
        "training": {
            "initial": {
                str(key): {name: value for name, value in item.items() if name != "image"}
                for key, item in initial.items()
            },
            "best": {
                str(key): {name: value for name, value in item.items() if name != "image"}
                for key, item in best.items()
            },
            "final": {
                str(key): {name: value for name, value in item.items() if name != "image"}
                for key, item in final.items()
            },
        },
        "controller_drive_gates": gates,
        "controller_feature_drive": controller_status,
        "zero_control_native_parity_exact": zero_control_native_parity,
        "strength_one_control_exactly_zero": strength_one_control_exactly_zero,
        "strength_one_native_latent_max_abs_error": dense[1.0]["latent_max_abs_vs_native"],
        "dense_curve": dense_curve,
        "visual_weak_mid_strong": "PENDING_INDEPENDENT_EVALUATION",
        "perceptual_percentage_calibration": "NOT_ESTABLISHED",
        "runtime_seconds": time.perf_counter() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
    }
    _json(output_dir / "feature_controller_report.json", report)
    print(
        json.dumps(
            _serialize(
                {
                    "controller_feature_drive": controller_status,
                    "initial_mean_error": initial_mean,
                    "best_mean_error": best_mean,
                    "best_iteration": checkpoint.iteration,
                    "dense_curve": dense_curve,
                }
            ),
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
