"""Validate endpoint-relative semantic progress and its terminal-control coupling.

The parser runs offline. This script accepts only an audited semantic-spec JSON;
pixel interpolation is used for probes/evaluation and never enters optimization.
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
from PIL import Image, ImageDraw

from diffusers.pipelines.rewardflow.pipeline_flux_kontext_terminal_control import (
    FluxKontextTerminalControlPipeline,
)
from diffusers.pipelines.rewardflow.rewards import Qwen25VQATeacherForcedScorer
from diffusers.pipelines.rewardflow.semantic_progress import (
    EndpointRelativeSemanticProgressReward,
    parse_semantic_progress_json,
)
from diffusers.pipelines.rewardflow.terminal_control import (
    blue_direction_score,
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
    parser.add_argument("--qwen-model", default=os.getenv("QWEN25_VL_MODEL_PATH", "Qwen/Qwen2.5-VL-3B-Instruct"))
    parser.add_argument("--reward-device", default="cuda:0")
    parser.add_argument("--semantic-spec", required=True)
    parser.add_argument("--semantic-spec-provenance", default="caller-supplied audited JSON")
    parser.add_argument("--oracle-output-dir", default=None)
    parser.add_argument("--audit-only", action="store_true", help="Run reward gates without controller optimization.")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--strengths", type=float, nargs="+", default=(0.2, 0.5, 0.8))
    parser.add_argument("--control-steps", type=int, default=4)
    parser.add_argument("--control-lr", type=float, default=0.1)
    parser.add_argument("--outer-iters", type=int, default=20)
    parser.add_argument("--lambda-control", type=float, default=1e-4)
    parser.add_argument("--use-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if not args.model:
        parser.error("Set FLUX_KONTEXT_MODEL_PATH or pass --model.")
    if args.control_steps != 4:
        parser.error("This fixed semantic experiment requires --control-steps 4.")
    if tuple(args.strengths) != (0.2, 0.5, 0.8):
        parser.error("This first semantic experiment requires --strengths 0.2 0.5 0.8.")
    if args.outer_iters < 0:
        parser.error("--outer-iters must be non-negative.")
    if args.control_lr <= 0 or args.lambda_control < 0:
        parser.error("Control LR must be positive and regularization must be non-negative.")
    if not Path(args.qwen_model).exists():
        parser.error("The Qwen model is not local. Set QWEN25_VL_MODEL_PATH or pass --qwen-model.")
    return args


def _pil_to_tensor(image, device):
    array = np.asarray(image, dtype=np.float32).copy() / 255
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def _tensor_to_pil(image):
    array = image.detach().clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array, mode="RGB")


def _serialize_diagnostics(diagnostics):
    return {
        key: {
            nested: value.detach().float().cpu().item() if torch.is_tensor(value) else value
            for nested, value in fields.items()
        }
        for key, fields in diagnostics.items()
    }


def _model_has_gradient(pipe):
    modules = (pipe.transformer, pipe.vae, pipe.text_encoder, pipe.text_encoder_2)
    return any(
        parameter.grad is not None for module in modules if module is not None for parameter in module.parameters()
    )


def _qwen_has_gradient(scorer):
    return any(parameter.grad is not None for parameter in scorer.model.parameters())


def _decode_with_clamp_audit(pipe, latent, inputs):
    unpacked = pipe._unpack_latents(latent, inputs.height, inputs.width, pipe.vae_scale_factor)
    unpacked = unpacked / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
    decoded = pipe.vae.decode(unpacked, return_dict=False)[0]
    raw = decoded / 2 + 0.5
    image = raw.clamp(0, 1)
    audit = {
        "fraction_below_zero_before_clamp": (raw < 0).float().mean().detach().item(),
        "fraction_above_one_before_clamp": (raw > 1).float().mean().detach().item(),
        "fraction_exactly_zero_entering_qwen": (image == 0).float().mean().detach().item(),
        "fraction_exactly_one_entering_qwen": (image == 1).float().mean().detach().item(),
    }
    return image, audit


def _rank_correlation(values):
    values = torch.tensor(values, dtype=torch.float64)
    ranks = values.argsort().argsort().to(torch.float64)
    expected = torch.arange(values.numel(), dtype=torch.float64)
    return torch.corrcoef(torch.stack([expected, ranks]))[0, 1].item()


def _write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _make_grid(source, full, strengths, initial, best, final):
    panels = [(source, "Source"), (full, "Native Full")]
    for strength in strengths:
        panels.extend(
            [
                (_tensor_to_pil(initial[strength]["image"][0]), f"Initial {strength:g}"),
                (_tensor_to_pil(best[strength]["image"][0]), f"Best {strength:g}"),
                (_tensor_to_pil(final[strength]["image"][0]), f"Final {strength:g}"),
            ]
        )
    width, height = source.size
    grid = Image.new("RGB", (width * len(panels), height + 30), "white")
    draw = ImageDraw.Draw(grid)
    for index, (image, label) in enumerate(panels):
        draw.text((index * width + 5, 8), label, fill="black")
        grid.paste(image.resize((width, height), Image.Resampling.LANCZOS), (index * width, 30))
    return grid


def _probe_reward(objective, source, full, output_dir):
    alphas = (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0)
    rows = []
    panels = []
    with torch.no_grad():
        for alpha in alphas:
            image = (1 - alpha) * source + alpha * full
            progress, diagnostics = objective.progress_vector(image)
            primitive = next(iter(diagnostics.values()))
            rows.append(
                {
                    "alpha": alpha,
                    "source_answer_score": primitive["source_answer_score_candidate"].item(),
                    "target_answer_score": primitive["target_answer_score_candidate"].item(),
                    "semantic_contrast": primitive["semantic_contrast"].item(),
                    "progress_raw": progress[0].item(),
                    "progress_clamped": progress[0].clamp(0, 1).item(),
                }
            )
            panels.append((_tensor_to_pil(image[0]), f"alpha={alpha:g}\np={progress[0].item():.3f}"))
    progress_values = [row["progress_raw"] for row in rows]
    inversions = sum(
        progress_values[left] > progress_values[right]
        for left in range(len(rows))
        for right in range(left + 1, len(rows))
    )
    summary = {
        "spearman_correlation": _rank_correlation(progress_values),
        "pairwise_inversion_count": inversions,
        "progress_dynamic_range": max(progress_values) - min(progress_values),
        "mae_to_alpha_diagnostic_only": sum(abs(row["progress_raw"] - row["alpha"]) for row in rows) / len(rows),
    }
    _write_csv(output_dir / "semantic_progress_probes.csv", rows)
    width, height = panels[0][0].size
    grid = Image.new("RGB", (width * len(panels), height + 42), "white")
    draw = ImageDraw.Draw(grid)
    for index, (image, label) in enumerate(panels):
        draw.multiline_text((index * width + 5, 5), label, fill="black")
        grid.paste(image, (index * width, 42))
    grid.save(output_dir / "probe_grid.png")
    return rows, summary


def _gradient_audit(objective, scorer, source, full):
    audit = {"qwen_parameters_require_grad": any(p.requires_grad for p in scorer.model.parameters()), "targets": {}}
    for target in (0.2, 0.8):
        image = ((source + full) / 2).detach().requires_grad_(True)
        output = objective(image, target)
        gradient = torch.autograd.grad(output.loss, image)[0]
        rms = gradient.float().square().mean().sqrt()
        trials = []
        for perturbation_rms in (1e-5, 3e-5, 1e-4, 3e-4):
            step_scale = perturbation_rms / rms.clamp_min(1e-12)
            stepped = (image.detach() - step_scale * gradient.detach()).clamp(0, 1)
            with torch.no_grad():
                stepped_output = objective(stepped, target)
            trials.append(
                {
                    "perturbation_rms": perturbation_rms,
                    "progress": stepped_output.achieved_score.item(),
                    "loss": stepped_output.loss.item(),
                    "loss_decreased": stepped_output.loss.item() < output.loss.item(),
                }
            )
        audit["targets"][str(target)] = {
            "initial_progress": output.achieved_score.detach().item(),
            "initial_loss": output.loss.detach().item(),
            "gradient_finite": bool(torch.isfinite(gradient).all().item()),
            "gradient_nonzero": bool((gradient.abs().sum() > 0).item()),
            "gradient_l2_norm": torch.linalg.vector_norm(gradient.float()).item(),
            "gradient_rms": rms.item(),
            "gradient_max_abs": gradient.abs().max().item(),
            "small_step_trials": trials,
            "any_small_step_decreased_loss": any(trial["loss_decreased"] for trial in trials),
        }
    audit["qwen_parameters_have_gradient"] = _qwen_has_gradient(scorer)
    return audit


def _score_oracle_outputs(objective, directory, strengths, device):
    if directory is None:
        return None
    directory = Path(directory)
    results = {}
    for strength in strengths:
        path = directory / f"final_{str(strength).replace('.', 'p')}.png"
        if not path.exists():
            return {"available": False, "missing": str(path)}
        image = _pil_to_tensor(Image.open(path).convert("RGB"), device)
        with torch.no_grad():
            output = objective(image, strength)
        results[str(strength)] = {
            "progress_raw": output.achieved_score.item(),
            "target_error": output.objective_error.item(),
        }
    values = [results[str(strength)]["progress_raw"] for strength in strengths]
    return {
        "available": True,
        "results": results,
        "strictly_increasing": all(left < right for left, right in zip(values, values[1:])),
    }


def _evaluate_controls(pipe, inputs, directions, masks, objective, strengths, source, full):
    results = {}
    with torch.no_grad():
        for strength in strengths:
            controls = masked_effective_controls(directions, masks, strength=strength)
            unroll = pipe.unroll_terminal_controls(inputs, controls, use_checkpointing=False)
            image, clamp = _decode_with_clamp_audit(pipe, unroll.final_latent, inputs)
            output = objective(image.to(next(iter(objective._anchors.values()))["denominator"].device), strength)
            pixel_target = (1 - strength) * source + strength * full
            results[strength] = {
                "image": image.detach().cpu(),
                "progress_raw": output.achieved_score.item(),
                "progress_clamped": output.achieved_score.clamp(0, 1).item(),
                "semantic_error": output.objective_error.item(),
                "semantic_loss": output.loss.item(),
                "primitive_diagnostics": _serialize_diagnostics(output.diagnostics),
                "clamp_audit": clamp,
                "pixel_oracle_mse_not_used_for_optimization": (image.float() - pixel_target).square().mean().item(),
                "blue_score_not_used_for_optimization": blue_direction_score(image).mean().item(),
            }
    return results


def _optimize(pipe, inputs, masks, objective, strengths, source, full, args):
    directions = initialize_velocity_controls(inputs.initial_latent, args.control_steps)
    optimizer = torch.optim.Adam(directions, lr=args.control_lr)
    initial = _evaluate_controls(pipe, inputs, directions, masks, objective, strengths, source, full)
    initial_mean = sum(item["semantic_error"] for item in initial.values()) / len(strengths)
    best = update_best_control_checkpoint(
        None, iteration=0, objective_error=torch.tensor(initial_mean), controls=directions
    )
    trace = []
    for iteration in range(1, args.outer_iters + 1):
        optimizer.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(torch.device(args.device))
            if torch.device(args.reward_device) != torch.device(args.device):
                torch.cuda.reset_peak_memory_stats(torch.device(args.reward_device))
        started = time.perf_counter()
        rows = []
        errors = []
        for strength in strengths:
            controls = masked_effective_controls(directions, masks, strength=strength)
            unroll = pipe.unroll_terminal_controls(inputs, controls, use_checkpointing=args.use_checkpointing)
            image, clamp = _decode_with_clamp_audit(pipe, unroll.final_latent, inputs)
            image.retain_grad()
            output = objective(image.to(args.reward_device), strength)
            (output.loss / len(strengths)).backward()
            image_gradient = image.grad
            if image_gradient is None or not torch.isfinite(image_gradient).all() or image_gradient.abs().sum() == 0:
                raise RuntimeError("Semantic loss did not produce a finite nonzero decoded-image gradient.")
            errors.append(output.objective_error.detach())
            rows.append(
                {
                    "target_strength": strength,
                    "progress_raw": output.achieved_score.detach().item(),
                    "progress_clamped": output.achieved_score.detach().clamp(0, 1).item(),
                    "semantic_error": output.objective_error.detach().item(),
                    "semantic_loss": output.loss.detach().item(),
                    "primitive_diagnostics": _serialize_diagnostics(output.diagnostics),
                    "image_gradient_norm": torch.linalg.vector_norm(image_gradient.detach().float()).item(),
                    **clamp,
                }
            )
            del controls, unroll, image, output, image_gradient
        effective_families = [masked_effective_controls(directions, masks, strength=s) for s in strengths]
        regularization = normalized_effective_control_energy(effective_families)
        (args.lambda_control * regularization).backward()
        if any(
            parameter.grad is None or not torch.isfinite(parameter.grad).all() or parameter.grad.abs().sum() == 0
            for parameter in directions
        ):
            raise RuntimeError("Shared directions have missing, zero, or non-finite gradients.")
        if _model_has_gradient(pipe) or _qwen_has_gradient(objective.scorer):
            raise RuntimeError("Frozen FLUX/VAE/Qwen parameters unexpectedly received gradients.")
        mean_error = torch.stack(errors).mean()
        best = update_best_control_checkpoint(
            best, iteration=iteration - 1, objective_error=mean_error, controls=directions
        )
        trace.append(
            {
                "iteration": iteration,
                "branches": rows,
                "mean_semantic_error_before_step": mean_error.item(),
                "total_semantic_loss_before_step": sum(row["semantic_loss"] for row in rows) / len(rows),
                "control_regularization": args.lambda_control * regularization.detach().item(),
                "control_energy": regularization.detach().item(),
                "direction_gradient_norms": [
                    torch.linalg.vector_norm(parameter.grad.detach().float()).item() for parameter in directions
                ],
                "flux_parameters_have_gradient": _model_has_gradient(pipe),
                "qwen_parameters_have_gradient": _qwen_has_gradient(objective.scorer),
                "elapsed_seconds": time.perf_counter() - started,
                "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(torch.device(args.device)),
                "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(torch.device(args.device)),
            }
        )
        optimizer.step()
    final = _evaluate_controls(pipe, inputs, directions, masks, objective, strengths, source, full)
    final_mean = sum(item["semantic_error"] for item in final.values()) / len(strengths)
    best = update_best_control_checkpoint(
        best, iteration=args.outer_iters, objective_error=torch.tensor(final_mean), controls=directions
    )
    best_results = _evaluate_controls(pipe, inputs, best.controls, masks, objective, strengths, source, full)
    return initial, best_results, final, best, trace


def main():
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Real semantic validation requires CUDA.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["single_primitive_mode"] = True
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    spec = parse_semantic_progress_json(Path(args.semantic_spec).read_text(encoding="utf-8"))
    if len(spec.primitives) != 1:
        raise ValueError("This validation script is explicitly single_primitive_mode and requires one primitive.")
    print("single_primitive_mode = true", flush=True)

    device = torch.device(args.device)
    reward_device = torch.device(args.reward_device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    source_pil = Image.open(args.source).convert("RGB").resize((args.width, args.height), Image.Resampling.LANCZOS)
    source_pil.save(output_dir / "source.png")
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
    masks, native_unroll = pipe.prepare_velocity_edit_masks(
        inputs, control_steps=4, mode="velocity-topk", topk_fraction=0.25
    )
    if not torch.equal(native_unroll.final_latent, inputs.native_final_latent):
        raise RuntimeError("Zero-control parity is not exact.")
    with torch.no_grad():
        full, native_clamp_audit = _decode_with_clamp_audit(pipe, inputs.native_final_latent, inputs)
    full_pil = _tensor_to_pil(full[0])
    full_pil.save(output_dir / "native_full.png")
    source = _pil_to_tensor(source_pil, device)

    scorer = Qwen25VQATeacherForcedScorer(
        args.qwen_model,
        device=reward_device,
        dtype=torch.bfloat16,
        local_files_only=True,
    )
    primitive = spec.primitives[0]
    middle_pil = _tensor_to_pil(((source + full) / 2)[0])
    fidelity = {
        "source": scorer.compare_with_official_processor(source_pil, primitive.question, primitive.target_answer),
        "native_full": scorer.compare_with_official_processor(full_pil, primitive.question, primitive.target_answer),
        "middle_probe": scorer.compare_with_official_processor(
            middle_pil, primitive.question, primitive.target_answer
        ),
    }
    (output_dir / "processor_fidelity.json").write_text(json.dumps(fidelity, indent=2), encoding="utf-8")

    objective = EndpointRelativeSemanticProgressReward(
        scorer,
        spec,
        source.to(reward_device),
        full.to(reward_device),
        fail_on_endpoint_validation=False,
    )
    endpoint_report = {
        "endpoint_semantic_validation_failed": objective.endpoint_semantic_validation_failed,
        "primitives": objective.endpoint_diagnostics,
    }
    (output_dir / "endpoint_semantic_scores.json").write_text(json.dumps(endpoint_report, indent=2), encoding="utf-8")
    if objective.endpoint_semantic_validation_failed:
        raise RuntimeError("Endpoint semantic validation failed; controller optimization was not started.")

    probe_rows, probe_summary = _probe_reward(objective, source.to(reward_device), full.to(reward_device), output_dir)
    gradient_audit = _gradient_audit(objective, scorer, source.to(reward_device), full.to(reward_device))
    (output_dir / "qwen_gradient_audit.json").write_text(json.dumps(gradient_audit, indent=2), encoding="utf-8")
    oracle_scores = _score_oracle_outputs(objective, args.oracle_output_dir, args.strengths, reward_device)
    reward_gates = {
        "probe_spearman_at_least_0_9": math.isfinite(probe_summary["spearman_correlation"])
        and probe_summary["spearman_correlation"] >= 0.9,
        "gradient_directionality_passed": all(
            item["any_small_step_decreased_loss"] for item in gradient_audit["targets"].values()
        ),
        "oracle_outputs_strictly_increasing": oracle_scores is None
        or (oracle_scores.get("available", False) and oracle_scores["strictly_increasing"]),
    }
    reward_gates["all_passed"] = all(reward_gates.values())
    if args.audit_only:
        report = {
            "scope": "semantic reward audit only; controller optimization not run",
            "method_assumption": "source-vs-target teacher-forced answer contrast represents semantic progress",
            "single_primitive_mode": True,
            "config": config,
            "semantic_spec": json.loads(Path(args.semantic_spec).read_text(encoding="utf-8")),
            "semantic_spec_provenance": args.semantic_spec_provenance,
            "processor_fidelity": fidelity,
            "endpoint_semantic_scores": endpoint_report,
            "probe_rows": probe_rows,
            "probe_summary": probe_summary,
            "gradient_audit": gradient_audit,
            "oracle_generated_output_scores": oracle_scores,
            "reward_gates": reward_gates,
            "zero_control_parity_exact": True,
            "native_clamp_audit": native_clamp_audit,
        }
        (output_dir / "semantic_progress_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("AUDIT_REPORT=" + json.dumps(report, sort_keys=True), flush=True)
        return
    if not reward_gates["all_passed"]:
        raise RuntimeError("Semantic reward gates failed; controller optimization was not started. Run --audit-only.")

    initial, best_results, final, best_checkpoint, trace = _optimize(
        pipe, inputs, masks.masks, objective, args.strengths, source, full, args
    )
    _write_csv(
        output_dir / "optimization_trace.csv",
        [
            {key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in row.items()}
            for row in trace
        ],
    )
    for strength in args.strengths:
        tag = str(strength).replace(".", "")
        for name, values in (("initial", initial), ("best", best_results), ("final", final)):
            _tensor_to_pil(values[strength]["image"][0]).save(output_dir / f"{name}_s{tag}.png")
    _make_grid(source_pil, full_pil, args.strengths, initial, best_results, final).save(
        output_dir / "grid_semantic_progress.png"
    )

    report = {
        "scope": "endpoint-relative semantic progress; pixel oracle and blue score are evaluation-only",
        "method_assumption": "source-vs-target teacher-forced answer contrast represents semantic progress",
        "single_primitive_mode": True,
        "config": config,
        "semantic_spec": json.loads(Path(args.semantic_spec).read_text(encoding="utf-8")),
        "semantic_spec_provenance": args.semantic_spec_provenance,
        "processor_fidelity": fidelity,
        "endpoint_semantic_scores": endpoint_report,
        "probe_rows": probe_rows,
        "probe_summary": probe_summary,
        "gradient_audit": gradient_audit,
        "oracle_generated_output_scores": oracle_scores,
        "zero_control_parity_exact": True,
        "native_clamp_audit": native_clamp_audit,
        "best_iteration": best_checkpoint.iteration,
        "best_mean_semantic_error": best_checkpoint.objective_error,
        "initial": {
            str(strength): {key: value for key, value in result.items() if key != "image"}
            for strength, result in initial.items()
        },
        "best": {
            str(strength): {key: value for key, value in result.items() if key != "image"}
            for strength, result in best_results.items()
        },
        "final": {
            str(strength): {key: value for key, value in result.items() if key != "image"}
            for strength, result in final.items()
        },
        "optimization_trace": trace,
        "flux_parameters_have_gradient": _model_has_gradient(pipe),
        "qwen_parameters_have_gradient": _qwen_has_gradient(scorer),
    }
    (output_dir / "semantic_progress_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("FINAL_REPORT=" + json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
