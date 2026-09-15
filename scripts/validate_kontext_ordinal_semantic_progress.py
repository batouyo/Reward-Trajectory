"""Bake off binary-v1 against five-stage ordinal semantic progress rewards.

The parser is offline. Pixel blends are probes only and never optimization
targets. Controller optimization starts only after every preregistered reward
gate passes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from diffusers.pipelines.rewardflow.ordinal_semantic_progress import (
    ORDINAL_CHOICE_LABELS,
    EndpointRelativeOrdinalSemanticProgressReward,
    parse_ordinal_semantic_progress_json,
)
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
PROBE_ALPHAS = (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0)
LABEL_PERMUTATION = ("C", "E", "A", "D", "B")


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.getenv("FLUX_KONTEXT_MODEL_PATH"))
    parser.add_argument("--qwen-model", default=os.getenv("QWEN25_VL_MODEL_PATH"))
    parser.add_argument("--source", required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--binary-spec", required=True)
    parser.add_argument("--ordinal-single-spec", required=True)
    parser.add_argument("--ordinal-ensemble-spec", required=True)
    parser.add_argument("--oracle-output-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reward-device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--strengths", type=float, nargs="+", default=(0.2, 0.5, 0.8))
    parser.add_argument("--control-steps", type=int, default=4)
    parser.add_argument("--control-lr", type=float, default=0.1)
    parser.add_argument("--outer-iters", type=int, default=20)
    parser.add_argument("--lambda-control", type=float, default=1e-4)
    parser.add_argument("--use-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if not args.model or not Path(args.model).exists():
        parser.error("Set FLUX_KONTEXT_MODEL_PATH or pass a local --model.")
    if not args.qwen_model or not Path(args.qwen_model).exists():
        parser.error("Set QWEN25_VL_MODEL_PATH or pass a local --qwen-model.")
    if tuple(args.strengths) != (0.2, 0.5, 0.8):
        parser.error("The preregistered bake-off requires --strengths 0.2 0.5 0.8.")
    if args.control_steps != 4:
        parser.error("The preregistered controller requires --control-steps 4.")
    if args.steps != 12 or args.seed != 20260914 or (args.height, args.width) != (256, 256):
        parser.error("The formal run requires steps=12, seed=20260914, and 256x256.")
    if args.guidance_scale != 2.5 or args.lambda_control != 1e-4:
        parser.error("The formal run requires guidance=2.5 and lambda_control=0.0001.")
    return args


def _pil_to_tensor(image, device):
    array = np.asarray(image, dtype=np.float32).copy() / 255
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def _tensor_to_pil(image):
    array = image.detach().clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array, mode="RGB")


def _serialize(value):
    if torch.is_tensor(value):
        detached = value.detach().float().cpu()
        return detached.item() if detached.numel() == 1 else detached.tolist()
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_serialize(item) for item in value]
    return value


def _write_json(path, value):
    path.write_text(json.dumps(_serialize(value), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _rank_correlation(values):
    values = torch.tensor(values, dtype=torch.float64)
    ranks = values.argsort().argsort().to(torch.float64)
    expected = torch.arange(values.numel(), dtype=torch.float64)
    return torch.corrcoef(torch.stack((expected, ranks)))[0, 1].item()


def _decode_with_clamp_audit(pipe, latent, inputs):
    unpacked = pipe._unpack_latents(latent, inputs.height, inputs.width, pipe.vae_scale_factor)
    unpacked = unpacked / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
    decoded = pipe.vae.decode(unpacked, return_dict=False)[0]
    raw = decoded / 2 + 0.5
    return raw.clamp(0, 1), {
        "fraction_below_zero_before_clamp": (raw < 0).float().mean().detach().item(),
        "fraction_above_one_before_clamp": (raw > 1).float().mean().detach().item(),
        "fraction_exactly_zero_entering_qwen": (raw <= 0).float().mean().detach().item(),
        "fraction_exactly_one_entering_qwen": (raw >= 1).float().mean().detach().item(),
    }


def _qwen_has_gradient(scorer):
    return any(parameter.grad is not None for parameter in scorer.model.parameters())


def _flux_has_gradient(pipe):
    modules = (pipe.transformer, pipe.vae, pipe.text_encoder, pipe.text_encoder_2)
    return any(parameter.grad is not None for module in modules if module for parameter in module.parameters())


def _question_rows(method, location, diagnostics):
    rows = []
    for primitive_id, primitive in diagnostics.items():
        if "questions" not in primitive:
            continue
        for question_id, question in primitive["questions"].items():
            row = {
                "method": method,
                "location": location,
                "primitive_id": primitive_id,
                "question_id": question_id,
                "top_choice": question["top_choice"],
                "top_stage": question["top_stage"],
                "normalized_entropy": _serialize(question["normalized_entropy"]),
                "raw_ordinal_expectation": _serialize(question["raw_ordinal_expectation"]),
                "endpoint_calibrated_progress_raw": _serialize(question["endpoint_calibrated_progress_raw"]),
                "progress_clamped": _serialize(question["progress_clamped"]),
            }
            logprobs = _serialize(question["choice_logprobs"])
            probabilities = _serialize(question["choice_probs"])
            for index, label in enumerate(ORDINAL_CHOICE_LABELS):
                row[f"logprob_{label}"] = logprobs[index]
                row[f"prob_{label}"] = probabilities[index]
            rows.append(row)
    return rows


def _endpoint_distribution_rows(method, endpoint_diagnostics):
    rows = []
    for primitive_id, primitive in endpoint_diagnostics.items():
        for question_id, question in primitive.items():
            for endpoint, progress in (("source", 0.0), ("full", 1.0)):
                probabilities = question[f"{endpoint}_choice_probs"]
                top_stage = question[f"{endpoint}_top_stage"]
                row = {
                    "method": method,
                    "location": endpoint,
                    "primitive_id": primitive_id,
                    "question_id": question_id,
                    "top_choice": ORDINAL_CHOICE_LABELS[top_stage],
                    "top_stage": top_stage,
                    "normalized_entropy": question[f"{endpoint}_normalized_entropy"],
                    "raw_ordinal_expectation": question[f"raw_{endpoint}"],
                    "endpoint_calibrated_progress_raw": progress,
                    "progress_clamped": progress,
                }
                for index, label in enumerate(ORDINAL_CHOICE_LABELS):
                    row[f"logprob_{label}"] = ""
                    row[f"prob_{label}"] = probabilities[index]
                rows.append(row)
    return rows


def _probe(objective, source, full, method):
    rows, distribution_rows = [], []
    with torch.no_grad():
        for alpha in PROBE_ALPHAS:
            image = (1 - alpha) * source + alpha * full
            _, diagnostics = objective.progress_vector(image)
            output = objective(image, alpha)
            rows.append(
                {
                    "alpha": alpha,
                    "overall_progress_raw": output.achieved_score.item(),
                    "overall_progress_clamped": output.achieved_score.clamp(0, 1).item(),
                    "objective_error": output.objective_error.item(),
                }
            )
            distribution_rows.extend(_question_rows(method, f"probe_alpha_{alpha:g}", diagnostics))
    values = [row["overall_progress_raw"] for row in rows]
    inversions = sum(
        values[left] > values[right] for left in range(len(values)) for right in range(left + 1, len(values))
    )
    return (
        rows,
        distribution_rows,
        {
            "spearman_correlation": _rank_correlation(values),
            "pairwise_inversion_count": inversions,
            "mae_to_alpha_diagnostic_only": sum(abs(row["overall_progress_raw"] - row["alpha"]) for row in rows)
            / len(rows),
            "progress_dynamic_range": max(values) - min(values),
        },
    )


def _binary_probe(objective, source, full):
    rows = []
    with torch.no_grad():
        for alpha in PROBE_ALPHAS:
            image = (1 - alpha) * source + alpha * full
            output = objective(image, alpha)
            primitive = next(iter(output.diagnostics.values()))
            rows.append(
                {
                    "alpha": alpha,
                    "source_answer_score": primitive["source_answer_score_candidate"].item(),
                    "target_answer_score": primitive["target_answer_score_candidate"].item(),
                    "semantic_contrast": primitive["semantic_contrast"].item(),
                    "overall_progress_raw": output.achieved_score.item(),
                    "overall_progress_clamped": output.achieved_score.clamp(0, 1).item(),
                }
            )
    values = [row["overall_progress_raw"] for row in rows]
    inversions = sum(
        values[left] > values[right] for left in range(len(values)) for right in range(left + 1, len(values))
    )
    return rows, {
        "spearman_correlation": _rank_correlation(values),
        "pairwise_inversion_count": inversions,
        "mae_to_alpha_diagnostic_only": sum(abs(row["overall_progress_raw"] - row["alpha"]) for row in rows)
        / len(rows),
    }


def _oracle_scores(objective, directory, strengths, device, method):
    rows, distribution_rows = [], []
    for strength in strengths:
        path = Path(directory) / f"final_{str(strength).replace('.', 'p')}.png"
        if not path.exists():
            raise FileNotFoundError(f"Missing preregistered oracle output: {path}")
        image = _pil_to_tensor(Image.open(path).convert("RGB"), device)
        with torch.no_grad():
            output = objective(image, strength)
        rows.append(
            {
                "strength": strength,
                "progress_raw": output.achieved_score.item(),
                "target_error": output.objective_error.item(),
            }
        )
        distribution_rows.extend(_question_rows(method, f"oracle_{strength:g}", output.diagnostics or {}))
    values = [row["progress_raw"] for row in rows]
    gaps = [right - left for left, right in zip(values, values[1:])]
    return (
        rows,
        distribution_rows,
        {
            "strictly_increasing": all(gap > 0 for gap in gaps),
            "spearman_correlation": _rank_correlation(values),
            "adjacent_progress_gaps": gaps,
            "minimum_adjacent_progress_gap": min(gaps),
        },
    )


def _gradient_audit(objective, scorer, source, full):
    audit = {"qwen_parameters_require_grad": any(p.requires_grad for p in scorer.model.parameters()), "targets": {}}
    for target in (0.2, 0.8):
        image = ((source + full) / 2).detach().requires_grad_(True)
        output = objective(image, target)
        gradient = torch.autograd.grad(output.loss, image)[0]
        rms = gradient.float().square().mean().sqrt()
        trials = []
        for perturbation_rms in (1e-5, 3e-5, 1e-4, 3e-4):
            step = (image.detach() - perturbation_rms * gradient.detach() / rms.clamp_min(1e-12)).clamp(0, 1)
            with torch.no_grad():
                stepped = objective(step, target)
            direction_correct = (
                stepped.achieved_score.item() < output.achieved_score.item()
                if target == 0.2
                else stepped.achieved_score.item() > output.achieved_score.item()
            )
            trials.append(
                {
                    "perturbation_rms": perturbation_rms,
                    "progress": stepped.achieved_score.item(),
                    "loss": stepped.loss.item(),
                    "loss_decreased": stepped.loss.item() < output.loss.item(),
                    "progress_direction_correct": direction_correct,
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
            "passed": bool(torch.isfinite(gradient).all().item())
            and bool((gradient.abs().sum() > 0).item())
            and any(trial["loss_decreased"] and trial["progress_direction_correct"] for trial in trials),
        }
    audit["qwen_parameters_have_gradient"] = _qwen_has_gradient(scorer)
    return audit


def _permuted_overall_progress(objective, image, permutation):
    primitive_values, primitive_weights = [], []
    for primitive in objective.spec.primitives:
        question_values, question_weights = [], []
        for question in primitive.questions:
            source_raw = objective._raw_question(objective._bias_source, question, stage_to_label=permutation)[
                "raw_expectation"
            ]
            full_raw = objective._raw_question(objective._bias_full, question, stage_to_label=permutation)[
                "raw_expectation"
            ]
            candidate_raw = objective._raw_question(image, question, stage_to_label=permutation)["raw_expectation"]
            question_values.append((candidate_raw - source_raw) / (full_raw - source_raw))
            question_weights.append(question.weight)
        values = torch.stack(question_values)
        weights = values.new_tensor(question_weights)
        primitive_values.append((values * weights / weights.sum()).sum())
        primitive_weights.append(primitive.weight)
    values = torch.stack(primitive_values)
    weights = values.new_tensor(primitive_weights)
    return (values * weights / weights.sum()).sum()


def _label_bias_audit(objective, source, full):
    objective._bias_source = source
    objective._bias_full = full
    locations = {"source": source, "middle": (source + full) / 2, "full": full}
    rows, changes = {}, []
    with torch.no_grad():
        for name, image in locations.items():
            normal = objective(image, 0.5).achieved_score.item()
            permuted = _permuted_overall_progress(objective, image, LABEL_PERMUTATION).item()
            rows[name] = {
                "normal_progress": normal,
                "permuted_progress": permuted,
                "absolute_change": abs(normal - permuted),
            }
            changes.append(abs(normal - permuted))
    del objective._bias_source, objective._bias_full
    return {
        "fixed_stage_to_label_permutation": LABEL_PERMUTATION,
        "locations": rows,
        "max_progress_change_under_permutation": max(changes),
        "diagnostic_only": True,
    }


def _ordinal_gates(objective, probe_summary, oracle_summary, gradient):
    endpoint_pass = not objective.endpoint_ordinal_validation_failed
    return {
        "endpoint_pass": endpoint_pass,
        "probe_spearman_at_least_0_9": math.isfinite(probe_summary["spearman_correlation"])
        and probe_summary["spearman_correlation"] >= 0.9,
        "probe_inversions_at_most_1": probe_summary["pairwise_inversion_count"] <= 1,
        "oracle_outputs_strictly_increasing": oracle_summary["strictly_increasing"],
        "gradient_target_0p2_pass": gradient["targets"]["0.2"]["passed"],
        "gradient_target_0p8_pass": gradient["targets"]["0.8"]["passed"],
    }


def _make_probe_grid(source, full, single_rows, ensemble_rows, output_path):
    images = []
    for alpha in PROBE_ALPHAS:
        image = _tensor_to_pil(((1 - alpha) * source + alpha * full)[0])
        single = next(row["overall_progress_raw"] for row in single_rows if row["alpha"] == alpha)
        ensemble = next(row["overall_progress_raw"] for row in ensemble_rows if row["alpha"] == alpha)
        images.append((image, f"a={alpha:g} single={single:.3f}\nensemble={ensemble:.3f}"))
    width, height = images[0][0].size
    grid = Image.new("RGB", (width * len(images), height + 42), "white")
    draw = ImageDraw.Draw(grid)
    for index, (image, label) in enumerate(images):
        draw.multiline_text((index * width + 4, 4), label, fill="black")
        grid.paste(image, (index * width, 42))
    grid.save(output_path)


def _select_method(single, ensemble):
    candidates = [item for item in (single, ensemble) if item["gates"]["all_passed"]]
    if not candidates:
        return None
    candidates.sort(
        key=lambda item: (
            item["oracle_summary"]["minimum_adjacent_progress_gap"],
            -item["probe_summary"]["pairwise_inversion_count"],
            item["probe_summary"]["spearman_correlation"],
            item["name"] == "ordinal-single",
        ),
        reverse=True,
    )
    return candidates[0]


def _run_ordinal_audit(name, objective, source, full, oracle_dir, strengths, scorer):
    probe_rows, distributions, probe_summary = _probe(objective, source, full, name)
    distributions = _endpoint_distribution_rows(name, objective.endpoint_diagnostics) + distributions
    oracle_rows, oracle_distributions, oracle_summary = _oracle_scores(
        objective, oracle_dir, strengths, source.device, name
    )
    distributions.extend(oracle_distributions)
    gradient = _gradient_audit(objective, scorer, source, full)
    label_bias = _label_bias_audit(objective, source, full)
    gates = _ordinal_gates(objective, probe_summary, oracle_summary, gradient)
    gates["all_passed"] = all(gates.values())
    return {
        "name": name,
        "endpoint": objective.endpoint_diagnostics,
        "probe_rows": probe_rows,
        "probe_summary": probe_summary,
        "oracle_rows": oracle_rows,
        "oracle_summary": oracle_summary,
        "gradient": gradient,
        "label_bias": label_bias,
        "gates": gates,
        "distribution_rows": distributions,
        "objective": objective,
    }


def _objective_device(objective):
    primitive_anchors = next(iter(objective._anchors.values()))
    question_anchor = next(iter(primitive_anchors.values()))
    return question_anchor["denominator"].device


def _evaluate_controls(pipe, inputs, directions, masks, objective, strengths, source, full):
    results = {}
    reward_device = _objective_device(objective)
    with torch.no_grad():
        for strength in strengths:
            controls = masked_effective_controls(directions, masks, strength=strength)
            unroll = pipe.unroll_terminal_controls(inputs, controls, use_checkpointing=False)
            image, clamp = _decode_with_clamp_audit(pipe, unroll.final_latent, inputs)
            output = objective(image.to(reward_device), strength)
            pixel_target = (1 - strength) * source + strength * full
            results[strength] = {
                "image": image.detach().cpu(),
                "progress_raw": output.achieved_score.item(),
                "progress_clamped": output.achieved_score.clamp(0, 1).item(),
                "semantic_error": output.objective_error.item(),
                "semantic_loss": output.loss.item(),
                "ordinal_diagnostics": _serialize(output.diagnostics),
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
    reward_device = _objective_device(objective)
    for iteration in range(1, args.outer_iters + 1):
        optimizer.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(torch.device(args.device))
            if reward_device != torch.device(args.device):
                torch.cuda.reset_peak_memory_stats(reward_device)
        rows, errors = [], []
        started = torch.cuda.Event(enable_timing=True)
        ended = torch.cuda.Event(enable_timing=True)
        started.record()
        for strength in strengths:
            controls = masked_effective_controls(directions, masks, strength=strength)
            unroll = pipe.unroll_terminal_controls(inputs, controls, use_checkpointing=args.use_checkpointing)
            image, clamp = _decode_with_clamp_audit(pipe, unroll.final_latent, inputs)
            image.retain_grad()
            output = objective(image.to(reward_device), strength)
            (output.loss / len(strengths)).backward()
            image_gradient = image.grad
            if image_gradient is None or not torch.isfinite(image_gradient).all() or image_gradient.abs().sum() == 0:
                raise RuntimeError("Ordinal loss did not produce a finite nonzero decoded-image gradient.")
            errors.append(output.objective_error.detach())
            rows.append(
                {
                    "target_strength": strength,
                    "overall_progress": output.achieved_score.detach().item(),
                    "semantic_error": output.objective_error.detach().item(),
                    "semantic_loss": output.loss.detach().item(),
                    "ordinal_diagnostics": _serialize(output.diagnostics),
                    "decoded_image_gradient_norm": torch.linalg.vector_norm(image_gradient.detach().float()).item(),
                    **clamp,
                }
            )
        effective = [masked_effective_controls(directions, masks, strength=value) for value in strengths]
        regularization = normalized_effective_control_energy(effective)
        (args.lambda_control * regularization).backward()
        if any(
            parameter.grad is None or not torch.isfinite(parameter.grad).all() or parameter.grad.abs().sum() == 0
            for parameter in directions
        ):
            raise RuntimeError("Shared control directions have missing, zero, or non-finite gradients.")
        if _flux_has_gradient(pipe) or _qwen_has_gradient(objective.scorer):
            raise RuntimeError("Frozen FLUX/VAE/Qwen parameters unexpectedly received gradients.")
        mean_error = torch.stack(errors).mean()
        best = update_best_control_checkpoint(
            best, iteration=iteration - 1, objective_error=mean_error, controls=directions
        )
        ended.record()
        torch.cuda.synchronize(torch.device(args.device))
        trace.append(
            {
                "iteration": iteration,
                "branches": rows,
                "mean_semantic_error_before_step": mean_error.item(),
                "control_energy": regularization.detach().item(),
                "control_regularization": args.lambda_control * regularization.detach().item(),
                "direction_gradient_norms": [
                    torch.linalg.vector_norm(parameter.grad.detach().float()).item() for parameter in directions
                ],
                "flux_parameters_have_gradient": _flux_has_gradient(pipe),
                "vae_parameters_have_gradient": any(p.grad is not None for p in pipe.vae.parameters()),
                "qwen_parameters_have_gradient": _qwen_has_gradient(objective.scorer),
                "elapsed_seconds": started.elapsed_time(ended) / 1000,
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
        raise RuntimeError("Real ordinal semantic validation requires CUDA.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    device, reward_device = torch.device(args.device), torch.device(args.reward_device)
    source_pil = Image.open(args.source).convert("RGB").resize((args.width, args.height), Image.Resampling.LANCZOS)
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
    masks, native = pipe.prepare_velocity_edit_masks(
        inputs, control_steps=args.control_steps, mode="velocity-topk", topk_fraction=0.25
    )
    if not torch.equal(native.final_latent, inputs.native_final_latent):
        raise RuntimeError("Zero-control parity is not exact.")
    with torch.no_grad():
        full, clamp_audit = _decode_with_clamp_audit(pipe, inputs.native_final_latent, inputs)
    source = _pil_to_tensor(source_pil, reward_device)
    full = full.to(reward_device)
    source_pil.save(output_dir / "source.png")
    _tensor_to_pil(full[0]).save(output_dir / "native_full.png")

    binary_spec = parse_semantic_progress_json(Path(args.binary_spec).read_text(encoding="utf-8"))
    single_spec = parse_ordinal_semantic_progress_json(Path(args.ordinal_single_spec).read_text(encoding="utf-8"))
    ensemble_spec = parse_ordinal_semantic_progress_json(Path(args.ordinal_ensemble_spec).read_text(encoding="utf-8"))
    if any(len(spec.primitives) != 1 for spec in (single_spec, ensemble_spec)):
        raise ValueError("The formal bake-off requires exactly one primitive in each ordinal spec.")
    if len(single_spec.primitives[0].questions) != 1 or len(ensemble_spec.primitives[0].questions) != 3:
        raise ValueError("Formal single/ensemble specs require one and three questions respectively.")
    scorer = Qwen25VQATeacherForcedScorer(
        args.qwen_model, device=reward_device, dtype=torch.bfloat16, local_files_only=True
    )

    binary = EndpointRelativeSemanticProgressReward(
        scorer, binary_spec, source, full, fail_on_endpoint_validation=False
    )
    binary_rows, binary_summary = _binary_probe(binary, source, full)
    binary_oracle_rows, _, binary_oracle_summary = _oracle_scores(
        binary, args.oracle_output_dir, args.strengths, reward_device, "binary-v1"
    )
    binary_gradient = _gradient_audit(binary, scorer, source, full)
    if binary.endpoint_semantic_validation_failed:
        raise RuntimeError("binary-v1 endpoint reproduction failed; ordinal audit stopped.")

    single_objective = EndpointRelativeOrdinalSemanticProgressReward(
        scorer, single_spec, source, full, temperature=1.0, fail_on_endpoint_validation=False
    )
    ensemble_objective = EndpointRelativeOrdinalSemanticProgressReward(
        scorer, ensemble_spec, source, full, temperature=1.0, fail_on_endpoint_validation=False
    )
    single = _run_ordinal_audit(
        "ordinal-single", single_objective, source, full, args.oracle_output_dir, args.strengths, scorer
    )
    ensemble = _run_ordinal_audit(
        "ordinal-ensemble", ensemble_objective, source, full, args.oracle_output_dir, args.strengths, scorer
    )
    selected = _select_method(single, ensemble)

    _write_csv(output_dir / "binary_probe_scores.csv", binary_rows)
    _write_csv(output_dir / "ordinal_single_probe_scores.csv", single["probe_rows"])
    _write_csv(output_dir / "ordinal_ensemble_probe_scores.csv", ensemble["probe_rows"])
    _write_csv(output_dir / "ordinal_single_oracle_scores.csv", single["oracle_rows"])
    _write_csv(output_dir / "ordinal_ensemble_oracle_scores.csv", ensemble["oracle_rows"])
    _write_csv(
        output_dir / "ordinal_choice_distributions.csv",
        single["distribution_rows"] + ensemble["distribution_rows"],
    )
    gradient_report = {"ordinal-single": single["gradient"], "ordinal-ensemble": ensemble["gradient"]}
    label_bias_report = {"ordinal-single": single["label_bias"], "ordinal-ensemble": ensemble["label_bias"]}
    _write_json(output_dir / "ordinal_gradient_audit.json", gradient_report)
    _write_json(output_dir / "ordinal_qwen_gradient_audit.json", gradient_report)
    _write_json(output_dir / "ordinal_label_bias_audit.json", label_bias_report)
    _make_probe_grid(source, full, single["probe_rows"], ensemble["probe_rows"], output_dir / "ordinal_probe_grid.png")

    report = {
        "scope": "binary-v1 versus ordinal-v2 reward bake-off",
        "existing_fact": "binary source-target contrast passed endpoints but failed continuous interior ordering",
        "method_hypothesis": "multiple-choice ordinal stage probabilities may define a more reliable semantic coordinate",
        "ordinal_stage_ground_truth_images_used": False,
        "pixel_blends_used_for_optimization": False,
        "temperature": 1.0,
        "config": vars(args),
        "native_clamp_audit": clamp_audit,
        "zero_control_parity_exact": True,
        "binary-v1": {
            "endpoint": binary.endpoint_diagnostics,
            "probe_rows": binary_rows,
            "probe_summary": binary_summary,
            "oracle_rows": binary_oracle_rows,
            "oracle_summary": binary_oracle_summary,
            "gradient": binary_gradient,
        },
        "ordinal-single": {
            key: value for key, value in single.items() if key not in ("objective", "distribution_rows")
        },
        "ordinal-ensemble": {
            key: value for key, value in ensemble.items() if key not in ("objective", "distribution_rows")
        },
        "selected_by_preregistered_rule": None if selected is None else selected["name"],
        "controller_run": False,
        "controller_result": None,
    }
    _write_json(output_dir / "reward_bakeoff_report.json", report)

    if selected is None:
        print("REWARD_GATES_FAILED_CONTROLLER_NOT_RUN", flush=True)
        return
    if args.audit_only:
        print(f"REWARD_GATES_PASSED_SELECTED={selected['name']}_AUDIT_ONLY", flush=True)
        return

    initial, best_results, final, best_checkpoint, trace = _optimize(
        pipe, inputs, masks.masks, selected["objective"], args.strengths, source.to(device), full.to(device), args
    )
    for name, values in (("initial", initial), ("best", best_results), ("final", final)):
        panels = []
        for strength in args.strengths:
            _tensor_to_pil(values[strength]["image"][0]).save(output_dir / f"ordinal_{name}_{strength:g}.png")
            panels.append((_tensor_to_pil(values[strength]["image"][0]), f"{name} s={strength:g}"))
        width, height = panels[0][0].size
        grid = Image.new("RGB", (width * len(panels), height + 30), "white")
        draw = ImageDraw.Draw(grid)
        for index, (image, label) in enumerate(panels):
            draw.text((index * width + 4, 5), label, fill="black")
            grid.paste(image, (index * width, 30))
        grid.save(output_dir / f"ordinal_{name}_grid.png")
    report["controller_run"] = True
    report["controller_result"] = {
        "selected_reward": selected["name"],
        "best_iteration": best_checkpoint.iteration,
        "best_mean_semantic_error": best_checkpoint.objective_error,
        "initial": {str(key): {k: v for k, v in value.items() if k != "image"} for key, value in initial.items()},
        "best": {str(key): {k: v for k, v in value.items() if k != "image"} for key, value in best_results.items()},
        "final": {str(key): {k: v for k, v in value.items() if k != "image"} for key, value in final.items()},
        "trace": trace,
        "flux_parameters_have_gradient": _flux_has_gradient(pipe),
        "qwen_parameters_have_gradient": _qwen_has_gradient(scorer),
    }
    _write_json(output_dir / "reward_bakeoff_report.json", report)


if __name__ == "__main__":
    main()
