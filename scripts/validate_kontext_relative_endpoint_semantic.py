"""Gate Relative Endpoint Semantic Reward v3 before unchanged Kontext control.

Held-out oracle images are model-generated evaluation probes. Pixel blends are
secondary diagnostics only and are never controller targets or reward anchors.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from diffusers.pipelines.rewardflow.pipeline_flux_kontext_terminal_control import FluxKontextTerminalControlPipeline
from diffusers.pipelines.rewardflow.relative_endpoint_parser import (
    TianyuAIRelativeEndpointParser,
    fingerprint_endpoint_image,
    make_human_relative_endpoint_parse_record,
    parse_relative_endpoint_semantic_json,
    save_cached_relative_endpoint_parse,
)
from diffusers.pipelines.rewardflow.relative_endpoint_semantic import (
    RELATIVE_ENDPOINT_CHOICES,
    RelativeEndpointSemanticReward,
    audit_endpoint_answers,
    build_relative_endpoint_comparison_prompt,
)
from diffusers.pipelines.rewardflow.rewards import Qwen25VQATeacherForcedScorer
from diffusers.pipelines.rewardflow.terminal_control import (
    initialize_velocity_controls,
    masked_effective_controls,
    normalized_effective_control_energy,
    update_best_control_checkpoint,
)


DEFAULT_MODEL = "/data15/hyp/weight/FLUX.1-Kontext-dev"
DEFAULT_SOURCE = "/data15/hyp/dataset/kontinuous_kontext/raw/source_images/source_000000.png"
DEFAULT_PROMPT = "Make the weighted training ball blue while preserving its shape, texture, lighting, and background."
DEFAULT_ORACLE = "/tmp/st_formal_shared_t4_mask"
DEFAULT_OUTPUT = "experiments/relative_endpoint_semantic_v3"
PROBE_ALPHAS = (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0)


def _args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.getenv("FLUX_KONTEXT_MODEL_PATH", DEFAULT_MODEL))
    parser.add_argument("--qwen-model", default=os.getenv("QWEN25_VL_MODEL_PATH"))
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--relative-spec")
    parser.add_argument("--auto-parse", action="store_true")
    parser.add_argument("--parser-cache", default="experiments/relative_endpoint_semantic_v3/parser_cache.json")
    parser.add_argument("--oracle-output-dir", default=DEFAULT_ORACLE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reward-device", default="cuda:0")
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
    if not args.relative_spec and not args.auto_parse:
        parser.error("Pass --relative-spec, or explicitly opt into one online call with --auto-parse.")
    if args.relative_spec and args.auto_parse:
        parser.error("Choose either --relative-spec or --auto-parse, not both.")
    for name in ("model", "qwen_model", "source", "oracle_output_dir"):
        value = getattr(args, name)
        if not value or not Path(value).exists():
            parser.error(f"`--{name.replace('_', '-')}` must point to an existing local path.")
    expected = ((args.steps, args.seed, args.height, args.width), tuple(args.strengths), args.control_steps)
    if expected != ((12, 20260914, 256, 256), (0.2, 0.5, 0.8), 4):
        parser.error(
            "Formal configuration requires steps=12, seed=20260914, 256x256, strengths=.2/.5/.8, control_steps=4."
        )
    if args.guidance_scale != 2.5 or args.control_lr != 0.1 or args.outer_iters != 20 or args.lambda_control != 1e-4:
        parser.error("Formal guidance/controller hyperparameters were changed.")
    return args


def _tensor(image, device):
    array = np.asarray(image.convert("RGB"), dtype=np.float32).copy() / 255
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def _pil(image):
    array = image.detach().clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array, mode="RGB")


def _serialize(value):
    if torch.is_tensor(value):
        value = value.detach().float().cpu()
        return value.item() if value.numel() == 1 else value.tolist()
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_serialize(item) for item in value]
    return value


def _json(path, value):
    path.write_text(json.dumps(_serialize(value), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _rank(values):
    values = torch.tensor(values, dtype=torch.float64)
    expected = torch.arange(values.numel(), dtype=torch.float64)
    return torch.corrcoef(torch.stack((values.argsort().argsort().double(), expected)))[0, 1].item()


def _decode(pipe, latent, inputs):
    unpacked = pipe._unpack_latents(latent, inputs.height, inputs.width, pipe.vae_scale_factor)
    unpacked = unpacked / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
    decoded = pipe.vae.decode(unpacked, return_dict=False)[0]
    raw = decoded / 2 + 0.5
    return raw.clamp(0, 1), {
        "fraction_below_zero_before_clamp": (raw < 0).float().mean().detach().item(),
        "fraction_above_one_before_clamp": (raw > 1).float().mean().detach().item(),
    }


def _oracle_path(directory, strength):
    return Path(directory) / f"final_{str(strength).replace('.', 'p')}.png"


def _diagnostic(objective, image, target=0.5):
    primitive = objective.spec.primitives[0]
    values = objective.primitive_margin(image, primitive)
    anchor = objective._anchors[primitive.id]
    dynamic_range = anchor["dynamic_range"].to(image.device)
    raw_margin = values["raw_margin"]
    if bool(torch.isfinite(dynamic_range).item()) and bool(
        (dynamic_range.abs() > objective.denominator_epsilon).item()
    ):
        target_margin = anchor["source_margin"].to(image.device) + target * dynamic_range
        residual = (raw_margin - target_margin) / dynamic_range
        coordinate = (raw_margin - anchor["source_margin"].to(image.device)) / dynamic_range
        error, loss = residual.abs(), residual.square()
    else:
        coordinate = error = loss = None
    return {
        "raw_margin": raw_margin,
        "relative_coordinate": coordinate,
        "semantic_residual_absolute": error,
        "semantic_loss": loss,
        "per_primitive": {primitive.id: values},
    }


def _oracle_audit(objective, directory, strengths, device):
    rows = []
    for strength in strengths:
        path = _oracle_path(directory, strength)
        image = _tensor(Image.open(path), device)
        with torch.no_grad():
            values = _diagnostic(objective, image, strength)
        rows.append(
            {
                "provenance": "held_out_model_generated_flux_controller_output",
                "used_for_reward_construction_or_calibration": False,
                "strength_label": strength,
                "raw_margin": _serialize(values["raw_margin"]),
                "relative_coordinate": _serialize(values["relative_coordinate"]),
                "semantic_residual_absolute": _serialize(values["semantic_residual_absolute"]),
                "semantic_loss": _serialize(values["semantic_loss"]),
            }
        )
    margins = [row["raw_margin"] for row in rows]
    gaps = [right - left for left, right in zip(margins, margins[1:])]
    return rows, {"strict_raw_margin_order": all(gap > 0 for gap in gaps), "adjacent_raw_margin_gaps": gaps}


def _pixel_probe(objective, source, full):
    rows = []
    for alpha in PROBE_ALPHAS:
        image = (1 - alpha) * source + alpha * full
        with torch.no_grad():
            values = _diagnostic(objective, image, alpha)
        rows.append(
            {
                "provenance": "PIXEL_BLEND_DIAGNOSTIC_NOT_MODEL_GENERATED_OUTPUT",
                "alpha": alpha,
                "raw_margin": _serialize(values["raw_margin"]),
                "relative_coordinate": _serialize(values["relative_coordinate"]),
            }
        )
    values = [row["raw_margin"] for row in rows]
    inversions = sum(
        values[left] >= values[right] for left in range(len(values)) for right in range(left + 1, len(values))
    )
    return rows, {"spearman": _rank(values), "inversions": inversions, "secondary_diagnostic_only": True}


def _pixel_grid(source, full, rows, path):
    width, height = source.shape[-1], source.shape[-2]
    top = 48
    grid = Image.new("RGB", (width * len(rows), height + top), "white")
    draw = ImageDraw.Draw(grid)
    draw.text((4, 3), "PIXEL BLEND DIAGNOSTIC — NOT MODEL-GENERATED OUTPUT", fill="red")
    for index, row in enumerate(rows):
        alpha = row["alpha"]
        image = _pil(((1 - alpha) * source + alpha * full)[0])
        draw.text((index * width + 4, 24), f"alpha={alpha:g} margin={row['raw_margin']:.3f}", fill="black")
        grid.paste(image, (index * width, top))
    grid.save(path)


def _gradient_direction_audit(objective, scorer, candidate):
    primitive = objective.spec.primitives[0]
    audit = {"candidate_provenance": "held_out_model_generated_oracle_0p5", "directions": {}}
    for direction, sign in (("toward_full", -1.0), ("toward_source", 1.0)):
        image = candidate.detach().requires_grad_(True)
        margin = objective.primitive_margin(image, primitive)["raw_margin"]
        loss = sign * margin
        gradient = torch.autograd.grad(loss, image)[0]
        rms = gradient.float().square().mean().sqrt()
        trials = []
        for step_rms in (1e-5, 3e-5, 1e-4, 3e-4):
            stepped_image = (image.detach() - step_rms * gradient.detach() / rms.clamp_min(1e-12)).clamp(0, 1)
            with torch.no_grad():
                stepped = objective.primitive_margin(stepped_image, primitive)["raw_margin"]
            correct = stepped > margin if direction == "toward_full" else stepped < margin
            trials.append({"step_rms": step_rms, "new_margin": stepped, "direction_correct": bool(correct.item())})
        audit["directions"][direction] = {
            "loss_definition": "-raw_margin" if sign < 0 else "+raw_margin",
            "initial_margin": margin,
            "gradient_finite": bool(torch.isfinite(gradient).all().item()),
            "gradient_nonzero": bool((gradient.abs().sum() > 0).item()),
            "gradient_l2_norm": torch.linalg.vector_norm(gradient.float()),
            "trials": trials,
            "passed": bool(torch.isfinite(gradient).all().item())
            and bool((gradient.abs().sum() > 0).item())
            and any(trial["direction_correct"] for trial in trials),
        }
    audit["qwen_parameters_require_grad"] = any(parameter.requires_grad for parameter in scorer.model.parameters())
    audit["qwen_parameters_have_grad"] = any(parameter.grad is not None for parameter in scorer.model.parameters())
    audit["source_requires_grad"] = objective.source_image.requires_grad
    audit["full_requires_grad"] = objective.full_image.requires_grad
    return audit


def _swapped_prompt(prompt):
    return prompt.replace("A. Image 1\nB. Image 3", "A. Image 3\nB. Image 1")


def _label_swap_margin(objective, image, primitive):
    prompt = build_relative_endpoint_comparison_prompt(objective.spec, primitive, first_reference="source")
    scores = objective.scorer.score_multi_image_single_token_choices(
        (objective.source_image, image, objective.full_image), _swapped_prompt(prompt), RELATIVE_ENDPOINT_CHOICES
    )
    return scores[0] - scores[1]


def _label_swap_audit(objective, oracle_dir, strengths, device):
    primitive = objective.spec.primitives[0]
    rows = []
    with torch.no_grad():
        for strength in strengths:
            image = _tensor(Image.open(_oracle_path(oracle_dir, strength)), device)
            production = objective.primitive_margin(image, primitive)["raw_margin"]
            swapped = _label_swap_margin(objective, image, primitive)
            rows.append(
                {
                    "strength": strength,
                    "production_margin": production,
                    "swapped_remapped_margin": swapped,
                    "change": swapped - production,
                }
            )
    swapped = [_serialize(row["swapped_remapped_margin"]) for row in rows]
    return {"diagnostic_only": True, "rows": rows, "swapped_ordering_strict": swapped[0] < swapped[1] < swapped[2]}


def _frozen_grad(module):
    return any(parameter.grad is not None for parameter in module.parameters())


def _objective_device(objective):
    return next(iter(objective._anchors.values()))["source_margin"].device


def _evaluate(pipe, inputs, directions, masks, objective, strengths):
    result = {}
    reward_device = _objective_device(objective)
    with torch.no_grad():
        for strength in strengths:
            controls = masked_effective_controls(directions, masks, strength=strength)
            unroll = pipe.unroll_terminal_controls(inputs, controls, use_checkpointing=False)
            image, clamp = _decode(pipe, unroll.final_latent, inputs)
            output = objective(image.to(reward_device), strength)
            result[strength] = {
                "image": image.cpu(),
                "relative_coordinate": output.achieved_score.item(),
                "semantic_error": output.objective_error.item(),
                "semantic_loss": output.loss.item(),
                "clamp": clamp,
            }
    return result


def _optimize(pipe, inputs, masks, objective, strengths, args):
    directions = initialize_velocity_controls(inputs.initial_latent, args.control_steps)
    optimizer = torch.optim.Adam(directions, lr=args.control_lr)
    initial = _evaluate(pipe, inputs, directions, masks, objective, strengths)
    best = update_best_control_checkpoint(
        None,
        iteration=0,
        objective_error=torch.tensor(sum(v["semantic_error"] for v in initial.values()) / len(strengths)),
        controls=directions,
    )
    trace = []
    reward_device = _objective_device(objective)
    for iteration in range(1, args.outer_iters + 1):
        optimizer.zero_grad(set_to_none=True)
        errors = []
        for strength in strengths:
            controls = masked_effective_controls(directions, masks, strength=strength)
            unroll = pipe.unroll_terminal_controls(inputs, controls, use_checkpointing=args.use_checkpointing)
            image, _ = _decode(pipe, unroll.final_latent, inputs)
            output = objective(image.to(reward_device), strength)
            (output.loss / len(strengths)).backward()
            errors.append(output.objective_error.detach())
        regularization = normalized_effective_control_energy(
            [masked_effective_controls(directions, masks, strength=value) for value in strengths]
        )
        (args.lambda_control * regularization).backward()
        if any(parameter.grad is None or not torch.isfinite(parameter.grad).all() for parameter in directions):
            raise RuntimeError("Controller directions have missing or non-finite gradients.")
        if _frozen_grad(pipe.transformer) or _frozen_grad(pipe.vae) or _frozen_grad(objective.scorer.model):
            raise RuntimeError("A frozen backbone unexpectedly accumulated parameter gradients.")
        mean_error = torch.stack(errors).mean()
        best = update_best_control_checkpoint(
            best, iteration=iteration - 1, objective_error=mean_error, controls=directions
        )
        trace.append({"iteration": iteration, "mean_semantic_error": mean_error, "control_energy": regularization})
        optimizer.step()
    final = _evaluate(pipe, inputs, directions, masks, objective, strengths)
    best = update_best_control_checkpoint(
        best,
        iteration=args.outer_iters,
        objective_error=torch.tensor(sum(v["semantic_error"] for v in final.values()) / len(strengths)),
        controls=directions,
    )
    return initial, _evaluate(pipe, inputs, best.controls, masks, objective, strengths), final, best, trace


def main():
    args = _args()
    if not torch.cuda.is_available():
        raise RuntimeError("Real relative endpoint validation requires CUDA.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device, reward_device = torch.device(args.device), torch.device(args.reward_device)
    source_pil = Image.open(args.source).convert("RGB").resize((args.width, args.height), Image.Resampling.LANCZOS)
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
    masks, native = pipe.prepare_velocity_edit_masks(inputs, control_steps=4, mode="velocity-topk", topk_fraction=0.25)
    if not torch.equal(native.final_latent, inputs.native_final_latent):
        raise RuntimeError("Zero-control parity is not exact.")
    with torch.no_grad():
        full, clamp = _decode(pipe, inputs.native_final_latent, inputs)
    source = _tensor(source_pil, reward_device)
    full = full.to(reward_device)
    source_pil.save(output_dir / "source_INPUT.png")
    _pil(full[0]).save(output_dir / "native_full_MODEL_GENERATED.png")

    if args.relative_spec:
        payload = json.loads(Path(args.relative_spec).read_text(encoding="utf-8"))
        if set(payload) >= {"spec", "provenance"}:
            spec = parse_relative_endpoint_semantic_json(json.dumps(payload["spec"]))
            provenance = payload["provenance"]
        else:
            spec = parse_relative_endpoint_semantic_json(json.dumps(payload))
            record = make_human_relative_endpoint_parse_record(
                spec, fingerprint_endpoint_image(args.source), hashlib_tensor(full)
            )
            provenance = record.provenance
    else:
        full_path = output_dir / "native_full_MODEL_GENERATED.png"
        record = TianyuAIRelativeEndpointParser().parse(args.source, full_path, args.prompt)
        save_cached_relative_endpoint_parse(args.parser_cache, record)
        spec, provenance = record.spec, record.provenance
    if len(spec.primitives) != 1 or spec.unresolved_instruction_items:
        raise ValueError("The formal ball audit requires exactly one primitive and no unresolved instruction items.")

    scorer = Qwen25VQATeacherForcedScorer(
        args.qwen_model, device=reward_device, dtype=torch.bfloat16, local_files_only=True
    )
    objective = RelativeEndpointSemanticReward(scorer, spec, source, full, fail_on_endpoint_validation=False)
    oracle_middle = _tensor(Image.open(_oracle_path(args.oracle_output_dir, 0.5)), reward_device)
    primitive = spec.primitives[0]
    comparison_prompt = build_relative_endpoint_comparison_prompt(spec, primitive, first_reference="source")
    fidelity = scorer.compare_multi_image_with_official_processor((source, oracle_middle, full), comparison_prompt)
    fidelity_gate = bool(
        fidelity["grid_equal"]
        and fidelity["pixel_cosine"] >= 0.999
        and math.isfinite(fidelity["choice_logprob_max_abs_difference"])
    )
    endpoint_vqa = audit_endpoint_answers(scorer, spec, source, full)
    endpoint_vqa_gate = all(item["valid"] for item in endpoint_vqa.values())
    endpoint_gate = not objective.endpoint_validation_failed and endpoint_vqa_gate
    oracle_rows, oracle_summary = _oracle_audit(objective, args.oracle_output_dir, args.strengths, reward_device)
    gradient = _gradient_direction_audit(objective, scorer, oracle_middle)
    label_bias = _label_swap_audit(objective, args.oracle_output_dir, args.strengths, reward_device)
    blend_rows, blend_summary = _pixel_probe(objective, source, full)
    _pixel_grid(source, full, blend_rows, output_dir / "pixel_blend_probe_NOT_MODEL_OUTPUT.png")
    gates = {
        "multi_image_processor_fidelity": fidelity_gate,
        "endpoint_semantic_preference_both_orders": endpoint_gate,
        "finite_nonzero_endpoint_dynamic_range": all(
            item["finite"] and item["dynamic_range_valid"] for item in objective.endpoint_diagnostics.values()
        ),
        "held_out_model_generated_oracle_strict_raw_margin_order": oracle_summary["strict_raw_margin_order"],
        "oracle_candidate_gradient_toward_full": gradient["directions"]["toward_full"]["passed"],
        "oracle_candidate_gradient_toward_source": gradient["directions"]["toward_source"]["passed"],
    }
    gates["all_passed"] = all(gates.values())
    report = {
        "scope": "Relative Endpoint Semantic Reward v3 preregistered gate",
        "semantic_spec": asdict(spec),
        "semantic_spec_provenance": provenance,
        "semantic_parser": {
            "online_call_requested": args.auto_parse,
            "provider": provenance.get("provider"),
            "model": provenance.get("model"),
            "base_url": provenance.get("base_url"),
        },
        "score_interpretation": "symmetrized relative endpoint preference margin; not human semantic strength",
        "oracle_provenance": "held-out model-generated outputs; never used to build or calibrate reward",
        "pixel_blend_provenance": "secondary NOT MODEL-GENERATED diagnostic; never optimized",
        "config": vars(args),
        "native_clamp_audit": clamp,
        "endpoint": objective.endpoint_diagnostics,
        "endpoint_vqa_audit_not_continuous_reward": endpoint_vqa,
        "multi_image_processor_fidelity": fidelity,
        "oracle": {"rows": oracle_rows, "summary": oracle_summary},
        "gradient": gradient,
        "label_swap_diagnostic": label_bias,
        "pixel_blend": {"rows": blend_rows, "summary": blend_summary},
        "gates": gates,
        "controller_run": False,
        "controller_result": None,
    }
    _json(
        output_dir / "relative_endpoint_endpoint_scores.json",
        {"endpoint": objective.endpoint_diagnostics, "endpoint_vqa": endpoint_vqa},
    )
    _json(output_dir / "relative_endpoint_gradient_audit.json", gradient)
    _json(output_dir / "relative_endpoint_order_bias.json", label_bias)
    _json(output_dir / "multi_image_processor_fidelity.json", fidelity)
    _csv(output_dir / "relative_endpoint_oracle_scores.csv", oracle_rows)
    _csv(output_dir / "pixel_blend_probe_NOT_MODEL_OUTPUT.csv", blend_rows)
    _json(output_dir / "relative_endpoint_report.json", report)
    if not gates["all_passed"]:
        print("RELATIVE_ENDPOINT_REWARD_GATES_FAILED", flush=True)
        print("CONTROLLER_NOT_RUN", flush=True)
        return
    if args.audit_only:
        print("RELATIVE_ENDPOINT_REWARD_GATES_PASSED_AUDIT_ONLY", flush=True)
        return

    initial, best, final, checkpoint, trace = _optimize(pipe, inputs, masks.masks, objective, args.strengths, args)
    for phase, values in (("initial", initial), ("best", best), ("final", final)):
        panels = []
        for strength in args.strengths:
            image = _pil(values[strength]["image"][0])
            filename = output_dir / f"relative_reward_controller_{phase}_{strength:g}_MODEL_GENERATED.png"
            image.save(filename)
            panels.append((image, f"MODEL GENERATED {phase} s={strength:g}"))
        width, height = panels[0][0].size
        grid = Image.new("RGB", (width * len(panels), height + 32), "white")
        draw = ImageDraw.Draw(grid)
        for index, (image, label) in enumerate(panels):
            draw.text((index * width + 4, 5), label, fill="black")
            grid.paste(image, (index * width, 32))
        grid.save(output_dir / f"relative_reward_controller_{phase}_MODEL_GENERATED_grid.png")
    report["controller_run"] = True
    report["controller_result"] = {
        "best_iteration": checkpoint.iteration,
        "best_mean_semantic_error": checkpoint.objective_error,
        "initial": {str(k): {x: y for x, y in v.items() if x != "image"} for k, v in initial.items()},
        "best": {str(k): {x: y for x, y in v.items() if x != "image"} for k, v in best.items()},
        "final": {str(k): {x: y for x, y in v.items() if x != "image"} for k, v in final.items()},
        "trace": trace,
    }
    _json(output_dir / "relative_endpoint_report.json", report)
    print("RELATIVE_ENDPOINT_REWARD_GATES_PASSED_CONTROLLER_COMPLETE", flush=True)


def hashlib_tensor(tensor):
    import hashlib

    return hashlib.sha256(tensor.detach().float().cpu().numpy().tobytes()).hexdigest()


if __name__ == "__main__":
    main()
