"""Run a minimal FLUX.1-Kontext generator-native T5 suppression diagnostic.

This is a single-pair diagnostic, not a paper-faithful contrastive
Difference-of-Means reproduction. Every factor reuses one captured B=1
native state and changes only T5 ``prompt_embeds``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from diffusers.pipelines.rewardflow.kontext_text_steering import (
    apply_kontext_pooled_steering,
    apply_kontext_text_steering,
    build_single_pair_clip_direction,
    build_single_pair_t5_direction,
    find_t5_phrase_alignment,
    reversion_fraction_to_alpha,
)
from diffusers.pipelines.rewardflow.pipeline_flux_kontext_terminal_control import (
    FluxKontextTerminalControlPipeline,
)


DEFAULT_MODEL = "/data15/hyp/weight/FLUX.1-Kontext-dev"
DEFAULT_SOURCE = "/data15/hyp/dataset/kontinuous_kontext/raw/source_images/source_000000.png"
DEFAULT_PROMPT = "Make the weighted training ball blue while preserving its shape, texture, lighting, and background."
DEFAULT_SOURCE_TEXT = "The weighted training ball is black."
DEFAULT_TARGET_TEXT = "The weighted training ball is blue."
DEFAULT_FACTORS = (-16.0, -8.0, -4.0, -2.0, -1.0, -0.5, -0.25, 0.0)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.getenv("FLUX_KONTEXT_MODEL_PATH", DEFAULT_MODEL))
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--source-semantic-text", default=DEFAULT_SOURCE_TEXT)
    parser.add_argument("--target-semantic-text", default=DEFAULT_TARGET_TEXT)
    parser.add_argument("--source-state-phrase", default="black")
    parser.add_argument("--target-state-phrase", default="blue")
    parser.add_argument("--edit-phrase", default="blue")
    parser.add_argument("--factors", type=float, nargs="+", default=None)
    parser.add_argument("--reversion-fractions", type=float, nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num-inference-steps", "--steps", dest="steps", type=int, default=12)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="experiments/kontext_text_suppression_v1")
    parser.add_argument("--parity-only", action="store_true", help="Stop after the alpha=0 exact parity gate.")
    parser.add_argument("--joint-clip", action="store_true", help="Compare matched T5-only against T5+CLIP.")
    args = parser.parse_args()
    if args.factors is not None and args.reversion_fractions is not None:
        parser.error("--factors and --reversion-fractions are mutually exclusive")
    if args.reversion_fractions is None:
        args.parameterization = "factor"
        args.factors = tuple(sorted(DEFAULT_FACTORS if args.factors is None else args.factors))
        scan_values = args.factors
    else:
        args.parameterization = "reversion-fraction"
        args.reversion_fractions = tuple(sorted(args.reversion_fractions))
        scan_values = args.reversion_fractions
    if not scan_values or any(not math.isfinite(value) for value in scan_values):
        parser.error("scan values must contain at least one finite number")
    if len(set(scan_values)) != len(scan_values):
        parser.error("scan values must be unique")
    if args.parameterization == "reversion-fraction" and any(value < 0 for value in scan_values):
        parser.error("reversion fractions must be nonnegative")
    if 0.0 not in scan_values and not args.parity_only:
        parser.error("scan values must include 0 as the Native Full anchor")
    if args.joint_clip and args.parameterization != "reversion-fraction":
        parser.error("--joint-clip requires --reversion-fractions")
    if args.joint_clip and (
        args.target_semantic_text != args.prompt
        or args.source_semantic_text.replace(args.source_state_phrase, args.target_state_phrase, 1)
        != args.target_semantic_text
    ):
        parser.error("joint CLIP requires instruction-matched source/target texts and target equal to prompt")
    if args.height <= 0 or args.width <= 0 or args.steps <= 0 or args.max_sequence_length <= 0:
        parser.error("height, width, steps, and max sequence length must be positive")
    if not math.isfinite(args.guidance_scale):
        parser.error("guidance scale must be finite")
    return args


def _factor_tag(factor: float) -> str:
    return f"{factor:+.3f}"


def _pil_to_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    array = image.detach().float().clamp(0, 1).mul(255).round().to(torch.uint8)
    return Image.fromarray(array.squeeze(0).permute(1, 2, 0).cpu().numpy(), mode="RGB")


def _distance(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    difference = actual.detach().float() - expected.detach().float()
    return {"mad": float(difference.abs().mean()), "mse": float(difference.square().mean())}


def _latent_distance(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    difference = actual.detach().float() - expected.detach().float()
    return {"l2": float(torch.linalg.vector_norm(difference)), "mad": float(difference.abs().mean())}


def _parity(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    difference = actual.detach().float() - expected.detach().float()
    return {"max_abs_diff": float(difference.abs().max()), "mean_abs_diff": float(difference.abs().mean())}


def _make_grid(
    source: Image.Image,
    native: Image.Image,
    factors: list[float],
    images: dict[float, Image.Image],
    labels: dict[float, str] | None = None,
):
    ordered = [(source, "Source")]
    ordered.extend((images[factor], labels[factor] if labels else f"factor {factor:+g}") for factor in factors)
    ordered.append((native, "Native Full"))
    width, height = source.size
    header_height = 42 if labels else 28
    grid = Image.new("RGB", (width * len(ordered), height + header_height), "white")
    draw = ImageDraw.Draw(grid)
    for index, (image, label) in enumerate(ordered):
        left = index * width
        draw.multiline_text((left + 5, 7), label, fill="black", spacing=1)
        grid.paste(image.resize((width, height), Image.Resampling.LANCZOS), (left, header_height))
    return grid


def _serializable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    return value


def _assert_only_text_conditioning_changed(inputs, changed_inputs, allowed_keys: set[str]) -> None:
    """Audit exact capture reuse for every non-steered tensor and value."""

    if (
        changed_inputs.initial_latent is not inputs.initial_latent
        or changed_inputs.native_final_latent is not inputs.native_final_latent
        or changed_inputs.timesteps is not inputs.timesteps
        or changed_inputs.sigmas is not inputs.sigmas
        or changed_inputs.source_clean_latent is not inputs.source_clean_latent
    ):
        raise AssertionError("Captured trajectory state was replaced")
    if changed_inputs.forward_kwargs.keys() != inputs.forward_kwargs.keys():
        raise AssertionError("Kontext forward key set changed")
    for key, original in inputs.forward_kwargs.items():
        if key in allowed_keys:
            continue
        candidate = changed_inputs.forward_kwargs[key]
        if torch.is_tensor(original):
            if not torch.is_tensor(candidate) or not torch.equal(original, candidate):
                raise AssertionError(f"Non-steered tensor changed: {key}")
        elif candidate != original:
            raise AssertionError(f"Non-steered forward value changed: {key}")


def _make_joint_grid(points, images_by_mode: dict[str, dict[float, Image.Image]], t5_norm: float, clip_norm: float):
    width, height = next(iter(images_by_mode["t5_only"].values())).size
    header = 44
    grid = Image.new("RGB", (width * len(points), 2 * (height + header)), "white")
    draw = ImageDraw.Draw(grid)
    for row, mode in enumerate(("t5_only", "joint_t5_clip")):
        for column, (_, fraction) in enumerate(points):
            alpha_t5 = reversion_fraction_to_alpha(fraction, t5_norm)
            alpha_clip = reversion_fraction_to_alpha(fraction, clip_norm) if row else 0.0
            label = f"{mode} r={fraction:.2f}\nT5={alpha_t5:.2f} CLIP={alpha_clip:.2f}"
            left, top = column * width, row * (height + header)
            draw.multiline_text((left + 5, top + 5), label, fill="black", spacing=1)
            grid.paste(images_by_mode[mode][fraction], (left, top + header))
    return grid


def _run_joint_experiment(
    args,
    pipe,
    inputs,
    edit_alignment,
    t5_direction,
    scan_points,
    native_reunroll,
    native_image,
    source_image,
    report,
    report_path,
    output_dir,
) -> int:
    base_t5 = inputs.forward_kwargs["prompt_embeds"]
    base_clip = inputs.forward_kwargs["pooled_prompt_embeds"]
    clip_direction = build_single_pair_clip_direction(
        pipe, args.source_semantic_text, args.target_semantic_text, base_clip
    )
    report["clip_direction"] = clip_direction.as_dict()
    report["method"].update(
        {
            "type": "matched_t5_only_vs_joint_t5_clip",
            "clip_pooled_modified": True,
            "only_t5_prompt_embeds_modified": False,
        }
    )
    zero_kwargs = dict(inputs.forward_kwargs)
    zero_kwargs["prompt_embeds"] = apply_kontext_text_steering(
        base_t5, edit_alignment.token_indices, t5_direction.direction, 0.0
    )
    zero_kwargs["pooled_prompt_embeds"] = apply_kontext_pooled_steering(base_clip, clip_direction.direction, 0.0)
    zero_inputs = replace(inputs, forward_kwargs=zero_kwargs)
    _assert_only_text_conditioning_changed(inputs, zero_inputs, {"prompt_embeds", "pooled_prompt_embeds"})
    joint_zero = pipe.unroll_terminal_controls(zero_inputs, controls=(), use_checkpointing=False)
    joint_zero_image = pipe.decode_terminal_latent(joint_zero.final_latent, zero_inputs).detach()
    report["parity"]["native_reunroll_vs_joint_zero_latent"] = _parity(
        native_reunroll.final_latent, joint_zero.final_latent
    )
    report["parity"]["native_reunroll_vs_joint_zero_decoded_image"] = _parity(native_image, joint_zero_image)
    if any(item["max_abs_diff"] != 0 for item in report["parity"].values()):
        report["status"] = "IMPLEMENTATION_PARITY_FAIL"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return 2

    source_image.save(output_dir / "source.png")
    native_pil = _tensor_to_pil(native_image)
    native_pil.save(output_dir / "native_full.png")
    source_tensor = _pil_to_tensor(source_image, torch.device(args.device))
    report["source_provenance"] = "SOURCE IMAGE"
    report["native_full_provenance"] = "MODEL GENERATED — NATIVE FULL"
    report["modes"] = {}
    images_by_mode = {}
    for mode in ("t5_only", "joint_t5_clip"):
        mode_dir = output_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        rows, images, labels, previous = [], {}, {}, None
        for alpha_t5, fraction in scan_points:
            alpha_clip = (
                reversion_fraction_to_alpha(fraction, clip_direction.raw_direction_norm)
                if mode == "joint_t5_clip"
                else 0.0
            )
            kwargs = dict(inputs.forward_kwargs)
            kwargs["prompt_embeds"] = apply_kontext_text_steering(
                base_t5, edit_alignment.token_indices, t5_direction.direction, alpha_t5
            )
            if mode == "joint_t5_clip":
                kwargs["pooled_prompt_embeds"] = apply_kontext_pooled_steering(
                    base_clip, clip_direction.direction, alpha_clip
                )
            current_inputs = replace(inputs, forward_kwargs=kwargs)
            _assert_only_text_conditioning_changed(
                inputs,
                current_inputs,
                {"prompt_embeds"} if mode == "t5_only" else {"prompt_embeds", "pooled_prompt_embeds"},
            )
            unroll = pipe.unroll_terminal_controls(current_inputs, controls=(), use_checkpointing=False)
            image = pipe.decode_terminal_latent(unroll.final_latent, current_inputs).detach()
            if not torch.isfinite(unroll.final_latent).all() or not torch.isfinite(image).all():
                raise RuntimeError(f"Non-finite {mode} output at r={fraction}")
            rendered = _tensor_to_pil(image)
            image_file = f"fraction_{fraction:.2f}.png"
            rendered.save(mode_dir / image_file)
            images[fraction] = rendered
            labels[alpha_t5] = f"r={fraction:.2f}\nT5={alpha_t5:.2f} CLIP={alpha_clip:.2f}"
            rows.append(
                {
                    "reversion_fraction": fraction,
                    "alpha_t5": alpha_t5,
                    "alpha_clip": alpha_clip,
                    "image_file": f"{mode}/{image_file}",
                    "finite": True,
                    "image_provenance": "MODEL GENERATED — TEXT STEERING DIAGNOSTIC",
                    "t5_actual_perturbation_norm": float(
                        torch.linalg.vector_norm((kwargs["prompt_embeds"] - base_t5).float())
                    ),
                    "clip_actual_perturbation_norm": float(
                        torch.linalg.vector_norm((kwargs["pooled_prompt_embeds"] - base_clip).float())
                    ),
                    "image_mad_mse_to_source": _distance(image, source_tensor),
                    "image_mad_mse_to_native_full": _distance(image, native_image),
                    "latent_distance_to_native_full": _latent_distance(
                        unroll.final_latent, inputs.native_final_latent
                    ),
                    "adjacent_image_mad": None if previous is None else _distance(image, previous)["mad"],
                }
            )
            previous = image
        _make_grid(
            source_image,
            native_pil,
            [alpha for alpha, _ in scan_points],
            {alpha: images[fraction] for alpha, fraction in scan_points},
            labels,
        ).save(mode_dir / "grid.png")
        report["modes"][mode] = rows
        images_by_mode[mode] = images
    _make_joint_grid(
        scan_points, images_by_mode, t5_direction.raw_direction_norm, clip_direction.raw_direction_norm
    ).save(output_dir / "comparison_grid.png")
    responded = any(
        row["image_mad_mse_to_native_full"]["mad"] > 0
        for rows in report["modes"].values()
        for row in rows
        if row["reversion_fraction"] != 0
    )
    report["status"] = "STEERING_RESPONSE_DETECTED" if responded else "NO_STEERING_RESPONSE"
    report["visual_status"] = "VISUAL_REVIEW_REQUIRED"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"status={report['status']} grid={output_dir / 'comparison_grid.png'} report={report_path}", flush=True)
    return 0


def _report(args, head: str, source: str) -> dict:
    return {
        "scope": (
            "Minimal FLUX-Kontext native-text suppression diagnostic. Tests whether negative T5 steering can retract "
            "a correct Native Full Edit. Not a semantic-strength calibration method and not a paper-faithful "
            "contrastive Difference-of-Means reproduction."
        ),
        "status": "RUNNING",
        "git_head": head,
        "model": os.path.realpath(args.model),
        "source": source,
        "prompt": args.prompt,
        "method": {
            "type": "single_pair_generator_native_t5_negative_steering",
            "paper_faithful_dom": False,
            "velocity_control_used": False,
            "reward_used": False,
            "elastic_band_used": False,
            "clip_pooled_modified": False,
            "only_t5_prompt_embeds_modified": True,
            "parameterization": args.parameterization,
        },
        "semantic_pair": {
            "source_text": args.source_semantic_text,
            "target_text": args.target_semantic_text,
            "source_phrase": args.source_state_phrase,
            "target_phrase": args.target_state_phrase,
            "edit_phrase": args.edit_phrase,
        },
        "generation_config": {
            "seed": args.seed,
            "height": args.height,
            "width": args.width,
            "num_inference_steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "dtype": args.dtype,
            "device": args.device,
            "max_sequence_length": args.max_sequence_length,
            "factors": list(args.factors or ()),
            "reversion_fractions": list(args.reversion_fractions or ()),
            "batching": "B=1 sequential unroll per factor; no factor batching",
        },
        "interpretation_guardrails": {
            "factor_is_not_semantic_percentage": True,
            "pixel_distance_is_not_semantic_strength": True,
            "visual_review_required": True,
        },
    }


def main() -> int:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = os.path.realpath(args.source)
    source_image = Image.open(args.source).convert("RGB").resize((args.width, args.height), Image.Resampling.LANCZOS)
    source_image.save(output_dir / "source.png")
    head = os.popen("git rev-parse HEAD").read().strip() or "unknown"
    report = _report(args, head, source_path)
    report_path = output_dir / "report.json"
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("The formal Kontext text suppression diagnostic requires CUDA.")
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    pipe = FluxKontextTerminalControlPipeline.from_pretrained(args.model, torch_dtype=dtype, local_files_only=True).to(
        device
    )
    pipe.set_progress_bar_config(disable=True)
    inputs = pipe.prepare_terminal_control_inputs(
        image=source_image,
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=torch.Generator(device=device).manual_seed(args.seed),
        max_sequence_length=args.max_sequence_length,
    )

    native_reunroll = pipe.unroll_terminal_controls(inputs, controls=(), use_checkpointing=False)
    base_prompt_embeds = inputs.forward_kwargs["prompt_embeds"]
    edit_alignment = find_t5_phrase_alignment(
        pipe.tokenizer_2, args.prompt, args.edit_phrase, args.max_sequence_length
    )
    direction = build_single_pair_t5_direction(
        pipe,
        args.source_semantic_text,
        args.target_semantic_text,
        args.source_state_phrase,
        args.target_state_phrase,
        args.max_sequence_length,
        base_prompt_embeds,
    )
    report["token_alignment"] = {
        "source_semantic": find_t5_phrase_alignment(
            pipe.tokenizer_2, args.source_semantic_text, args.source_state_phrase, args.max_sequence_length
        ).as_dict(),
        "target_semantic": find_t5_phrase_alignment(
            pipe.tokenizer_2, args.target_semantic_text, args.target_state_phrase, args.max_sequence_length
        ).as_dict(),
        "edit": edit_alignment.as_dict(),
    }
    report["direction"] = direction.as_dict()
    if args.parameterization == "reversion-fraction":
        scan_points = [
            (reversion_fraction_to_alpha(fraction, direction.raw_direction_norm), fraction)
            for fraction in args.reversion_fractions
        ]
        args.factors = tuple(alpha for alpha, _ in scan_points)
        report["generation_config"]["factors"] = list(args.factors)
    else:
        scan_points = [(factor, None) for factor in args.factors]
    zero_prompt = apply_kontext_text_steering(
        base_prompt_embeds, edit_alignment.token_indices, direction.direction, 0.0
    )
    zero_kwargs = dict(inputs.forward_kwargs)
    zero_kwargs["prompt_embeds"] = zero_prompt
    zero_inputs = replace(inputs, forward_kwargs=zero_kwargs)
    zero_reunroll = pipe.unroll_terminal_controls(zero_inputs, controls=(), use_checkpointing=False)
    captured_native = inputs.native_final_latent
    native_image = pipe.decode_terminal_latent(native_reunroll.final_latent, inputs).detach()
    zero_image = pipe.decode_terminal_latent(zero_reunroll.final_latent, zero_inputs).detach()
    report["parity"] = {
        "captured_native_vs_native_reunroll_latent": _parity(captured_native, native_reunroll.final_latent),
        "native_reunroll_vs_factor_zero_latent": _parity(native_reunroll.final_latent, zero_reunroll.final_latent),
        "native_reunroll_vs_factor_zero_decoded_image": _parity(native_image, zero_image),
    }
    if any(item["max_abs_diff"] != 0 for item in report["parity"].values()):
        report["status"] = "IMPLEMENTATION_PARITY_FAIL"
        report_path.write_text(json.dumps(_serializable(report), indent=2), encoding="utf-8")
        print(f"IMPLEMENTATION_PARITY_FAIL report={report_path}", flush=True)
        return 2
    if args.parity_only:
        report["status"] = "VISUAL_REVIEW_REQUIRED"
        report["parity_gate_passed"] = True
        report_path.write_text(json.dumps(_serializable(report), indent=2), encoding="utf-8")
        print(f"alpha=0 exact parity PASS report={report_path}", flush=True)
        return 0
    if args.joint_clip:
        return _run_joint_experiment(
            args,
            pipe,
            inputs,
            edit_alignment,
            direction,
            scan_points,
            native_reunroll,
            native_image,
            source_image,
            report,
            report_path,
            output_dir,
        )
    report["source_provenance"] = "SOURCE IMAGE"
    report["native_full_provenance"] = "MODEL GENERATED — NATIVE FULL"
    source_tensor = _pil_to_tensor(source_image, device)
    native_pil = _tensor_to_pil(native_image)
    native_pil.save(output_dir / "native_full.png")
    result_images, results, labels = {}, [], {}
    previous_image = None
    for factor, fraction in scan_points:
        steered = apply_kontext_text_steering(
            base_prompt_embeds, edit_alignment.token_indices, direction.direction, factor
        )
        forward_kwargs = dict(inputs.forward_kwargs)
        forward_kwargs["prompt_embeds"] = steered
        factor_inputs = replace(inputs, forward_kwargs=forward_kwargs)
        unroll = pipe.unroll_terminal_controls(factor_inputs, controls=(), use_checkpointing=False)
        image = pipe.decode_terminal_latent(unroll.final_latent, factor_inputs).detach()
        if not torch.isfinite(unroll.final_latent).all() or not torch.isfinite(image).all():
            raise RuntimeError(f"Non-finite output at factor {factor}.")
        image_pil = _tensor_to_pil(image)
        image_name = (
            f"factor_{_factor_tag(factor)}.png"
            if fraction is None
            else f"fraction_{fraction:.2f}_alpha_{_factor_tag(factor)}.png"
        )
        image_pil.save(output_dir / image_name)
        result_images[factor] = image_pil
        if fraction is not None:
            labels[factor] = f"r={fraction:.2f}\nalpha={factor:.2f}"
        delta = steered.detach().float() - base_prompt_embeds.detach().float()
        results.append(
            {
                "factor": factor,
                "reversion_fraction": fraction,
                "image_file": image_name,
                "image_provenance": "MODEL GENERATED — TEXT STEERING DIAGNOSTIC",
                "finite": True,
                "prompt_embedding_perturbation_norm": float(torch.linalg.vector_norm(delta)),
                "image_mad_mse_to_source": _distance(image, source_tensor),
                "image_mad_mse_to_native_full": _distance(image, native_image),
                "latent_distance_to_native_full": _latent_distance(unroll.final_latent, captured_native),
                "adjacent_image_mad": None if previous_image is None else _distance(image, previous_image)["mad"],
            }
        )
        previous_image = image
    report["results"] = results
    full = [row["image_mad_mse_to_native_full"]["mad"] for row in results]
    source = [row["image_mad_mse_to_source"]["mad"] for row in results]
    report["reversion_signal"] = {
        "factor_order_as_provided": list(args.factors),
        "scan_order": "increasing_reversion_fraction" if args.reversion_fractions is not None else "increasing_factor",
        "distance_to_full_over_input_order": full,
        "distance_to_source_over_input_order": source,
        "distance_to_full_non_increasing_in_input_order": all(a >= b for a, b in zip(full, full[1:])),
        "distance_to_full_non_decreasing_in_input_order": all(a <= b for a, b in zip(full, full[1:])),
        "distance_to_source_non_decreasing_in_input_order": all(a <= b for a, b in zip(source, source[1:])),
        "distance_to_source_non_increasing_in_input_order": all(a >= b for a, b in zip(source, source[1:])),
    }
    nonzero = [row for row in results if row["factor"] != 0]
    response = any(row["image_mad_mse_to_native_full"]["mad"] > 0 for row in nonzero)
    report["status"] = "STEERING_RESPONSE_DETECTED" if response else "NO_STEERING_RESPONSE"
    report["visual_status"] = "VISUAL_REVIEW_REQUIRED"
    _make_grid(source_image, native_pil, list(args.factors), result_images, labels or None).save(
        output_dir / "grid.png"
    )
    report_path.write_text(json.dumps(_serializable(report), indent=2), encoding="utf-8")
    print(f"status={report['status']} grid={output_dir / 'grid.png'} report={report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
