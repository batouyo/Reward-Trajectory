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
    apply_kontext_text_steering,
    build_single_pair_t5_direction,
    find_t5_phrase_alignment,
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
    parser.add_argument("--factors", type=float, nargs="+", default=DEFAULT_FACTORS)
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
    args = parser.parse_args()
    if not args.factors or any(not math.isfinite(value) for value in args.factors):
        parser.error("--factors must contain at least one finite number")
    if len(set(args.factors)) != len(args.factors):
        parser.error("--factors must be unique")
    if 0.0 not in args.factors and not args.parity_only:
        parser.error("--factors must include 0 as the Native Full anchor")
    args.factors = tuple(sorted(args.factors))
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


def _make_grid(source: Image.Image, native: Image.Image, factors: list[float], images: dict[float, Image.Image]):
    ordered = [(source, "Source")]
    ordered.extend((images[factor], f"factor {factor:+g}") for factor in factors)
    ordered.append((native, "Native Full"))
    width, height = source.size
    grid = Image.new("RGB", (width * len(ordered), height + 28), "white")
    draw = ImageDraw.Draw(grid)
    for index, (image, label) in enumerate(ordered):
        left = index * width
        draw.text((left + 5, 7), label, fill="black")
        grid.paste(image.resize((width, height), Image.Resampling.LANCZOS), (left, 28))
    return grid


def _serializable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    return value


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
            "factors": list(args.factors),
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
    report["source_provenance"] = "SOURCE IMAGE"
    report["native_full_provenance"] = "MODEL GENERATED — NATIVE FULL"
    source_tensor = _pil_to_tensor(source_image, device)
    native_pil = _tensor_to_pil(native_image)
    native_pil.save(output_dir / "native_full.png")
    result_images, results = {}, []
    previous_image = None
    for factor in args.factors:
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
        image_pil.save(output_dir / f"factor_{_factor_tag(factor)}.png")
        result_images[factor] = image_pil
        delta = steered.detach().float() - base_prompt_embeds.detach().float()
        results.append(
            {
                "factor": factor,
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
        "distance_to_full_over_input_order": full,
        "distance_to_source_over_input_order": source,
        "distance_to_full_non_increasing_in_input_order": all(a >= b for a, b in zip(full, full[1:])),
        "distance_to_source_non_decreasing_in_input_order": all(a <= b for a, b in zip(source, source[1:])),
    }
    nonzero = [row for row in results if row["factor"] != 0]
    response = any(row["image_mad_mse_to_native_full"]["mad"] > 0 for row in nonzero)
    report["status"] = "STEERING_RESPONSE_DETECTED" if response else "NO_STEERING_RESPONSE"
    report["visual_status"] = "VISUAL_REVIEW_REQUIRED"
    _make_grid(source_image, native_pil, list(args.factors), result_images).save(output_dir / "grid.png")
    report_path.write_text(json.dumps(_serializable(report), indent=2), encoding="utf-8")
    print(f"status={report['status']} grid={output_dir / 'grid.png'} report={report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
