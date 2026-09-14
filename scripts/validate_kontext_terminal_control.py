"""Validate long-horizon early velocity control on a real FLUX.1-Kontext model.

This is an opt-in mechanism experiment. Its endpoint blue/pixel objectives are
diagnostic controllability probes, not semantic edit-strength rewards.
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
    BlueEndpointTargetLoss,
    EndpointPixelTargetLoss,
    blue_direction_score,
    endpoint_soft_mask,
    initialize_velocity_controls,
    normalized_control_energy,
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
    parser.add_argument("--control-steps", type=int, default=2)
    parser.add_argument("--outer-iters", type=int, default=6)
    parser.add_argument("--control-lr", type=float, default=0.1)
    parser.add_argument("--lambda-control", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--terminal-objective", choices=("blue", "pixel"), default="blue")
    parser.add_argument("--endpoint-soft-mask", action="store_true")
    parser.add_argument(
        "--use-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    if not args.model:
        parser.error("Set FLUX_KONTEXT_MODEL_PATH or pass --model.")
    if args.steps < 1:
        parser.error("--steps must be positive.")
    if not 1 <= args.control_steps <= args.steps:
        parser.error("--control-steps must lie in [1, steps].")
    if args.outer_iters < 0:
        parser.error("--outer-iters must be non-negative.")
    if args.control_lr <= 0 or not math.isfinite(args.control_lr):
        parser.error("--control-lr must be finite and positive.")
    if args.lambda_control < 0 or not math.isfinite(args.lambda_control):
        parser.error("--lambda-control must be finite and non-negative.")
    if args.grad_clip is not None and (args.grad_clip <= 0 or not math.isfinite(args.grad_clip)):
        parser.error("--grad-clip must be finite and positive when provided.")
    if any(not 0 <= strength <= 1 for strength in args.strengths):
        parser.error("Every strength must lie in [0, 1].")
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


def _control_diagnostics(controls, native_velocities) -> dict[str, list[float | None]]:
    control_norms = []
    native_norms = []
    ratios = []
    cosines = []
    for control, native in zip(controls, native_velocities):
        control_flat = control.detach().float().flatten()
        native_flat = native.detach().float().flatten()
        control_norm = torch.linalg.vector_norm(control_flat).item()
        native_norm = torch.linalg.vector_norm(native_flat).item()
        control_norms.append(control_norm)
        native_norms.append(native_norm)
        ratios.append(control_norm / native_norm if native_norm > 0 else None)
        cosines.append(
            F.cosine_similarity(control_flat[None], native_flat[None]).item()
            if control_norm > 0 and native_norm > 0
            else None
        )
    return {
        "control_norms": control_norms,
        "native_velocity_norms": native_norms,
        "control_native_ratios": ratios,
        "control_native_cosines": cosines,
    }


def _make_objective(name, source_image, full_image, weight):
    if name == "blue":
        return BlueEndpointTargetLoss(source_image, full_image, weight=weight)
    return EndpointPixelTargetLoss(source_image, full_image, weight=weight)


def _optimize_strength(pipe, inputs, objective, args, strength: float):
    controls = initialize_velocity_controls(inputs.initial_latent, args.control_steps)
    optimizer = torch.optim.Adam(controls, lr=args.control_lr)
    trace = []
    initial_image = None

    with torch.no_grad():
        initial_unroll = pipe.unroll_terminal_controls(inputs, controls, use_checkpointing=False)
        initial_image = pipe.decode_terminal_latent(initial_unroll.final_latent, inputs).detach()
        initial_objective = objective(initial_image, strength)
        initial_target_error = (initial_objective.achieved_score - initial_objective.target_score).abs().mean().item()

    for outer_iter in range(args.outer_iters):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(args.device)
        torch.cuda.synchronize(args.device)
        started = time.perf_counter()

        unroll = pipe.unroll_terminal_controls(
            inputs,
            controls,
            use_checkpointing=args.use_checkpointing,
        )
        final_image = pipe.decode_terminal_latent(unroll.final_latent, inputs)
        objective_output = objective(final_image, strength)
        control_regularization = normalized_control_energy(controls)
        total_loss = objective_output.loss + args.lambda_control * control_regularization
        total_loss.backward()

        gradients = [control.grad for control in controls]
        if any(gradient is None for gradient in gradients):
            raise RuntimeError("Terminal loss did not reach every configured early control.")
        if any(not torch.isfinite(gradient).all() for gradient in gradients):
            raise RuntimeError("A terminal-control gradient contains non-finite values.")
        if not any(gradient.abs().sum() > 0 for gradient in gradients):
            raise RuntimeError("Terminal loss produced only zero early-control gradients.")
        if _model_has_gradient(pipe):
            raise RuntimeError("Frozen Kontext/VAE/text parameters unexpectedly received gradients.")

        grad_norms = [torch.linalg.vector_norm(gradient.detach().float()).item() for gradient in gradients]
        diagnostics = _control_diagnostics(controls, unroll.native_control_velocities)
        if args.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(controls, args.grad_clip)
        optimizer.step()
        torch.cuda.synchronize(args.device)
        elapsed = time.perf_counter() - started

        achieved = objective_output.achieved_score.detach().mean().item()
        target = objective_output.target_score.detach().mean().item()
        trace.append(
            {
                "strength": strength,
                "outer_iter": outer_iter,
                "terminal_loss": total_loss.detach().item(),
                "terminal_strength_loss": objective_output.loss.detach().item(),
                "control_regularization": control_regularization.detach().item(),
                "weighted_control_regularization": (args.lambda_control * control_regularization).detach().item(),
                "source_score": objective_output.source_score.detach().mean().item(),
                "full_score": objective_output.full_score.detach().mean().item(),
                "target_score": target,
                "achieved_score": achieved,
                "absolute_target_error": abs(achieved - target),
                "control_gradient_norms": grad_norms,
                **diagnostics,
                "final_latent_norm": torch.linalg.vector_norm(unroll.final_latent.detach().float()).item(),
                "elapsed_seconds": elapsed,
                "peak_cuda_allocated_mb": torch.cuda.max_memory_allocated(args.device) / 2**20,
                "peak_cuda_reserved_mb": torch.cuda.max_memory_reserved(args.device) / 2**20,
            }
        )
        del final_image, total_loss, objective_output, unroll

    with torch.no_grad():
        final_unroll = pipe.unroll_terminal_controls(inputs, controls, use_checkpointing=False)
        final_image = pipe.decode_terminal_latent(final_unroll.final_latent, inputs).detach()
        final_objective = objective(final_image, strength)
        final_target_error = (final_objective.achieved_score - final_objective.target_score).abs().mean().item()
        final_diagnostics = _control_diagnostics(controls, final_unroll.native_control_velocities)

    return {
        "initial_image": initial_image.cpu(),
        "final_image": final_image.cpu(),
        "initial_target_error": initial_target_error,
        "final_target_error": final_target_error,
        "final_strength_loss": final_objective.loss.item(),
        "source_score": final_objective.source_score.mean().item(),
        "full_score": final_objective.full_score.mean().item(),
        "target_score": final_objective.target_score.mean().item(),
        "achieved_score": final_objective.achieved_score.mean().item(),
        "final_latent_norm": torch.linalg.vector_norm(final_unroll.final_latent.float()).item(),
        "final_diagnostics": final_diagnostics,
        "trace": trace,
    }


def _write_trace_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    normalized = []
    for row in rows:
        normalized.append({key: json.dumps(value) if isinstance(value, list) else value for key, value in row.items()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(normalized[0]))
        writer.writeheader()
        writer.writerows(normalized)


def _make_grid(source: Image.Image, native: Image.Image, results: dict[float, dict]) -> Image.Image:
    images = [source, native, *[_tensor_to_pil(result["final_image"][0]) for result in results.values()]]
    labels = ["Source", "Native Full", *(f"s={strength:g}" for strength in results)]
    width, height = source.size
    header = 30
    grid = Image.new("RGB", (width * len(images), height + header), "white")
    draw = ImageDraw.Draw(grid)
    for index, (image, label) in enumerate(zip(images, labels)):
        left = index * width
        draw.text((left + 6, 8), label, fill="black")
        grid.paste(image.resize((width, height), Image.Resampling.LANCZOS), (left, header))
    return grid


def main():
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Real Kontext terminal-control validation requires CUDA.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source = Image.open(args.source).convert("RGB").resize((args.width, args.height), Image.Resampling.LANCZOS)
    source.save(output_dir / "source.png")

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    torch.cuda.set_device(device)
    print(f"Loading {args.model} on {device} as {dtype}", flush=True)
    pipe = FluxKontextTerminalControlPipeline.from_pretrained(
        args.model,
        torch_dtype=dtype,
        local_files_only=True,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    common = {
        "image": source,
        "prompt": args.prompt,
        "height": args.height,
        "width": args.width,
        "max_area": args.height * args.width,
        "_auto_resize": False,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "output_type": "latent",
    }
    inputs = pipe.prepare_terminal_control_inputs(
        image=source,
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=torch.Generator(device=device).manual_seed(args.seed),
    )

    # Hard parity gate: official scheduler, captured K=1 native path, and the
    # independent zero-control unroll all start from the same seeded latent.
    official = FluxKontextPipeline.__call__(
        pipe,
        **common,
        generator=torch.Generator(device=device).manual_seed(args.seed),
    ).images
    zero_controls = initialize_velocity_controls(inputs.initial_latent, args.control_steps)
    with torch.no_grad():
        zero_unroll = pipe.unroll_terminal_controls(inputs, zero_controls, use_checkpointing=False)
    official_vs_native = _latent_parity(inputs.native_final_latent, official)
    zero_vs_native = _latent_parity(zero_unroll.final_latent, inputs.native_final_latent)
    print(
        json.dumps({"official_vs_native": official_vs_native, "zero_vs_native": zero_vs_native}, indent=2), flush=True
    )
    if official_vs_native["max_absolute_error"] != 0 or zero_vs_native["max_absolute_error"] != 0:
        raise RuntimeError("Zero-control parity failed; refusing to start terminal-control optimization.")

    with torch.no_grad():
        native_full_image = pipe.decode_terminal_latent(inputs.native_final_latent, inputs).detach()
    native_full_pil = _tensor_to_pil(native_full_image[0])
    native_full_pil.save(output_dir / "native_full.png")
    source_tensor = _pil_to_tensor(source, device)
    weight = endpoint_soft_mask(source_tensor, native_full_image) if args.endpoint_soft_mask else None
    objective = _make_objective(args.terminal_objective, source_tensor, native_full_image, weight)

    torch.save(
        {
            "initial_latent": inputs.initial_latent.detach().cpu(),
            "native_final_latent": inputs.native_final_latent.detach().cpu(),
            "sigmas": inputs.sigmas.detach().cpu(),
            "timesteps": torch.stack([t.detach().cpu() for t in inputs.timesteps]),
        },
        output_dir / "fixed_trajectory.pt",
    )

    results = {}
    trace_rows = []
    for strength in args.strengths:
        print(f"Optimizing terminal controls for strength={strength:g}", flush=True)
        result = _optimize_strength(pipe, inputs, objective, args, float(strength))
        results[float(strength)] = result
        trace_rows.extend(result["trace"])
        tag = _float_tag(strength)
        _tensor_to_pil(result["initial_image"][0]).save(output_dir / f"strength_{tag}_iter_initial.png")
        _tensor_to_pil(result["final_image"][0]).save(output_dir / f"strength_{tag}_final.png")
        print(
            json.dumps(
                {
                    "strength": strength,
                    **{k: v for k, v in result.items() if k not in {"trace", "initial_image", "final_image"}},
                },
                indent=2,
            ),
            flush=True,
        )

    _make_grid(source, native_full_pil, results).save(output_dir / "terminal_control_grid.png")
    _write_trace_csv(output_dir / "terminal_control_trace.csv", trace_rows)
    with (output_dir / "terminal_control_trace.json").open("w", encoding="utf-8") as handle:
        json.dump(trace_rows, handle, indent=2)

    source_score = blue_direction_score(source_tensor, weight).item()
    full_score = blue_direction_score(native_full_image, weight).item()
    achieved = [results[float(strength)]["achieved_score"] for strength in args.strengths]
    direction = 1 if full_score >= source_score else -1
    ordered = all(direction * left < direction * right for left, right in zip(achieved, achieved[1:]))
    report = {
        "scope": "terminal early-velocity control with diagnostic endpoint targets",
        "model_path": os.path.realpath(args.model),
        "source_path": os.path.realpath(args.source),
        "prompt": args.prompt,
        "gpu_name": torch.cuda.get_device_name(device),
        "device": str(device),
        "dtype": str(dtype),
        "steps": args.steps,
        "seed": args.seed,
        "guidance_scale": args.guidance_scale,
        "control_steps": args.control_steps,
        "outer_iters": args.outer_iters,
        "control_lr": args.control_lr,
        "lambda_control": args.lambda_control,
        "use_checkpointing": args.use_checkpointing,
        "terminal_objective": args.terminal_objective,
        "endpoint_soft_mask": args.endpoint_soft_mask,
        "official_vs_native_parity": official_vs_native,
        "zero_control_vs_native_parity": zero_vs_native,
        "model_parameters_have_gradient": _model_has_gradient(pipe),
        "source_score": source_score,
        "full_score": full_score,
        "achieved_scores_in_strength_order": achieved,
        "endpoint_direction_ordered": ordered,
        "results": {
            str(strength): {
                key: value for key, value in result.items() if key not in {"trace", "initial_image", "final_image"}
            }
            for strength, result in results.items()
        },
    }
    with (output_dir / "terminal_control_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print("FINAL_REPORT=" + json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
