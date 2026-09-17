"""Run the independent-control RewardSlider V1 diagnostics and smoke experiment.

The runner keeps controls as independent [K,tokens,channels] FP32 parameters.
``D`` is used only for initialization and weak soft priors, never as a control
parameterization or a control-trajectory endpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from diffusers.pipelines.rewardflow.coupled_terminal_control import (
    control_band_loss,
    initialize_independent_coupled_controls,
    scalar_direction_residual_diagnostics,
)
from diffusers.pipelines.rewardflow.dreamsim_adapter import DreamSimAdapter
from diffusers.pipelines.rewardflow.pipeline_flux_kontext_coupled_control import FluxKontextCoupledControlPipeline
from diffusers.pipelines.rewardflow.semantic_feature_scorers import CLIPImageFeatureScorer
from diffusers.pipelines.rewardflow.terminal_control import initialize_velocity_controls
from diffusers.pipelines.rewardflow.trajectory_objectives import CoupledTrajectoryObjective, RewardSliderV1LossWeights


FORMAL_MANIFEST = "experiments/semantic_embedding_geometry_v6/formal_stage/suite_manifest.json"
DEFAULT_CLIP = "/data15/hyp/weight/reward_models/clip-vit-large-patch14"
DEFAULT_DREAMSIM = "/data15/hyp/weight/dreamsim_ckpts"


def _args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=FORMAL_MANIFEST)
    parser.add_argument("--model", default=os.getenv("FLUX_KONTEXT_MODEL_PATH"))
    parser.add_argument("--clip-model", default=DEFAULT_CLIP)
    parser.add_argument("--dreamsim-model", default=DEFAULT_DREAMSIM)
    parser.add_argument("--output-dir", default="experiments/coupled_reward_trajectory_v1_reviewed")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--control-steps", type=int, default=2)
    parser.add_argument("--branches", type=int, default=None)
    parser.add_argument("--num-trajectory-nodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--control-init-noise", type=float, default=0.0)
    parser.add_argument("--outer-iters", type=int, default=8)
    parser.add_argument("--ablation-iters", type=int, default=6)
    parser.add_argument("--control-lr", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--use-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--backward-mode", choices=("one_pass", "two_pass_vjp"), default="one_pass")
    parser.add_argument("--vjp-microbatch-size", type=int, default=1)
    parser.add_argument(
        "--run-vjp-parity",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Measure real one-pass versus exact two-pass control gradients before optimization.",
    )
    parser.add_argument("--run-ablations", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--benchmark-repeats", type=int, default=1)
    parser.add_argument("--lambda-rank", type=float, default=1.0, help="Legacy alias for semantic-order weight.")
    parser.add_argument("--lambda-sem-coverage", type=float, default=1.0)
    parser.add_argument("--lambda-gap", type=float, default=1.0, help="Legacy alias for fine-jump weight.")
    parser.add_argument("--lambda-coarse-gap", type=float, default=1.0)
    parser.add_argument("--lambda-second", type=float, default=0.25)
    parser.add_argument("--lambda-preserve", type=float, default=1.0)
    parser.add_argument("--lambda-ctrl", type=float, default=0.0)
    parser.add_argument("--lambda-band", type=float, default=0.01)
    parser.add_argument("--lambda-spatial", type=float, default=0.05)
    parser.add_argument("--lambda-energy", type=float, default=0.0)
    parser.add_argument("--rank-margin", type=float, default=0.0, help="Fine semantic numerical order margin.")
    parser.add_argument("--sem-min-fraction", type=float, default=0.05)
    parser.add_argument("--sem-max-fraction", type=float, default=0.55)
    parser.add_argument("--fine-max-jump-fraction", type=float, default=0.55)
    parser.add_argument("--coarse-min-fraction", type=float, default=0.05)
    parser.add_argument("--coarse-max-fraction", type=float, default=0.55)
    parser.add_argument("--init-mode", choices=("linear_D", "collapsed_D", "reversed_D"), default="linear_D")
    parser.add_argument("--parity-min-cosine", type=float, default=0.999)
    parser.add_argument("--parity-max-mean-error", type=float, default=0.01)
    parser.add_argument("--try-fp32-first-step", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    if not args.model:
        parser.error("Pass --model or set FLUX_KONTEXT_MODEL_PATH.")
    if args.num_trajectory_nodes is not None:
        if args.num_trajectory_nodes < 3:
            parser.error("`--num-trajectory-nodes` must be at least three.")
        inferred = args.num_trajectory_nodes - 2
        if args.branches is not None and args.branches != inferred:
            parser.error("`--branches` and `--num-trajectory-nodes` disagree.")
        args.branches = inferred
    elif args.branches is None:
        args.branches = 3
    args.num_trajectory_nodes = args.branches + 2
    if args.branches < 1 or args.control_steps < 1 or args.steps < args.control_steps:
        parser.error("Require branches/control-steps >= 1 and steps >= control-steps.")
    if args.outer_iters < 0 or args.ablation_iters < 0 or args.control_lr <= 0:
        parser.error("Iteration counts must be non-negative and --control-lr must be positive.")
    if not 1 <= args.vjp_microbatch_size <= args.branches:
        parser.error("`--vjp-microbatch-size` must lie in [1, branches].")
    if not (
        0 <= args.sem_min_fraction <= args.sem_max_fraction <= 1
        and 0 <= args.coarse_min_fraction <= args.coarse_max_fraction <= 1
    ):
        parser.error("Coverage fractions must satisfy 0 <= min <= max <= 1.")
    return args


def _commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _jsonable(value):
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if torch.is_tensor(value):
        return value.detach().float().cpu().tolist() if value.ndim else float(value.detach().float().cpu())
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _json(path: Path, payload) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")


def _pil_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(image, dtype=np.float32).copy() / 255).permute(2, 0, 1).unsqueeze(0).to(device)


def _to_pil(image: torch.Tensor) -> Image.Image:
    return Image.fromarray(image.detach().clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy(), "RGB")


def _errors(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    difference = (actual.float() - expected.float()).abs()
    return {
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "rms": float(difference.square().mean().sqrt()),
        "cosine": float(
            torch.nn.functional.cosine_similarity(actual.float().flatten()[None], expected.float().flatten()[None])
        ),
    }


def _no_parameter_grad(*modules) -> bool:
    return all(parameter.grad is None for module in modules if module is not None for parameter in module.parameters())


def _make_grid(
    path: Path, source: torch.Tensor, candidates: torch.Tensor, native: torch.Tensor, labels: list[str]
) -> None:
    panels = [_to_pil(source[0]), *[_to_pil(image) for image in candidates], _to_pil(native[0])]
    width, height = panels[0].size
    canvas = Image.new("RGB", (width * len(panels), height + 28), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (image, label) in enumerate(zip(panels, labels)):
        draw.text((index * width + 4, 7), label, fill="black")
        canvas.paste(image, (index * width, 28))
    canvas.save(path)


def _save_relevance(output: Path, inputs) -> None:
    rows = []
    for index, (direction, score, relevance) in enumerate(
        zip(inputs.prior.directions, inputs.prior.raw_scores, inputs.prior.relevance)
    ):
        map_ = relevance[0].reshape(inputs.native.sampling_token_height, inputs.native.sampling_token_width)
        pixels = map_.mul(255).round().byte().cpu().numpy()
        Image.fromarray(pixels, "L").resize(
            (inputs.native.width, inputs.native.height), Image.Resampling.NEAREST
        ).save(output / f"relevance_step_{index}.png")
        probability = relevance / relevance.sum().clamp_min(1e-8)
        rows.append(
            {
                "step": index,
                "direction_l2": direction.float().norm(),
                "raw_min": score.min(),
                "raw_mean": score.mean(),
                "raw_max": score.max(),
                "raw_std": score.std(unbiased=False),
                "relevance_min": relevance.min(),
                "relevance_mean": relevance.mean(),
                "relevance_max": relevance.max(),
                "relevance_std": relevance.std(unbiased=False),
                "relevance_entropy": -(probability * probability.clamp_min(1e-8).log()).sum(),
                "fraction_above_0p25": (relevance > 0.25).float().mean(),
                "fraction_above_0p5": (relevance > 0.5).float().mean(),
                "fraction_above_0p75": (relevance > 0.75).float().mean(),
                "is_binary": bool(torch.all((relevance == 0) | (relevance == 1))),
            }
        )
    _json(
        output / "prior_diagnostics.json",
        {"soft_prior": "score / max(score) per timestep; no threshold or hard mask", "steps": rows},
    )


def _control_diagnostics(controls, prior) -> dict:
    _, band = control_band_loss(controls, prior.directions, prior.relevance)
    scalar = scalar_direction_residual_diagnostics(controls, prior.directions)
    steps = []
    for step, (control, direction, relevance, band_row, scalar_row) in enumerate(
        zip(controls, prior.directions, prior.relevance, band, scalar)
    ):
        direction = direction.to(control)
        coefficient = (control.float() * direction.float()).sum(-1) / direction.float().square().sum(-1).clamp_min(
            1e-8
        )
        orthogonal = control.float() - coefficient[..., None] * direction.float()
        high = relevance.to(control) >= relevance.to(control).median()
        low = ~high
        branches = []
        for branch in range(control.shape[0]):
            item = control[branch].float()
            branches.append(
                {
                    "branch": branch,
                    "grad_norm": None if control.grad is None else control.grad[branch].detach().float().norm(),
                    "control_norm": item.norm(),
                    "cosine_to_D": torch.nn.functional.cosine_similarity(
                        item.flatten()[None], direction[0].float().flatten()[None]
                    ).squeeze(),
                    "coefficient_mean": coefficient[branch].mean(),
                    "coefficient_std": coefficient[branch].std(unbiased=False),
                    "orthogonal_ratio": orthogonal[branch].norm() / item.norm().clamp_min(1e-8),
                    "control_to_D_norm": item.norm() / direction.float().norm().clamp_min(1e-8),
                    "scalar_beta_star": scalar_row["beta_star"][branch],
                    "scalar_D_residual_ratio": scalar_row["residual_ratio"][branch],
                    "low_relevance_energy": item[low[0]].square().mean(),
                    "high_relevance_energy": item[high[0]].square().mean(),
                }
            )
        steps.append({"step": step, "aggregate": band_row, "branches": branches})
    return {"steps": steps}


def _objective_summary(result, controls, prior) -> dict:
    return {
        "losses": result.components,
        "total": result.total,
        "clip_q": result.progress,
        "semantic_gaps": result.adjacent_semantic_gaps,
        "coarse_anchor_indices": result.coarse_anchor_indices,
        "coarse_semantic_gaps": result.coarse_semantic_gaps,
        "strict_semantic_order": bool(torch.all(result.adjacent_semantic_gaps > 0)),
        "nondecreasing_semantic_order": bool(torch.all(result.adjacent_semantic_gaps >= 0)),
        "dreamsim_gaps": result.dreamsim_gaps,
        "dreamsim_gap_fractions": result.dreamsim_gap_fractions,
        "dreamsim_worst_jump": result.dreamsim_worst_jump,
        "dreamsim_worst_jump_index": result.dreamsim_worst_jump_index,
        "coarse_dreamsim_gaps": result.coarse_dreamsim_gaps,
        "coarse_dreamsim_fractions": result.coarse_dreamsim_fractions,
        "triangle_raw_deficits": result.triangle_raw_deficits,
        "triangle_normalized_deficits": result.triangle_normalized_deficits,
        "triangle_mean": result.triangle_normalized_deficits.mean(),
        "triangle_max": result.triangle_normalized_deficits.max(),
        "controls": _control_diagnostics(controls, prior),
    }


def _evaluate(pipe, inputs, controls, objective, *, checkpointing: bool):
    unroll = pipe.unroll_coupled_controls(inputs, controls, use_checkpointing=checkpointing)
    candidates = pipe.decode_coupled_terminal_latent(unroll.final_latent, inputs)
    result = objective(candidates, controls)
    return candidates, result


def _initial_betas(args) -> tuple[float, ...]:
    linear = tuple(1 - (index + 1) / (args.branches + 1) for index in range(args.branches))
    if args.init_mode == "linear_D":
        return linear
    if args.init_mode == "collapsed_D":
        return (0.5,) * args.branches
    return tuple(reversed(linear))


def _make_independent_controls(inputs, args):
    return initialize_independent_coupled_controls(
        inputs.prior.directions,
        num_branches=args.branches,
        betas=_initial_betas(args),
        noise_std=args.control_init_noise,
    )


def _two_pass_backward(pipe, inputs, controls, objective, args):
    """Exact frozen-Kontext image VJP, replayed branchwise after reward Pass A."""
    with torch.no_grad():
        detached = pipe.decode_coupled_terminal_latent(
            pipe.unroll_coupled_controls(inputs, controls, use_checkpointing=False).final_latent, inputs
        ).detach()
    reward_images = detached.requires_grad_(True)
    reward_result = objective(reward_images, controls)
    image_total = objective.image_total(reward_result.components)
    image_grad = torch.autograd.grad(image_total, reward_images)[0].detach()
    del reward_images, image_total
    torch.cuda.empty_cache()
    for start in range(0, args.branches, args.vjp_microbatch_size):
        end = min(start + args.vjp_microbatch_size, args.branches)
        unroll = pipe.unroll_coupled_control_microbatch(
            inputs, controls, slice(start, end), use_checkpointing=args.use_checkpointing
        )
        replay = pipe.decode_coupled_terminal_latent(unroll.final_latent, inputs)
        torch.autograd.backward(replay, grad_tensors=image_grad[start:end])
        del unroll, replay
    control_total, _, _ = objective.control_total(controls)
    control_total.backward()
    return detached, reward_result


def _gradient_difference(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float | bool]:
    difference = reference.float() - candidate.float()
    reference_norm = reference.float().norm()
    candidate_norm = candidate.float().norm()
    return {
        "reference_norm": float(reference_norm),
        "candidate_norm": float(candidate_norm),
        "difference_l2": float(difference.norm()),
        "relative_l2": float(difference.norm() / reference_norm.clamp_min(1e-12)),
        "max_abs": float(difference.abs().max()),
        "cosine": float(
            torch.nn.functional.cosine_similarity(reference.float().flatten()[None], candidate.float().flatten()[None])
        ),
        "all_finite": bool(torch.isfinite(reference).all() and torch.isfinite(candidate).all()),
    }


def _real_two_pass_vjp_parity(pipe, inputs, objective, args, output: Path) -> dict:
    """Compare full real-backbone control gradients without changing optimization.

    This is deliberately an observation, not a gate: activation checkpointing
    and BF16 may introduce small replay differences, which must be reported
    rather than hidden by a made-up tolerance.
    """
    one_pass_controls = _make_independent_controls(inputs, args)
    two_pass_controls = [torch.nn.Parameter(control.detach().clone()) for control in one_pass_controls]
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(args.device)
    torch.cuda.synchronize(args.device)
    started = time.perf_counter()
    one_images, one_result = _evaluate(
        pipe, inputs, one_pass_controls, objective, checkpointing=args.use_checkpointing
    )
    one_result.total.backward()
    torch.cuda.synchronize(args.device)
    one_pass_seconds = time.perf_counter() - started
    one_pass_peak = torch.cuda.max_memory_allocated(args.device)
    one_pass_gradients = [control.grad.detach().clone() for control in one_pass_controls]
    one_total = one_result.total.detach()
    del one_images, one_result

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(args.device)
    torch.cuda.synchronize(args.device)
    started = time.perf_counter()
    two_images, two_result = _two_pass_backward(pipe, inputs, two_pass_controls, objective, args)
    torch.cuda.synchronize(args.device)
    two_pass_seconds = time.perf_counter() - started
    two_pass_peak = torch.cuda.max_memory_allocated(args.device)
    two_pass_gradients = [control.grad.detach().clone() for control in two_pass_controls]
    two_total = two_result.total.detach()
    payload = {
        "description": "Exact real FLUX-Kontext replay comparison; no tolerance is used to mask differences.",
        "checkpointing": bool(args.use_checkpointing),
        "vjp_microbatch_size": args.vjp_microbatch_size,
        "one_pass": {"seconds": one_pass_seconds, "peak_allocated_vram": one_pass_peak, "total": one_total},
        "two_pass_vjp": {"seconds": two_pass_seconds, "peak_allocated_vram": two_pass_peak, "total": two_total},
        "total_difference": float((one_total.float() - two_total.float()).abs()),
        "per_control_step": [
            _gradient_difference(reference, candidate)
            for reference, candidate in zip(one_pass_gradients, two_pass_gradients)
        ],
    }
    _json(output / "real_two_pass_vjp_parity.json", payload)
    del two_images, two_result
    torch.cuda.empty_cache()
    return payload


def _parity_localization(pipe, inputs, args, output: Path) -> dict:
    native = inputs.native
    with torch.no_grad():
        velocity_one = pipe._predict_kontext_velocity(
            native.initial_latent, native.timesteps[0], **native.forward_kwargs
        )
        velocity_three = pipe._predict_kontext_velocity(
            inputs.initial_latent, native.timesteps[0], **inputs.forward_kwargs
        )
        controls = initialize_independent_coupled_controls(
            inputs.prior.directions, num_branches=args.branches, betas=(0.0,) * args.branches
        )
        batched = pipe.unroll_coupled_controls(inputs, controls, use_checkpointing=False)
        sequential = pipe.unroll_terminal_controls(
            native, initialize_velocity_controls(native.initial_latent, args.control_steps), use_checkpointing=False
        )
    first = {
        "B3_branch_vs_branch": _errors(velocity_three, velocity_three[:1].expand_as(velocity_three)),
        "B3_branch0_vs_B1": _errors(velocity_three[:1], velocity_one),
    }
    rows = []
    for step, (batched_state, one_state) in enumerate(zip(batched.states, sequential.states)):
        row = {"step": step, **_errors(batched_state[:1], one_state)}
        rows.append(row)
    with (output / "parity_by_step.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        f = native.forward_kwargs
        legacy_values = pipe._materialize_kontext_strength_batch(
            args.branches,
            native.initial_latent,
            f["image_latents"],
            f["prompt_embeds"],
            f["pooled_prompt_embeds"],
            f["guidance"],
            f["negative_prompt_embeds"],
            f["negative_pooled_prompt_embeds"],
            f["image_embeds"],
            f["negative_image_embeds"],
        )
        (
            legacy_initial,
            legacy_image_latents,
            legacy_prompt,
            legacy_pooled,
            legacy_guidance,
            legacy_negative_prompt,
            legacy_negative_pooled,
            legacy_image_embeds,
            legacy_negative_image_embeds,
        ) = legacy_values
        legacy_kwargs = dict(f)
        legacy_kwargs.update(
            {
                "image_latents": legacy_image_latents,
                "prompt_embeds": legacy_prompt,
                "pooled_prompt_embeds": legacy_pooled,
                "guidance": legacy_guidance,
                "negative_prompt_embeds": legacy_negative_prompt,
                "negative_pooled_prompt_embeds": legacy_negative_pooled,
                "image_embeds": legacy_image_embeds,
                "negative_image_embeds": legacy_negative_image_embeds,
            }
        )
        legacy_velocity = pipe._predict_kontext_velocity(legacy_initial, native.timesteps[0], **legacy_kwargs)
    legacy = {
        "materialization": "FluxKontextStrengthTrajectoryPipeline._materialize_kontext_strength_batch",
        "initial_latent_vs_coupled": _errors(legacy_initial, inputs.initial_latent),
        "velocity_vs_coupled": _errors(legacy_velocity, velocity_three),
        "note": "This explicitly re-executes the legacy materialization helper; image_ids and text_ids remain shared sequence tensors.",
    }
    fp32 = {"attempted": bool(args.try_fp32_first_step), "available": False}
    if args.try_fp32_first_step:
        try:
            original_dtype = next(pipe.transformer.parameters()).dtype
            pipe.transformer.float()
            kwargs_one = {
                key: value.float() if torch.is_tensor(value) and value.is_floating_point() else value
                for key, value in native.forward_kwargs.items()
            }
            kwargs_three = {
                key: value.float() if torch.is_tensor(value) and value.is_floating_point() else value
                for key, value in inputs.forward_kwargs.items()
            }
            one = pipe._predict_kontext_velocity(native.initial_latent.float(), native.timesteps[0], **kwargs_one)
            three = pipe._predict_kontext_velocity(inputs.initial_latent.float(), native.timesteps[0], **kwargs_three)
            fp32 = {"attempted": True, "available": True, "B3_branch0_vs_B1": _errors(three[:1], one)}
            pipe.transformer.to(dtype=original_dtype)
        except RuntimeError as error:
            fp32["reason"] = str(error)
    parity = {
        "first_transformer_forward": first,
        "legacy_k_batched_path": legacy,
        "fp32_first_step": fp32,
        "zero_control_branch_max_difference": (batched.final_latent - batched.final_latent[:1]).abs().max(),
        "final_B3_branch0_vs_B1": _errors(batched.final_latent[:1], sequential.final_latent),
        "gate": {
            "branch_identity": bool((batched.final_latent - batched.final_latent[:1]).abs().max() == 0),
            "minimum_cosine": args.parity_min_cosine,
            "maximum_mean_error": args.parity_max_mean_error,
            "B3_vs_B1_cosine_pass": rows[-1]["cosine"] >= args.parity_min_cosine,
            "B3_vs_B1_mean_error_pass": rows[-1]["mean_abs"] <= args.parity_max_mean_error,
        },
    }
    parity["gate"]["passed"] = all(
        value for key, value in parity["gate"].items() if key.endswith("pass") or key == "branch_identity"
    )
    _json(output / "parity_localization.json", parity)
    return parity


def _compute_benchmark(pipe, inputs, args, output: Path) -> dict:
    """Compare one K=3 batched unroll against K separate B=1 unrolls.

    This is a timing and peak-memory observation only; it makes no claim about
    FLOP reduction and uses zero controls to preserve the native trajectory.
    """

    controls = initialize_independent_coupled_controls(
        inputs.prior.directions,
        num_branches=args.branches,
        betas=(0.0,) * args.branches,
    )

    def measure(call):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(args.device)
        torch.cuda.synchronize(args.device)
        started = time.perf_counter()
        with torch.no_grad():
            value = call()
        torch.cuda.synchronize(args.device)
        return {
            "seconds": time.perf_counter() - started,
            "peak_allocated_vram": torch.cuda.max_memory_allocated(args.device),
            "peak_reserved_vram": torch.cuda.max_memory_reserved(args.device),
            "final_latent_norm": value.final_latent.float().norm(),
        }

    batched = [
        measure(lambda: pipe.unroll_coupled_controls(inputs, controls, use_checkpointing=False))
        for _ in range(args.benchmark_repeats)
    ]

    def separate():
        output_states = []
        for branch in range(args.branches):
            branch_controls = [control[branch : branch + 1] for control in controls]
            output_states.append(
                pipe.unroll_terminal_controls(inputs.native, branch_controls, use_checkpointing=False)
            )
        return output_states[-1]

    separate_runs = [measure(separate) for _ in range(args.benchmark_repeats)]
    payload = {
        "description": "K=3 batched versus K separate B=1 zero-control unrolls; timing only, not a FLOP claim.",
        "repeats": args.benchmark_repeats,
        "batched": batched,
        "separate_b1": separate_runs,
    }
    _json(output / "compute_benchmark.json", payload)
    return payload


def _run_optimization(
    pipe, inputs, objective, args, output: Path, name: str, weights: RewardSliderV1LossWeights, iterations: int
):
    run_dir = output / name
    run_dir.mkdir(parents=True, exist_ok=True)
    objective.weights = weights
    controls = _make_independent_controls(inputs, args)
    optimizer = torch.optim.Adam(controls, lr=args.control_lr)
    if {id(parameter) for parameter in optimizer.param_groups[0]["params"]} != {id(control) for control in controls}:
        raise RuntimeError("Only independent velocity controls may enter the optimizer.")
    with torch.no_grad():
        initial_images, initial_result = _evaluate(pipe, inputs, controls, objective, checkpointing=False)
    initial_summary = _objective_summary(initial_result, controls, inputs.prior)
    _make_grid(
        run_dir / "trajectory_initial.png",
        objective.source_image,
        initial_images,
        objective.native_full_image,
        ["Source", *[f"Interior-{index + 1}" for index in range(args.branches)], "NativeFull"],
    )
    for label, image in enumerate(initial_images):
        _to_pil(image).save(run_dir / f"initial_{label + 1}.png")
    trace = []
    for iteration in range(1, iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(args.device)
        started = time.perf_counter()
        if args.backward_mode == "one_pass":
            images, result = _evaluate(pipe, inputs, controls, objective, checkpointing=args.use_checkpointing)
            result.total.backward()
        else:
            images, result = _two_pass_backward(pipe, inputs, controls, objective, args)
        gradients = [control.grad for control in controls]
        present_gradients = [gradient for gradient in gradients if gradient is not None]
        if any(not torch.isfinite(gradient).all() for gradient in present_gradients):
            raise RuntimeError("An independent control received a non-finite gradient.")
        if not present_gradients or not any(gradient.abs().sum() > 0 for gradient in present_gradients):
            raise RuntimeError("All independent control gradients are zero.")
        frozen_ok = _no_parameter_grad(
            pipe.transformer,
            pipe.vae,
            pipe.text_encoder,
            pipe.text_encoder_2,
            objective.feature_encoder.model,
            objective.dreamsim.model,
        )
        if not frozen_ok:
            raise RuntimeError("A frozen inference/reward parameter unexpectedly received a gradient.")
        before_step = _objective_summary(result, controls, inputs.prior)
        if args.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(controls, args.grad_clip)
        optimizer.step()
        trace.append(
            {
                "iteration": iteration,
                "elapsed_seconds": time.perf_counter() - started,
                "peak_allocated_vram": torch.cuda.max_memory_allocated(args.device),
                "peak_reserved_vram": torch.cuda.max_memory_reserved(args.device),
                "frozen_gradient_audit": frozen_ok,
                "backward_mode": args.backward_mode,
                "vjp_microbatch_size": args.vjp_microbatch_size,
                "missing_control_gradient_steps": [
                    index for index, gradient in enumerate(gradients) if gradient is None
                ],
                **before_step,
            }
        )
        del images, result
    with torch.no_grad():
        final_images, final_result = _evaluate(pipe, inputs, controls, objective, checkpointing=False)
    final_summary = _objective_summary(final_result, controls, inputs.prior)
    for label, image in enumerate(final_images):
        _to_pil(image).save(run_dir / f"final_{label + 1}.png")
    _make_grid(
        run_dir / "trajectory_final.png",
        objective.source_image,
        final_images,
        objective.native_full_image,
        ["Source", *[f"Interior-{index + 1}" for index in range(args.branches)], "NativeFull"],
    )
    _write_jsonl(run_dir / "optimization_trace.jsonl", trace)
    _json(
        run_dir / "summary.json",
        {
            "name": name,
            "iterations": iterations,
            "weights": weights,
            "initial": initial_summary,
            "final": final_summary,
            "loss_decreased": float(final_result.total) < float(initial_result.total),
            "nan_detected": False,
        },
    )
    return {"initial": initial_summary, "final": final_summary, "iterations": iterations}


def main():
    args = _args()
    if not torch.cuda.is_available():
        raise RuntimeError("The real Kontext diagnostics require CUDA.")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    case = manifest["cases"][0]
    seed = case["seed"] if args.seed is None else args.seed
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _json(
        output / "config.json",
        {
            **vars(args),
            "case_id": case["case_id"],
            "source": case["source"],
            "prompt": case["instruction"],
            "seed": seed,
            "git_commit": _commit(),
        },
    )
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    source_pil = Image.open(case["source"]).convert("RGB").resize((args.width, args.height), Image.Resampling.LANCZOS)
    source_pil.save(output / "source.png")
    pipe = FluxKontextCoupledControlPipeline.from_pretrained(args.model, torch_dtype=dtype, local_files_only=True).to(
        device
    )
    pipe.set_progress_bar_config(disable=True)
    inputs = pipe.prepare_coupled_control_inputs(
        num_branches=args.branches,
        control_steps=args.control_steps,
        image=source_pil,
        prompt=case["instruction"],
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=torch.Generator(device=device).manual_seed(seed),
    )
    with torch.no_grad():
        native_image = pipe.decode_terminal_latent(inputs.native.native_final_latent, inputs.native)
    _to_pil(native_image[0]).save(output / "native_full.png")
    _save_relevance(output, inputs)
    parity = _parity_localization(pipe, inputs, args, output)
    benchmark = _compute_benchmark(pipe, inputs, args, output)
    dreamsim = DreamSimAdapter(model_path=args.dreamsim_model, device=device)
    _json(output / "dreamsim_fidelity.json", dreamsim.fidelity_audit(_pil_tensor(source_pil, device), native_image))
    scorer = CLIPImageFeatureScorer(args.clip_model, device=device, local_files_only=True)
    weights = RewardSliderV1LossWeights(
        semantic_order=args.lambda_rank,
        semantic_coverage=args.lambda_sem_coverage,
        fine_jump=args.lambda_gap,
        coarse_gap=args.lambda_coarse_gap,
        second=args.lambda_second,
        preserve=args.lambda_preserve,
        ctrl=args.lambda_ctrl,
        band=args.lambda_band,
        spatial=args.lambda_spatial,
        energy=args.lambda_energy,
    )
    objective = CoupledTrajectoryObjective(
        feature_encoder=scorer,
        dreamsim=dreamsim,
        source_image=_pil_tensor(source_pil, device),
        native_full_image=native_image,
        prior=inputs.prior,
        token_height=inputs.native.sampling_token_height,
        token_width=inputs.native.sampling_token_width,
        weights=weights,
        order_margin=args.rank_margin,
        sem_min_fraction=args.sem_min_fraction,
        sem_max_fraction=args.sem_max_fraction,
        fine_max_jump_fraction=args.fine_max_jump_fraction,
        coarse_min_fraction=args.coarse_min_fraction,
        coarse_max_fraction=args.coarse_max_fraction,
    )
    vjp_parity = None
    if args.run_vjp_parity:
        vjp_parity = _real_two_pass_vjp_parity(pipe, inputs, objective, args, output)
    initialization = _run_optimization(pipe, inputs, objective, args, output, "initialization", weights, 0)
    _json(output / "initialization_diagnostics.json", initialization)
    results = {"initialization": initialization, "parity": parity, "compute_benchmark": benchmark}
    if vjp_parity is not None:
        results["real_two_pass_vjp_parity"] = vjp_parity
    if parity["gate"]["passed"]:
        results["full_v1"] = _run_optimization(
            pipe, inputs, objective, args, output, "full_v1", weights, args.outer_iters
        )
        if args.run_ablations:
            results["no_band"] = _run_optimization(
                pipe,
                inputs,
                objective,
                args,
                output,
                "no_band",
                RewardSliderV1LossWeights(**{**weights.__dict__, "band": 0.0}),
                args.ablation_iters,
            )
            results["prior_only"] = _run_optimization(
                pipe,
                inputs,
                objective,
                args,
                output,
                "prior_only",
                RewardSliderV1LossWeights(
                    semantic_order=0.0,
                    semantic_coverage=0.0,
                    fine_jump=0.0,
                    coarse_gap=0.0,
                    second=0.0,
                    preserve=0.0,
                    ctrl=weights.ctrl,
                    band=weights.band,
                    spatial=weights.spatial,
                    energy=weights.energy,
                ),
                args.ablation_iters,
            )
            results["no_spatial"] = _run_optimization(
                pipe,
                inputs,
                objective,
                args,
                output,
                "no_spatial",
                RewardSliderV1LossWeights(**{**weights.__dict__, "spatial": 0.0}),
                args.ablation_iters,
            )
    else:
        results["optimization_blocked"] = (
            "Numerical parity gate did not pass; diagnostics were retained and no optimization was run."
        )
    _json(output / "summary.json", results)
    print(
        "FINAL_REPORT="
        + json.dumps(
            _jsonable({"output_dir": str(output), "parity_gate": parity["gate"], "runs": list(results)}),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
