"""Run auditable RewardSlider V1 prior, parity, and compute diagnostics.

This runner deliberately stops before image-trajectory optimization when the
official differentiable DreamSim path is unavailable.  It still emits A/B/C/F
artifacts so a reviewer can verify independent controls and Kontext batching.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from diffusers.pipelines.rewardflow.coupled_terminal_control import (
    adjacent_ranking_loss,
    control_band_loss,
    control_smoothness_loss,
    gap_bound_loss,
    initialize_independent_coupled_controls,
    spatial_prior_loss,
    triangle_deficit_loss,
)
from diffusers.pipelines.rewardflow.dreamsim_adapter import DreamSimAdapter
from diffusers.pipelines.rewardflow.pipeline_flux_kontext_coupled_control import FluxKontextCoupledControlPipeline
from diffusers.pipelines.rewardflow.terminal_control import initialize_velocity_controls


FORMAL_MANIFEST = "experiments/semantic_embedding_geometry_v6/formal_stage/suite_manifest.json"


def _args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=FORMAL_MANIFEST)
    parser.add_argument("--model", default=os.getenv("FLUX_KONTEXT_MODEL_PATH"))
    parser.add_argument("--output-dir", default="experiments/coupled_reward_trajectory_v1")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--control-steps", type=int, default=2)
    parser.add_argument("--branches", type=int, default=3)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--control-init-noise", type=float, default=0.0)
    parser.add_argument("--use-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--benchmark-repeats", type=int, default=2)
    # Defaults are smoke-test weights only, not claimed as a final method setting.
    parser.add_argument("--lambda-rank", type=float, default=1.0)
    parser.add_argument("--lambda-gap", type=float, default=1.0)
    parser.add_argument("--lambda-second", type=float, default=0.25)
    parser.add_argument("--lambda-preserve", type=float, default=1.0)
    parser.add_argument("--lambda-ctrl", type=float, default=0.05)
    parser.add_argument("--lambda-band", type=float, default=0.05)
    parser.add_argument("--lambda-spatial", type=float, default=0.05)
    parser.add_argument("--lambda-energy", type=float, default=1e-4)
    parser.add_argument("--rank-margin", type=float, default=0.01)
    parser.add_argument("--min-gap-ratio", type=float, default=0.25)
    parser.add_argument("--max-gap-ratio", type=float, default=2.0)
    parser.add_argument("--dreamsim-model", default=None)
    args = parser.parse_args()
    if not args.model:
        parser.error("Pass --model or set FLUX_KONTEXT_MODEL_PATH.")
    if args.branches < 1 or args.control_steps < 1 or args.steps < args.control_steps:
        parser.error("Require branches/control-steps >= 1 and steps >= control-steps.")
    if args.control_init_noise < 0 or args.benchmark_repeats < 1:
        parser.error("--control-init-noise must be non-negative and repeats positive.")
    return args


def _pil_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(image, dtype=np.float32).copy() / 255).permute(2, 0, 1).unsqueeze(0).to(device)


def _to_pil(image: torch.Tensor) -> Image.Image:
    return Image.fromarray(image.detach().clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy(), "RGB")


def _json(path: Path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _module_diagnostics() -> dict[str, float]:
    direction = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    relevance = (torch.tensor([[1.0, 0.0]]),)
    good, _ = control_band_loss((0.5 * direction,), (direction,), relevance)
    wrong, _ = control_band_loss((-0.5 * direction,), (direction,), relevance)
    orth, _ = control_band_loss((0.5 * direction + torch.tensor([[[0.0, 2.0], [0.0, 2.0]]]),), (direction,), relevance)
    high = spatial_prior_loss((torch.tensor([[[2.0], [0.0]]]),), relevance)
    low = spatial_prior_loss((torch.tensor([[[0.0], [2.0]]]),), relevance)
    linear = (torch.tensor([[[0.75], [0.75]], [[0.5], [0.5]], [[0.25], [0.25]]]),)
    zigzag = (linear[0] + torch.tensor([[[0.4], [0.4]], [[-0.4], [-0.4]], [[0.4], [0.4]]]),)
    smooth = control_smoothness_loss(linear, (torch.ones(1, 2, 1),))
    curved = control_smoothness_loss(zigzag, (torch.ones(1, 2, 1),))
    normal_gap, _, _ = gap_bound_loss(torch.tensor([0.25, 0.3, 0.2, 0.25]), torch.tensor(1.0), min_ratio=0.25, max_ratio=2.0)
    collapse_gap, _, _ = gap_bound_loss(torch.tensor([0.01, 0.25, 0.25, 0.25]), torch.tensor(1.0), min_ratio=0.25, max_ratio=2.0)
    jump_gap, _, _ = gap_bound_loss(torch.tensor([0.25, 0.25, 0.25, 0.8]), torch.tensor(1.0), min_ratio=0.25, max_ratio=2.0)
    straight, _ = triangle_deficit_loss(torch.tensor([1.0, 1.0]), torch.tensor([2.0]), torch.tensor(2.0))
    detour, _ = triangle_deficit_loss(torch.tensor([1.0, 1.0]), torch.tensor([0.5]), torch.tensor(2.0))
    return {key: float(value) for key, value in {
        "band_good": good, "band_wrong": wrong, "band_orthogonal": orth,
        "spatial_high_relevance": high, "spatial_low_relevance": low,
        "control_smooth_linear": smooth, "control_smooth_zigzag": curved,
        "rank_ordered": adjacent_ranking_loss(torch.tensor([0., .2, .5, .8, 1.]), margin=.01),
        "rank_inverted": adjacent_ranking_loss(torch.tensor([0., .5, .3]), margin=.01),
        "gap_normal": normal_gap, "gap_collapse": collapse_gap, "gap_jump": jump_gap,
        "triangle_straight": straight, "triangle_detour": detour,
    }.items()}


def _save_relevance(output: Path, inputs):
    rows = []
    for index, (direction, score, relevance) in enumerate(zip(inputs.prior.directions, inputs.prior.raw_scores, inputs.prior.relevance)):
        map_ = relevance[0].reshape(inputs.native.sampling_token_height, inputs.native.sampling_token_width)
        pixels = map_.mul(255).round().byte().cpu().numpy()
        Image.fromarray(pixels, "L").resize((inputs.native.width, inputs.native.height), Image.Resampling.NEAREST).save(output / f"relevance_step_{index}.png")
        rows.append({
            "step": index, "direction_l2": float(direction.float().norm()),
            "raw_min": float(score.min()), "raw_mean": float(score.mean()), "raw_max": float(score.max()), "raw_std": float(score.std(unbiased=False)),
            "relevance_min": float(relevance.min()), "relevance_mean": float(relevance.mean()), "relevance_max": float(relevance.max()), "relevance_std": float(relevance.std(unbiased=False)),
            "fraction_above_0p25": float((relevance > .25).float().mean()), "fraction_above_0p5": float((relevance > .5).float().mean()), "fraction_above_0p75": float((relevance > .75).float().mean()),
            "is_binary": bool(torch.all((relevance == 0) | (relevance == 1))),
        })
    _json(output / "prior_diagnostics.json", {"soft_prior": "score / max(score) per timestep; no threshold or hard mask", "steps": rows})


def _errors(actual: torch.Tensor, expected: torch.Tensor):
    difference = (actual.float() - expected.float()).abs()
    return {"max_abs": float(difference.max()), "mean_abs": float(difference.mean()), "cosine": float(torch.nn.functional.cosine_similarity(actual.float().flatten()[None], expected.float().flatten()[None]))}


def main():
    args = _args()
    if not torch.cuda.is_available():
        raise RuntimeError("The real Kontext diagnostics require CUDA.")
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    case = manifest["cases"][0]
    seed = case["seed"] if args.seed is None else args.seed
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _json(output / "module_diagnostics.json", _module_diagnostics())
    _json(output / "config.json", {**vars(args), "case_id": case["case_id"], "source": case["source"], "prompt": case["instruction"], "seed": seed, "git_commit": os.popen("git rev-parse HEAD").read().strip()})
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    source_pil = Image.open(case["source"]).convert("RGB").resize((args.width, args.height), Image.Resampling.LANCZOS)
    source_pil.save(output / "source.png")
    pipe = FluxKontextCoupledControlPipeline.from_pretrained(args.model, torch_dtype=dtype, local_files_only=True).to(device)
    pipe.set_progress_bar_config(disable=True)
    inputs = pipe.prepare_coupled_control_inputs(num_branches=args.branches, control_steps=args.control_steps, image=source_pil, prompt=case["instruction"], height=args.height, width=args.width, num_inference_steps=args.steps, guidance_scale=args.guidance_scale, generator=torch.Generator(device=device).manual_seed(seed))
    with torch.no_grad():
        native_image = pipe.decode_terminal_latent(inputs.native.native_final_latent, inputs.native)
    _to_pil(native_image[0]).save(output / "native_full.png")
    _save_relevance(output, inputs)
    zero = initialize_independent_coupled_controls(inputs.prior.directions, num_branches=args.branches, betas=(0.0,) * args.branches)
    with torch.no_grad():
        batched = pipe.unroll_coupled_controls(inputs, zero, use_checkpointing=False)
        sequential = pipe.unroll_terminal_controls(inputs.native, initialize_velocity_controls(inputs.native.initial_latent, args.control_steps), use_checkpointing=False)
        batched_images = pipe.decode_coupled_terminal_latent(batched.final_latent, inputs)
        sequential_image = pipe.decode_terminal_latent(sequential.final_latent, inputs.native)
    branch_errors = [_errors(batched.final_latent[index:index + 1], sequential.final_latent) for index in range(args.branches)]
    image_errors = [_errors(batched_images[index:index + 1], sequential_image) for index in range(args.branches)]
    _json(output / "batched_parity.json", {"K": args.branches, "branch_latent_vs_single": branch_errors, "branch_image_vs_single": image_errors, "max_within_batched_latent": float((batched.final_latent - batched.final_latent[:1]).abs().max())})
    # Benchmark only zero controls. It is an implementation-cost measurement, not an optimization result.
    del batched, sequential, batched_images, sequential_image
    gc.collect()
    torch.cuda.empty_cache()
    benchmark_controls = initialize_independent_coupled_controls(
        inputs.prior.directions, num_branches=args.branches, betas=(0.0,) * args.branches
    )
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for _ in range(args.benchmark_repeats):
        with torch.no_grad():
            pipe.unroll_coupled_controls(inputs, benchmark_controls, use_checkpointing=args.use_checkpointing)
    batched_seconds = (time.perf_counter() - start) / args.benchmark_repeats
    batched_peak = {"allocated": int(torch.cuda.max_memory_allocated(device)), "reserved": int(torch.cuda.max_memory_reserved(device))}
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for _ in range(args.benchmark_repeats):
        for _branch in range(args.branches):
            with torch.no_grad():
                pipe.unroll_terminal_controls(
                    inputs.native,
                    initialize_velocity_controls(inputs.native.initial_latent, args.control_steps),
                    use_checkpointing=args.use_checkpointing,
                )
    sequential_seconds = (time.perf_counter() - start) / args.benchmark_repeats
    _json(output / "compute_benchmark.json", {"sequential_seconds": sequential_seconds, "batched_seconds": batched_seconds, "batched_peak_vram": batched_peak, "note": "Batched paths reduce repeated transformer forwards; checkpointing trades memory for recomputation time."})
    blocker = DreamSimAdapter.availability_reason(model_path=args.dreamsim_model)
    _json(output / "dreamsim_blocker.json", {"blocked": blocker is not None, "reason": blocker, "consequence": "Experiment D initialization geometry and Experiment E optimization were intentionally not run without official differentiable DreamSim."})
    print("FINAL_REPORT=" + json.dumps({"output_dir": str(output), "dreamsim_blocked": blocker is not None, "parity": branch_errors, "benchmark_seconds": {"sequential": sequential_seconds, "batched": batched_seconds}}, sort_keys=True))


if __name__ == "__main__":
    main()
