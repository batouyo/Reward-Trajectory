#!/usr/bin/env python3
"""Compare single-control LeapAlign bridge estimates to full reward gradients."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rewardflow_calibration.diagnostics.backward_direction import cosine_similarity  # noqa: E402
from rewardflow_calibration.diagnostics.leapalign_proxy import mask_gradient_estimators  # noqa: E402
from rewardflow_calibration.diagnostics.velo_timed_leap import (  # noqa: E402
    extract_control_step_gradient,
    leap_gradient_for_control_step,
)
from rewardflow_calibration.metrics.dreamsim import DreamSimDistance  # noqa: E402
from rewardflow_calibration.optimization.backward_reward import (  # noqa: E402
    BackwardReward,
    BackwardRewardConfig,
)
from rewardflow_calibration.optimization.backward_trajectory_optimizer import (  # noqa: E402
    build_image_space_masks,
)
from rewardflow_calibration.rollout.veloedit import (  # noqa: E402
    VeloEditCompatibleRollout,
    VeloEditRolloutConfig,
)


DEFAULT_IMAGE = "/home/hyp/Code/VeloEdit/testdata/9.jpg"
DEFAULT_MODEL = "/data15/hyp/weight/FLUX.1-Kontext-dev"
DEFAULT_PROMPT = "Change the car to a modern sport car"
DEFAULT_OUTPUT = ROOT / "outputs/velo_timed_leapalign_car_seed42"
CONTROL_STEP_INDICES = (0, 1, 2, 3)
BRIDGE_STEP_NUMBERS = (4, 8, 12)
NESTED_GRAD_COE = 0.3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--goal-steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--max-area", type=int, default=1024 * 1024)
    parser.add_argument("--similarity-threshold", type=float, default=0.8)
    parser.add_argument("--siglip-model-path", default=BackwardReward.DEFAULT_SIGLIP_PATH)
    parser.add_argument("--dino-model-path", default=BackwardReward.DEFAULT_DINO_PATH)
    parser.add_argument("--dreamsim-cache-dir", default="/data15/hyp/weight/dreamsim_ckpts")
    parser.add_argument("--cache-dir", default="/data15/hyp/weight")
    parser.add_argument("--semantic-floor-fraction", type=float, default=0.5)
    parser.add_argument("--semantic-anchor-min-gap", type=float, default=0.02)
    parser.add_argument("--source-weight", type=float, default=1.0)
    parser.add_argument("--semantic-weight", type=float, default=1.0)
    parser.add_argument("--keep-weight", type=float, default=1.0)
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def _save_image(image: torch.Tensor, path: Path) -> None:
    import numpy as np

    pixels = image[0].detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    Image.fromarray(np.rint(pixels * 255).astype("uint8"), mode="RGB").save(path)


def _loss_components(reward, image, source, keep_mask) -> dict[str, torch.Tensor]:
    values = reward.evaluate(image, source, keep_mask)
    return {"source": values.source_loss, "total": values.total}


def _full_gradients(losses: dict[str, torch.Tensor], residual: torch.Tensor) -> dict[str, torch.Tensor]:
    gradients: dict[str, torch.Tensor] = {}
    for index, name in enumerate(("source", "total")):
        gradient = torch.autograd.grad(
            losses[name], residual, retain_graph=index == 0, allow_unused=True
        )[0]
        gradients[name] = torch.zeros_like(residual) if gradient is None else gradient.detach()
    return gradients


def _rms(value: torch.Tensor) -> float:
    return float(value.detach().float().square().mean().sqrt().cpu())


def _native_edit_masks(trace: dict[str, Any], prepared, threshold: float):
    rows: list[torch.Tensor] = []
    states = trace["latent_states"]
    sigmas = trace["sigmas"]
    velocities = trace["native_velocities"]
    for index in CONTROL_STEP_INDICES:
        state = states[index]
        velocity = velocities[index]
        reference_velocity = (
            (state.float() - prepared.reference_latent.float())
            / (torch.as_tensor(sigmas[index], device=state.device).float() + 1e-8)
        )
        reference_abs = reference_velocity.abs() + 1e-8
        similarity = reference_abs / (reference_abs + (velocity.float() - reference_velocity).abs())
        rows.append(~(similarity >= threshold).squeeze(0).bool())
    return torch.stack(rows)


def main() -> None:
    args = parse_args()
    if args.steps < max(BRIDGE_STEP_NUMBERS):
        raise SystemExit("steps must cover bridge steps 4, 8, and 12")
    if args.goal_steps != len(CONTROL_STEP_INDICES):
        raise SystemExit("this diagnostic uses the four VeloEdit-timed control steps")
    for label, path in (
        ("model", args.model_path), ("source image", args.image),
        ("SigLIP", args.siglip_model_path), ("DINOv2", args.dino_model_path),
        ("DreamSim cache", args.dreamsim_cache_dir),
    ):
        if not Path(path).exists():
            raise FileNotFoundError(f"{label} path is unavailable: {path}")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dtype = torch.bfloat16
    # Native FLUX-Kontext trajectory: VeloEdit interventions and its sigma
    # schedule alignment are disabled. FIRST_STEP_ALIGN_STEPS is independent
    # of intervention timing; only the first-four-step timing prior is reused.
    rollout_config = VeloEditRolloutConfig(
        steps=args.steps,
        seed=args.seed,
        guidance_scale=args.guidance_scale,
        first_step_align_steps=0,
        preserve_steps=0,
        edit_steps=0,
        similarity_threshold=args.similarity_threshold,
        max_area=args.max_area,
        accumulate_latents_fp32=False,
    )
    reward_config = BackwardRewardConfig(
        semantic_floor_fraction=args.semantic_floor_fraction,
        semantic_anchor_min_gap=args.semantic_anchor_min_gap,
        source_weight=args.source_weight,
        semantic_weight=args.semantic_weight,
        keep_weight=args.keep_weight,
    )
    config = {
        "experiment": "VeloEdit-timed single-control LeapAlign bridge sweep",
        "base_commit": "397eccb02893c4d9d6525d08ca399bee197e0bf7",
        "model_path": args.model_path,
        "image": str(Path(args.image).resolve()),
        "prompt": args.prompt,
        "output_dir": str(output),
        "device": args.device,
        "seed": args.seed,
        "steps": args.steps,
        "control_steps": [1, 2, 3, 4],
        "control_step_indices_zero_based": list(CONTROL_STEP_INDICES),
        "bridge_steps": list(BRIDGE_STEP_NUMBERS),
        "bridge_state_index_definition": (
            "Step N means the latent immediately before velocity update N; "
            "the zero-based state index is N-1."
        ),
        "veloedit_official_reference": {
            "repository": "https://github.com/xmulzq/VeloEdit",
            "benchmark_intervention_flux_defaults": {
                "preserve_intervention_steps": 4,
                "edit_intervention_steps": 4,
                "first_step_align_steps": 4,
            },
            "config_py_flux_defaults": {
                "model_name": "flux",
                "model_path": "black-forest-labs/FLUX.1-Kontext-dev",
                "dtype": "bfloat16",
                "num_inference_steps": 6,
                "guidance_scale": 2.5,
                "defines_intervention_steps": False,
                "defines_first_step_align_steps": False,
            },
            "sampler_activation": "preserve_active = i < preserve_steps; edit_active = i < edit_steps",
            "effective_intervention_steps_zero_based": [0, 1, 2, 3],
            "first_step_align_is_sigma_schedule_adjustment": True,
            "first_step_align_semantics": "schedule adjustment, not a control-step selector",
            "config_py_scope": "FLUX model/sampling defaults; it does not define intervention step counts or first-step alignment.",
        },
        "prior_used": "timing only: first four effective velocity updates",
        "veloedit_velocity_direction_used": False,
        "veloedit_keep_edit_velocity_used": False,
        "veloedit_alpha_used": False,
        "native_rollout": asdict(rollout_config),
        "nested_grad_coe": NESTED_GRAD_COE,
        "reward": asdict(reward_config),
        "primary_comparison": "raw",
        "masked_comparison": "secondary frozen-mask diagnostic only",
        "spatial_mask_is_core_prior": False,
        "optimizer_gate_line_search_or_correction_used": False,
    }
    _write_json(output / "config.json", config)

    rollout = VeloEditCompatibleRollout(
        args.model_path, device=device, dtype=dtype, local_files_only=True
    )
    source_pil = Image.open(args.image).convert("RGB")
    prepared = rollout.prepare(source_pil, args.prompt, config=rollout_config, seed=args.seed)
    pixels = rollout.pipeline.image_processor.preprocess(
        prepared.working_image, prepared.height, prepared.width
    ).to(device=device, dtype=torch.float32)
    source = ((pixels + 1.0) / 2.0).clamp(0, 1).detach()

    with torch.no_grad():
        native_full, true_trace = rollout.rollout_native(
            prepared, config=rollout_config, return_latent_trace=True
        )
        native_proxy, _ = rollout.rollout_native(
            prepared,
            config=rollout_config,
            early_stop_steps=args.goal_steps,
            goal_residual=torch.zeros(
                (args.goal_steps, *prepared.latents.shape[1:]), device=device
            ),
            return_latent_trace=True,
        )
    true_states = tuple(state.detach() for state in true_trace["latent_states"])
    sigmas = tuple(true_trace["sigmas"])
    true_final = true_trace["true_final_latent"].detach()
    if true_final.requires_grad or any(state.requires_grad for state in true_states):
        raise AssertionError("full-rollout anchors must be detached")

    hard_mask = _native_edit_masks(true_trace, prepared, args.similarity_threshold).to(device)
    _, keep_mask = build_image_space_masks(
        hard_mask,
        height=prepared.height,
        width=prepared.width,
        vae_scale_factor=rollout.pipeline.vae_scale_factor,
        latent_ids=prepared.latent_ids,
    )
    keep_mask = keep_mask.to(device)

    dreamsim = DreamSimDistance(device, cache_dir=args.dreamsim_cache_dir)
    reward = BackwardReward(
        args.prompt,
        device=device,
        config=reward_config,
        siglip_model_path=args.siglip_model_path,
        dino_model_path=args.dino_model_path,
        cache_dir=args.cache_dir,
        dreamsim_distance=dreamsim.distance,
    )
    anchors = reward.set_anchors(source, native_full, native_proxy)
    _save_image(source, output / "source.png")
    _save_image(native_proxy, output / "native_proxy.png")
    _save_image(native_full, output / "native_full.png")

    residual = torch.zeros(
        (args.goal_steps, *prepared.latents.shape[1:]),
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    full_image = rollout.rollout_native(
        prepared, config=rollout_config, goal_residual=residual, early_stop_steps=None
    )
    full_gradients = _full_gradients(
        _loss_components(reward, full_image, source, keep_mask), residual
    )
    del full_image
    if device.type == "cuda":
        torch.cuda.empty_cache()

    def velocity_fn(latent: torch.Tensor, sigma: torch.Tensor | float) -> torch.Tensor:
        return rollout.native_velocity(prepared, latent, sigma)

    comparison: dict[str, Any] = {}
    approximation: list[dict[str, Any]] = []
    sweep_rows: list[dict[str, Any]] = []
    for control_index in CONTROL_STEP_INDICES:
        control_key = f"control_step_{control_index + 1}"
        comparison[control_key] = {}
        for component in ("source", "total"):
            full_step = extract_control_step_gradient(
                full_gradients[component], control_step_index=control_index
            )
            comparison[control_key][component] = {
                "full_gradient_rms": _rms(full_step),
                "full_gradient_shape": list(full_step.shape),
                "bridges": {},
            }
            for bridge_step in BRIDGE_STEP_NUMBERS:
                bridge_index = bridge_step - 1

                def objective(latent, component_name=component):
                    image = rollout.decode_latents(prepared, latent)
                    return _loss_components(reward, image, source, keep_mask)[component_name]

                leap_gradient, details = leap_gradient_for_control_step(
                    torch.zeros_like(full_step),
                    control_step_index=control_index,
                    bridge_step_index=bridge_index,
                    true_states=true_states,
                    sigmas=sigmas,
                    true_final=true_final,
                    velocity_fn=velocity_fn,
                    objective_fn=objective,
                    nested_grad_coe=NESTED_GRAD_COE,
                )
                masked_leap = mask_gradient_estimators(
                    {"gradient": leap_gradient.unsqueeze(0)},
                    hard_mask[control_index : control_index + 1],
                )["gradient"].squeeze(0)
                masked_full = mask_gradient_estimators(
                    {"gradient": full_step.unsqueeze(0)},
                    hard_mask[control_index : control_index + 1],
                )["gradient"].squeeze(0)
                raw_cosine = cosine_similarity(leap_gradient, full_step)
                masked_cosine = cosine_similarity(masked_leap, masked_full)
                pair_key = f"bridge_{bridge_step}"
                stats = {
                    "control_step": control_index + 1,
                    "bridge_step": bridge_step,
                    "control_to_bridge_relative_error": details["control_to_bridge"]["relative_rms_error"],
                    "bridge_to_final_relative_error": details["bridge_to_final"]["relative_rms_error"],
                    "raw_cosine_vs_full": raw_cosine,
                    "masked_cosine_vs_full": masked_cosine,
                    "leap_gradient_rms": _rms(leap_gradient),
                    "full_gradient_rms": _rms(full_step),
                    "control_sigma": details["control_sigma"],
                    "bridge_sigma": details["bridge_sigma"],
                    "connector_forward_max_abs_error": max(
                        details["bridge_connected_forward_max_abs_error"],
                        details["final_connected_forward_max_abs_error"],
                    ),
                    "bridge_connected_dtype": details["bridge_connected_dtype"],
                    "final_connected_dtype": details["final_connected_dtype"],
                }
                comparison[control_key][component]["bridges"][pair_key] = stats
                sweep_rows.append({"component": component, **stats})
                if component == "source":
                    approximation.append({**details, "component": component})
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    _write_json(output / "gradient_comparison.json", {
        "primary_comparison": "raw",
        "masked_is_secondary": True,
        "control_steps": comparison,
    })
    _write_json(output / "leap_approximation.json", {
        "state_index_definition": config["bridge_state_index_definition"],
        "bridge_sweep": approximation,
    })
    _write_json(output / "bridge_sweep_summary.json", {
        "primary_comparison": "raw",
        "bridge_steps": list(BRIDGE_STEP_NUMBERS),
        "rows": sweep_rows,
        "automatic_best_bridge_selection": False,
    })

    summary = {
        "status": "complete",
        "base_reward_anchors": anchors,
        "primary_comparison": "raw",
        "masked_comparison": "secondary",
        "timing_prior_only": True,
        "veloedit_velocity_direction_used": False,
        "official_intervention_steps_zero_based": [0, 1, 2, 3],
        "control_steps": [1, 2, 3, 4],
        "bridge_steps": list(BRIDGE_STEP_NUMBERS),
        "true_final_ground_truth": True,
        "optimizer_or_correction_used": False,
        "output_dir": str(output),
        "outputs": {
            name: str((output / name).resolve())
            for name in (
                "config.json", "gradient_comparison.json", "leap_approximation.json",
                "bridge_sweep_summary.json", "source.png", "native_proxy.png", "native_full.png",
            )
        },
    }
    _write_json(output / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
