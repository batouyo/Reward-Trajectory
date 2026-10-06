#!/usr/bin/env python3
"""Run the experimental masked reward-guided backward trajectory controller."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rewardflow_calibration.optimization.backward_reward import (
    BackwardReward,
    BackwardRewardConfig,
    InvalidSemanticAnchorError,
)
from rewardflow_calibration.optimization.backward_trajectory_optimizer import (
    BackwardTrajectoryConfig,
    BackwardTrajectoryOptimizer,
    build_image_space_masks,
    freeze_native_control_context,
)
from rewardflow_calibration.optimization.trajectory_gate import TrajectoryGateConfig
from rewardflow_calibration.rollout.veloedit import VeloEditCompatibleRollout, VeloEditRolloutConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--goal-steps", type=int, default=4)
    parser.add_argument("--max-iterations", type=int, default=16)
    parser.add_argument("--semantic-floor-fraction", type=float, default=0.5)
    parser.add_argument("--semantic-anchor-min-gap", type=float, default=0.02)
    parser.add_argument("--semantic-tolerance", type=float, default=0.005)
    parser.add_argument("--min-visible-dreamsim", type=float, default=0.01)
    parser.add_argument("--max-second-order-deficit", type=float, default=0.25)
    parser.add_argument("--keep-l1-tolerance", type=float, default=0.01)
    parser.add_argument("--line-search-ratios", nargs="+", type=float, default=[0.05, 0.02, 0.01, 0.005, 0.002])
    parser.add_argument("--max-total-residual-ratio", type=float, default=0.10)
    parser.add_argument("--max-active-residual-ratio", type=float, default=0.5)
    parser.add_argument("--similarity-threshold", type=float, default=0.8)
    parser.add_argument("--sourceward-tolerance", type=float, default=1e-5)
    parser.add_argument("--gradient-epsilon", type=float, default=1e-10)
    parser.add_argument("--source-weight", type=float, default=1.0)
    parser.add_argument("--semantic-weight", type=float, default=1.0)
    parser.add_argument("--keep-weight", type=float, default=1.0)
    parser.add_argument("--siglip-model-path", default=BackwardReward.DEFAULT_SIGLIP_PATH)
    parser.add_argument("--dino-model-path", default=BackwardReward.DEFAULT_DINO_PATH)
    parser.add_argument("--cache-dir", default="/data15/hyp/weight")
    parser.add_argument("--max-area", type=int, default=1024 * 1024)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--verify-final", action="store_true", default=False)
    parser.add_argument("--save-debug-tensors", action="store_true", default=False)
    parser.add_argument("--float32", action="store_true")
    return parser.parse_args()


def _save_image(image: torch.Tensor, path: Path) -> None:
    pixels = image[0].detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    from PIL import Image
    import numpy as np
    Image.fromarray(np.rint(pixels * 255).astype(np.uint8), mode="RGB").save(path)


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.steps < 1 or not 1 <= args.goal_steps <= args.steps:
        raise SystemExit("goal-steps must be in [1, steps]")
    if args.max_iterations < 1:
        raise SystemExit("max-iterations must be positive")
    if not args.line_search_ratios or any(value <= 0 for value in args.line_search_ratios):
        raise SystemExit("line-search-ratios must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reward_config = BackwardRewardConfig(
        semantic_floor_fraction=args.semantic_floor_fraction,
        semantic_anchor_min_gap=args.semantic_anchor_min_gap,
        source_weight=args.source_weight,
        semantic_weight=args.semantic_weight,
        keep_weight=args.keep_weight,
    )
    backward_config = BackwardTrajectoryConfig(
        goal_steps=args.goal_steps,
        max_iterations=args.max_iterations,
        line_search_ratios=tuple(args.line_search_ratios),
        max_total_residual_ratio=args.max_total_residual_ratio,
        gradient_epsilon=args.gradient_epsilon,
        sourceward_tolerance=args.sourceward_tolerance,
    )
    gate_config = TrajectoryGateConfig(
        semantic_tolerance=args.semantic_tolerance,
        semantic_floor=0.0,
        min_visible_dreamsim=args.min_visible_dreamsim,
        sourceward_tolerance=args.sourceward_tolerance,
        max_second_order_deficit=args.max_second_order_deficit,
        keep_l1_tolerance=args.keep_l1_tolerance,
        max_total_residual_ratio=args.max_total_residual_ratio,
        max_active_residual_ratio=args.max_active_residual_ratio,
    )
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
    config_json = {
        "model_path": args.model_path,
        "image": str(Path(args.image).resolve()),
        "prompt": args.prompt,
        "output_dir": str(args.output_dir.resolve()),
        "device": args.device,
        "seed": args.seed,
        "steps": args.steps,
        "goal_steps": args.goal_steps,
        "dtype": "float32" if args.float32 else "bfloat16",
        "fp32_latent_accumulation": False,
        "verify_final": args.verify_final,
        "save_debug_tensors": args.save_debug_tensors,
        "backward_trajectory": asdict(backward_config),
        "reward": asdict(reward_config),
        "trajectory_gate": asdict(gate_config),
        "rollout": asdict(rollout_config),
        "siglip_model_path": args.siglip_model_path,
        "dino_model_path": args.dino_model_path,
        "cache_dir": args.cache_dir,
        "mask_source": "elementwise low-similarity mask from native full trajectory; frozen before optimization",
        "candidate_selection": "evaluate all line-search ratios; choose accepted candidate with lowest DreamSim-to-source, tie-break by smaller increment ratio",
        "line_search_ratio_semantics": "global increment RMS / global native velocity RMS; not slider strength",
    }
    _write_json(args.output_dir / "config.json", config_json)

    dtype = torch.float32 if args.float32 else torch.bfloat16
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    rollout = VeloEditCompatibleRollout(
        args.model_path, device=device, dtype=dtype, local_files_only=True
    )
    source_pil = Image.open(args.image).convert("RGB")
    prepared = rollout.prepare(source_pil, args.prompt, config=rollout_config, seed=args.seed)
    pixels = rollout.pipeline.image_processor.preprocess(
        prepared.working_image, prepared.height, prepared.width
    ).to(device=device, dtype=torch.float32)
    source = ((pixels + 1.0) / 2.0).clamp(0, 1)
    native_trace: list[dict[str, torch.Tensor]] = []
    with torch.no_grad():
        native_full = rollout.rollout_native(
            prepared, config=rollout_config, velocity_trace=native_trace
        )
        native_proxy = rollout.rollout_native(
            prepared, config=rollout_config, early_stop_steps=args.goal_steps
        )
    _save_image(source, args.output_dir / "source.png")
    _save_image(native_full, args.output_dir / "native_full.png")
    _save_image(native_proxy, args.output_dir / "native_proxy.png")

    hard_mask, hard_keep_mask, native_rms, native_velocity = freeze_native_control_context(native_trace, args.goal_steps)
    image_edit_mask, image_keep_mask = build_image_space_masks(
        hard_mask,
        height=prepared.height,
        width=prepared.width,
        vae_scale_factor=rollout.pipeline.vae_scale_factor,
        latent_ids=prepared.latent_ids,
    )
    mask_diagnostics = {
        "mask_global_coverage": float(hard_mask.float().mean().cpu()),
        "mask_coverage_per_step": hard_mask.float().mean(dim=(1, 2)).cpu().tolist(),
        "native_velocity_rms_per_step": native_rms,
        "image_edit_mask_coverage": float(image_edit_mask.mean().cpu()),
        "token_grid": [prepared.height // (rollout.pipeline.vae_scale_factor * 2), prepared.width // (rollout.pipeline.vae_scale_factor * 2)],
        "token_count": int(prepared.latents.shape[1]),
    }
    reward = BackwardReward(
        args.prompt,
        device=device,
        config=reward_config,
        siglip_model_path=args.siglip_model_path,
        dino_model_path=args.dino_model_path,
        cache_dir=args.cache_dir,
    )
    try:
        anchor_diagnostics = reward.set_anchors(source, native_full, native_proxy)
    except InvalidSemanticAnchorError as exc:
        anchor_diagnostics = {**exc.diagnostics, **mask_diagnostics}
        _write_json(args.output_dir / "anchors.json", anchor_diagnostics)
        _write_json(args.output_dir / "summary.json", {
            "stop_reason": "invalid_semantic_anchor",
            "accepted_count": 0,
            "rejection_reason_counts": {},
            "mask_diagnostics": mask_diagnostics,
        })
        (args.output_dir / "trajectory.jsonl").write_text(
            json.dumps({"event": "stop", "stop_reason": "invalid_semantic_anchor"}) + "\n",
            encoding="utf-8",
        )
        return
    anchor_diagnostics.update(mask_diagnostics)
    _write_json(args.output_dir / "anchors.json", anchor_diagnostics)
    config_json["trajectory_gate"]["semantic_floor"] = float(reward.semantic_floor)
    _write_json(args.output_dir / "config.json", config_json)
    gate_config = TrajectoryGateConfig(
        semantic_tolerance=args.semantic_tolerance,
        semantic_floor=float(reward.semantic_floor),
        min_visible_dreamsim=args.min_visible_dreamsim,
        sourceward_tolerance=args.sourceward_tolerance,
        max_second_order_deficit=args.max_second_order_deficit,
        keep_l1_tolerance=args.keep_l1_tolerance,
        max_total_residual_ratio=args.max_total_residual_ratio,
        max_active_residual_ratio=args.max_active_residual_ratio,
    )
    optimizer = BackwardTrajectoryOptimizer(
        rollout,
        prepared=prepared,
        source_image=source,
        native_full_image=native_full,
        native_proxy_image=native_proxy,
        reward=reward,
        hard_edit_mask=hard_mask.to(device),
        image_keep_mask=image_keep_mask.to(device),
        native_velocity_rms_per_step=native_rms,
        native_velocity_per_step=native_velocity.to(device),
        rollout_config=rollout_config,
        config=backward_config,
        gate_config=gate_config,
    )
    summary = optimizer.run(
        args.output_dir,
        verify_final=args.verify_final,
        save_debug_tensors=args.save_debug_tensors,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
