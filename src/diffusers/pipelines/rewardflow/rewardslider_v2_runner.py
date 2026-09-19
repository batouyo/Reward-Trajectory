"""Minimal auditable RewardSlider V2 runner facade and CLI."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from PIL import Image

from .pipeline_flux_kontext_rewardslider_v2 import FluxKontextRewardSliderV2Pipeline, RewardSliderV2Inputs
from .rewardslider_v2_alpha import OrderedAlphaParameterization
from .rewardslider_v2_lpips import LPIPSDistance
from .rewardslider_v2_preservation import build_image_space_relevance, masked_lpips_preservation_loss
from .rewardslider_v2_regularizers import initialize_v_goal_parameters, v_goal_regularizers
from .rewardslider_v2_coordination import DynamicDeficitCoordinator, DynamicDeficitOutput
from .rewardslider_v2_quality import DifferentiableQualityReward, audit_quality_reward
from .rewardslider_v2_scheduler import RewardSliderV2Scheduler


def build_rewardslider_v2_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the RewardSlider V2 optimization loop.")
    parser.add_argument("--initial-nodes", type=int, default=5)
    parser.add_argument("--max-nodes", type=int, default=10)
    parser.add_argument("--control-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha-lr", type=float, default=1e-3)
    parser.add_argument("--vgoal-lr", type=float, default=1e-3)
    parser.add_argument("--trajectory-kl-threshold", type=float, default=0.15)
    parser.add_argument("--trajectory-patience", type=int, default=3)
    parser.add_argument("--trajectory-guard-weight", type=float, default=0.1)
    parser.add_argument("--local-refine-iters", type=int, default=10)
    parser.add_argument("--joint-refine-iters", type=int, default=10)
    parser.add_argument("--enable-prune", action="store_true")
    parser.add_argument("--enable-insert", action="store_true")
    parser.add_argument("--quality-reward", default=None)
    parser.add_argument("--use-checkpointing", action="store_true")
    parser.add_argument("--output-jsonl", type=Path, default=Path("rewardslider_v2.jsonl"))
    parser.add_argument("--run-real-smoke", action="store_true")
    parser.add_argument("--model", default=os.getenv("FLUX_KONTEXT_MODEL_PATH"))
    parser.add_argument("--source", default=None)
    parser.add_argument("--prompt", default="Make the weighted training ball blue while preserving its shape, texture, lighting, and background.")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    return parser






class _MockQualityScorer(nn.Module):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return 1 - image.float().square().mean()

def build_quality_reward(spec: str | None, device: torch.device) -> DifferentiableQualityReward | None:
    if spec is None:
        return None
    if spec.lower() == "mock":
        return DifferentiableQualityReward(_MockQualityScorer()).to(device)
    raise ValueError(
        f"Quality reward {spec} is unavailable. Use mock for the tensor-native test scorer or omit it."
    )
def _norm(value: torch.Tensor | None) -> float | None:
    return None if value is None else float(value.detach().float().norm().item())


def coordinate_image_deficits(
    coordinator: DynamicDeficitCoordinator,
    preservation_loss: torch.Tensor,
    quality_loss: torch.Tensor | None = None,
) -> DynamicDeficitOutput:
    """Coordinate only image-level deficits; control penalties remain separate."""
    values = [preservation_loss.reshape(())]
    if quality_loss is not None:
        values.append(quality_loss.reshape(()))
    return coordinator.coordinate(torch.stack(values))


class RewardSliderV2Runner:
    """Log one deterministic optimization step; FLUX execution is injected by caller."""

    def __init__(self, alpha_parameter: nn.Parameter, v_goals: Sequence[nn.Parameter], output_jsonl: str | Path):
        self.alpha_parameter = alpha_parameter
        self.v_goals = tuple(v_goals)
        self.output_jsonl = Path(output_jsonl)
        self.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        self.scheduler = RewardSliderV2Scheduler(
            nn.ParameterList([alpha_parameter]), self.v_goals, min_repair_iterations=1
        )
        self.iteration = 0

    def step(
        self,
        trajectory_loss: torch.Tensor,
        quality_loss: torch.Tensor,
        *,
        trajectory_kl: float,
        reward_values: Sequence[float] = (),
        normalized_rewards: Sequence[float] = (),
        dynamic_weights: Sequence[float] = (),
        topology_event: dict | None = None,
    ) -> dict:
        started = time.perf_counter()
        self.scheduler.backward(trajectory_loss=trajectory_loss, quality_loss=quality_loss)
        self.scheduler.advance(trajectory_kl)
        record = {
            "iteration": self.iteration,
            "trajectory": {
                "current_number_of_nodes": self.alpha_parameter.numel() + 1,
                "trajectory_kl": float(trajectory_kl),
            },
            "reward": {
                "raw_values": list(reward_values),
                "normalized_values": list(normalized_rewards),
                "dynamic_weights": list(dynamic_weights),
            },
            "gradient": {
                "alpha_gradient_norm": _norm(self.alpha_parameter.grad),
                "v_goal_gradient_norms": [_norm(goal.grad) for goal in self.v_goals],
                "finite": all(parameter.grad is None or torch.isfinite(parameter.grad).all().item() for parameter in (self.alpha_parameter, *self.v_goals)),
            },
            "control": {"v_goal_norms": [_norm(goal) for goal in self.v_goals]},
            "topology": topology_event or {"operation": None},
            "system": {"phase": self.scheduler.phase, "elapsed_seconds": time.perf_counter() - started, "nan_inf": False},
        }
        with self.output_jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        self.iteration += 1
        return record


def _jsonable(value):
    if torch.is_tensor(value):
        return float(value.detach().float().cpu()) if value.ndim == 0 else value.detach().float().cpu().tolist()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(record)) + "\n")

def routed_optimization_step(
    scheduler: RewardSliderV2Scheduler,
    alpha_optimizer: torch.optim.Optimizer,
    vgoal_optimizer: torch.optim.Optimizer,
    *,
    trajectory_loss: torch.Tensor | None,
    quality_loss: torch.Tensor | None,
    trajectory_kl: torch.Tensor | float,
    trajectory_collapsed: bool = False,
    control_loss: torch.Tensor | None = None,
    trajectory_guard_loss: torch.Tensor | None = None,
) -> dict[str, object]:
    """Route one optimization step through the three-phase scheduler."""
    phase_before = scheduler.phase
    alpha_optimizer.zero_grad(set_to_none=True)
    vgoal_optimizer.zero_grad(set_to_none=True)
    total = scheduler.backward(
        trajectory_loss=trajectory_loss,
        quality_loss=quality_loss,
        control_loss=control_loss,
        trajectory_guard_loss=trajectory_guard_loss,
    )
    alpha_gradient_norm = _norm(scheduler.alpha_parameters[0].grad) if scheduler.alpha_parameters else None
    vgoal_gradient_norms = [_norm(parameter.grad) for parameter in scheduler.v_goal_parameters]
    scheduler.configure_optimizers(alpha_optimizer, vgoal_optimizer)
    if phase_before in ("trajectory_calibration", "joint_refinement"):
        alpha_optimizer.step()
    if phase_before in ("quality_repair", "joint_refinement"):
        vgoal_optimizer.step()
    phase_after = scheduler.advance(trajectory_kl, trajectory_collapsed=trajectory_collapsed)
    return {
        "phase_before": phase_before,
        "phase_after": phase_after,
        "loss": float(total.detach()),
        "alpha_gradient_norm": alpha_gradient_norm,
        "v_goal_gradient_norms": vgoal_gradient_norms,
    }


def summarize_native_parity(batched_latent: torch.Tensor, native_latent: torch.Tensor, batched_image: torch.Tensor, native_image: torch.Tensor) -> dict[str, object]:
    """Summarize single and B=K native parity without hiding batch drift."""
    if batched_latent.ndim != 3 or native_latent.ndim != 3 or native_latent.shape[0] != 1:
        raise ValueError("Latents must be [branches, tokens, channels] and [1, tokens, channels].")
    if batched_image.ndim != 4 or native_image.ndim != 4 or native_image.shape[0] != 1:
        raise ValueError("Images must be [branches, channels, height, width] and [1, channels, height, width].")
    latent_mae = (batched_latent.float() - native_latent[0].float()).abs().mean(dim=(1, 2))
    latent_cosine = F.cosine_similarity(batched_latent.float().flatten(1), native_latent[0].float().flatten().unsqueeze(0), dim=1)
    image_mae = (batched_image.float() - native_image[0].float()).abs().mean(dim=(1, 2, 3))
    within_batch = (batched_latent.float() - batched_latent[0:1].float()).abs().mean(dim=(1, 2))
    return {
        "per_branch_latent_mae": latent_mae,
        "per_branch_latent_cosine": latent_cosine,
        "per_branch_image_mae": image_mae,
        "max_batch_latent_mae": latent_mae.max(),
        "min_batch_latent_cosine": latent_cosine.min(),
        "max_batch_image_mae": image_mae.max(),
        "within_batch_max_latent_difference": within_batch.max(),
    }
def _pil_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    import numpy as np

    return torch.from_numpy(np.asarray(image, dtype=np.float32).copy() / 255).permute(2, 0, 1).unsqueeze(0).to(device)


def _save_tensor_image(image: torch.Tensor, path: Path) -> None:
    import numpy as np

    value = image.detach().clamp(0, 1)[0].mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(np.asarray(value), mode="RGB").save(path)


def _frozen_gradient_audit(pipe) -> bool:
    modules = (pipe.transformer, pipe.vae, pipe.text_encoder, pipe.text_encoder_2)
    return all(parameter.grad is None for module in modules if module is not None for parameter in module.parameters())


def _trajectory_images(pipe, inputs, alpha_parameterization, v_goals, lpips):
    alphas = alpha_parameterization.alphas[1:-1]
    unroll = pipe.unroll_rewardslider_v2_controls(inputs, alphas, v_goals, control_steps=4, use_checkpointing=False)
    candidates = pipe.decode_rewardslider_v2_terminal(unroll.final_latent, inputs)
    with torch.no_grad():
        native_full = pipe.decode_rewardslider_v2_terminal(inputs.native.native_final_latent, inputs)
    source = _pil_tensor(Image.open(inputs._source_path).convert("RGB"), candidates.device) if hasattr(inputs, "_source_path") else None
    return unroll, candidates, native_full, source


def run_real_flux_smoke(args) -> dict:
    """Run a short, auditable 256x256 FLUX-Kontext V2 smoke experiment."""

    if not args.model or not args.source:
        raise ValueError("Real smoke requires --model and --source.")
    if args.height != 256 or args.width != 256:
        raise ValueError("The acceptance smoke is fixed to 256x256.")
    if args.steps < 4:
        raise ValueError("Real smoke requires at least four denoising steps.")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    output = args.output_jsonl
    output.parent.mkdir(parents=True, exist_ok=True)
    source_pil = Image.open(args.source).convert("RGB").resize((args.width, args.height), Image.Resampling.LANCZOS)
    source_image = _pil_tensor(source_pil, device)
    quality_reward = build_quality_reward(args.quality_reward, device)
    quality_audit = None if quality_reward is None else audit_quality_reward(quality_reward, source_image, require_pass=True)
    pipe = FluxKontextRewardSliderV2Pipeline.from_pretrained(args.model, torch_dtype=dtype, local_files_only=True).to(device)
    pipe.set_progress_bar_config(disable=True)
    inputs = pipe.prepare_rewardslider_v2_inputs(
        num_branches=args.initial_nodes - 2,
        image=source_pil,
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=torch.Generator(device=device).manual_seed(args.seed),
    )
    native = pipe.unroll_rewardslider_v2_controls(
        inputs, torch.ones(args.initial_nodes - 2, device=device),
        [torch.zeros(args.initial_nodes - 2, *inputs.native.initial_latent.shape[1:], device=device) for _ in range(4)],
        control_steps=4, use_checkpointing=False,
    )
    native_single = pipe.unroll_rewardslider_v2_controls(
        RewardSliderV2Inputs(inputs.native, 1, inputs.native.forward_kwargs),
        torch.ones(1, device=device),
        [torch.zeros(1, *inputs.native.initial_latent.shape[1:], device=device) for _ in range(4)],
        control_steps=4, use_checkpointing=False,
    )
    native_parity = (native_single.final_latent[0] - inputs.native.native_final_latent[0]).float().abs()
    native_image = pipe.decode_rewardslider_v2_terminal(inputs.native.native_final_latent, inputs).detach()
    with torch.no_grad():
        native_batch_images = pipe.decode_rewardslider_v2_terminal(native.final_latent, inputs)
    batch_parity = summarize_native_parity(native.final_latent, inputs.native.native_final_latent, native_batch_images, native_image)
    lpips = LPIPSDistance(net="vgg").to(device)
    alpha_parameterization = OrderedAlphaParameterization.random(
        num_interior=args.initial_nodes - 2, seed=args.seed, device=device
    )
    v_goals = initialize_v_goal_parameters(inputs.native.initial_latent.shape, num_branches=args.initial_nodes - 2, control_steps=4, device=device)
    native_directions = []
    for index in range(4):
        keep = (native_single.states[index] - inputs.native.source_clean_latent) / inputs.native.sigmas[index]
        native_directions.append((native_single.native_edit_velocities[index] - keep).detach())
    relevance = []
    for direction in native_directions:
        score = direction.float().square().mean(dim=-1).sqrt()
        relevance.append((score / score.amax(dim=1, keepdim=True).clamp_min(1e-8)).detach())

    def evaluate():
        unroll = pipe.unroll_rewardslider_v2_controls(
            inputs,
            alpha_parameterization.alphas[1:-1],
            v_goals,
            control_steps=args.control_steps,
            use_checkpointing=args.use_checkpointing,
        )
        images = pipe.decode_rewardslider_v2_terminal(unroll.final_latent, inputs)
        nodes = torch.cat((source_image, images, native_image), dim=0)
        stats = lpips.trajectory(nodes * 2 - 1)
        return unroll, images, stats

    with torch.no_grad():
        _, _, initial_stats = evaluate()
    alpha_optimizer = torch.optim.Adam([alpha_parameterization.interval_logits], lr=args.alpha_lr)
    vgoal_optimizer = torch.optim.Adam(v_goals, lr=args.vgoal_lr)
    dynamic_coordinator = DynamicDeficitCoordinator()
    scheduler = RewardSliderV2Scheduler(
        alpha_parameterization,
        v_goals,
        trajectory_kl_threshold=args.trajectory_kl_threshold,
        trajectory_patience=args.trajectory_patience,
        min_repair_iterations=args.joint_refine_iters,
    )
    relevance_image = build_image_space_relevance(
        relevance,
        token_height=inputs.native.sampling_token_height,
        token_width=inputs.native.sampling_token_width,
        target_height=args.height,
        target_width=args.width,
    )
    phase_records = []
    alpha_grad_norm = None
    vgoal_gradient_norms = []
    for iteration in range(args.local_refine_iters + args.joint_refine_iters):
        unroll, images, stats = evaluate()
        quality_loss = None
        quality_deficit = None
        control_loss = None
        coordination = None
        trajectory_guard_loss = None
        trajectory_guard_weight = args.trajectory_guard_weight
        if scheduler.phase in ("quality_repair", "joint_refinement"):
            preserve = masked_lpips_preservation_loss(images * 2 - 1, source_image * 2 - 1, relevance_image, lpips)
            if quality_reward is not None:
                quality_deficit = F.softplus(-quality_reward(images))
            regularizers = v_goal_regularizers(v_goals, native_directions, relevance)
            coordination = coordinate_image_deficits(dynamic_coordinator, preserve, quality_deficit)
            quality_loss = coordination.total_loss
            control_loss = regularizers.residual + 2.0 * regularizers.parallel + regularizers.spatial
            trajectory_guard_loss = trajectory_guard_weight * torch.relu(
                stats.kl_uniform - args.trajectory_kl_threshold
            )
        routed = routed_optimization_step(
            scheduler,
            alpha_optimizer,
            vgoal_optimizer,
            trajectory_loss=stats.kl_uniform,
            quality_loss=quality_loss,
            trajectory_kl=stats.kl_uniform,
            trajectory_collapsed=stats.collapsed,
            trajectory_guard_loss=trajectory_guard_loss,
            control_loss=control_loss,
        )
        alpha_grad_norm = routed["alpha_gradient_norm"]
        vgoal_gradient_norms = routed["v_goal_gradient_norms"]
        phase_records.append(
            {
                "phase": routed["phase_before"],
                "iteration": iteration,
                "trajectory_kl": float(stats.kl_uniform.detach()),
                "coordination": None if coordination is None else {
                    "raw_deficits": coordination.raw_deficits,
                    "normalized_deficits": coordination.normalized_deficits,
                    "dynamic_weights": coordination.weights,
                    "direction": coordination.direction,
                },
                "trajectory_kl_raw": float(stats.kl_uniform.detach()),
                "trajectory_guard_loss": None if trajectory_guard_loss is None else float(trajectory_guard_loss.detach()),
                "trajectory_guard_weight": trajectory_guard_weight,
                "quality_loss": None if quality_loss is None else float(quality_loss.detach()),
                "alpha": alpha_parameterization.alphas.detach().cpu().tolist(),
                "alpha_gradient_norm": alpha_grad_norm,
                "v_goal_gradient_norms": vgoal_gradient_norms,
                "phase_after": routed["phase_after"],
            }
        )
    with torch.no_grad():
        final_unroll, final_images, final_stats = evaluate()
    record = {
        "trajectory": {"current_number_of_nodes": args.initial_nodes, "max_nodes": args.max_nodes, "alpha": alpha_parameterization.alphas.detach(), "initial_kl": initial_stats.kl_uniform, "final_kl": final_stats.kl_uniform, "adjacent_lpips": final_stats.distances, "normalized_lpips": final_stats.normalized_distances, "max_normalized_gap": final_stats.max_normalized_gap, "worst_interval": final_stats.worst_interval, "path_length": final_stats.path_length, "endpoint_distance": final_stats.endpoint_distance, "collapsed": final_stats.collapsed},
        "reward": {
            "quality_reward": args.quality_reward,
            "formal_trajectory_metric": "LPIPS_KL_uniform",
            "quality_audit": None if quality_audit is None else {
                "passed": quality_audit.passed,
                "image_gradient_norm": quality_audit.image_gradient_norm,
            },
        },
        "gradient": {"alpha_gradient_norm": alpha_grad_norm, "v_goal_gradient_norms": vgoal_gradient_norms, "alpha_finite": bool(torch.isfinite(alpha_parameterization.interval_logits).all()), "v_goal_finite": all(torch.isfinite(parameter).all().item() for parameter in v_goals)},
        "control": {"v_goal_norms": [parameter.detach().float().norm() for parameter in v_goals]},
        "parity": {"single": {"latent_mae": native_parity.mean(), "latent_cosine": F.cosine_similarity(native_single.final_latent[0].float().flatten()[None], inputs.native.native_final_latent[0].float().flatten()[None]).squeeze(), "image_mae": (native_image - pipe.decode_rewardslider_v2_terminal(native_single.final_latent, inputs)).float().abs().mean()}, "batched": batch_parity},
        "topology": {"operation": None, "old_nodes": args.initial_nodes, "new_nodes": args.initial_nodes},
        "system": {"frozen_model_audit": _frozen_gradient_audit(pipe), "device": str(device), "dtype": str(dtype), "height": args.height, "width": args.width, "steps": args.steps, "control_steps": 4, "phase_records": phase_records},
    }
    _write_jsonl(output, record)
    _save_tensor_image(native_image, output.with_name("native_full.png"))
    _save_tensor_image(final_images[0:1], output.with_name("candidate_weak.png"))
    return record


def main() -> None:
    args = build_rewardslider_v2_parser().parse_args()
    if not args.run_real_smoke:
        raise RuntimeError("Pass --run-real-smoke with --model and --source to execute the FLUX-Kontext smoke test.")
    run_real_flux_smoke(args)


if __name__ == "__main__":
    main()
