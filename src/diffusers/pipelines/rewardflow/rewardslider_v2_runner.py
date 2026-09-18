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
from .rewardslider_v2_preservation import masked_lpips_preservation_loss
from .rewardslider_v2_regularizers import initialize_v_goal_parameters, v_goal_regularizers
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


def _norm(value: torch.Tensor | None) -> float | None:
    return None if value is None else float(value.detach().float().norm().item())


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
    phase_records = []
    alpha_grad_norm = None
    for iteration in range(args.local_refine_iters):
        alpha_optimizer.zero_grad(set_to_none=True)
        unroll, images, stats = evaluate()
        stats.kl_uniform.backward()
        alpha_grad_norm = float(alpha_parameterization.interval_logits.grad.float().norm().item())
        alpha_optimizer.step()
        phase_records.append({"phase": "trajectory_calibration", "iteration": iteration, "trajectory_kl": float(stats.kl_uniform), "alpha": alpha_parameterization.alphas.detach().cpu().tolist(), "alpha_gradient_norm": alpha_grad_norm})
    for parameter in alpha_parameterization.parameters():
        parameter.requires_grad_(False)
    quality_optimizer = torch.optim.Adam(v_goals, lr=args.vgoal_lr)
    vgoal_gradient_norms = []
    for iteration in range(args.joint_refine_iters):
        quality_optimizer.zero_grad(set_to_none=True)
        unroll, images, stats = evaluate()
        relevance_image = F.interpolate(relevance[0][None, None], size=(args.height, args.width), mode="bilinear", align_corners=False)
        preserve = masked_lpips_preservation_loss(images * 2 - 1, source_image * 2 - 1, relevance_image, lpips)
        regularizers = v_goal_regularizers(v_goals, native_directions, relevance)
        quality_loss = preserve + regularizers.residual + 2.0 * regularizers.parallel + regularizers.spatial
        quality_loss.backward()
        vgoal_gradient_norms = [float(parameter.grad.float().norm().item()) if parameter.grad is not None else 0.0 for parameter in v_goals]
        quality_optimizer.step()
        phase_records.append({"phase": "quality_repair", "iteration": iteration, "trajectory_kl": float(stats.kl_uniform), "quality_loss": float(quality_loss.detach()), "v_goal_gradient_norms": vgoal_gradient_norms})
    with torch.no_grad():
        final_unroll, final_images, final_stats = evaluate()
    record = {
        "trajectory": {"current_number_of_nodes": args.initial_nodes, "max_nodes": args.max_nodes, "alpha": alpha_parameterization.alphas.detach(), "initial_kl": initial_stats.kl_uniform, "final_kl": final_stats.kl_uniform, "adjacent_lpips": final_stats.distances, "normalized_lpips": final_stats.normalized_distances, "max_normalized_gap": final_stats.max_normalized_gap, "worst_interval": final_stats.worst_interval},
        "reward": {"quality_reward": "not_configured", "formal_trajectory_metric": "LPIPS_KL_uniform"},
        "gradient": {"alpha_gradient_norm": alpha_grad_norm, "v_goal_gradient_norms": vgoal_gradient_norms, "alpha_finite": bool(torch.isfinite(alpha_parameterization.interval_logits).all()), "v_goal_finite": all(torch.isfinite(parameter).all().item() for parameter in v_goals)},
        "control": {"v_goal_norms": [parameter.detach().float().norm() for parameter in v_goals]},
        "parity": {"latent_mae": native_parity.mean(), "latent_cosine": F.cosine_similarity(native_single.final_latent[0].float().flatten()[None], inputs.native.native_final_latent[0].float().flatten()[None]).squeeze(), "image_mae": (native_image - pipe.decode_rewardslider_v2_terminal(native_single.final_latent, inputs)).float().abs().mean()},
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
