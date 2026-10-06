"""Masked, reward-guided backward control on the first native trajectory steps."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .backward_reward import BackwardReward
from .trajectory_gate import (
    GateDecision,
    TrajectoryGateConfig,
    TrajectoryState,
    assess_candidate,
    rejection_counts,
    stop_reason_for_rejections,
)
from .velocity_control import scale_gradient_to_global_native_ratio
from rewardflow_calibration.rollout.veloedit import (
    PreparedVeloEdit,
    VeloEditCompatibleRollout,
    VeloEditRolloutConfig,
)


@dataclass(frozen=True)
class BackwardTrajectoryConfig:
    goal_steps: int = 4
    max_iterations: int = 16
    line_search_ratios: tuple[float, ...] = (0.05, 0.02, 0.01, 0.005, 0.002)
    max_total_residual_ratio: float = 0.10
    gradient_epsilon: float = 1e-10
    sourceward_tolerance: float = 1e-5

    def validate(self, total_steps: int) -> None:
        if not 1 <= self.goal_steps <= total_steps:
            raise ValueError("goal_steps must be in [1, total rollout steps]")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if not self.line_search_ratios or any(
            not np.isfinite(value) or value <= 0 for value in self.line_search_ratios
        ):
            raise ValueError("line_search_ratios must contain finite positive values")
        if self.max_total_residual_ratio <= 0 or self.gradient_epsilon <= 0:
            raise ValueError("residual cap and gradient epsilon must be positive")
        if self.sourceward_tolerance < 0:
            raise ValueError("sourceward_tolerance must be non-negative")


def freeze_native_edit_mask(
    velocity_trace: list[dict[str, torch.Tensor]], goal_steps: int
) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    """Freeze elementwise native low-similarity masks and native velocity RMS."""
    if len(velocity_trace) < goal_steps:
        raise ValueError("native velocity trace does not cover every controlled timestep")
    masks: list[torch.Tensor] = []
    velocity_rms: list[float] = []
    for index, row in enumerate(velocity_trace[:goal_steps]):
        mask = row.get("hard_edit_mask", row.get("low_similarity_mask"))
        if mask is None:
            raise ValueError(f"native velocity trace step {index} has no elementwise hard edit mask")
        mask = torch.as_tensor(mask).detach().bool()
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)
        if mask.ndim != 2:
            raise ValueError("each hard edit mask must have shape [tokens, channels]")
        masks.append(mask)
        value = float(row["native_velocity_rms"].detach().float().cpu())
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"native velocity RMS is invalid at step {index}")
        velocity_rms.append(value)
    frozen = torch.stack(masks).detach().clone()
    keep = (~frozen).detach().clone()
    return frozen, keep, velocity_rms


def freeze_native_control_context(velocity_trace: list[dict[str, torch.Tensor]], goal_steps: int):
    """Freeze hard masks, per-step RMS, and the exact native velocities."""
    hard, keep, rms = freeze_native_edit_mask(velocity_trace, goal_steps)
    velocities = []
    for i, row in enumerate(velocity_trace[:goal_steps]):
        velocity = row.get("native_velocity")
        if velocity is None:
            raise ValueError(f"native velocity trace step {i} has no native_velocity")
        velocity = torch.as_tensor(velocity).detach().float()
        if velocity.ndim == 3 and velocity.shape[0] == 1:
            velocity = velocity.squeeze(0)
        if velocity.shape != hard[i].shape or not torch.isfinite(velocity).all():
            raise ValueError(f"native velocity invalid or mismatched at step {i}")
        velocities.append(velocity.clone())
    return hard, keep, rms, torch.stack(velocities).detach()


def apply_frozen_edit_mask(gradient: torch.Tensor, hard_edit_mask: torch.Tensor) -> torch.Tensor:
    """Apply a previously frozen elementwise mask without changing time weighting."""
    if gradient.shape != hard_edit_mask.shape:
        raise ValueError("gradient and frozen hard_edit_mask must have identical shapes")
    return gradient * hard_edit_mask.to(device=gradient.device, dtype=gradient.dtype)


def build_image_space_masks(
    hard_edit_mask: torch.Tensor,
    *,
    height: int,
    width: int,
    vae_scale_factor: int,
    latent_ids: torch.Tensor | None = None,
    pack_factor: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Restore Flux 2x2-packed VAE tokens to an image-space soft edit mask.

    Raises rather than guessing if the VAE/packing grid or token ordering differs.
    """
    if hard_edit_mask.ndim != 3:
        raise ValueError("hard_edit_mask must have shape [steps, tokens, channels]")
    divisor = vae_scale_factor * pack_factor
    if height % divisor or width % divisor:
        raise ValueError("image dimensions are not divisible by VAE scale × Flux packing factor")
    grid_h, grid_w = height // divisor, width // divisor
    token_count = hard_edit_mask.shape[1]
    if grid_h * grid_w != token_count:
        raise ValueError(
            f"packed latent token count mismatch: got {token_count}, expected {grid_h}x{grid_w}={grid_h * grid_w}"
        )
    if latent_ids is not None:
        ids = torch.as_tensor(latent_ids).detach().cpu()
        if ids.ndim == 3 and ids.shape[0] == 1:
            ids = ids.squeeze(0)
        if ids.ndim != 2 or ids.shape[1] < 3 or ids.shape[0] not in (token_count, 2 * token_count):
            raise ValueError("latent_ids do not match packed token count/grid")
        expected = torch.stack(torch.meshgrid(
            torch.arange(grid_h), torch.arange(grid_w), indexing="ij"
        ), dim=-1).reshape(-1, 2)
        for offset in range(0, ids.shape[0], token_count):
            if not torch.equal(ids[offset:offset + token_count, -2:].long(), expected):
                raise ValueError("latent_ids ordering does not match the expected row-major packed grid")
    token_probability = hard_edit_mask.float().mean(dim=(0, 2))
    token_grid = token_probability.reshape(1, 1, grid_h, grid_w)
    edit = F.interpolate(token_grid, size=(height, width), mode="bilinear", align_corners=False)
    edit = edit.clamp(0, 1).detach()
    keep = (1.0 - edit).detach()
    return edit, keep


def _save_image(image: torch.Tensor, path: Path) -> None:
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
        raise ValueError("saved trajectory images must have shape [1, 3, H, W]")
    pixels = image[0].detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    Image.fromarray(np.rint(pixels * 255).astype(np.uint8), mode="RGB").save(path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _append_jsonl(path: Path, row: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_safe(row), ensure_ascii=False) + "\n")


class BackwardTrajectoryOptimizer:
    """No-optimizer-state gradient descent with mask, global trust scale, and gate."""

    def __init__(
        self,
        rollout: VeloEditCompatibleRollout,
        *,
        prepared: PreparedVeloEdit,
        source_image: torch.Tensor,
        native_full_image: torch.Tensor,
        native_proxy_image: torch.Tensor,
        reward: BackwardReward,
        hard_edit_mask: torch.Tensor,
        image_keep_mask: torch.Tensor,
        native_velocity_rms_per_step: list[float] | torch.Tensor,
        rollout_config: VeloEditRolloutConfig,
        native_velocity_per_step: torch.Tensor | None = None,
        config: BackwardTrajectoryConfig | None = None,
        gate_config: TrajectoryGateConfig | None = None,
    ) -> None:
        self.rollout = rollout
        self.prepared = prepared
        self.source = source_image.detach()
        self.native_full = native_full_image.detach()
        self.native_proxy = native_proxy_image.detach()
        self.reward = reward
        self.hard_edit_mask = hard_edit_mask.detach().bool().clone()
        self.image_keep_mask = image_keep_mask.detach().float().clone()
        self.native_rms = torch.as_tensor(
            native_velocity_rms_per_step, device=rollout.device, dtype=torch.float32
        ).flatten().detach()
        self.native_velocity = (
            None
            if native_velocity_per_step is None
            else torch.as_tensor(native_velocity_per_step, device=rollout.device)
            .detach()
            .float()
            .clone()
        )
        self.rollout_config = rollout_config
        self.config = config or BackwardTrajectoryConfig(goal_steps=self.hard_edit_mask.shape[0])
        self.gate_config = gate_config or TrajectoryGateConfig(
            semantic_floor=reward.semantic_floor or 0.0,
            max_total_residual_ratio=self.config.max_total_residual_ratio,
            sourceward_tolerance=self.config.sourceward_tolerance,
            max_active_residual_ratio=0.5,
        )
        self.config.validate(len(prepared.sigma_schedule) - 1)
        self.gate_config.validate()
        expected = (self.config.goal_steps, *prepared.latents.shape[1:])
        if tuple(self.hard_edit_mask.shape) != expected:
            raise ValueError(f"hard edit mask shape {tuple(self.hard_edit_mask.shape)} != residual shape {expected}")
        if self.native_rms.numel() != self.config.goal_steps:
            raise ValueError("native_velocity_rms_per_step length must equal goal_steps")
        if self.native_velocity is not None and tuple(self.native_velocity.shape) != expected:
            raise ValueError("native_velocity_per_step must match residual shape")
        if self.image_keep_mask.shape != (1, 1, prepared.height, prepared.width):
            raise ValueError("image_keep_mask must be [1, 1, prepared.height, prepared.width]")
        if rollout_config.first_step_align_steps or rollout_config.preserve_steps or rollout_config.edit_steps:
            raise ValueError("backward controller requires an unmodified native rollout config")

    def _total_residual_ratios(self, residual: torch.Tensor) -> tuple[float, list[float]]:
        global_native = self.native_rms.square().mean().sqrt().clamp_min(1e-12)
        global_ratio = float(residual.detach().float().square().mean().sqrt().cpu() / global_native.cpu())
        per_step = residual.detach().float().square().mean(dim=(1, 2)).sqrt() / self.native_rms.clamp_min(1e-12)
        return global_ratio, per_step.cpu().tolist()

    def _active_residual_stats(self, residual: torch.Tensor) -> dict[str, object]:
        if self.native_velocity is None:
            return {"active_residual_rms": None, "active_native_velocity_rms": None,
                    "active_residual_native_ratio": None,
                    "active_residual_ratio_per_step": [None] * self.config.goal_steps}
        mask, native = self.hard_edit_mask.to(residual.device), self.native_velocity.to(residual.device)
        rparts, nparts, per = [], [], []
        for i in range(self.config.goal_steps):
            active = mask[i]
            if not bool(active.any()):
                per.append(None)
                continue
            r, n = residual.detach().float()[i][active], native[i][active]
            rr, nr = r.square().mean().sqrt(), n.square().mean().sqrt()
            per.append(None if float(nr.cpu()) <= 1e-12 else float((rr/nr).cpu()))
            rparts.append(r.reshape(-1)); nparts.append(n.reshape(-1))
        if not rparts:
            return {"active_residual_rms": None, "active_native_velocity_rms": None,
                    "active_residual_native_ratio": None, "active_residual_ratio_per_step": per}
        rr, nr = torch.cat(rparts).square().mean().sqrt(), torch.cat(nparts).square().mean().sqrt()
        return {
            "active_residual_rms": float(rr.cpu()),
            "active_native_velocity_rms": float(nr.cpu()),
            "active_residual_native_ratio": (
                None if float(nr.cpu()) <= 1e-12 else float((rr / nr).cpu())
            ),
            "active_residual_ratio_per_step": per,
        }

    def _measure_state(
        self,
        name: str,
        image: torch.Tensor,
        previous: TrajectoryState | None,
        residual: torch.Tensor,
        line_search_ratio: float | None = None,
        increment_stats: dict[str, object] | None = None,
        reward_values: Any | None = None,
    ) -> TrajectoryState:
        if reward_values is None:
            with torch.no_grad():
                reward_values = self.reward.evaluate(image, self.source, self.image_keep_mask)
        semantic = float(reward_values.semantic_score.detach().cpu())
        keep_l1_value = float(reward_values.keep_loss.detach().cpu())
        d_source = self.reward.dreamsim(self.source, image)
        residual_global, residual_per_step = self._total_residual_ratios(residual)
        cumulative = 0.0 if previous is None else previous.cumulative_dreamsim
        active_stats = self._active_residual_stats(residual)
        metrics = dict(increment_stats or {}); metrics.update(active_stats)
        state = TrajectoryState(
            name=name,
            image=image.detach(),
            semantic_score=semantic,
            dreamsim_to_source=d_source,
            keep_l1=keep_l1_value,
            residual_global_ratio=residual_global,
            residual_ratio_per_step=residual_per_step,
            active_residual_native_ratio=active_stats["active_residual_native_ratio"],
            active_residual_ratio_per_step=active_stats["active_residual_ratio_per_step"],
            line_search_ratio=line_search_ratio,
            cumulative_dreamsim=cumulative,
            metrics=metrics,
        )
        return state

    def _verify_final(self, accepted_states, residuals):
        proxy_sem = [float(self.reward.semantic_score(self.native_proxy).detach().cpu())]
        proxy_sem += [s.semantic_score for s in accepted_states[1:]]
        proxy_src = [self.reward.dreamsim(self.source, self.native_proxy)]
        proxy_src += [s.dreamsim_to_source for s in accepted_states[1:]]
        final_images = [self.native_full]
        for residual in residuals:
            with torch.no_grad():
                final_images.append(self.rollout.rollout_native(
                    self.prepared, config=self.rollout_config, goal_residual=residual).detach())
        final_sem, final_src = [], []
        for image in final_images:
            with torch.no_grad():
                final_sem.append(float(self.reward.semantic_score(image).detach().cpu()))
            final_src.append(self.reward.dreamsim(self.source, image))
        def ordered(values):
            return all(left >= right for left, right in zip(values, values[1:]))

        sem_pairs = list(zip(proxy_sem, proxy_sem[1:], final_sem, final_sem[1:]))
        source_pairs = list(zip(proxy_src, proxy_src[1:], final_src, final_src[1:]))
        sem_agreement = (
            sum((a >= b) == (c >= d) for a, b, c, d in sem_pairs) / len(sem_pairs)
            if sem_pairs
            else None
        )
        source_agreement = (
            sum((a >= b) == (c >= d) for a, b, c, d in source_pairs) / len(source_pairs)
            if source_pairs
            else None
        )
        return {
            "accepted_count": len(residuals), "proxy_semantic_scores": proxy_sem,
            "final_semantic_scores": final_sem, "proxy_dreamsim_to_source": proxy_src,
            "final_dreamsim_to_source": final_src,
            "proxy_final_semantic_gap": float(
                np.mean(np.abs(np.asarray(proxy_sem) - np.asarray(final_sem)))
            ),
            "proxy_final_dreamsim_gap": float(
                np.mean(np.abs(np.asarray(proxy_src) - np.asarray(final_src)))
            ),
            "adjacent_semantic_order_agreement": sem_agreement,
            "adjacent_source_distance_order_agreement": source_agreement,
            "proxy_semantic_ordered": ordered(proxy_sem),
            "final_semantic_ordered": ordered(final_sem),
            "proxy_source_distance_ordered": ordered(proxy_src),
            "final_source_distance_ordered": ordered(final_src),
            "proxy_final_mismatch": any(
                value is not None and value < 1.0 - 1e-6
                for value in (sem_agreement, source_agreement)
            ),
            "final_images": final_images,
        }

    def run(
        self,
        output_dir: str | Path,
        *,
        verify_final: bool = False,
        save_debug_tensors: bool = False,
    ) -> dict[str, object]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        trajectory_path = output / "trajectory.jsonl"
        trajectory_path.write_text("", encoding="utf-8")
        zero = torch.zeros_like(self.hard_edit_mask, dtype=torch.float32, device=self.rollout.device)
        current_residual = zero
        baseline_proxy = self._measure_state("native_proxy", self.native_proxy, None, zero)
        accepted_states = [baseline_proxy]
        accepted_residuals: list[torch.Tensor] = []
        _save_image(self.source, output / "source.png")
        _save_image(self.native_full, output / "native_full.png")
        _save_image(self.native_proxy, output / "native_proxy.png")
        _append_jsonl(trajectory_path, {
            "event": "native_full",
            "semantic_score": self.reward.semantic_full,
            "dreamsim_to_source": self.reward.anchor_diagnostics.get("dreamsim_source_native_full"),
            "image": str((output / "native_full.png").resolve()),
        })
        _append_jsonl(trajectory_path, {
            "event": "native_proxy",
            "semantic_score": baseline_proxy.semantic_score,
            "dreamsim_to_source": baseline_proxy.dreamsim_to_source,
            "dreamsim_to_previous": self.reward.anchor_diagnostics.get("dreamsim_native_proxy_native_full"),
            "image": str((output / "native_proxy.png").resolve()),
            "residual_global_ratio": 0.0,
        })

        rejection_events: list[GateDecision] = []
        iteration_diagnostics: list[dict[str, object]] = []
        stop_reason = "max_iterations"
        previous_masked_gradient: torch.Tensor | None = None

        if float(self.hard_edit_mask.float().mean().cpu()) <= 0:
            stop_reason = "mask_gradient_dead"
        else:
            for iteration in range(self.config.max_iterations):
                residual_var = current_residual.detach().clone().requires_grad_(True)
                velocity_trace: list[dict[str, torch.Tensor]] = []
                proxy = self.rollout.rollout_native(
                    self.prepared,
                    config=self.rollout_config,
                    goal_residual=residual_var,
                    early_stop_steps=self.config.goal_steps,
                    velocity_trace=velocity_trace,
                )
                guide = self.reward.evaluate(proxy, self.source, self.image_keep_mask)
                raw_gradient = torch.autograd.grad(guide.total, residual_var)[0]
                if not torch.isfinite(raw_gradient).all():
                    stop_reason = "gradient_dead"
                    break
                raw_rms = raw_gradient.float().square().mean().sqrt()
                if float(raw_rms.detach().cpu()) <= self.config.gradient_epsilon:
                    stop_reason = "gradient_dead"
                    break
                masked_gradient = apply_frozen_edit_mask(
                    raw_gradient.float(), self.hard_edit_mask.to(raw_gradient.device).float()
                )
                masked_rms = masked_gradient.square().mean().sqrt()
                active_gradient = masked_gradient[self.hard_edit_mask.to(masked_gradient.device)]
                active_gradient_rms = (
                    active_gradient.square().mean().sqrt()
                    if active_gradient.numel()
                    else masked_rms.new_zeros(())
                )
                if (
                    float(masked_rms.detach().cpu()) <= self.config.gradient_epsilon
                    or float(active_gradient_rms.detach().cpu()) <= self.config.gradient_epsilon
                ):
                    stop_reason = "mask_gradient_dead"
                    break
                retained_energy = masked_gradient.square().sum() / raw_gradient.float().square().sum().clamp_min(1e-20)
                per_step_gradient_rms = masked_gradient.square().mean(dim=(1, 2)).sqrt()
                cosine = None
                if previous_masked_gradient is not None:
                    cosine = float(F.cosine_similarity(
                        masked_gradient.flatten(), previous_masked_gradient.flatten(), dim=0
                    ).detach().cpu())
                trace_rows = velocity_trace[:self.config.goal_steps]
                native_rms_observed = [float(row["native_velocity_rms"].detach().float().cpu()) for row in trace_rows]
                gradient_record: dict[str, object] = {
                    "event": "gradient",
                    "iteration": iteration + 1,
                    **guide.diagnostics(),
                    "raw_gradient_rms": float(raw_rms.detach().cpu()),
                    "masked_gradient_rms": float(masked_rms.detach().cpu()),
                    "active_gradient_rms": float(active_gradient_rms.detach().cpu()),
                    "masked_raw_gradient_energy_ratio": float(retained_energy.detach().cpu()),
                    "gradient_rms_per_step": per_step_gradient_rms.detach().cpu().tolist(),
                    "gradient_cosine_with_previous": cosine,
                    "native_velocity_rms_per_step": native_rms_observed,
                    "mask_coverage_per_step": self.hard_edit_mask.float().mean(dim=(1, 2)).cpu().tolist(),
                    "mask_coverage": float(self.hard_edit_mask.float().mean().cpu()),
                }
                _append_jsonl(trajectory_path, gradient_record)
                iteration_diagnostics.append(gradient_record)
                if save_debug_tensors:
                    torch.save({
                        "raw_gradient": raw_gradient.detach().cpu(),
                        "masked_gradient": masked_gradient.detach().cpu(),
                    }, output / f"debug_gradient_{iteration + 1:03d}.pt")

                decisions: list[GateDecision] = []
                candidates: list[tuple[TrajectoryState, torch.Tensor, GateDecision, dict[str, object]]] = []
                direction = -masked_gradient.detach()
                for trial_index, requested_ratio in enumerate(self.config.line_search_ratios, start=1):
                    increment, increment_info = scale_gradient_to_global_native_ratio(
                        direction, self.native_rms, requested_ratio
                    )
                    trial_residual = current_residual + increment.to(current_residual)
                    residual_global, residual_per_step = self._total_residual_ratios(trial_residual)
                    active_inc = self._active_residual_stats(increment)
                    active_total = self._active_residual_stats(trial_residual)
                    trial_proxy = self.rollout.rollout_native(
                        self.prepared,
                        config=self.rollout_config,
                        goal_residual=trial_residual,
                        early_stop_steps=self.config.goal_steps,
                    )
                    with torch.no_grad():
                        trial_values = self.reward.evaluate(trial_proxy, self.source, self.image_keep_mask)
                    increment_stats = {
                        "actual_increment_global_ratio": increment_info["actual_global_ratio"],
                        "actual_increment_ratio_per_step": increment_info["actual_ratio_per_step"],
                        "scale_factor": increment_info["scale_factor"],
                        "active_increment_rms": active_inc["active_residual_rms"],
                        "active_native_velocity_rms": active_inc["active_native_velocity_rms"],
                        "active_increment_native_ratio": active_inc["active_residual_native_ratio"],
                        "active_total_residual_rms": active_total["active_residual_rms"],
                        "active_total_residual_native_ratio": active_total["active_residual_native_ratio"],
                        "active_residual_ratio_per_step": active_total["active_residual_ratio_per_step"],
                    }
                    state = self._measure_state(
                        f"trial_{iteration + 1:03d}_{trial_index:02d}",
                        trial_proxy,
                        accepted_states[-1],
                        trial_residual,
                        requested_ratio,
                        increment_stats,
                        reward_values=trial_values,
                    )
                    decision = assess_candidate(
                        accepted_states[-1], state, accepted_states,
                        self.reward.dreamsim_distance, self.gate_config,
                    )
                    decisions.append(decision)
                    rejection_events.append(decision)
                    trial_record = {
                        "event": "trial",
                        "iteration": iteration + 1,
                        "trial": trial_index,
                        **decision.metrics,
                        "guide_total": float(trial_values.total.detach().cpu()),
                        "source_loss": float(trial_values.source_loss.detach().cpu()),
                        "semantic_gate_loss": float(trial_values.semantic_gate_loss.detach().cpu()),
                        "keep_loss": float(trial_values.keep_loss.detach().cpu()),
                        "accepted": decision.accepted,
                        "rejection_reasons": decision.reasons,
                    }
                    _append_jsonl(trajectory_path, trial_record)
                    if decision.accepted:
                        candidates.append((state, trial_residual.detach(), decision, trial_record))

                if not candidates:
                    stop_reason = stop_reason_for_rejections(decisions)
                    break
                chosen, next_residual, chosen_decision, chosen_record = min(
                    candidates,
                    key=lambda item: (
                        item[0].dreamsim_to_source,
                        float(item[0].metrics["actual_increment_global_ratio"]),
                    ),
                )
                chosen.name = f"accepted_{len(accepted_residuals) + 1:03d}"
                chosen.cumulative_dreamsim = accepted_states[-1].cumulative_dreamsim + float(
                    chosen_decision.metrics["dreamsim_to_previous"]
                )
                accepted_states.append(chosen)
                accepted_residuals.append(next_residual.clone())
                current_residual = next_residual
                previous_masked_gradient = masked_gradient.detach().clone()
                image_path = output / f"accepted_proxy_{len(accepted_residuals):03d}.png"
                residual_path = output / f"accepted_residual_{len(accepted_residuals):03d}.pt"
                _save_image(chosen.image, image_path)
                torch.save(current_residual.detach().cpu(), residual_path)
                chosen_record.update({
                    "event": "accepted",
                    "name": chosen.name,
                    "gradient_diagnostics": gradient_record,
                    "image": str(image_path.resolve()),
                    "residual": str(residual_path.resolve()),
                    "cumulative_dreamsim": chosen.cumulative_dreamsim,
                })
                _append_jsonl(trajectory_path, chosen_record)

        verify: dict[str, object] | None = None
        if verify_final:
            verify = self._verify_final(accepted_states, accepted_residuals)
            final_images = verify.pop("final_images")
            for index, image in enumerate(final_images[1:], start=1):
                _save_image(image, output / f"accepted_final_{index:03d}.png")
        summary: dict[str, object] = {
            "stop_reason": stop_reason,
            "accepted_count": len(accepted_residuals),
            "iterations_completed": len(iteration_diagnostics),
            "rejection_reason_counts": rejection_counts(rejection_events),
            "mask_coverage": float(self.hard_edit_mask.float().mean().cpu()),
            "mask_coverage_per_step": self.hard_edit_mask.float().mean(dim=(1, 2)).cpu().tolist(),
            "native_velocity_rms_per_step": self.native_rms.cpu().tolist(),
            "final_residual_global_ratio": self._total_residual_ratios(current_residual)[0],
            "final_residual_ratio_per_step": self._total_residual_ratios(current_residual)[1],
            "max_active_residual_ratio": self.gate_config.max_active_residual_ratio,
            "final_active_residual_native_ratio": self._active_residual_stats(current_residual)["active_residual_native_ratio"],
            "accepted_states": [
                {
                    "name": state.name,
                    "semantic_score": state.semantic_score,
                    "dreamsim_to_source": state.dreamsim_to_source,
                    "keep_l1": state.keep_l1,
                    "line_search_ratio": state.line_search_ratio,
                    "residual_global_ratio": state.residual_global_ratio,
                    "residual_ratio_per_step": state.residual_ratio_per_step,
                    "active_residual_native_ratio": state.active_residual_native_ratio,
                    "active_residual_ratio_per_step": state.active_residual_ratio_per_step,
                    "cumulative_dreamsim": state.cumulative_dreamsim,
                }
                for state in accepted_states[1:]
            ],
            "gradient_iterations": iteration_diagnostics,
            "proxy_final_verification": verify,
        }
        (output / "summary.json").write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
        return summary
