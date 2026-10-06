"""Optimize early native-Kontext velocity residuals toward requested progress levels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from rewardflow_calibration.rollout.veloedit import PreparedVeloEdit, VeloEditCompatibleRollout, VeloEditRolloutConfig
from .goal_residual import GoalVelocityResidual
from .progress_objective import ProgressLossConfig, progress_control_loss
from .progress_reward import ProgressEstimator
from .velocity_control import normalized_feedback_update


@dataclass
class ProgressResidualResult:
    source_image: torch.Tensor
    native_full_image: torch.Tensor
    initial_proxy_image: torch.Tensor
    optimized_proxy_image: torch.Tensor
    optimized_final_image: torch.Tensor
    residual: torch.Tensor
    initial_proxy: dict[str, float | str | None]
    optimized_proxy: dict[str, float | str | None]
    optimized_final: dict[str, float | str | None]
    history: list[dict[str, object]]
    velocity_diagnostics: dict[str, object]
    projection_diagnostics: dict[str, object]
    warnings: list[str]
    best_iteration: int
    best_loss: float
    last_iteration_loss: float
    best_final_progress: float
    last_final_progress: float
    best_target_error: float
    last_target_error: float
    stop_reason: str
    converged: bool
    best_loss_progress: float
    best_loss_target_error: float
    best_loss_drift: float
    best_target_error_so_far: float
    best_target_error_iteration: int


class ProgressResidualOptimizer:
    """Per-image target-level controller on a native FLUX-Kontext trajectory."""

    def __init__(
        self,
        rollout: VeloEditCompatibleRollout,
        *,
        prepared: PreparedVeloEdit,
        native_full_image: torch.Tensor,
        progress_estimator: ProgressEstimator,
        target_strength: float,
        rollout_config: VeloEditRolloutConfig,
        loss_config: ProgressLossConfig | None = None,
        goal_steps: int = 4,
        proxy_steps: int = 4,
        learning_rate: float = 1e-3,
        iterations: int = 4,
        clip_grad_norm: float = 1.0,
        reward_mode: str = "proxy",
        target_tolerance: float = 0.03,
        patience: int = 4,
        min_improvement: float = 1e-4,
        optimizer_mode: str = "adam",
        feedback_step_size: float = 1e-3,
        auxiliary_step_size: float = 1e-3,
        progress_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        if not 0.0 <= target_strength <= 1.0:
            raise ValueError("target_strength must be in [0, 1]")
        if rollout_config.first_step_align_steps or rollout_config.preserve_steps or rollout_config.edit_steps:
            raise ValueError("native rollout config must set first_step_align_steps, preserve_steps, edit_steps to 0")
        if goal_steps < 1 or goal_steps > rollout_config.steps:
            raise ValueError("goal_steps must be in [1, rollout steps]")
        if not goal_steps <= proxy_steps <= rollout_config.steps:
            raise ValueError("proxy_steps must cover goal_steps and not exceed rollout steps")
        if iterations < 1 or learning_rate <= 0 or clip_grad_norm <= 0:
            raise ValueError("iterations, learning_rate, and clip_grad_norm must be positive")
        if reward_mode not in {"proxy", "final"}:
            raise ValueError("reward_mode must be 'proxy' or 'final'")
        if target_tolerance < 0 or patience < 1 or min_improvement < 0:
            raise ValueError("target_tolerance and min_improvement must be non-negative; patience must be positive")
        if optimizer_mode not in {"adam", "sgd", "feedback"}:
            raise ValueError("optimizer_mode must be 'adam', 'sgd', or 'feedback'")
        if feedback_step_size <= 0 or auxiliary_step_size < 0:
            raise ValueError("feedback_step_size must be positive and auxiliary_step_size non-negative")
        self.rollout = rollout
        self.prepared = prepared
        self.native_full_image = native_full_image.detach()
        self.estimator = progress_estimator
        self.target_strength = float(target_strength)
        self.config = rollout_config
        self.loss_config = loss_config or ProgressLossConfig()
        self.loss_config.validate()
        self.goal_steps = int(goal_steps)
        self.proxy_steps = int(proxy_steps)
        self.learning_rate = float(learning_rate)
        self.iterations = int(iterations)
        self.clip_grad_norm = float(clip_grad_norm)
        self.reward_mode = reward_mode
        self.target_tolerance = float(target_tolerance)
        self.patience = int(patience)
        self.min_improvement = float(min_improvement)
        self.optimizer_mode = optimizer_mode
        self.feedback_step_size = float(feedback_step_size)
        self.auxiliary_step_size = float(auxiliary_step_size)
        self.progress_callback = progress_callback

    def prepared_source_tensor(self) -> torch.Tensor:
        pixels = self.rollout.pipeline.image_processor.preprocess(
            self.prepared.working_image, self.prepared.height, self.prepared.width
        ).to(device=self.rollout.device, dtype=torch.float32)
        return ((pixels + 1.0) / 2.0).clamp(0, 1)

    @staticmethod
    def _all_finite(tensor: torch.Tensor, name: str) -> None:
        if not torch.isfinite(tensor).all():
            raise FloatingPointError(f"{name} contains NaN or Inf")

    def run(self) -> ProgressResidualResult:
        for component in self.rollout.pipeline.components.values():
            if isinstance(component, torch.nn.Module):
                component.eval()
                for parameter in component.parameters():
                    parameter.requires_grad_(False)
        self.prepared.prompt_embeds = self.prepared.prompt_embeds.detach()
        self.prepared.pooled_prompt_embeds = self.prepared.pooled_prompt_embeds.detach()
        source = self.prepared_source_tensor()
        if self.estimator.source_features is None:
            self.estimator.set_anchors(source, self.native_full_image)

        with torch.no_grad():
            initial_proxy_image = self.rollout.rollout_native(
                self.prepared, config=self.config, early_stop_steps=self.proxy_steps
            ).detach()
            initial_values = self.estimator(initial_proxy_image)
        residual = GoalVelocityResidual(
            steps=self.goal_steps,
            latent_shape=tuple(self.prepared.latents.shape[1:]),
            device=self.rollout.device,
        )
        optimizer = None
        if self.optimizer_mode == "adam":
            optimizer = torch.optim.Adam([residual.velocity], lr=self.learning_rate)
        elif self.optimizer_mode == "sgd":
            optimizer = torch.optim.SGD([residual.velocity], lr=self.learning_rate, momentum=0.0)
        history: list[dict[str, object]] = []
        best_residual: torch.Tensor | None = None
        best_loss = float("inf")
        best_iteration = 0
        stale_iterations = 0
        last_residual = residual.velocity.detach().clone()
        last_iteration_loss = float("inf")
        stop_reason = "max_iterations"
        best_loss_progress = float("nan")
        best_loss_target_error = float("nan")
        best_loss_drift = float("nan")
        best_target_error_so_far = float("inf")
        best_target_error_iteration = 0
        best_target_error_progress = float("nan")
        best_target_error_drift = float("nan")

        for iteration in range(self.iterations):
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            residual_before = residual.velocity.detach().clone()
            velocity_trace: list[dict[str, torch.Tensor]] = []
            supervision_image = self.rollout.rollout_native(
                self.prepared, config=self.config, goal_residual=residual.velocity,
                early_stop_steps=self.proxy_steps if self.reward_mode == "proxy" else None,
                velocity_trace=velocity_trace,
            )
            values = self.estimator(supervision_image)
            loss = progress_control_loss(
                values.raw_progress, self.target_strength, values.drift,
                residual.velocity, self.loss_config,
            )
            self._all_finite(loss.total, "total loss")
            progress = values.raw_progress.float().mean()
            signed_error = progress - self.target_strength
            progress_gradient = torch.autograd.grad(
                progress, residual.velocity, retain_graph=True
            )[0]
            if progress_gradient is None:
                raise RuntimeError("progress estimator produced no gradient for velocity residual")
            self._all_finite(progress_gradient, "progress gradient")
            per_step_grad = progress_gradient.float().flatten(1).norm(dim=1)
            grad_norm = progress_gradient.detach().float().norm()
            feedback_update = torch.zeros_like(residual.velocity, dtype=torch.float32)
            auxiliary_update = torch.zeros_like(residual.velocity, dtype=torch.float32)
            feedback_stats = {
                "progress_gradient_rms": float(progress_gradient.detach().float().square().mean().sqrt().cpu()),
                "progress_gradient_norm": float(grad_norm.cpu()),
            }
            record_per_step_grad = per_step_grad

            if self.optimizer_mode == "feedback":
                feedback_update, feedback_stats = normalized_feedback_update(
                    progress_gradient, signed_error.detach(), self.feedback_step_size
                )
                auxiliary_loss = (
                    self.loss_config.drift_weight * values.drift.float().mean()
                    + self.loss_config.regularization_weight * residual.velocity.float().square().mean()
                )
                if self.auxiliary_step_size > 0:
                    auxiliary_gradient = torch.autograd.grad(
                        auxiliary_loss, residual.velocity, allow_unused=True
                    )[0]
                    if auxiliary_gradient is not None:
                        self._all_finite(auxiliary_gradient, "auxiliary gradient")
                        auxiliary_update = -self.auxiliary_step_size * auxiliary_gradient.float()
                with torch.no_grad():
                    residual.velocity.add_(feedback_update + auxiliary_update)
                grad_norm = torch.as_tensor(feedback_stats["progress_gradient_norm"], device=residual.velocity.device)
            else:
                loss.total.backward()
                grad = residual.velocity.grad
                if grad is None:
                    raise RuntimeError("progress loss produced no gradient for velocity residual")
                self._all_finite(grad, "residual gradient")
                per_step_grad_loss = grad.float().flatten(1).norm(dim=1)
                record_per_step_grad = per_step_grad_loss
                if float(per_step_grad_loss.max().detach().cpu()) <= 0:
                    raise RuntimeError("progress loss produced a zero residual gradient")
                grad_norm = torch.nn.utils.clip_grad_norm_([residual.velocity], self.clip_grad_norm)
                self._all_finite(residual.velocity, "residual before update")
                assert optimizer is not None
                optimizer.step()
            self._all_finite(residual.velocity, "residual after update")
            last_residual = residual.velocity.detach().clone()
            actual_update = last_residual - residual_before
            feedback_update_rms = float(feedback_update.square().mean().sqrt().cpu())
            auxiliary_update_rms = float(auxiliary_update.square().mean().sqrt().cpu())
            total_update_rms = float(actual_update.float().square().mean().sqrt().cpu())
            residual_before_rms = float(residual_before.float().square().mean().sqrt().cpu())
            residual_after_rms = float(last_residual.float().square().mean().sqrt().cpu())
            native_step_rms = torch.stack([
                row["native_velocity_rms"].detach().float().to(residual.velocity.device)
                for row in velocity_trace[:self.goal_steps]
            ])
            native_rms_mean = native_step_rms.mean().clamp_min(1e-12)
            update_to_native_ratio = total_update_rms / float(native_rms_mean.cpu())
            residual_to_native_ratio = residual_after_rms / float(native_rms_mean.cpu())

            with torch.no_grad():
                post_image = self.rollout.rollout_native(
                    self.prepared, config=self.config, goal_residual=residual.velocity.detach(),
                    early_stop_steps=self.proxy_steps if self.reward_mode == "proxy" else None,
                )
                post_values = self.estimator(post_image)
                post_loss = progress_control_loss(
                    post_values.raw_progress, self.target_strength, post_values.drift,
                    residual.velocity.detach(), self.loss_config,
                )
            self._all_finite(post_loss.total, "post-update total loss")
            post_loss_value = float(post_loss.total.detach().cpu())
            post_progress = float(post_values.raw_progress.detach().mean().cpu())
            post_signed_error = post_progress - self.target_strength
            post_target_error = abs(post_signed_error)
            post_drift = float(post_values.drift.detach().mean().cpu())
            last_iteration_loss = post_loss_value
            previous_best_loss = best_loss
            if post_loss_value < best_loss:
                best_loss = post_loss_value
                best_iteration = iteration + 1
                best_residual = residual.velocity.detach().clone()
                best_loss_progress = post_progress
                best_loss_target_error = post_target_error
                best_loss_drift = post_drift

            previous_best_target_error = best_target_error_so_far
            if post_target_error < best_target_error_so_far:
                best_target_error_so_far = post_target_error
                best_target_error_iteration = iteration + 1
                best_target_error_progress = post_progress
                best_target_error_drift = post_drift
            if self.optimizer_mode == "feedback":
                if previous_best_target_error == float("inf") or previous_best_target_error - best_target_error_so_far >= self.min_improvement:
                    stale_iterations = 0
                else:
                    stale_iterations += 1
            else:
                loss_improvement = previous_best_loss - post_loss_value
                if previous_best_loss == float("inf") or loss_improvement >= self.min_improvement:
                    stale_iterations = 0
                else:
                    stale_iterations += 1

            per_step_rms = residual.velocity.detach().float().square().mean(dim=(1, 2)).sqrt()
            record: dict[str, object] = {
                "iteration": iteration,
                "iteration_number": iteration + 1,
                "reward_mode": self.reward_mode,
                "optimizer_mode": self.optimizer_mode,
                "target_strength": self.target_strength,
                "progress": post_progress,
                "signed_error": post_signed_error,
                "absolute_error": post_target_error,
                "progress_gradient_rms": feedback_stats["progress_gradient_rms"],
                "progress_gradient_norm": feedback_stats["progress_gradient_norm"],
                "feedback_update_rms": feedback_update_rms,
                "auxiliary_update_rms": auxiliary_update_rms,
                "total_update_rms": total_update_rms,
                "residual_rms_before": residual_before_rms,
                "residual_rms_after": residual_after_rms,
                "update_to_native_velocity_ratio": update_to_native_ratio,
                "residual_to_native_velocity_ratio": residual_to_native_ratio,
                "best_target_error_so_far": best_target_error_so_far,
                "best_target_error_progress": best_target_error_progress,
                "best_target_error_drift": best_target_error_drift,
                "best_target_error_iteration": best_target_error_iteration,
                "post_update": True,
                "supervision_progress": post_progress,
                "total_loss": post_loss_value,
                "progress_loss": float(post_loss.progress.detach().cpu()),
                "drift_loss": float(post_loss.drift.detach().cpu()),
                "regularization": float(post_loss.regularization.detach().cpu()),
                "raw_progress": post_progress,
                "target_error": float(post_loss.target_error.detach().cpu()),
                "gradient_norm_total": float(grad_norm.detach().cpu()),
                "gradient_norm_per_step": per_step_grad.detach().cpu().tolist(),
                "gradient_norm_pre_update": float(grad_norm.detach().cpu()),
                "gradient_norm_per_step_pre_update": per_step_grad.detach().cpu().tolist(),
                "residual_rms": float(residual.velocity.detach().float().square().mean().sqrt().cpu()),
                "residual_max_abs": float(residual.velocity.detach().float().abs().max().cpu()),
                "residual_rms_per_step": per_step_rms.cpu().tolist(),
                "progress_diagnostics": post_values.diagnostics(),
                "best_loss_so_far": best_loss,
                "best_iteration_so_far": best_iteration,
                "stale_iterations": stale_iterations,
            }
            history.append(record)
            if self.progress_callback:
                self.progress_callback(record)

            if abs(post_progress - self.target_strength) <= self.target_tolerance:
                stop_reason = "target_tolerance"
                break
            if stale_iterations >= self.patience:
                stop_reason = "patience"
                break

        if best_residual is None:
            raise RuntimeError("optimizer completed without evaluating a best residual")
        self._all_finite(best_residual, "best residual")
        with torch.no_grad():
            optimized_proxy_image = self.rollout.rollout_native(
                self.prepared, config=self.config, goal_residual=best_residual,
                early_stop_steps=self.proxy_steps,
            ).detach()
            optimized_proxy_values = self.estimator(optimized_proxy_image)
            optimized_final_trace: list[dict[str, torch.Tensor]] = []
            optimized_final_image = self.rollout.rollout_native(
                self.prepared, config=self.config, goal_residual=best_residual,
                velocity_trace=optimized_final_trace,
            ).detach()
            optimized_final_values = self.estimator(optimized_final_image)
            last_final_image = self.rollout.rollout_native(
                self.prepared, config=self.config, goal_residual=last_residual,
            ).detach()
            last_final_values = self.estimator(last_final_image)
        best_final_progress = float(optimized_final_values.raw_progress.detach().mean().cpu())
        last_final_progress = float(last_final_values.raw_progress.detach().mean().cpu())
        best_target_error = abs(best_final_progress - self.target_strength)
        last_target_error = abs(last_final_progress - self.target_strength)
        converged = abs(float(history[-1]["raw_progress"]) - self.target_strength) <= self.target_tolerance

        velocity_rows = []
        warnings_list: list[str] = []
        for row in optimized_final_trace[:self.goal_steps]:
            ratio = float(row["residual_native_rms_ratio"].cpu())
            step = int(row["step"].item())
            velocity_rows.append({
                "step": step,
                "native_velocity_rms": float(row["native_velocity_rms"].cpu()),
                "residual_velocity_rms": float(row["residual_velocity_rms"].cpu()),
                "residual_native_rms_ratio": ratio,
            })
            if ratio > 3.0:
                warnings_list.append(f"step {step}: residual/native velocity RMS ratio={ratio:.3f} (>3)")

        from .optimizer import projection_statistics
        directions = [row["edit_direction"].detach() for row in optimized_final_trace[:self.goal_steps]]
        projection = projection_statistics(best_residual.detach(), directions)
        detailed_steps = []
        residual_energy_total = 0.0
        parallel_energy_total = 0.0
        orthogonal_energy_total = 0.0
        negative_energy_total = 0.0
        cosines = []
        for step, direction in enumerate(directions):
            value = best_residual.detach()[step].float()
            direction = direction.to(device=value.device, dtype=torch.float32).squeeze(0)
            denom = direction.square().sum(dim=-1, keepdim=True)
            coefficient = (value * direction).sum(dim=-1, keepdim=True) / denom.clamp_min(1e-12)
            parallel = coefficient * direction
            orthogonal = value - parallel
            value_energy = float(value.square().sum().cpu())
            parallel_energy = float(parallel.square().sum().cpu())
            orthogonal_energy = float(orthogonal.square().sum().cpu())
            negative_energy = float((parallel.square() * (coefficient < 0)).sum().cpu())
            cosine = float(torch.nn.functional.cosine_similarity(
                value.reshape(1, -1), direction.reshape(1, -1), dim=-1
            ).item())
            cosines.append(cosine)
            residual_energy_total += value_energy
            parallel_energy_total += parallel_energy
            orthogonal_energy_total += orthogonal_energy
            negative_energy_total += negative_energy
            detailed_steps.append({
                "step": step,
                "cosine_similarity": cosine,
                "parallel_energy_fraction": parallel_energy / max(value_energy, 1e-12),
                "orthogonal_energy_fraction": orthogonal_energy / max(value_energy, 1e-12),
                "negative_projection_energy_fraction": negative_energy / max(value_energy, 1e-12),
                "negative_projection_token_fraction": float((coefficient < 0).float().mean().cpu()),
            })
        denominator = max(residual_energy_total, 1e-12)
        projection.update({
            "cosine_similarity_mean": sum(cosines) / max(len(cosines), 1),
            "parallel_energy_fraction_of_residual": parallel_energy_total / denominator,
            "orthogonal_energy_fraction_of_residual": orthogonal_energy_total / denominator,
            "negative_projection_energy_fraction_of_residual": negative_energy_total / denominator,
            "per_step_details": detailed_steps,
        })
        if float(projection.get("parallel_energy_fraction_of_residual", 0.0)) >= 0.9:
            warnings_list.append("Reward control is degenerating toward VeloEdit-like rescaling.")
        diagnostics = {
            "steps": velocity_rows,
            "max_residual_native_rms_ratio": max(
                (row["residual_native_rms_ratio"] for row in velocity_rows), default=0.0
            ),
            "warning_threshold": 3.0,
        }
        return ProgressResidualResult(
            source_image=source.detach(),
            native_full_image=self.native_full_image,
            initial_proxy_image=initial_proxy_image,
            optimized_proxy_image=optimized_proxy_image,
            optimized_final_image=optimized_final_image,
            residual=best_residual.detach().cpu(),
            initial_proxy=initial_values.diagnostics(),
            optimized_proxy=optimized_proxy_values.diagnostics(),
            optimized_final=optimized_final_values.diagnostics(),
            history=history,
            velocity_diagnostics=diagnostics,
            projection_diagnostics=projection,
            warnings=warnings_list,
            best_iteration=best_iteration,
            best_loss=best_loss,
            last_iteration_loss=last_iteration_loss,
            best_final_progress=best_final_progress,
            last_final_progress=last_final_progress,
            best_target_error=best_target_error,
            last_target_error=last_target_error,
            stop_reason=stop_reason,
            converged=converged,
            best_loss_progress=best_loss_progress,
            best_loss_target_error=best_loss_target_error,
            best_loss_drift=best_loss_drift,
            best_target_error_so_far=best_target_error_so_far,
            best_target_error_iteration=best_target_error_iteration,
        )
