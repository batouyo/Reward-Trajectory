"""Optimize a per-sample V_goal while keeping alpha and FLUX weights fixed."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from rewardflow_calibration.rollout.veloedit import (
    PreparedVeloEdit,
    VeloEditCompatibleRollout,
    VeloEditRolloutConfig,
)

from .goal_residual import GoalVelocityResidual
from .objectives import GoalLossConfig, RewardFn, goal_residual_loss

MAX_GOAL_RESIDUAL_ITERATIONS = 16


def project_negative_edit_component(
    residual: torch.Tensor,
    edit_directions: list[torch.Tensor],
) -> torch.Tensor:
    """Remove only the negative component parallel to cached V_edit-V_keep."""
    projected = []
    for step, direction in enumerate(edit_directions):
        direction = direction.to(device=residual.device, dtype=torch.float32).squeeze(0)
        value = residual[step].float()
        denom = direction.square().sum(dim=-1, keepdim=True)
        coefficient = (value * direction).sum(dim=-1, keepdim=True) / denom.clamp_min(1e-12)
        negative_parallel = coefficient.clamp(max=0.0) * direction
        projected.append(value - negative_parallel)
    return torch.stack(projected).to(dtype=residual.dtype)


def projection_statistics(
    residual: torch.Tensor,
    edit_directions: list[torch.Tensor],
) -> dict[str, object]:
    """Summarize signed parallel projections per edit timestep and overall."""
    per_step = []
    total_energy = residual.float().square().sum().clamp_min(1e-12)
    total_parallel = torch.zeros((), device=residual.device)
    total_negative = torch.zeros((), device=residual.device)
    total_coefficients = []
    total_negative_tokens = 0
    total_active_tokens = 0
    for step, direction in enumerate(edit_directions):
        direction = direction.to(device=residual.device, dtype=torch.float32).squeeze(0)
        value = residual[step].float()
        denom = direction.square().sum(dim=-1)
        dot = (value * direction).sum(dim=-1)
        active = denom > 1e-12
        coefficient = dot / denom.clamp_min(1e-12)
        parallel_energy = coefficient.square() * denom
        negative_energy = coefficient.clamp(max=0.0).square() * denom
        per_step.append({
            "step": step,
            "mean_projection_coefficient": float(coefficient[active].mean().item()) if active.any() else 0.0,
            "negative_active_token_fraction": float((coefficient[active] < 0).float().mean().item()) if active.any() else 0.0,
            "parallel_energy_fraction_of_residual": float((parallel_energy.sum() / value.square().sum().clamp_min(1e-12)).item()),
            "negative_projection_energy_fraction_of_residual": float((negative_energy.sum() / value.square().sum().clamp_min(1e-12)).item()),
        })
        total_parallel += parallel_energy.sum()
        total_negative += negative_energy.sum()
        total_coefficients.append(coefficient[active])
        total_negative_tokens += int((coefficient[active] < 0).sum().item())
        total_active_tokens += int(active.sum().item())
    coefficients = [value for value in total_coefficients if value.numel()]
    mean_coefficient = float(torch.cat(coefficients).mean().item()) if coefficients else 0.0
    return {
        "mean_projection_coefficient": mean_coefficient,
        "negative_active_token_fraction": total_negative_tokens / max(1, total_active_tokens),
        "parallel_energy_fraction_of_residual": float((total_parallel / total_energy).item()),
        "negative_projection_energy_fraction_of_residual": float((total_negative / total_energy).item()),
        "per_step": per_step,
    }


@dataclass
class GoalResidualResult:
    baseline_image: torch.Tensor
    optimized_image: torch.Tensor
    baseline_edit_score: float
    optimized_edit_score: float
    baseline_preservation_loss: float
    optimized_preservation_loss: float
    residual: torch.Tensor
    history: list[dict[str, float]]
    baseline_proxy_edit_score: float | None = None
    optimized_proxy_edit_score: float | None = None
    baseline_proxy_preservation_loss: float | None = None
    optimized_proxy_preservation_loss: float | None = None
    projection_diagnostics: dict[str, object] | None = None


class GoalResidualOptimizer:
    """Test-time optimize early velocity residuals for one fixed image/edit."""

    def __init__(
        self,
        rollout: VeloEditCompatibleRollout,
        *,
        prepared: PreparedVeloEdit,
        alpha: float,
        rollout_config: VeloEditRolloutConfig,
        edit_reward: RewardFn,
        preservation_loss: RewardFn,
        loss_config: GoalLossConfig | None = None,
        goal_steps: int = 4,
        learning_rate: float = 1e-3,
        iterations: int = 16,
        proxy_steps: int | None = None,
        preservation_reference: torch.Tensor | None = None,
        restrict_negative_edit_projection: bool = False,
        progress_callback: Callable[[dict[str, float]], None] | None = None,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if goal_steps < 1 or goal_steps > rollout_config.steps:
            raise ValueError("goal_steps must be in [1, rollout steps]")
        if proxy_steps is not None and not 1 <= proxy_steps <= rollout_config.steps:
            raise ValueError("proxy_steps must be in [1, rollout steps]")
        if proxy_steps is not None and goal_steps > proxy_steps:
            raise ValueError("proxy_steps must cover every optimized goal step")
        if learning_rate <= 0 or iterations < 1:
            raise ValueError("learning_rate and iterations must be positive")
        if iterations > MAX_GOAL_RESIDUAL_ITERATIONS:
            raise ValueError(
                f"iterations must not exceed {MAX_GOAL_RESIDUAL_ITERATIONS}"
            )
        if preservation_reference is not None:
            expected = (1, 3, prepared.height, prepared.width)
            if tuple(preservation_reference.shape) != expected:
                raise ValueError(f"preservation_reference must have shape {expected}")
            if not torch.isfinite(preservation_reference).all():
                raise ValueError("preservation_reference must contain only finite values")
        self.rollout = rollout
        self.prepared = prepared
        self.alpha = float(alpha)
        self.rollout_config = rollout_config
        self.edit_reward = edit_reward
        self.preservation_loss = preservation_loss
        self.loss_config = loss_config or GoalLossConfig()
        self.goal_steps = goal_steps
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.proxy_steps = proxy_steps
        self.preservation_reference = (
            None if preservation_reference is None
            else preservation_reference.detach().to(device=rollout.device, dtype=torch.float32)
        )
        self.restrict_negative_edit_projection = restrict_negative_edit_projection
        self.progress_callback = progress_callback

    def run(self) -> GoalResidualResult:
        pipeline = self.rollout.pipeline
        for component in pipeline.components.values():
            if isinstance(component, torch.nn.Module):
                component.eval()
                for parameter in component.parameters():
                    parameter.requires_grad_(False)

        # ``prepare`` may have encoded the prompt before the pipeline was
        # frozen, leaving a text-encoder autograd graph attached to these
        # constant embeddings. Reusing that graph across optimization steps
        # causes the second backward() to fail; only V_goal should receive
        # gradients in this experiment.
        self.prepared.prompt_embeds = self.prepared.prompt_embeds.detach()
        self.prepared.pooled_prompt_embeds = self.prepared.pooled_prompt_embeds.detach()

        with torch.no_grad():
            baseline_trace: list[dict[str, torch.Tensor]] = []
            baseline = self.rollout.rollout(
                self.prepared,
                [self.alpha],
                config=self.rollout_config,
                velocity_trace=baseline_trace,
            ).detach()
            edit_directions = [entry["edit_direction"] for entry in baseline_trace[: self.goal_steps]]
            if len(edit_directions) < self.goal_steps:
                raise RuntimeError("rollout did not record edit directions for all goal steps")
            source = self.prepared_image_tensor()
            preservation_reference = self.preservation_reference_tensor()
            baseline_edit_score = self.edit_reward(baseline, source)
            if baseline_edit_score.numel() != 1:
                raise ValueError("edit_reward must return a scalar for one fixed-alpha image")
            baseline_edit_score = baseline_edit_score.detach().reshape(())
            baseline_preservation_loss = self.preservation_loss(
                baseline, preservation_reference
            ).detach().reshape(())
            baseline_proxy_edit_score = None
            baseline_proxy_preservation_loss = None
            if self.proxy_steps is not None:
                baseline_proxy = self.rollout.rollout(
                    self.prepared,
                    [self.alpha],
                    config=self.rollout_config,
                    early_stop_steps=self.proxy_steps,
                ).detach()
                baseline_proxy_edit_score = self.edit_reward(baseline_proxy, source).detach().reshape(())
                baseline_proxy_preservation_loss = self.preservation_loss(
                    baseline_proxy, preservation_reference
                ).detach().reshape(())
            objective_baseline_edit_score = (
                baseline_edit_score if baseline_proxy_edit_score is None else baseline_proxy_edit_score
            )

        residual = GoalVelocityResidual(
            steps=self.goal_steps,
            latent_shape=tuple(self.prepared.latents.shape[1:]),
            device=self.rollout.device,
        )
        optimizer = torch.optim.Adam([residual.velocity], lr=self.learning_rate)
        history: list[dict[str, float]] = []

        for iteration in range(self.iterations):
            optimizer.zero_grad(set_to_none=True)
            applied_residual = (
                project_negative_edit_component(residual.velocity, edit_directions)
                if self.restrict_negative_edit_projection
                else residual.velocity
            )
            generated = self.rollout.rollout(
                self.prepared,
                [self.alpha],
                config=self.rollout_config,
                goal_residual=applied_residual,
                early_stop_steps=self.proxy_steps,
            )
            values = goal_residual_loss(
                generated,
                self.prepared_image_tensor(),
                objective_baseline_edit_score,
                self.edit_reward,
                self.preservation_loss,
                applied_residual,
                self.loss_config,
                preservation_reference=preservation_reference,
            )
            values.total.backward()
            if residual.velocity.grad is None:
                raise RuntimeError("loss produced no gradient for V_goal")
            if not torch.isfinite(residual.velocity.grad).all():
                raise FloatingPointError("non-finite gradient encountered in V_goal optimization")
            optimizer.step()
            record = {
                "iteration": float(iteration),
                "total_loss": float(values.total.detach()),
                "edit_loss": float(values.edit.detach()),
                "preservation_loss": float(values.preservation.detach()),
                "regularization": float(values.regularization.detach()),
                "edit_score": float(values.edit_score.detach()),
                "objective_uses_early_proxy": float(self.proxy_steps is not None),
                "residual_rms": float(residual.velocity.detach().square().mean().sqrt()),
            }
            history.append(record)
            if self.progress_callback is not None:
                self.progress_callback(record)

        with torch.no_grad():
            final_applied_residual = (
                project_negative_edit_component(residual.velocity, edit_directions)
                if self.restrict_negative_edit_projection
                else residual.velocity.detach()
            )
            optimized_trace: list[dict[str, torch.Tensor]] = []
            optimized = self.rollout.rollout(
                self.prepared,
                [self.alpha],
                config=self.rollout_config,
                goal_residual=final_applied_residual,
                velocity_trace=optimized_trace,
            ).detach()
            source = self.prepared_image_tensor()
            optimized_score = self.edit_reward(optimized, source)
            preservation_reference = self.preservation_reference_tensor()
            optimized_preservation_loss = self.preservation_loss(
                optimized, preservation_reference
            )
            optimized_proxy_edit_score = None
            optimized_proxy_preservation_loss = None
            if self.proxy_steps is not None:
                optimized_proxy = self.rollout.rollout(
                    self.prepared,
                    [self.alpha],
                    config=self.rollout_config,
                    goal_residual=residual.velocity.detach(),
                    early_stop_steps=self.proxy_steps,
                ).detach()
                optimized_proxy_edit_score = self.edit_reward(optimized_proxy, source)
                optimized_proxy_preservation_loss = self.preservation_loss(
                    optimized_proxy, preservation_reference
                )
            optimized_directions = [entry["edit_direction"] for entry in optimized_trace[: self.goal_steps]]
            projection_report = {
                "projection_basis": "zero-residual VeloEdit trajectory; active low-similarity interpolation elements",
                "negative_projection_restricted": self.restrict_negative_edit_projection,
                "against_baseline_directions": projection_statistics(
                    final_applied_residual, edit_directions
                ),
                "against_optimized_trajectory_directions": projection_statistics(
                    final_applied_residual, optimized_directions
                ),
            }

        return GoalResidualResult(
            baseline_image=baseline,
            optimized_image=optimized,
            baseline_edit_score=float(baseline_edit_score),
            optimized_edit_score=float(optimized_score.reshape(())),
            baseline_preservation_loss=float(baseline_preservation_loss),
            optimized_preservation_loss=float(optimized_preservation_loss.reshape(())),
            residual=final_applied_residual.detach().cpu(),
            history=history,
            baseline_proxy_edit_score=(
                None if baseline_proxy_edit_score is None else float(baseline_proxy_edit_score)
            ),
            optimized_proxy_edit_score=(
                None if optimized_proxy_edit_score is None else float(optimized_proxy_edit_score.reshape(()))
            ),
            baseline_proxy_preservation_loss=(
                None if baseline_proxy_preservation_loss is None else float(baseline_proxy_preservation_loss)
            ),
            optimized_proxy_preservation_loss=(
                None if optimized_proxy_preservation_loss is None
                else float(optimized_proxy_preservation_loss.reshape(()))
            ),
            projection_diagnostics=projection_report,
        )

    def prepared_image_tensor(self) -> torch.Tensor:
        """Return resized source as `[1, 3, H, W]` in the rollout's image range."""
        source = self.rollout.pipeline.image_processor.preprocess(
            self.prepared.working_image, self.prepared.height, self.prepared.width
        )
        source = source.to(device=self.rollout.device, dtype=torch.float32)
        return ((source + 1.0) / 2.0).clamp(0, 1)

    def preservation_reference_tensor(self) -> torch.Tensor:
        """Return the safe-alpha anchor, or original input if none was provided."""
        if self.preservation_reference is not None:
            return self.preservation_reference
        return self.prepared_image_tensor()
