"""Three-phase RewardSlider V2 optimizer routing and deterministic scheduler."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class RewardSliderV2Scheduler:
    PHASES = ("trajectory_calibration", "quality_repair", "joint_refinement")

    def __init__(
        self,
        alpha_parameterization: nn.Module,
        v_goal_parameters: Sequence[nn.Parameter],
        *,
        trajectory_kl_threshold: float = 0.15,
        trajectory_patience: int = 3,
        trajectory_tolerance: float = 0.02,
        rollback_patience: int = 3,
        min_repair_iterations: int = 10,
        joint_alpha_lr_scale: float = 0.1,
    ):
        if not 0 < trajectory_kl_threshold:
            raise ValueError("`trajectory_kl_threshold` must be positive.")
        if trajectory_patience < 1 or rollback_patience < 1 or min_repair_iterations < 1:
            raise ValueError("Patience and minimum iteration values must be positive.")
        if trajectory_tolerance < 0 or not 0 < joint_alpha_lr_scale <= 1:
            raise ValueError("Tolerance must be non-negative and alpha LR scale must be in (0, 1].")
        self.alpha_parameterization = alpha_parameterization
        self.v_goal_parameters = tuple(v_goal_parameters)
        self.trajectory_kl_threshold = float(trajectory_kl_threshold)
        self.trajectory_patience = int(trajectory_patience)
        self.trajectory_tolerance = float(trajectory_tolerance)
        self.rollback_patience = int(rollback_patience)
        self.min_repair_iterations = int(min_repair_iterations)
        self.joint_alpha_lr_scale = float(joint_alpha_lr_scale)
        self.phase = "trajectory_calibration"
        self.phase_iterations = 0
        self.phase_reference_kl: float | None = None
        self._healthy_streak = 0
        self._bad_streak = 0
        self.topology_events = []
        self._set_parameter_permissions()

    def config_dict(self) -> dict[str, object]:
        """Return constructor configuration, excluding mutable phase state."""
        return {
            "trajectory_kl_threshold": self.trajectory_kl_threshold,
            "trajectory_patience": self.trajectory_patience,
            "trajectory_tolerance": self.trajectory_tolerance,
            "rollback_patience": self.rollback_patience,
            "min_repair_iterations": self.min_repair_iterations,
            "joint_alpha_lr_scale": self.joint_alpha_lr_scale,
        }

    get_rebuild_kwargs = config_dict

    def configure_optimizers(self, alpha_optimizer, v_goal_optimizer) -> None:
        """Apply phase-specific learning rates without changing optimizer membership."""
        for group in alpha_optimizer.param_groups:
            group.setdefault("_rewardslider_v2_base_lr", group["lr"])
            group["lr"] = (
                group["_rewardslider_v2_base_lr"] * self.joint_alpha_lr_scale
                if self.phase == "joint_refinement"
                else group["_rewardslider_v2_base_lr"]
            )
        for group in v_goal_optimizer.param_groups:
            group.setdefault("_rewardslider_v2_base_lr", group["lr"])
            group["lr"] = group["_rewardslider_v2_base_lr"]

    @property
    def alpha_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(parameter for parameter in self.alpha_parameterization.parameters())

    @property
    def trainable_parameters(self) -> tuple[nn.Parameter, ...]:
        return self.alpha_parameters + self.v_goal_parameters

    def _set_parameter_permissions(self) -> None:
        alpha_on = self.phase in ("trajectory_calibration", "joint_refinement")
        goal_on = self.phase in ("quality_repair", "joint_refinement")
        for parameter in self.alpha_parameters:
            parameter.requires_grad_(alpha_on)
            parameter.grad = None
        for parameter in self.v_goal_parameters:
            parameter.requires_grad_(goal_on)
            parameter.grad = None

    def force_phase(self, phase: str) -> None:
        if phase not in self.PHASES:
            raise ValueError(f"Unknown V2 scheduler phase: {phase}")
        self.phase = phase
        self.phase_iterations = 0
        self._healthy_streak = 0
        self._bad_streak = 0
        self._set_parameter_permissions()

    @staticmethod
    def _sum_losses(*losses: torch.Tensor | None) -> torch.Tensor | None:
        present = [loss for loss in losses if loss is not None]
        return None if not present else torch.stack([loss.reshape(()) for loss in present]).sum()

    @staticmethod
    def _write_gradients(loss: torch.Tensor | None, parameters: Sequence[nn.Parameter]) -> None:
        if loss is None or not loss.requires_grad:
            return
        active = [parameter for parameter in parameters if parameter.requires_grad]
        if not active:
            return
        gradients = torch.autograd.grad(loss, active, allow_unused=True, retain_graph=True)
        for parameter, gradient in zip(active, gradients):
            if gradient is not None:
                parameter.grad = gradient.detach().clone()

    def backward(
        self,
        *,
        trajectory_loss: torch.Tensor | None = None,
        quality_loss: torch.Tensor | None = None,
        control_loss: torch.Tensor | None = None,
        trajectory_guard_loss: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Populate routed gradients; callers then invoke their optimizer."""

        for parameter in self.trainable_parameters:
            parameter.grad = None
        quality_and_control = self._sum_losses(quality_loss, control_loss)
        if self.phase == "trajectory_calibration":
            self._write_gradients(trajectory_loss, self.alpha_parameters)
            total = trajectory_loss
        elif self.phase == "quality_repair":
            self._write_gradients(self._sum_losses(quality_and_control, trajectory_guard_loss), self.v_goal_parameters)
            total = self._sum_losses(quality_and_control, trajectory_guard_loss)
        else:
            self._write_gradients(trajectory_loss, self.alpha_parameters)
            self._write_gradients(self._sum_losses(quality_and_control, trajectory_guard_loss), self.v_goal_parameters)
            total = self._sum_losses(trajectory_loss, quality_and_control, trajectory_guard_loss)
        if total is None:
            raise ValueError("At least one routed loss is required.")
        return total

    def advance(self, trajectory_kl: float | torch.Tensor, *, trajectory_collapsed: bool = False) -> str:
        value = float(trajectory_kl.detach().cpu().item()) if torch.is_tensor(trajectory_kl) else float(trajectory_kl)
        if not torch.isfinite(torch.tensor(value)) or value < 0:
            raise ValueError("Trajectory KL must be finite and non-negative.")
        self.phase_iterations += 1
        if self.phase == "trajectory_calibration":
            self._healthy_streak = self._healthy_streak + 1 if not trajectory_collapsed and value <= self.trajectory_kl_threshold else 0
            if self._healthy_streak >= self.trajectory_patience:
                self.phase_reference_kl = value
                self.force_phase("quality_repair")
        else:
            self._bad_streak = self._bad_streak + 1 if trajectory_collapsed or value > self.trajectory_kl_threshold + self.trajectory_tolerance else 0
            if self._bad_streak >= self.rollback_patience:
                self.force_phase("trajectory_calibration")
            elif self.phase == "quality_repair" and self.phase_iterations >= self.min_repair_iterations:
                self.force_phase("joint_refinement")
        return self.phase
