from types import SimpleNamespace

import torch

from rewardflow_calibration.optimization.progress_objective import ProgressLossConfig
from rewardflow_calibration.optimization.progress_optimizer import ProgressResidualOptimizer
from rewardflow_calibration.rollout.veloedit import VeloEditRolloutConfig


class _FakeValues:
    def __init__(self, progress):
        self.raw_progress = progress.reshape(1)
        self.clamped_progress = self.raw_progress.clamp(0.0, 1.0)
        self.drift = self.raw_progress * 0.0

    def diagnostics(self):
        return {
            "raw_progress": float(self.raw_progress.detach().mean()),
            "clamped_progress": float(self.clamped_progress.detach().mean()),
            "drift": float(self.drift.detach().mean()),
        }


class _FakeEstimator:
    def __init__(self):
        self.source_features = None

    def set_anchors(self, source, target):
        self.source_features = torch.ones(1, 1)
        return {"source_progress": 0.0, "target_progress": 1.0}

    def __call__(self, image):
        return _FakeValues(image.mean())


class _FakeImageProcessor:
    def preprocess(self, image, height, width):
        return torch.zeros(1, 3, height, width)


class _FakeRollout:
    def __init__(self):
        self.device = torch.device("cpu")
        self.pipeline = SimpleNamespace(
            components={}, image_processor=_FakeImageProcessor()
        )

    def rollout_native(self, prepared, *, config, goal_residual=None,
                       early_stop_steps=None, velocity_trace=None):
        residual_mean = 0.0 if goal_residual is None else goal_residual.mean()
        image = torch.ones(1, 1, 1, 1) * (1.0 + 2.0 * residual_mean)
        if velocity_trace is not None:
            residual_rms = (
                torch.zeros(()) if goal_residual is None
                else goal_residual[0].float().square().mean().sqrt().detach()
            )
            velocity_trace.append({
                "step": torch.tensor(0),
                "native_velocity_rms": torch.tensor(1.0),
                "residual_velocity_rms": residual_rms,
                "residual_native_rms_ratio": residual_rms,
                "edit_direction": torch.ones(1, 1, 1),
            })
        return image


def _run_one_feedback_step(target):
    rollout = _FakeRollout()
    prepared = SimpleNamespace(
        working_image=None,
        height=1,
        width=1,
        latents=torch.zeros(1, 1, 1),
        prompt_embeds=torch.zeros(1, 1),
        pooled_prompt_embeds=torch.zeros(1, 1),
    )
    config = VeloEditRolloutConfig(
        steps=1, first_step_align_steps=0, preserve_steps=0, edit_steps=0
    )
    optimizer = ProgressResidualOptimizer(
        rollout,
        prepared=prepared,
        native_full_image=torch.ones(1, 1, 1, 1),
        progress_estimator=_FakeEstimator(),
        target_strength=target,
        rollout_config=config,
        loss_config=ProgressLossConfig(),
        goal_steps=1,
        proxy_steps=1,
        iterations=1,
        reward_mode="final",
        optimizer_mode="feedback",
        feedback_step_size=0.1,
        auxiliary_step_size=0.0,
        target_tolerance=0.0,
    )
    return optimizer.run()


def test_feedback_optimizer_integration_scales_update_by_target_error():
    results = {target: _run_one_feedback_step(target) for target in (0.25, 0.50, 0.75)}
    magnitudes = [float(results[t].residual.float().square().mean().sqrt()) for t in (0.25, 0.50, 0.75)]
    assert magnitudes[0] > magnitudes[1] > magnitudes[2]

    for target, result in results.items():
        assert result.best_final_progress < 1.0
        assert abs(result.best_final_progress - target) < abs(1.0 - target)
    assert 1.0 - results[0.25].best_final_progress > 1.0 - results[0.75].best_final_progress

    row = results[0.5].history[0]
    assert row["gradient_norm_per_step"] == row["gradient_norm_per_step_pre_update"]
    assert row["progress_gradient_norm_per_step"] == row["optimizer_gradient_norm_per_step"]
