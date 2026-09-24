"""Inference-time optimization of small VeloEdit velocity residuals."""

from .goal_residual import GoalVelocityResidual
from .objectives import GoalLossConfig, GoalLossValues, goal_residual_loss
from .optimizer import GoalResidualOptimizer, GoalResidualResult
from .reward_plugin import GoalRewardBundle, load_reward_bundle

__all__ = [
    "GoalLossConfig",
    "GoalLossValues",
    "GoalResidualOptimizer",
    "GoalResidualResult",
    "GoalRewardBundle",
    "GoalVelocityResidual",
    "goal_residual_loss",
    "load_reward_bundle",
]
