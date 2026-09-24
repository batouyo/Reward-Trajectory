"""Small adapter contract for reusing differentiable rewards from RewardFlow."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class GoalRewardBundle:
    """Callbacks consume [B,3,H,W] images and return differentiable scalars."""

    edit_score: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    preservation_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def load_reward_bundle(
    spec: str,
    *,
    device: torch.device | str,
    prompt: str,
) -> GoalRewardBundle:
    """Load ``module:function`` returning a GoalRewardBundle.

    The factory receives ``device`` and the edit ``prompt`` as keyword
    arguments. This lets the
    standalone calibration package reuse installed RewardFlow reward models
    without copying their implementation or adding model dependencies here.
    """
    if ":" not in spec:
        raise ValueError("reward plugin must be specified as module:function")
    module_name, function_name = spec.split(":", maxsplit=1)
    factory = getattr(importlib.import_module(module_name), function_name)
    bundle = factory(device=device, prompt=prompt)
    if not isinstance(bundle, GoalRewardBundle):
        raise TypeError("reward plugin factory must return GoalRewardBundle")
    return bundle
