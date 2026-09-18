import pytest
import torch
from torch import nn

from diffusers.pipelines.rewardflow.rewardslider_v2_quality import (
    DifferentiableQualityReward,
    audit_quality_reward,
)


class _GoodReward(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0))

    def forward(self, image):
        return self.scale * image.square().mean()


def test_quality_reward_returns_scalar_and_preserves_image_gradient():
    scorer = _GoodReward()
    reward = DifferentiableQualityReward(scorer)
    image = torch.ones(2, 3, 4, 4, requires_grad=True)
    value = reward(image)
    assert value.ndim == 0 and value.requires_grad
    value.backward()
    assert image.grad is not None and torch.isfinite(image.grad).all() and image.grad.abs().sum() > 0
    assert not scorer.scale.requires_grad


def test_quality_gradient_audit_passes_for_differentiable_reward_and_freezes_model():
    reward = DifferentiableQualityReward(_GoodReward())
    image = torch.randn(2, 3, 4, 4)
    result = audit_quality_reward(reward, image)
    assert result.passed
    assert result.reward_requires_grad
    assert result.image_gradient_norm > 0
    assert result.finite


class _DetachedReward(nn.Module):
    def forward(self, image):
        return image.detach().mean()


class _ConstantReward(nn.Module):
    def forward(self, image):
        return torch.ones((), device=image.device)


@pytest.mark.parametrize("scorer", [_DetachedReward(), _ConstantReward()])
def test_quality_gradient_audit_rejects_non_differentiable_or_zero_gradient_reward(scorer):
    reward = DifferentiableQualityReward(scorer)
    with pytest.raises(RuntimeError, match="gradient audit"):
        audit_quality_reward(reward, torch.ones(1, 3, 2, 2), require_pass=True)

