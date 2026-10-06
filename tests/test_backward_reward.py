import pytest
import torch
from torch import nn

from rewardflow_calibration.optimization.backward_reward import (
    BackwardReward,
    BackwardRewardConfig,
    InvalidSemanticAnchorError,
    keep_region_l1,
    semantic_floor_from_anchors,
    source_attraction_loss,
)
from rewardflow_calibration.optimization.backward_trajectory_optimizer import apply_frozen_edit_mask


class ToySemanticEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, image):
        mean = image.mean(dim=(1, 2, 3))
        return torch.stack([mean * self.scale, 1.0 - mean], dim=-1)


class ToyDinoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, image):
        mean = image.mean(dim=(1, 2, 3))
        return torch.stack([mean * self.scale, 1.0 - mean], dim=-1)


def _toy_reward(fraction=0.5):
    semantic = ToySemanticEncoder()
    dino = ToyDinoEncoder()
    reward = BackwardReward(
        "target",
        device="cpu",
        config=BackwardRewardConfig(
            semantic_floor_fraction=fraction,
            semantic_anchor_min_gap=0.01,
            source_weight=1.0,
            semantic_weight=1.0,
            keep_weight=1.0,
        ),
        semantic_image_encoder=semantic,
        semantic_text_encoder=lambda _: torch.tensor([[1.0, 0.0]]),
        dino_image_encoder=dino,
        dreamsim_distance=lambda a, b: (a - b).abs().mean().reshape(1),
    )
    source = torch.full((1, 3, 2, 2), 0.2)
    full = torch.full_like(source, 0.8)
    reward.set_anchors(source, full, full)
    return reward, source, full, semantic, dino


def test_semantic_floor_is_instance_relative_not_absolute():
    floor = semantic_floor_from_anchors(0.2, 0.8, 0.25)
    assert floor == pytest.approx(0.35)
    second = semantic_floor_from_anchors(-0.1, 0.1, 0.5)
    assert second == pytest.approx(0.0)


def test_degenerate_semantic_anchor_is_rejected():
    with pytest.raises(InvalidSemanticAnchorError) as error:
        semantic_floor_from_anchors(0.4, 0.405, 0.5, min_gap=0.01)
    assert error.value.code == "invalid_semantic_anchor"
    assert error.value.diagnostics["semantic_gap"] == pytest.approx(0.005)


def test_candidate_semantic_gradient_reaches_image_not_frozen_parameters():
    reward, source, full, semantic, dino = _toy_reward(0.8)
    candidate = torch.full_like(source, 0.55, requires_grad=True)
    keep = torch.zeros((1, 1, 2, 2))
    values = reward.evaluate(candidate, source, keep)
    values.total.backward()
    assert candidate.grad is not None
    assert torch.isfinite(candidate.grad).all()
    assert candidate.grad.abs().sum() > 0
    for module in (semantic, dino):
        assert all(not parameter.requires_grad for parameter in module.parameters())
        assert all(parameter.grad is None for parameter in module.parameters())


def test_source_attraction_gradient_moves_features_toward_source():
    source_features = torch.tensor([[1.0, 0.0]])
    candidate_features = torch.tensor([[0.6, 0.8]], requires_grad=True)
    before = source_attraction_loss(candidate_features, source_features)
    gradient = torch.autograd.grad(before, candidate_features)[0]
    updated = (candidate_features.detach() - 0.1 * gradient).requires_grad_(False)
    after = source_attraction_loss(updated, source_features)
    assert after < before.detach()


def test_keep_region_l1_normalizes_by_valid_keep_mass():
    source = torch.zeros(1, 3, 2, 2)
    candidate = torch.ones_like(source)
    keep = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
    assert keep_region_l1(candidate, source, keep).item() == pytest.approx(1.0)


def test_frozen_edit_mask_zeros_update_outside_edit_region():
    gradient = torch.arange(24, dtype=torch.float32).reshape(4, 2, 3)
    mask = torch.zeros_like(gradient, dtype=torch.bool)
    mask[:, 0, 1] = True
    update = apply_frozen_edit_mask(gradient, mask)
    assert torch.count_nonzero(update[~mask]) == 0
    torch.testing.assert_close(update[mask], gradient[mask])


def test_reward_anchor_validation_returns_invalid_anchor_status():
    semantic = ToySemanticEncoder()
    dino = ToyDinoEncoder()
    reward = BackwardReward(
        "target",
        device="cpu",
        config=BackwardRewardConfig(semantic_anchor_min_gap=0.01),
        semantic_image_encoder=semantic,
        semantic_text_encoder=lambda _: torch.tensor([[1.0, 0.0]]),
        dino_image_encoder=dino,
        dreamsim_distance=lambda a, b: (a - b).abs().mean().reshape(1),
    )
    source = torch.full((1, 3, 2, 2), 0.5)
    almost_full = torch.full_like(source, 0.505)
    with pytest.raises(InvalidSemanticAnchorError):
        reward.set_anchors(source, almost_full, almost_full)
    assert reward.anchor_diagnostics["status"] == "invalid_semantic_anchor"
