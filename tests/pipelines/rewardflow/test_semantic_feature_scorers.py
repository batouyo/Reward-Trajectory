from types import SimpleNamespace

import torch
import torch.nn.functional as F

from diffusers.pipelines.rewardflow.rewards import RegionCLIPReward, SigLIPReward
from diffusers.pipelines.rewardflow.semantic_feature_scorers import (
    CLIPImageFeatureScorer,
    QwenHiddenFeatureScorer,
    SigLIPImageFeatureScorer,
)


class _TinyProjectedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(3, 4, bias=False)
        self.config = SimpleNamespace(_commit_hash="tiny")

    def get_image_features(self, pixel_values):
        return self.projection(pixel_values.mean(dim=(-2, -1)))


class _TinyProcessor:
    do_resize = True
    size = {"height": 4, "width": 4}
    do_center_crop = False
    crop_size = None
    do_rescale = True
    rescale_factor = 1 / 255
    do_normalize = True
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]


def _scorer(cls):
    return cls("tiny", model=_TinyProjectedModel(), image_processor=_TinyProcessor())


def test_clip_and_siglip_projected_embeddings_are_normalized_frozen_and_differentiable():
    for scorer_class in (CLIPImageFeatureScorer, SigLIPImageFeatureScorer):
        scorer = _scorer(scorer_class)
        image = torch.rand(1, 3, 6, 8, requires_grad=True)
        feature = scorer.encode_image(image)
        gradient = torch.autograd.grad(feature.sum(), image)[0]
        assert feature.shape == (4,)
        torch.testing.assert_close(torch.linalg.vector_norm(feature), torch.tensor(1.0))
        assert all(not parameter.requires_grad and parameter.grad is None for parameter in scorer.model.parameters())
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


class _ExistingQwenScorer:
    def __init__(self):
        self.model = torch.nn.Linear(1, 1)
        self.calls = []

    def focus_conditioned_representation(self, image, prompt):
        self.calls.append((image, prompt))
        value = image.mean()
        return torch.stack((value, 1 - value))


def test_qwen_wrapper_reuses_existing_api_and_comparison_focus():
    existing = _ExistingQwenScorer()
    wrapper = QwenHiddenFeatureScorer(existing, "visible surface texture")
    image = torch.rand(1, 3, 4, 4, requires_grad=True)
    feature = wrapper.encode_image(image)
    torch.autograd.grad(feature.sum(), image)
    assert existing.calls[0][0] is image
    assert "visible surface texture" in existing.calls[0][1]
    assert "Focus only on:" in existing.calls[0][1]


def test_legacy_siglip_and_region_clip_preprocessing_remains_unchanged():
    image = torch.linspace(-0.2, 1.2, 3 * 3 * 5).reshape(1, 3, 3, 5)
    for reward_class in (SigLIPReward, RegionCLIPReward):
        reward = reward_class.__new__(reward_class)
        reward.image_size = 4
        reward.image_mean = (0.5, 0.5, 0.5)
        reward.image_std = (0.5, 0.5, 0.5)
        expected = F.interpolate(image.clamp(0, 1), size=(4, 4), mode="bicubic", align_corners=False)
        expected = (expected - 0.5) / 0.5
        torch.testing.assert_close(reward._preprocess_image(image), expected)
