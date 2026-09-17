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
        self.text_projection = torch.nn.Embedding(32, 4)
        self.config = SimpleNamespace(_commit_hash="tiny", text_config=SimpleNamespace(max_position_embeddings=8))

    def get_image_features(self, pixel_values):
        return self.projection(pixel_values.mean(dim=(-2, -1)))

    def get_text_features(self, input_ids, attention_mask):
        tokens = self.text_projection(input_ids)
        return (tokens * attention_mask[..., None]).sum(dim=1)


class _TinyTokenizer:
    model_max_length = 8

    def __call__(
        self,
        text,
        *,
        add_special_tokens=True,
        truncation=False,
        max_length=None,
        padding=None,
        return_tensors=None,
    ):
        ids = [1] + [2 + len(token) % 20 for token in text.split()] + [2]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        mask = [1] * len(ids)
        if padding == "max_length":
            ids += [0] * (max_length - len(ids))
            mask += [0] * (max_length - len(mask))
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids]), "attention_mask": torch.tensor([mask])}
        return {"input_ids": ids, "attention_mask": mask}


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
    return cls(
        "tiny",
        model=_TinyProjectedModel(),
        image_processor=_TinyProcessor(),
        tokenizer=_TinyTokenizer(),
    )


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


def test_batched_image_encoding_matches_single_image_api_and_keeps_candidate_gradients():
    for scorer_class in (CLIPImageFeatureScorer, SigLIPImageFeatureScorer):
        scorer = _scorer(scorer_class)
        images = torch.rand(3, 3, 6, 8, requires_grad=True)
        batched = scorer.encode_images(images)
        singles = torch.stack([scorer.encode_image(images[index : index + 1]) for index in range(3)])
        # Batched reductions may use a different, still FP32, reduction tree
        # than three singleton calls. This is a numerical-equivalence check.
        torch.testing.assert_close(batched, singles, rtol=5e-3, atol=2e-3)
        gradient = torch.autograd.grad(batched.sum(), images)[0]
        assert batched.shape == (3, 4)
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_clip_and_siglip_use_official_projected_text_api_and_detach_features():
    for scorer_class in (CLIPImageFeatureScorer, SigLIPImageFeatureScorer):
        scorer = _scorer(scorer_class)
        feature = scorer.encode_text("a vivid blue weighted training ball")
        assert feature.shape == (4,)
        assert feature.grad_fn is None and not feature.requires_grad
        torch.testing.assert_close(torch.linalg.vector_norm(feature), torch.tensor(1.0))
        assert scorer.last_text_metadata == {
            "raw_tokenized_length": 8,
            "effective_tokenized_length": 8,
            "model_max_length": 8,
            "truncated": False,
        }
        assert all(not parameter.requires_grad and parameter.grad is None for parameter in scorer.model.parameters())


def test_text_metadata_reports_truncation_without_silently_extending_model_limit():
    scorer = _scorer(CLIPImageFeatureScorer)
    scorer.encode_text("one two three four five six seven eight nine")
    assert scorer.last_text_metadata["raw_tokenized_length"] == 11
    assert scorer.last_text_metadata["effective_tokenized_length"] == 8
    assert scorer.last_text_metadata["truncated"] is True


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
