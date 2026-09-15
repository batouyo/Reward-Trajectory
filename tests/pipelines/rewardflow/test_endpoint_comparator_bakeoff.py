import json
import os
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from diffusers.pipelines.rewardflow.endpoint_comparator_metrics import (
    average_ranks,
    gradient_direction_gate,
    ordering_diagnostics,
    spearman_correlation,
)
from diffusers.pipelines.rewardflow.endpoint_feature_distance import FeatureEndpointDistanceReward
from diffusers.pipelines.rewardflow.pairwise_endpoint_semantic import (
    PairwiseEndpointSemanticReward,
    build_pairwise_endpoint_affinity_prompt,
)
from diffusers.pipelines.rewardflow.relative_endpoint_parser import parse_relative_endpoint_semantic_json
from diffusers.pipelines.rewardflow.rewards import Qwen25VQATeacherForcedScorer


def _spec(count=1):
    return parse_relative_endpoint_semantic_json(
        json.dumps(
            {
                "edit_instruction": "change the object",
                "primitives": [
                    {
                        "id": f"attribute_{index}",
                        "object": "object",
                        "attribute": f"channel {index}",
                        "edit_description": "change the channel",
                        "comparison_focus": f"visible state of channel {index}",
                        "source_state": "dark",
                        "target_state": "bright",
                        "endpoint_question": "What is the state?",
                        "source_answer": "Dark.",
                        "target_answer": "Bright.",
                        "weight": 1.0,
                    }
                    for index in range(count)
                ],
                "preserve_constraints": [],
                "unresolved_instruction_items": [],
            }
        )
    )


class SmoothPairScorer:
    def score_multi_image_answer(self, images, question, answer, **kwargs):
        del answer, kwargs
        channel = int(question.split("channel ")[1][0])
        left, right = (image[:, channel].mean() for image in images)
        return -(left - right).square() + 0.05 * left


def _pairwise(count=1):
    source = torch.zeros(1, 3, 4, 4)
    full = torch.ones_like(source)
    spec = _spec(count)
    statements = {primitive.id: "The two images show a similar visible state." for primitive in spec.primitives}
    return PairwiseEndpointSemanticReward(SmoothPairScorer(), spec, source, full, statements=statements), source, full


def test_pairwise_prompt_contains_no_endpoint_or_choice_labels():
    prompt = build_pairwise_endpoint_affinity_prompt("visible object color")
    assert "visible object color" in prompt
    assert "SOURCE" not in prompt and "FULL" not in prompt
    assert "A." not in prompt and "B." not in prompt
    assert "percentage" in prompt


def test_pairwise_endpoint_coordinate_is_monotonic_symmetric_and_unclamped():
    reward, source, full = _pairwise()
    assert torch.allclose(reward(source, 0).achieved_score, torch.tensor(0.0))
    assert torch.allclose(reward(full, 1).achieved_score, torch.tensor(1.0))
    interior = torch.full_like(source, 0.4)
    values = reward.primitive_affinities(interior, reward.spec.primitives[0])
    assert torch.allclose(
        values["raw_margin"], 0.5 * (values["candidate_first_margin"] + values["reference_first_margin"])
    )
    assert 0 < reward(interior, 0.5).achieved_score < 1
    beyond = torch.full_like(source, 1.2, requires_grad=True)
    output = reward(beyond, 0.5)
    gradient = torch.autograd.grad(output.loss, beyond)[0]
    assert output.achieved_score > 1
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_pairwise_multi_primitive_weighted_targets():
    source = torch.zeros(1, 3, 4, 4)
    full = torch.ones_like(source)
    base_spec = _spec(3)
    weighted_spec = replace(
        base_spec,
        primitives=tuple(
            replace(primitive, weight=index + 1.0) for index, primitive in enumerate(base_spec.primitives)
        ),
    )
    statements = {
        primitive.id: "The two images show a similar visible state." for primitive in weighted_spec.primitives
    }
    reward = PairwiseEndpointSemanticReward(SmoothPairScorer(), weighted_spec, source, full, statements=statements)
    image = source.clone().requires_grad_(True)
    image.data[:, 0] = 0.2
    image.data[:, 1] = 0.5
    image.data[:, 2] = 0.8
    output = reward(image, {"attribute_0": 0.3, "attribute_1": 0.6, "attribute_2": 0.7})
    gradient = torch.autograd.grad(output.loss, image)[0]
    expected = (1 * 0.2 + 2 * 0.5 + 3 * 0.8) / 6
    assert torch.allclose(output.achieved_score, torch.tensor(expected))
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


class SmoothFeatureScorer:
    def __init__(self):
        self.model = torch.nn.Linear(1, 1, bias=False)
        self.model.requires_grad_(False)

    def focus_conditioned_representation(self, image, prompt):
        channel = int(prompt.split("channel ")[1][0])
        value = image[:, channel].mean()
        return torch.stack((1 - value, value))


def test_feature_distance_values_gradient_and_detached_endpoint_cache():
    source = torch.zeros(1, 3, 4, 4)
    full = torch.ones_like(source)
    reward = FeatureEndpointDistanceReward(SmoothFeatureScorer(), _spec(), source, full)
    assert torch.allclose(reward(source, 0).achieved_score, torch.tensor(0.0), atol=1e-7)
    assert torch.allclose(reward(full, 1).achieved_score, torch.tensor(1.0), atol=1e-7)
    candidate = torch.full_like(source, 0.4, requires_grad=True)
    output = reward(candidate, 0.7)
    gradient = torch.autograd.grad(output.loss, candidate)[0]
    diagnostics = output.diagnostics["attribute_0"]
    assert 0 < diagnostics["cosine_distance_ratio"] < 1
    assert 0 < diagnostics["euclidean_distance_ratio"] < 1
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert all(
        not feature.requires_grad for endpoints in reward._endpoint_features.values() for feature in endpoints.values()
    )
    assert source.grad is None and full.grad is None
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in reward.model.parameters())


def test_average_rank_spearman_and_ordering_separate_ties():
    values = [0, 0, 1, 1, 2]
    assert average_ranks(values) == [0.5, 0.5, 2.5, 2.5, 4.0]
    assert 0 < spearman_correlation(values) < 1
    diagnostics = ordering_diagnostics([0.0, 0.0, -1.0, 2.0])
    assert not diagnostics["strict_order_pass"]
    assert diagnostics["tie_count"] == 1
    assert diagnostics["descending_inversion_count"] == 2
    assert diagnostics["non_strict_violation_count"] == 3


def test_gradient_gate_requires_two_of_three_small_steps():
    def trials(flags, large=True):
        return [
            {"step_rms": 1e-5, "direction_correct": flags[0]},
            {"step_rms": 3e-5, "direction_correct": flags[1]},
            {"step_rms": 1e-4, "direction_correct": flags[2]},
            {"step_rms": 3e-4, "direction_correct": large},
        ]

    assert not gradient_direction_gate(trials((True, False, False)))["passed"]
    assert gradient_direction_gate(trials((True, True, False)))["passed"]
    assert gradient_direction_gate(trials((True, True, True)))["passed"]
    assert not gradient_direction_gate(trials((False, False, False), large=True))["passed"]


def _answer_scorer():
    scorer = Qwen25VQATeacherForcedScorer.__new__(Qwen25VQATeacherForcedScorer)
    scorer.max_answer_tokens = 70
    scorer.model = torch.nn.Linear(1, 1, bias=False)
    scorer.model.requires_grad_(False)
    scorer._differentiable_multi_image_inputs = lambda images: (
        torch.stack([image.mean() for image in images]).unsqueeze(1),
        torch.ones(len(images), 3, dtype=torch.long),
    )

    def aligned(pixel_values, image_grid_thw, question, answer):
        del image_grid_thw, question, answer
        scale = pixel_values.mean() * scorer.model.weight.flatten()[0]
        logits = torch.stack((scale, -scale, scale * 0.5)).repeat(2, 1)
        return logits, torch.tensor((0, 2))

    scorer._aligned_answer_logits = aligned
    return scorer


def test_qwen_multi_image_answer_supports_two_and_more_images_with_gradient():
    scorer = _answer_scorer()
    source = torch.zeros(1, 3, 2, 2)
    candidate = torch.rand_like(source, requires_grad=True)
    full = torch.ones_like(source)
    two = scorer.score_multi_image_answer((candidate, source), "question", "answer")
    more = scorer.score_multi_image_answer((candidate, source, full), "question", "answer")
    single = scorer.score_answer(candidate, "question", "answer")
    gradient = torch.autograd.grad(two + more + single, candidate)[0]
    assert all(score.ndim == 0 for score in (two, more, single))
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert source.grad is None and full.grad is None
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in scorer.model.parameters())


def test_qwen_focus_representation_uses_last_layer_last_prompt_token_with_gradient():
    scorer = Qwen25VQATeacherForcedScorer.__new__(Qwen25VQATeacherForcedScorer)

    class HiddenModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.frozen = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)

        def forward(self, input_ids, pixel_values, output_hidden_states, **kwargs):
            del kwargs
            assert output_hidden_states
            scale = pixel_values.mean() * self.frozen
            hidden = torch.arange(input_ids.shape[1] * 3, dtype=scale.dtype, device=scale.device).reshape(
                1, input_ids.shape[1], 3
            )
            hidden = hidden + scale * torch.tensor((1.0, 2.0, 3.0)).view(1, 1, 3)
            return SimpleNamespace(hidden_states=(hidden * 0.5, hidden))

    scorer.model = HiddenModel()
    scorer._model_device_and_dtype = lambda: (torch.device("cpu"), torch.float32)
    scorer._differentiable_image_inputs = lambda image: (image.reshape(-1, 3), torch.ones(1, 3, dtype=torch.long))
    scorer._build_prompt_prefix_inputs = lambda grid, device, prompt: (
        torch.ones(1, 4, dtype=torch.long),
        torch.ones(1, 4, dtype=torch.long),
    )
    image = torch.rand(1, 3, 2, 2, requires_grad=True)
    representation = scorer.focus_conditioned_representation(image, "focus")
    gradient = torch.autograd.grad(representation[0], image)[0]
    scale = image.detach().mean()
    expected = torch.nn.functional.normalize(torch.tensor((9.0, 10.0, 11.0)) + scale * torch.tensor((1, 2, 3)), dim=0)
    assert torch.allclose(representation, expected)
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in scorer.model.parameters())


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float32), ids=("bf16", "fp32"))
@pytest.mark.skipif(
    not os.getenv("QWEN25_VL_MODEL_PATH") or not torch.cuda.is_available(),
    reason="requires QWEN25_VL_MODEL_PATH and CUDA",
)
def test_real_qwen_pairwise_answer_and_feature_gradients_when_local_model_is_available(dtype):
    scorer = Qwen25VQATeacherForcedScorer(
        os.environ["QWEN25_VL_MODEL_PATH"], device="cuda", dtype=dtype, local_files_only=True
    )
    source = torch.zeros(1, 3, 56, 56, device="cuda")
    candidate = torch.rand_like(source, requires_grad=True)
    answer_score = scorer.score_multi_image_answer(
        (candidate, source),
        "Do the two images show a similar visible object color?",
        "The two images show a similar visible object color.",
    )
    answer_gradient = torch.autograd.grad(answer_score, candidate)[0]
    candidate = torch.rand_like(source, requires_grad=True)
    representation = scorer.focus_conditioned_representation(
        candidate, "Focus only on visible object color. Represent this attribute."
    )
    feature_gradient = torch.autograd.grad(representation[0], candidate)[0]
    assert torch.isfinite(answer_gradient).all() and answer_gradient.abs().sum() > 0
    assert torch.isfinite(feature_gradient).all() and feature_gradient.abs().sum() > 0
    assert source.grad is None
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in scorer.model.parameters())
