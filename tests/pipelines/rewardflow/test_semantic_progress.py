from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from diffusers.pipelines.rewardflow import rewards as rewards_module
from diffusers.pipelines.rewardflow.rewards import Qwen25VQAReward, Qwen25VQATeacherForcedScorer
from diffusers.pipelines.rewardflow.semantic_progress import (
    EndpointRelativeSemanticProgressReward,
    EndpointSemanticValidationError,
    make_semantic_progress_cache_key,
    parse_semantic_progress_json,
)


def _payload(primitive_count=1):
    primitives = [
        {
            "id": "ball_color",
            "edit_description": "change ball color to blue",
            "question": "What color is the weighted training ball?",
            "source_answer": "black",
            "target_answer": "blue",
            "weight": 2.0,
        }
    ]
    if primitive_count == 2:
        primitives.append(
            {
                "id": "ball_finish",
                "edit_description": "change ball finish",
                "question": "What finish does the ball have?",
                "source_answer": "matte",
                "target_answer": "glossy",
                "weight": 1.0,
            }
        )
    return {
        "edit_instruction": "Make the weighted training ball blue.",
        "primitives": primitives,
        "preserve_constraints": ["shape", "background"],
    }


class LinearContrastScorer:
    def __init__(self):
        self.model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.model.weight.fill_(1)
        self.model.requires_grad_(False)

    def score_answer(self, image, question, answer):
        value = self.model(image.mean().reshape(1, 1)).squeeze()
        return value if answer in {"blue", "glossy"} else 1 - value


def _spec(primitive_count=1):
    return parse_semantic_progress_json(json.dumps(_payload(primitive_count)))


def _reward(primitive_count=1):
    scorer = LinearContrastScorer()
    source = torch.zeros(1, 3, 2, 2)
    full = torch.ones_like(source)
    return EndpointRelativeSemanticProgressReward(scorer, _spec(primitive_count), source, full), scorer


def test_semantic_progress_parser_parses_one_primitive():
    spec = _spec()

    assert len(spec.primitives) == 1
    assert spec.primitives[0].id == "ball_color"
    assert spec.primitives[0].source_answer == "black"
    assert spec.primitives[0].target_answer == "blue"


def test_semantic_progress_parser_supports_multiple_primitives():
    spec = _spec(2)

    assert [primitive.id for primitive in spec.primitives] == ["ball_color", "ball_finish"]


def test_semantic_progress_cache_key_binds_both_endpoints_instruction_and_version():
    baseline = make_semantic_progress_cache_key("source-a", "full-a", "turn blue", "parser-a")

    assert baseline != make_semantic_progress_cache_key("source-b", "full-a", "turn blue", "parser-a")
    assert baseline != make_semantic_progress_cache_key("source-a", "full-b", "turn blue", "parser-a")
    assert baseline != make_semantic_progress_cache_key("source-a", "full-a", "turn red", "parser-a")
    assert baseline != make_semantic_progress_cache_key("source-a", "full-a", "turn blue", "parser-b")


def test_qwen_vqa_reward_remains_a_fixed_question_answer_compatibility_wrapper(monkeypatch):
    calls = []

    class FakeScorer:
        def __init__(self, **kwargs):
            self.model = torch.nn.Linear(1, 1)
            self.processor = object()

        def score_answer(self, image, question, answer, **kwargs):
            calls.append((question, answer, kwargs))
            return image.mean()

        def compare_with_official_processor(self, image, question, answer, **kwargs):
            return {"question": question, "answer": answer, **kwargs}

    monkeypatch.setattr(rewards_module, "Qwen25VQATeacherForcedScorer", FakeScorer)
    reward = Qwen25VQAReward("Question?", "Answer.", margin=0.5, lambda_margin=0.25)
    image = torch.ones(1, 3, 2, 2, requires_grad=True)

    value = reward(image, "unused")

    assert value.item() == 1
    assert calls == [("Question?", "Answer.", {"margin": 0.5, "lambda_margin": 0.25})]
    assert reward.compare_with_official_processor(image)["margin"] == 0.5


def test_semantic_contrast_and_endpoint_normalization_increase_from_source_to_full():
    reward, _ = _reward()
    source_progress, _ = reward.progress_vector(torch.zeros(1, 3, 2, 2))
    middle_progress, _ = reward.progress_vector(torch.full((1, 3, 2, 2), 0.4))
    full_progress, _ = reward.progress_vector(torch.ones(1, 3, 2, 2))

    torch.testing.assert_close(source_progress, torch.tensor([0.0]))
    torch.testing.assert_close(middle_progress, torch.tensor([0.4]))
    torch.testing.assert_close(full_progress, torch.tensor([1.0]))


def test_semantic_progress_loss_has_finite_nonzero_image_gradient_and_frozen_model():
    reward, scorer = _reward()
    image = torch.full((1, 3, 2, 2), 0.5, requires_grad=True)

    output = reward(image, 0.8)
    output.loss.backward()

    assert image.grad is not None
    assert torch.isfinite(image.grad).all()
    assert image.grad.abs().sum() > 0
    assert all(not parameter.requires_grad for parameter in scorer.model.parameters())
    assert all(parameter.grad is None for parameter in scorer.model.parameters())


def test_unclamped_out_of_range_progress_retains_loss_gradient():
    reward, _ = _reward()
    image = torch.full((1, 3, 2, 2), 1.5, requires_grad=True)

    output = reward(image, 0.8)
    gradient = torch.autograd.grad(output.loss, image)[0]

    assert output.achieved_score.item() == pytest.approx(1.5)
    assert output.diagnostics["ball_color"]["progress_clamped"].item() == 1
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_two_primitive_weighted_aggregation_is_numerically_correct():
    reward, _ = _reward(2)
    image = torch.full((1, 3, 2, 2), 0.6)

    output = reward(image, {"ball_color": 0.2, "ball_finish": 0.9})

    expected_loss = (2 * (0.6 - 0.2) ** 2 + (0.6 - 0.9) ** 2) / 3
    expected_error = (2 * abs(0.6 - 0.2) + abs(0.6 - 0.9)) / 3
    assert output.loss.item() == pytest.approx(expected_loss)
    assert output.objective_error.item() == pytest.approx(expected_error)
    assert output.achieved_score.item() == pytest.approx(0.6)


def test_endpoint_semantic_validation_fails_without_source_preference():
    class AlwaysTargetScorer:
        def score_answer(self, image, question, answer):
            return image.mean() + (1 if answer == "blue" else 0)

    with pytest.raises(EndpointSemanticValidationError):
        EndpointRelativeSemanticProgressReward(
            AlwaysTargetScorer(),
            _spec(),
            torch.zeros(1, 3, 2, 2),
            torch.ones(1, 3, 2, 2),
        )


def test_real_qwen_semantic_contrast_image_gradient_when_local_model_is_available():
    model_path = os.getenv("QWEN25_VL_MODEL_PATH")
    if not model_path or not Path(model_path).exists():
        pytest.skip("Set QWEN25_VL_MODEL_PATH to run the real semantic-gradient integration test.")
    if not torch.cuda.is_available():
        pytest.skip("Real Qwen semantic-gradient integration requires CUDA.")
    scorer = Qwen25VQATeacherForcedScorer(
        model_path,
        device="cuda",
        dtype=torch.bfloat16,
        local_files_only=True,
    )
    image = torch.rand(1, 3, 56, 56, device="cuda", requires_grad=True)

    source_score = scorer.score_answer(image, "What color is the square?", "black")
    target_score = scorer.score_answer(image, "What color is the square?", "blue")
    semantic_loss = ((target_score - source_score) - 0.25).square()
    gradient = torch.autograd.grad(semantic_loss, image)[0]

    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0
    assert all(not parameter.requires_grad for parameter in scorer.model.parameters())
    assert all(parameter.grad is None for parameter in scorer.model.parameters())
