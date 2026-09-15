import json
import os
from types import SimpleNamespace

import pytest
import torch

from diffusers.pipelines.rewardflow.relative_endpoint_parser import parse_relative_endpoint_semantic_json
from diffusers.pipelines.rewardflow.relative_endpoint_semantic import (
    RelativeEndpointSemanticReward,
    RelativeEndpointValidationError,
    audit_endpoint_answers,
    build_relative_endpoint_comparison_prompt,
)
from diffusers.pipelines.rewardflow.rewards import Qwen25VQATeacherForcedScorer


def _spec(count=1):
    primitives = []
    for index in range(count):
        primitives.append(
            {
                "id": f"attribute_{index}",
                "object": "object",
                "attribute": f"attribute {index}",
                "edit_description": "change the object",
                "comparison_focus": f"object channel {index}",
                "source_state": "dark",
                "target_state": "bright",
                "endpoint_question": "Is the object dark or bright?",
                "source_answer": "Dark.",
                "target_answer": "Bright.",
                "weight": 1.0,
            }
        )
    return parse_relative_endpoint_semantic_json(
        json.dumps(
            {
                "edit_instruction": "brighten the object",
                "primitives": primitives,
                "preserve_constraints": ["background"],
                "unresolved_instruction_items": [],
            }
        )
    )


class SmoothRelativeScorer:
    def __init__(self):
        self.model = torch.nn.Linear(1, 1, bias=False)
        self.model.requires_grad_(False)

    def score_multi_image_single_token_choices(self, images, question, choices=("A", "B")):
        del choices
        channel = int(question.split("object channel ")[1][0]) if "object channel " in question else 0
        first, candidate, third = [image[:, channel].mean() for image in images]
        distance_first = (candidate - first).square()
        distance_third = (candidate - third).square()
        return torch.stack((-distance_first, -distance_third))

    def score_answer(self, image, question, answer):
        del question
        value = image.mean()
        target = value.new_tensor(0.0 if answer == "Dark." else 1.0)
        return -(value - target).square()


def _reward(count=1, fail=True):
    source = torch.zeros(1, 3, 4, 4)
    full = torch.ones_like(source)
    return (
        RelativeEndpointSemanticReward(
            SmoothRelativeScorer(), _spec(count), source, full, fail_on_endpoint_validation=fail
        ),
        source,
        full,
    )


def test_prompt_tracks_image_order_without_percent_or_stage_language():
    spec = _spec()
    forward = build_relative_endpoint_comparison_prompt(spec, spec.primitives[0], first_reference="source")
    reverse = build_relative_endpoint_comparison_prompt(spec, spec.primitives[0], first_reference="full")
    assert "Image 1 is the SOURCE" in forward and "Image 3 is the NATIVE FULL-EDIT" in forward
    assert "Image 1 is the NATIVE FULL-EDIT" in reverse and "Image 3 is the SOURCE" in reverse
    assert "%" not in forward and "20%" not in forward


def test_endpoints_calibrate_exactly_and_interior_is_unclamped():
    reward, source, full = _reward()
    assert torch.allclose(reward(source, 0).achieved_score, torch.tensor(0.0))
    assert torch.allclose(reward(full, 1).achieved_score, torch.tensor(1.0))
    beyond = torch.full_like(source, 1.2, requires_grad=True)
    output = reward(beyond, 0.5)
    gradient = torch.autograd.grad(output.loss, beyond)[0]
    assert output.achieved_score > 1
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_symmetrized_margin_and_target_margin_loss_are_correct():
    reward, source, full = _reward()
    image = torch.full_like(source, 0.25)
    output = reward(image, 0.75)
    diagnostic = output.diagnostics["attribute_0"]
    expected_margin = 0.5 * (diagnostic["forward_1_margin"] + diagnostic["forward_2_margin"])
    assert torch.allclose(diagnostic["raw_margin"], expected_margin)
    expected_residual = (diagnostic["raw_margin"] - diagnostic["target_margin"]) / diagnostic["dynamic_range"]
    assert torch.allclose(output.loss, expected_residual.square())


def test_multi_primitive_targets_and_gradients_are_supported():
    reward, source, _ = _reward(3)
    image = source.clone().requires_grad_(True)
    image.data[:, 0] = 0.2
    image.data[:, 1] = 0.5
    image.data[:, 2] = 0.8
    output = reward(image, {"attribute_0": 0.3, "attribute_1": 0.6, "attribute_2": 0.7})
    gradient = torch.autograd.grad(output.loss, image)[0]
    assert len(output.diagnostics) == 3
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_endpoint_validation_fails_on_reversed_or_collapsed_range():
    class AlwaysFullScorer(SmoothRelativeScorer):
        def score_multi_image_single_token_choices(self, images, question, choices=("A", "B")):
            del images, question, choices
            return torch.tensor((0.0, 1.0))

    source = torch.ones(1, 3, 2, 2)
    full = torch.zeros_like(source)
    with pytest.raises(RelativeEndpointValidationError):
        RelativeEndpointSemanticReward(AlwaysFullScorer(), _spec(), source, full)
    with pytest.raises(RelativeEndpointValidationError):
        RelativeEndpointSemanticReward(SmoothRelativeScorer(), _spec(), source, source)


def test_endpoint_teacher_forced_audit_is_separate_from_continuous_reward():
    _, source, full = _reward()
    report = audit_endpoint_answers(SmoothRelativeScorer(), _spec(), source, full)
    assert report["attribute_0"]["valid"]
    assert report["attribute_0"]["used_for_continuous_reward"] is False


def test_qwen_multi_image_choices_use_one_forward_and_preserve_candidate_gradient():
    scorer = Qwen25VQATeacherForcedScorer.__new__(Qwen25VQATeacherForcedScorer)

    class Tokenizer:
        def __call__(self, text, **kwargs):
            del kwargs
            mapping = {"A": [1], "B": [2]}
            return SimpleNamespace(input_ids=torch.tensor([mapping.get(text, [7, 8])]))

    calls = []

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.frozen = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)

        def forward(self, input_ids, pixel_values, **kwargs):
            del kwargs
            calls.append(1)
            scale = pixel_values.mean() * self.frozen
            vocabulary = torch.arange(10, dtype=scale.dtype, device=scale.device)
            return SimpleNamespace(logits=scale * vocabulary.view(1, 1, -1).expand(1, input_ids.shape[1], -1))

    scorer.processor = SimpleNamespace(
        tokenizer=Tokenizer(),
        image_token="<image>",
        image_processor=SimpleNamespace(merge_size=1),
        apply_chat_template=lambda messages, **kwargs: "<image>" * len(messages[0]["content"][:-1]) + "prompt",
    )
    scorer.model = Model()
    scorer._model_device_and_dtype = lambda: (torch.device("cpu"), torch.float32)
    scorer._differentiable_image_inputs = lambda image: (image.reshape(-1, 3), torch.tensor([[1, 1, 1]]))
    source = torch.zeros(1, 3, 2, 2)
    candidate = torch.rand_like(source, requires_grad=True)
    full = torch.ones_like(source)
    scores = scorer.score_multi_image_single_token_choices((source, candidate, full), "question")
    gradient = torch.autograd.grad(scores[1] - scores[0], candidate)[0]
    assert scores.shape == (2,) and len(calls) == 1
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in scorer.model.parameters())
    single = scorer.score_single_token_choices(candidate, "question", ("A", "B"))
    one_image_multi = scorer.score_multi_image_single_token_choices((candidate,), "question", ("A", "B"))
    assert torch.equal(single, one_image_multi)


def _preprocessing_scorer():
    scorer = Qwen25VQATeacherForcedScorer.__new__(Qwen25VQATeacherForcedScorer)
    scorer.processor = SimpleNamespace(
        image_processor=SimpleNamespace(
            patch_size=2,
            temporal_patch_size=2,
            merge_size=1,
            min_pixels=4,
            max_pixels=4096,
            image_mean=(0.5, 0.5, 0.5),
            image_std=(0.5, 0.5, 0.5),
        )
    )
    return scorer


def test_three_image_preprocessing_has_three_grid_rows_and_candidate_gradient():
    scorer = _preprocessing_scorer()
    source = torch.zeros(1, 3, 4, 4)
    candidate = torch.rand(1, 3, 6, 6, requires_grad=True)
    full = torch.ones(1, 3, 8, 4)
    pixels, grids = scorer._differentiable_multi_image_inputs((source, candidate, full))
    gradient = torch.autograd.grad(pixels.square().mean(), candidate)[0]
    assert grids.shape == (3, 3)
    assert grids[:, 1:].tolist() == [[2, 2], [3, 3], [4, 2]]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert source.grad is None and full.grad is None


def test_multi_image_prompt_requires_one_placeholder_per_image():
    scorer = _preprocessing_scorer()
    scorer.processor.image_token = "<image>"
    scorer.processor.tokenizer = lambda *args, **kwargs: SimpleNamespace(input_ids=torch.tensor([[1, 2]]))
    scorer.processor.apply_chat_template = lambda *args, **kwargs: "<image>only-one"
    with pytest.raises(NotImplementedError, match="exactly 3"):
        scorer._build_prompt_prefix_inputs(torch.ones(3, 3, dtype=torch.long), torch.device("cpu"), "question")


@pytest.mark.skipif(
    not os.getenv("QWEN25_VL_MODEL_PATH") or not torch.cuda.is_available(),
    reason="requires QWEN25_VL_MODEL_PATH and CUDA",
)
def test_real_qwen_relative_endpoint_candidate_gradient_when_local_model_is_available():
    scorer = Qwen25VQATeacherForcedScorer(
        os.environ["QWEN25_VL_MODEL_PATH"], device="cuda", dtype=torch.bfloat16, local_files_only=True
    )
    source = torch.zeros(1, 3, 56, 56, device="cuda")
    candidate = torch.rand_like(source, requires_grad=True)
    full = torch.ones_like(source)
    scores = scorer.score_multi_image_single_token_choices(
        (source, candidate, full),
        "Which reference is visually closer to Image 2?\nA. Image 1\nB. Image 3\nAnswer with A or B.",
    )
    gradient = torch.autograd.grad(scores[1] - scores[0], candidate)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert source.grad is None and full.grad is None
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in scorer.model.parameters())
