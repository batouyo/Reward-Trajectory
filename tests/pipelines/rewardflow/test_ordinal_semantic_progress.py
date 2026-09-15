import json
import os
from types import SimpleNamespace

import pytest
import torch

from diffusers.pipelines.rewardflow.ordinal_semantic_progress import (
    EndpointRelativeOrdinalSemanticProgressReward,
    build_ordinal_semantic_progress_parser_prompt,
    load_cached_ordinal_semantic_progress_spec,
    make_ordinal_semantic_progress_cache_key,
    ordinal_choice_distribution,
    parse_ordinal_semantic_progress_json,
    save_cached_ordinal_semantic_progress_spec,
)
from diffusers.pipelines.rewardflow.rewards import Qwen25VQATeacherForcedScorer


def _question(question_id="color_state", question="Which visual color state best matches the ball?", weight=1.0):
    return {
        "id": question_id,
        "question": question,
        "weight": weight,
        "stages": [
            "original dark appearance",
            "mostly dark with emerging blue",
            "balanced dark and blue appearance",
            "predominantly blue with residual dark appearance",
            "full target blue appearance",
        ],
    }


def _payload(question_count=1, primitive_count=1):
    primitives = []
    for primitive_index in range(primitive_count):
        questions = [
            _question(
                f"p{primitive_index}_q{question_index}",
                f"Which visual state best matches primitive {primitive_index} view {question_index}?",
                question_index + 1,
            )
            for question_index in range(question_count)
        ]
        primitives.append(
            {
                "id": f"primitive_{primitive_index}",
                "edit_description": f"change primitive {primitive_index}",
                "weight": primitive_index + 1,
                "questions": questions,
            }
        )
    return {
        "edit_instruction": "change the ball color",
        "primitives": primitives,
        "preserve_constraints": ["shape", "background"],
    }


class SmoothOrdinalScorer:
    def __init__(self):
        self.model = torch.nn.Linear(1, 1, bias=False)
        self.model.requires_grad_(False)

    def score_single_token_choices(self, image, question, choices=("A", "B", "C", "D", "E")):
        channel = 1 if "primitive 1" in question else 0
        value = image[:, channel].mean()
        nodes = value.new_tensor((0.0, 0.25, 0.5, 0.75, 1.0))
        return -32 * (value - nodes).square()


def _reward(question_count=1, primitive_count=1):
    spec = parse_ordinal_semantic_progress_json(json.dumps(_payload(question_count, primitive_count)))
    source = torch.zeros(1, 3, 4, 4)
    full = torch.ones_like(source)
    return EndpointRelativeOrdinalSemanticProgressReward(SmoothOrdinalScorer(), spec, source, full), source, full


def test_qwen_single_token_choice_scoring_returns_k_values_and_image_gradient():
    scorer = Qwen25VQATeacherForcedScorer.__new__(Qwen25VQATeacherForcedScorer)

    class Tokenizer:
        ids = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}

        def __call__(self, text, **kwargs):
            del kwargs
            values = [self.ids[text]] if text in self.ids else [7, 8]
            return SimpleNamespace(input_ids=torch.tensor([values]))

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.frozen = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)

        def forward(self, input_ids, pixel_values, **kwargs):
            del kwargs
            scale = pixel_values.mean() * self.frozen
            vocabulary = torch.arange(12, device=scale.device, dtype=scale.dtype)
            logits = scale * vocabulary.view(1, 1, -1).expand(1, input_ids.shape[1], -1)
            return SimpleNamespace(logits=logits)

    scorer.processor = SimpleNamespace(
        tokenizer=Tokenizer(),
        image_token="<image>",
        image_processor=SimpleNamespace(merge_size=1),
        apply_chat_template=lambda *args, **kwargs: "<image>prompt",
    )
    scorer.model = Model()
    scorer._model_device_and_dtype = lambda: (torch.device("cpu"), torch.float32)
    scorer._differentiable_image_inputs = lambda image: (image.reshape(-1, 3), torch.tensor([[1, 1, 1]]))
    image = torch.rand(1, 3, 2, 2, requires_grad=True)
    scores = scorer.score_single_token_choices(image, "question")
    gradient = torch.autograd.grad(scores.sum(), image)[0]

    assert scores.shape == (5,)
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in scorer.model.parameters())


def test_qwen_choice_labels_must_be_single_distinct_tokens():
    scorer = Qwen25VQATeacherForcedScorer.__new__(Qwen25VQATeacherForcedScorer)

    class Tokenizer:
        def __call__(self, text, **kwargs):
            del kwargs
            mapping = {"A": [1], "B": [2], "duplicate": [1], "long": [3, 4]}
            return SimpleNamespace(input_ids=torch.tensor([mapping[text]]))

    scorer.processor = SimpleNamespace(tokenizer=Tokenizer())
    assert scorer._choice_token_ids(("A", "B"), torch.device("cpu")).shape == (2,)
    with pytest.raises(ValueError, match="exactly one token"):
        scorer._choice_token_ids(("A", "long"), torch.device("cpu"))
    with pytest.raises(ValueError, match="distinct"):
        scorer._choice_token_ids(("A", "duplicate"), torch.device("cpu"))


def test_ordinal_parser_supports_single_multi_question_and_future_multi_primitive():
    single = parse_ordinal_semantic_progress_json(json.dumps(_payload()))
    ensemble = parse_ordinal_semantic_progress_json(json.dumps(_payload(question_count=3)))
    multi = parse_ordinal_semantic_progress_json(json.dumps(_payload(question_count=2, primitive_count=2)))
    assert len(single.primitives[0].questions) == 1
    assert len(ensemble.primitives[0].questions) == 3
    assert len(multi.primitives) == 2
    assert "exactly five ordered stage descriptions" in build_ordinal_semantic_progress_parser_prompt("turn blue")


@pytest.mark.parametrize("forbidden", ("50% blue", "fifty percent blue", "high strength blue"))
def test_ordinal_parser_rejects_wrong_duplicate_or_forbidden_stages(forbidden):
    payload = _payload()
    payload["primitives"][0]["questions"][0]["stages"] = ["a", "b", "c", "d"]
    with pytest.raises(ValueError, match="exactly five"):
        parse_ordinal_semantic_progress_json(json.dumps(payload))
    payload = _payload()
    payload["primitives"][0]["questions"][0]["stages"][1] = payload["primitives"][0]["questions"][0]["stages"][0]
    with pytest.raises(ValueError, match="unique"):
        parse_ordinal_semantic_progress_json(json.dumps(payload))
    payload = _payload()
    payload["primitives"][0]["questions"][0]["stages"][2] = forbidden
    with pytest.raises(ValueError, match="percentages or strength"):
        parse_ordinal_semantic_progress_json(json.dumps(payload))


def test_ordinal_cache_key_binds_both_endpoints_instruction_and_version(tmp_path):
    base = make_ordinal_semantic_progress_cache_key("source", "full", "turn blue")
    assert base != make_ordinal_semantic_progress_cache_key("other", "full", "turn blue")
    assert base != make_ordinal_semantic_progress_cache_key("source", "other", "turn blue")
    assert base != make_ordinal_semantic_progress_cache_key("source", "full", "turn red")
    assert base != make_ordinal_semantic_progress_cache_key("source", "full", "turn blue", "v3")
    spec = parse_ordinal_semantic_progress_json(json.dumps(_payload()))
    path = tmp_path / "ordinal.json"
    save_cached_ordinal_semantic_progress_spec(path, "source", "full", spec)
    assert load_cached_ordinal_semantic_progress_spec(path, "source", "full", spec.edit_instruction) == spec
    assert load_cached_ordinal_semantic_progress_spec(path, "source", "wrong", spec.edit_instruction) is None


def test_ordinal_choice_probabilities_sum_to_one_and_expectation_is_correct():
    probabilities = torch.tensor((0.05, 0.15, 0.55, 0.20, 0.05))
    label_probs, stage_probs, expectation = ordinal_choice_distribution(probabilities.log())
    assert torch.allclose(label_probs.sum(), torch.tensor(1.0))
    assert torch.allclose(stage_probs, probabilities)
    assert torch.allclose(expectation, torch.tensor(0.5125))


def test_ordinal_endpoint_calibration_and_intermediate_value():
    reward, source, full = _reward()
    source_progress, _ = reward.progress_vector(source)
    full_progress, _ = reward.progress_vector(full)
    middle_progress, diagnostics = reward.progress_vector((source + full) / 2)
    assert torch.allclose(source_progress, torch.zeros_like(source_progress), atol=1e-6)
    assert torch.allclose(full_progress, torch.ones_like(full_progress), atol=1e-6)
    assert torch.allclose(middle_progress, torch.full_like(middle_progress, 0.5), atol=1e-5)
    question = diagnostics["primitive_0"]["questions"]["p0_q0"]
    assert torch.allclose(question["choice_probs"].sum(), torch.tensor(1.0))


def test_out_of_range_progress_is_unclamped_for_loss_and_retains_gradient():
    reward, _, _ = _reward()
    image = torch.full((1, 3, 4, 4), 1.2, requires_grad=True)
    output = reward(image, 0.2)
    gradient = torch.autograd.grad(output.loss, image)[0]
    raw = output.diagnostics["primitive_0"]["primitive_progress"]
    assert raw > 1
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_multi_question_and_multi_primitive_weighted_aggregation():
    reward, source, full = _reward(question_count=3, primitive_count=2)
    image = source.clone()
    image[:, 0] = 0.25
    image[:, 1] = 0.75
    progress, diagnostics = reward.progress_vector(image)
    assert 0 < progress[0] < 0.5 < progress[1] < 1
    assert len(diagnostics["primitive_0"]["questions"]) == 3
    p0_questions = torch.stack(
        [question["endpoint_calibrated_progress_raw"] for question in diagnostics["primitive_0"]["questions"].values()]
    )
    p0_weights = p0_questions.new_tensor((1.0, 2.0, 3.0))
    assert torch.allclose(progress[0], (p0_questions * p0_weights / p0_weights.sum()).sum())
    output = reward(image, {"primitive_0": 0.2, "primitive_1": 0.8})
    expected_progress = (1 * progress[0] + 2 * progress[1]) / 3
    expected_loss = ((progress[0] - 0.2).square() + 2 * (progress[1] - 0.8).square()) / 3
    assert torch.allclose(output.achieved_score, expected_progress)
    assert torch.allclose(output.loss, expected_loss)


@pytest.mark.skipif(
    not os.getenv("QWEN25_VL_MODEL_PATH") or not torch.cuda.is_available(),
    reason="requires QWEN25_VL_MODEL_PATH and CUDA",
)
def test_real_qwen_ordinal_choice_gradient_when_local_model_is_available():
    pytest.importorskip("transformers")
    scorer = Qwen25VQATeacherForcedScorer(
        os.environ["QWEN25_VL_MODEL_PATH"], device="cuda", dtype=torch.bfloat16, local_files_only=True
    )
    image = torch.rand(1, 3, 112, 112, device="cuda", requires_grad=True)
    logprobs = scorer.score_single_token_choices(
        image, "Which option fits?\nA. one\nB. two\nC. three\nD. four\nE. five"
    )
    _, _, expectation = ordinal_choice_distribution(logprobs)
    gradient = torch.autograd.grad((expectation - 0.5).square(), image)[0]
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in scorer.model.parameters())
