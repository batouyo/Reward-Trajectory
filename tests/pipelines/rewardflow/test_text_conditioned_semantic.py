import json
from types import SimpleNamespace

import pytest
import torch

from diffusers.pipelines.rewardflow.text_conditioned_semantic import (
    TEXT_CONDITIONED_SEMANTIC_PARSER_VERSION,
    TextConditionedParseRecord,
    TextConditionedSemanticGeometry,
    TextSemanticEndpointDirectionError,
    TianyuAITextConditionedSemanticParser,
    load_cached_text_conditioned_parse,
    make_text_conditioned_semantic_cache_key,
    parse_text_conditioned_semantic_json,
    save_cached_text_conditioned_parse,
)


def _payload():
    return {
        "edit_instruction": "Turn the ball blue.",
        "primitives": [
            {
                "id": "ball_color",
                "object": "weighted training ball",
                "attribute": "visible surface color",
                "edit_description": "change the weighted training ball from dark gray to vivid blue",
                "comparison_focus": "visible surface color of the weighted training ball",
                "source_state": "dark gray",
                "target_state": "vivid blue",
                "source_semantic_text": "a dark gray weighted training ball",
                "target_semantic_text": "a vivid blue weighted training ball",
                "endpoint_question": "What color is the weighted training ball?",
                "source_answer": "Dark gray.",
                "target_answer": "Vivid blue.",
                "weight": 1.0,
            }
        ],
        "preserve_constraints": ["ball shape", "background composition"],
        "unresolved_instruction_items": [],
    }


def test_parser_requires_distinct_nonempty_state_only_matched_text_and_unit_weight():
    parsed = parse_text_conditioned_semantic_json(json.dumps(_payload()))
    assert parsed.primitives[0].source_semantic_text == "a dark gray weighted training ball"
    for field, value in (
        ("source_semantic_text", ""),
        ("target_semantic_text", "a dark gray weighted training ball"),
        ("target_semantic_text", "the final ball at 80 percent strength"),
    ):
        payload = _payload()
        payload["primitives"][0][field] = value
        with pytest.raises(ValueError):
            parse_text_conditioned_semantic_json(json.dumps(payload))
    payload = _payload()
    payload["primitives"][0]["weight"] = 0.5
    with pytest.raises(ValueError, match="exactly 1.0"):
        parse_text_conditioned_semantic_json(json.dumps(payload))


def test_cache_identity_binds_endpoints_instruction_parser_and_provider_without_credentials(tmp_path):
    base = make_text_conditioned_semantic_cache_key("source", "full", "turn blue")
    assert base != make_text_conditioned_semantic_cache_key("changed", "full", "turn blue")
    assert base != make_text_conditioned_semantic_cache_key("source", "changed", "turn blue")
    assert base != make_text_conditioned_semantic_cache_key("source", "full", "turn red")
    assert base != make_text_conditioned_semantic_cache_key("source", "full", "turn blue", model="other")
    assert base != make_text_conditioned_semantic_cache_key("source", "full", "turn blue", provider="other")
    assert base != make_text_conditioned_semantic_cache_key("source", "full", "turn blue", parser_version="other")
    spec = parse_text_conditioned_semantic_json(json.dumps(_payload()))
    record = TextConditionedParseRecord(
        spec,
        {
            "provider": "tianyuai",
            "model": "gpt-5.6-luna",
            "parser_version": TEXT_CONDITIONED_SEMANTIC_PARSER_VERSION,
            "source_fingerprint": "source",
            "full_fingerprint": "full",
            "edit_instruction": spec.edit_instruction,
            "created_at": "2026-09-16T00:00:00+00:00",
            "base_url": "https://tianyuai.lol/v1",
            "cache_hit": False,
        },
    )
    cache = tmp_path / "cache.json"
    save_cached_text_conditioned_parse(cache, record)
    loaded = load_cached_text_conditioned_parse(cache, "source", "full", spec.edit_instruction)
    assert loaded.spec == spec and loaded.provenance["cache_hit"] is True
    serialized = cache.read_text().casefold()
    assert "api_key" not in serialized and "secret" not in serialized


@pytest.mark.parametrize(
    "content_wrapper",
    [
        lambda value: value,
        lambda value: [{"type": "text", "text": value}],
        lambda value: [
            {"type": "reasoning", "text": "private reasoning is not parser JSON"},
            {"type": "output_text", "text": value},
        ],
        lambda value: [
            {"type": "output_text", "text": value},
            {"type": "output_text", "text": value},
        ],
    ],
)
def test_tianyu_parser_uses_one_call_with_only_ordered_endpoints_and_instruction(tmp_path, content_wrapper):
    from PIL import Image

    source = tmp_path / "source.png"
    full = tmp_path / "full.png"
    Image.new("RGB", (2, 2), "black").save(source)
    Image.new("RGB", (2, 2), "blue").save(full)
    calls = []

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content_wrapper(json.dumps(_payload()))))]
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    record = TianyuAITextConditionedSemanticParser(client=client).parse(source, full, "Turn the ball blue.")
    assert len(calls) == 1
    content = calls[0]["messages"][0]["content"]
    assert [item["type"] for item in content] == ["image_url", "image_url", "text"]
    assert record.provenance["parser_version"] == TEXT_CONDITIONED_SEMANTIC_PARSER_VERSION


class _SyntheticEncoder:
    def __init__(self, reverse_endpoint=False):
        self.reverse_endpoint = reverse_endpoint
        self.last_text_metadata = {"truncated": False}

    def encode_text(self, text):
        return torch.tensor([1.0, 0.0]) if "source" in text else torch.tensor([0.0, 1.0])

    def encode_image(self, image):
        value = image.mean()
        if self.reverse_endpoint:
            value = 1 - value
        feature = torch.stack((1 - value, value))
        return torch.nn.functional.normalize(feature, dim=0)


def test_text_margin_geometry_normalizes_source_full_and_keeps_candidate_gradient():
    source = torch.zeros(1, 3, 2, 2, requires_grad=True)
    full = torch.ones(1, 3, 2, 2, requires_grad=True)
    geometry = TextConditionedSemanticGeometry(_SyntheticEncoder(), source, full, "source state", "target state")
    source_output = geometry(source.detach())
    full_output = geometry(full.detach())
    middle = torch.full((1, 3, 2, 2), 0.5, requires_grad=True)
    middle_output = geometry(middle)
    gradient = torch.autograd.grad(middle_output.progress, middle)[0]
    torch.testing.assert_close(source_output.progress, torch.tensor(0.0), atol=1e-6, rtol=0)
    torch.testing.assert_close(full_output.progress, torch.tensor(1.0), atol=1e-6, rtol=0)
    assert 0 < float(middle_output.progress) < 1
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert not geometry.source_feature.requires_grad
    assert not geometry.full_feature.requires_grad
    assert not geometry.source_text_feature.requires_grad
    assert not geometry.target_text_feature.requires_grad
    assert source.grad is None and full.grad is None


def test_negative_endpoint_direction_fails_without_sign_flip():
    with pytest.raises(TextSemanticEndpointDirectionError) as error:
        TextConditionedSemanticGeometry(
            _SyntheticEncoder(reverse_endpoint=True),
            torch.zeros(1, 3, 2, 2),
            torch.ones(1, 3, 2, 2),
            "source state",
            "target state",
        )
    assert error.value.diagnostics["endpoint_dynamic_range"] < 0
