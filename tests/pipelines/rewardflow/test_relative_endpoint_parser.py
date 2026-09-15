import json
import os
from types import SimpleNamespace

import pytest

from diffusers.pipelines.rewardflow.relative_endpoint_parser import (
    DEFAULT_RELATIVE_ENDPOINT_PARSER_MODEL,
    TianyuAIRelativeEndpointParser,
    build_relative_endpoint_semantic_parser_prompt,
    load_cached_relative_endpoint_parse,
    make_human_relative_endpoint_parse_record,
    make_relative_endpoint_cache_key,
    parse_relative_endpoint_semantic_json,
    save_cached_relative_endpoint_parse,
)


def _primitive(primitive_id="ball_color"):
    return {
        "id": primitive_id,
        "object": "foreground ball",
        "attribute": "color",
        "edit_description": "change the ball from dark gray to blue",
        "comparison_focus": "the visible color of the foreground ball",
        "source_state": "the ball is dark gray",
        "target_state": "the ball is vivid blue",
        "endpoint_question": "What is the visible color of the foreground ball?",
        "source_answer": "Dark gray.",
        "target_answer": "Blue.",
        "weight": 1.0,
    }


def _payload(count=1):
    primitives = [_primitive("ball_color")]
    for index in range(1, count):
        primitive = _primitive(f"car_attribute_{index}")
        primitive.update(
            object="car",
            attribute=f"attribute {index}",
            edit_description=f"change car attribute {index}",
            comparison_focus=f"visible car attribute {index}",
            source_state=f"car attribute {index} is original",
            target_state=f"car attribute {index} is edited",
            endpoint_question=f"What is car attribute {index}?",
            source_answer=f"Original value {index}.",
            target_answer=f"Edited value {index}.",
        )
        primitives.append(primitive)
    return {
        "edit_instruction": "Turn the ball blue.",
        "primitives": primitives,
        "preserve_constraints": ["ball shape", "background composition"],
        "unresolved_instruction_items": [],
    }


def test_parser_accepts_single_and_four_primitives_and_preserves_auxiliary_lists():
    single = parse_relative_endpoint_semantic_json(json.dumps(_payload()))
    car = parse_relative_endpoint_semantic_json(json.dumps(_payload(4)))
    assert len(single.primitives) == 1
    assert len(car.primitives) == 4
    assert single.preserve_constraints == ("ball shape", "background composition")
    assert single.unresolved_instruction_items == ()
    prompt = build_relative_endpoint_semantic_parser_prompt(single.edit_instruction)
    assert "Source image" in prompt and "Native Full Edit" in prompt
    assert "Never invent middle stages" in prompt


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("comparison_focus", "ball at 50% blue", "forbidden"),
        ("source_state", "strength is low", "forbidden"),
        ("target_state", "stage 5", "forbidden"),
        ("endpoint_question", "Which percentage is visible?", "forbidden"),
        ("source_answer", "stage one", "forbidden"),
        ("target_answer", "100 percent", "forbidden"),
    ],
)
def test_parser_rejects_percentage_strength_and_stage_leakage(field, value, match):
    payload = _payload()
    payload["primitives"][0][field] = value
    with pytest.raises(ValueError, match=match):
        parse_relative_endpoint_semantic_json(json.dumps(payload))


def test_parser_rejects_duplicates_nonunit_weight_and_equal_endpoint_semantics():
    payload = _payload(2)
    payload["primitives"][1]["id"] = payload["primitives"][0]["id"]
    with pytest.raises(ValueError, match="unique"):
        parse_relative_endpoint_semantic_json(json.dumps(payload))
    payload = _payload()
    payload["primitives"][0]["weight"] = 0.5
    with pytest.raises(ValueError, match="exactly 1.0"):
        parse_relative_endpoint_semantic_json(json.dumps(payload))
    payload = _payload()
    payload["primitives"][0]["target_state"] = " THE BALL IS DARK GRAY "
    with pytest.raises(ValueError, match="states must differ"):
        parse_relative_endpoint_semantic_json(json.dumps(payload))
    payload = _payload()
    payload["primitives"][0]["target_answer"] = " dark GRAY. "
    with pytest.raises(ValueError, match="answers must differ"):
        parse_relative_endpoint_semantic_json(json.dumps(payload))


def test_parser_rejects_duplicate_constraints_and_unresolved_items():
    for field in ("preserve_constraints", "unresolved_instruction_items"):
        payload = _payload()
        payload[field] = ["same item", " Same  item "]
        with pytest.raises(ValueError, match="unique"):
            parse_relative_endpoint_semantic_json(json.dumps(payload))


def test_cache_key_binds_both_images_instruction_parser_version_and_model(tmp_path):
    base = make_relative_endpoint_cache_key("source", "full", "turn blue")
    assert base != make_relative_endpoint_cache_key("other", "full", "turn blue")
    assert base != make_relative_endpoint_cache_key("source", "other", "turn blue")
    assert base != make_relative_endpoint_cache_key("source", "full", "turn red")
    assert base != make_relative_endpoint_cache_key("source", "full", "turn blue", parser_version="v2")
    assert base != make_relative_endpoint_cache_key("source", "full", "turn blue", model="other")
    assert base != make_relative_endpoint_cache_key("source", "full", "turn blue", provider="other")
    assert base != make_relative_endpoint_cache_key("source", "full", "turn blue", base_url="https://other/v1")
    spec = parse_relative_endpoint_semantic_json(json.dumps(_payload()))
    record = make_human_relative_endpoint_parse_record(spec, "source", "full")
    cache = tmp_path / "cache.json"
    save_cached_relative_endpoint_parse(cache, record)
    loaded = load_cached_relative_endpoint_parse(
        cache,
        "source",
        "full",
        spec.edit_instruction,
        model="human-audited",
        provider="human",
        base_url="local",
    )
    assert loaded == record
    assert "api_key" not in cache.read_text().casefold()


def test_tianyu_provider_uses_one_structured_chat_completion_and_ordered_images(tmp_path):
    from PIL import Image

    source = tmp_path / "source.png"
    full = tmp_path / "full.png"
    Image.new("RGB", (2, 2), "black").save(source)
    Image.new("RGB", (2, 2), "blue").save(full)
    calls = []

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            message = SimpleNamespace(content=json.dumps(_payload()))
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    parser = TianyuAIRelativeEndpointParser(client=client)
    record = parser.parse(source, full, "Turn the ball blue.")
    assert len(calls) == 1
    assert calls[0]["model"] == DEFAULT_RELATIVE_ENDPOINT_PARSER_MODEL
    assert calls[0]["response_format"]["json_schema"]["strict"] is True
    content = calls[0]["messages"][0]["content"]
    assert [item["type"] for item in content] == ["image_url", "image_url", "text"]
    assert record.provenance["provider"] == "tianyuai"
    assert record.provenance["base_url"] == "https://tianyuai.lol/v1"


@pytest.mark.skipif(
    not os.getenv("TIANYUAI_API_KEY") or os.getenv("RUN_TIANYUAI_SEMANTIC_PARSER_INTEGRATION") != "1",
    reason="requires TIANYUAI_API_KEY and explicit RUN_TIANYUAI_SEMANTIC_PARSER_INTEGRATION=1",
)
def test_live_tianyu_semantic_parser_only_when_explicitly_enabled(tmp_path):
    from PIL import Image

    source = tmp_path / "source.png"
    full = tmp_path / "full.png"
    Image.new("RGB", (32, 32), "black").save(source)
    Image.new("RGB", (32, 32), "blue").save(full)
    record = TianyuAIRelativeEndpointParser().parse(source, full, "Turn the square blue.")
    assert record.provenance["provider"] == "tianyuai"
    assert record.spec.primitives
