"""Offline semantic specification and optional TianyuAI parser for v3.

This research parser is independent of RewardFlow's paper semantic parser. It
describes only Source/Native-Full endpoint semantics and never invents
intermediate strength stages.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import PIL.Image


RELATIVE_ENDPOINT_PARSER_VERSION = "relative-endpoint-semantic-v3-parser-v1"
RELATIVE_ENDPOINT_CACHE_SCHEMA_VERSION = 2
DEFAULT_RELATIVE_ENDPOINT_PARSER_MODEL = "gpt-5.6-luna"
DEFAULT_RELATIVE_ENDPOINT_PARSER_PROVIDER = "tianyuai"
DEFAULT_RELATIVE_ENDPOINT_PARSER_BASE_URL = "https://tianyuai.lol/v1"
_ID = re.compile(r"[a-z][a-z0-9_]*\Z")
_FORBIDDEN = re.compile(
    r"%|\bpercent(?:age)?\b|\bstrength\b|\bstage\s*(?:[1-5]|one|two|three|four|five)\b",
    re.IGNORECASE,
)
_PRIMITIVE_FIELDS = {
    "id",
    "object",
    "attribute",
    "edit_description",
    "comparison_focus",
    "source_state",
    "target_state",
    "endpoint_question",
    "source_answer",
    "target_answer",
    "weight",
}


@dataclass(frozen=True)
class RelativeEndpointPrimitiveSpec:
    id: str
    object: str
    attribute: str
    edit_description: str
    comparison_focus: str
    source_state: str
    target_state: str
    endpoint_question: str
    source_answer: str
    target_answer: str
    weight: float = 1.0


@dataclass(frozen=True)
class RelativeEndpointSemanticSpec:
    edit_instruction: str
    primitives: tuple[RelativeEndpointPrimitiveSpec, ...]
    preserve_constraints: tuple[str, ...]
    unresolved_instruction_items: tuple[str, ...]


@dataclass(frozen=True)
class RelativeEndpointParseRecord:
    spec: RelativeEndpointSemanticSpec
    provenance: dict[str, Any]


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"`{name}` must be a non-empty string.")
    return value.strip()


def _normalized(value: str) -> str:
    return " ".join(value.casefold().split())


def _string_list(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"`{name}` must be a list.")
    result = tuple(_string(item, f"{name}[{index}]") for index, item in enumerate(value))
    normalized = [_normalized(item) for item in result]
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"`{name}` entries must be unique.")
    return result


def relative_endpoint_json_schema() -> dict[str, Any]:
    string = {"type": "string", "minLength": 1}
    properties = {name: string for name in _PRIMITIVE_FIELDS if name != "weight"}
    properties["weight"] = {"type": "number", "const": 1.0}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["edit_instruction", "primitives", "preserve_constraints", "unresolved_instruction_items"],
        "properties": {
            "edit_instruction": string,
            "primitives": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": sorted(_PRIMITIVE_FIELDS),
                    "properties": properties,
                },
            },
            "preserve_constraints": {"type": "array", "items": string},
            "unresolved_instruction_items": {"type": "array", "items": string},
        },
    }


def build_relative_endpoint_semantic_parser_prompt(edit_instruction: str) -> str:
    instruction = _string(edit_instruction, "edit_instruction")
    return f"""Relative Endpoint Semantic Parser v1.

You receive exactly two images in this order:
1. Source image.
2. Native Full Edit generated from that Source.

EDIT_INSTRUCTION: {instruction}

Decompose only the semantic changes requested by the instruction and grounded in the two endpoints. Each primitive
must name one object and one changed visual attribute. Describe the Source state and Native Full state directly.
Create one endpoint question whose source_answer correctly describes Source and whose target_answer correctly
describes Native Full. Keep preservation requirements separate. Put ambiguous or ungrounded instruction items in
unresolved_instruction_items instead of guessing.

Never invent middle stages, percentages, numeric strength levels, or stage labels. Do not use the word strength.
Every primitive ID must be unique lower_snake_case. Every weight must be exactly 1.0.

Return only JSON matching the supplied strict schema."""


def parse_relative_endpoint_semantic_json(text: str) -> RelativeEndpointSemanticSpec:
    if not isinstance(text, str):
        raise TypeError("Relative endpoint parser output must be a string.")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Relative endpoint parser output is not valid JSON: {exc.msg}.") from exc
    required = {"edit_instruction", "primitives", "preserve_constraints", "unresolved_instruction_items"}
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("Relative endpoint JSON contains missing or unexpected top-level fields.")
    instruction = _string(payload["edit_instruction"], "edit_instruction")
    raw_primitives = payload["primitives"]
    if not isinstance(raw_primitives, list) or not raw_primitives:
        raise ValueError("`primitives` must contain at least one primitive.")
    primitives = []
    ids: set[str] = set()
    audited_fields = (
        "comparison_focus",
        "source_state",
        "target_state",
        "endpoint_question",
        "source_answer",
        "target_answer",
    )
    for index, raw in enumerate(raw_primitives):
        name = f"primitives[{index}]"
        if not isinstance(raw, dict) or set(raw) != _PRIMITIVE_FIELDS:
            raise ValueError(f"`{name}` contains missing or unexpected fields.")
        primitive_id = _string(raw["id"], f"{name}.id")
        if not _ID.fullmatch(primitive_id):
            raise ValueError(f"`{name}.id` must be lower_snake_case.")
        if primitive_id in ids:
            raise ValueError("Primitive IDs must be unique.")
        ids.add(primitive_id)
        strings = {field: _string(raw[field], f"{name}.{field}") for field in _PRIMITIVE_FIELDS - {"weight"}}
        for field in audited_fields:
            if _FORBIDDEN.search(strings[field]):
                raise ValueError(f"`{name}.{field}` contains forbidden strength, percentage, or stage wording.")
        if _normalized(strings["source_state"]) == _normalized(strings["target_state"]):
            raise ValueError("Source and target states must differ.")
        if _normalized(strings["source_answer"]) == _normalized(strings["target_answer"]):
            raise ValueError("Source and target answers must differ.")
        weight = raw["weight"]
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(float(weight)):
            raise ValueError("Primitive weight must be exactly 1.0.")
        if float(weight) != 1.0:
            raise ValueError("Primitive weight must be exactly 1.0.")
        primitives.append(RelativeEndpointPrimitiveSpec(**strings, weight=1.0))
    return RelativeEndpointSemanticSpec(
        edit_instruction=instruction,
        primitives=tuple(primitives),
        preserve_constraints=_string_list(payload["preserve_constraints"], "preserve_constraints"),
        unresolved_instruction_items=_string_list(
            payload["unresolved_instruction_items"], "unresolved_instruction_items"
        ),
    )


def fingerprint_endpoint_image(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_relative_endpoint_cache_key(
    source_fingerprint: str,
    full_fingerprint: str,
    edit_instruction: str,
    *,
    parser_version: str = RELATIVE_ENDPOINT_PARSER_VERSION,
    model: str = DEFAULT_RELATIVE_ENDPOINT_PARSER_MODEL,
    provider: str = DEFAULT_RELATIVE_ENDPOINT_PARSER_PROVIDER,
    base_url: str = DEFAULT_RELATIVE_ENDPOINT_PARSER_BASE_URL,
) -> str:
    identity = {
        "source_fingerprint": _string(source_fingerprint, "source_fingerprint"),
        "full_fingerprint": _string(full_fingerprint, "full_fingerprint"),
        "edit_instruction": " ".join(_string(edit_instruction, "edit_instruction").split()),
        "parser_version": _string(parser_version, "parser_version"),
        "model": _string(model, "model"),
        "provider": _string(provider, "provider"),
        "base_url": _string(base_url, "base_url").rstrip("/"),
        "cache_schema_version": RELATIVE_ENDPOINT_CACHE_SCHEMA_VERSION,
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": RELATIVE_ENDPOINT_CACHE_SCHEMA_VERSION, "entries": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != RELATIVE_ENDPOINT_CACHE_SCHEMA_VERSION or not isinstance(
        payload.get("entries"), dict
    ):
        raise ValueError(f"Relative endpoint parser cache `{path}` has an unsupported schema.")
    return payload


def save_cached_relative_endpoint_parse(
    cache_path: str | Path,
    record: RelativeEndpointParseRecord,
) -> str:
    if not isinstance(record, RelativeEndpointParseRecord):
        raise TypeError("`record` must be a RelativeEndpointParseRecord.")
    provenance = dict(record.provenance)
    required = {
        "provider",
        "model",
        "parser_version",
        "source_fingerprint",
        "full_fingerprint",
        "edit_instruction",
        "created_at",
        "base_url",
    }
    if not required.issubset(provenance):
        raise ValueError("Parse provenance is incomplete.")
    if any("key" in name.casefold() or "secret" in name.casefold() for name in provenance):
        raise ValueError("Parse provenance must not contain credentials.")
    key = make_relative_endpoint_cache_key(
        provenance["source_fingerprint"],
        provenance["full_fingerprint"],
        provenance["edit_instruction"],
        parser_version=provenance["parser_version"],
        model=provenance["model"],
        provider=provenance["provider"],
        base_url=provenance["base_url"],
    )
    path = Path(cache_path)
    payload = _cache(path)
    payload["entries"][key] = {"spec": asdict(record.spec), "provenance": provenance}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return key


def load_cached_relative_endpoint_parse(
    cache_path: str | Path,
    source_fingerprint: str,
    full_fingerprint: str,
    edit_instruction: str,
    *,
    parser_version: str = RELATIVE_ENDPOINT_PARSER_VERSION,
    model: str = DEFAULT_RELATIVE_ENDPOINT_PARSER_MODEL,
    provider: str = DEFAULT_RELATIVE_ENDPOINT_PARSER_PROVIDER,
    base_url: str = DEFAULT_RELATIVE_ENDPOINT_PARSER_BASE_URL,
) -> RelativeEndpointParseRecord | None:
    path = Path(cache_path)
    if not path.exists():
        return None
    key = make_relative_endpoint_cache_key(
        source_fingerprint,
        full_fingerprint,
        edit_instruction,
        parser_version=parser_version,
        model=model,
        provider=provider,
        base_url=base_url,
    )
    entry = _cache(path)["entries"].get(key)
    if entry is None:
        return None
    spec = parse_relative_endpoint_semantic_json(json.dumps(entry["spec"]))
    return RelativeEndpointParseRecord(spec=spec, provenance=dict(entry["provenance"]))


def make_human_relative_endpoint_parse_record(
    spec: RelativeEndpointSemanticSpec,
    source_fingerprint: str,
    full_fingerprint: str,
    *,
    model: str = "human-audited",
) -> RelativeEndpointParseRecord:
    return RelativeEndpointParseRecord(
        spec=spec,
        provenance={
            "provider": "human",
            "base_url": "local",
            "model": model,
            "parser_version": RELATIVE_ENDPOINT_PARSER_VERSION,
            "source_fingerprint": source_fingerprint,
            "full_fingerprint": full_fingerprint,
            "edit_instruction": spec.edit_instruction,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def _image_data_url(path: str | Path) -> str:
    image = PIL.Image.open(path).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


class TianyuAIRelativeEndpointParser:
    """One-call TianyuAI Chat Completions structured-output adapter.

    The service is OpenAI-compatible but is not operated by OpenAI. Credentials
    are read from ``TIANYUAI_API_KEY`` and never included in provenance.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        client=None,
    ):
        self.model = (
            model
            or os.getenv("TIANYUAI_SEMANTIC_PARSER_MODEL")
            or os.getenv("OPENAI_SEMANTIC_PARSER_MODEL", DEFAULT_RELATIVE_ENDPOINT_PARSER_MODEL)
        )
        self.provider = DEFAULT_RELATIVE_ENDPOINT_PARSER_PROVIDER
        self.base_url = (base_url or os.getenv("TIANYUAI_BASE_URL", DEFAULT_RELATIVE_ENDPOINT_PARSER_BASE_URL)).rstrip(
            "/"
        )
        if client is None:
            api_key = api_key or os.getenv("TIANYUAI_API_KEY")
            if not api_key:
                raise ValueError("Set `TIANYUAI_API_KEY` before using the online TianyuAI semantic parser.")
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError("Install the official `openai` package to use the online semantic parser.") from exc
            client = OpenAI(api_key=api_key, base_url=self.base_url)
        self.client = client

    def parse(
        self,
        source_image: str | Path,
        full_image: str | Path,
        edit_instruction: str,
    ) -> RelativeEndpointParseRecord:
        source_fingerprint = fingerprint_endpoint_image(source_image)
        full_fingerprint = fingerprint_endpoint_image(full_image)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": _image_data_url(source_image)}},
                        {"type": "image_url", "image_url": {"url": _image_data_url(full_image)}},
                        {"type": "text", "text": build_relative_endpoint_semantic_parser_prompt(edit_instruction)},
                    ],
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "relative_endpoint_semantic_spec",
                    "strict": True,
                    "schema": relative_endpoint_json_schema(),
                },
            },
        )
        try:
            output_text = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise RuntimeError("TianyuAI Chat Completions returned no assistant message.") from exc
        if not isinstance(output_text, str) or not output_text.strip():
            raise RuntimeError("TianyuAI Chat Completions returned no structured JSON text.")
        spec = parse_relative_endpoint_semantic_json(output_text)
        if _normalized(spec.edit_instruction) != _normalized(edit_instruction):
            raise ValueError("Parser output changed the edit instruction.")
        return RelativeEndpointParseRecord(
            spec=spec,
            provenance={
                "provider": self.provider,
                "base_url": self.base_url,
                "model": self.model,
                "parser_version": RELATIVE_ENDPOINT_PARSER_VERSION,
                "source_fingerprint": source_fingerprint,
                "full_fingerprint": full_fingerprint,
                "edit_instruction": spec.edit_instruction,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )


# Backward-compatible import name. Calls now use the explicitly configured
# TianyuAI Chat Completions provider rather than OpenAI Responses.
OpenAIRelativeEndpointParser = TianyuAIRelativeEndpointParser
