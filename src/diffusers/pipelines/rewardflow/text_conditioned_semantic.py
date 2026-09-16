"""Text-conditioned endpoint geometry for the independent v7 experiment.

This module is research code, not a RewardFlow paper component. It keeps the
v7 parser/cache separate from the existing relative-endpoint v3 schema.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import torch

from .relative_endpoint_parser import (
    DEFAULT_RELATIVE_ENDPOINT_PARSER_BASE_URL,
    DEFAULT_RELATIVE_ENDPOINT_PARSER_MODEL,
    DEFAULT_RELATIVE_ENDPOINT_PARSER_PROVIDER,
    TianyuAIRelativeEndpointParser,
    fingerprint_endpoint_image,
)


TEXT_CONDITIONED_SEMANTIC_PARSER_VERSION = "text-conditioned-semantic-v7-parser-v1"
TEXT_CONDITIONED_SEMANTIC_CACHE_SCHEMA_VERSION = 1
_ID = re.compile(r"[a-z][a-z0-9_]*\Z")
_FORBIDDEN_SEMANTIC_TEXT = re.compile(
    r"%|\bpercent(?:age)?\b|\bstrength\b|\bstage\b|\bsource image\b|\btarget image\b|"
    r"\bedited image\b|\bbefore\b|\bafter\b|\boriginal\b|\bfinal\b|\bpartially edited\b|\bfully edited\b",
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
    "source_semantic_text",
    "target_semantic_text",
    "endpoint_question",
    "source_answer",
    "target_answer",
    "weight",
}


@dataclass(frozen=True)
class TextConditionedPrimitiveSpec:
    id: str
    object: str
    attribute: str
    edit_description: str
    comparison_focus: str
    source_state: str
    target_state: str
    source_semantic_text: str
    target_semantic_text: str
    endpoint_question: str
    source_answer: str
    target_answer: str
    weight: float = 1.0


@dataclass(frozen=True)
class TextConditionedSemanticSpec:
    edit_instruction: str
    primitives: tuple[TextConditionedPrimitiveSpec, ...]
    preserve_constraints: tuple[str, ...]
    unresolved_instruction_items: tuple[str, ...]


@dataclass(frozen=True)
class TextConditionedParseRecord:
    spec: TextConditionedSemanticSpec
    provenance: dict[str, Any]


@dataclass
class TextSemanticGeometryOutput:
    progress: torch.Tensor
    semantic_margin: torch.Tensor
    source_similarity: torch.Tensor
    target_similarity: torch.Tensor
    endpoint_dynamic_range: torch.Tensor
    text_axis_norm: torch.Tensor
    image_delta_text_alignment: torch.Tensor
    orthogonal_ratio: torch.Tensor


class TextFeatureEncoder(Protocol):
    def encode_image(self, image: torch.Tensor) -> torch.Tensor: ...

    def encode_text(self, text: str) -> torch.Tensor: ...


class TextSemanticEndpointDirectionError(ValueError):
    """Raised when target text does not increase from Source to Full."""

    def __init__(self, diagnostics: dict[str, float | bool]):
        self.diagnostics = diagnostics
        super().__init__("TEXT_SEMANTIC_ENDPOINT_DIRECTION_FAIL: m_full must be greater than m_source.")


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
    if len({_normalized(item) for item in result}) != len(result):
        raise ValueError(f"`{name}` entries must be unique.")
    return result


def _assistant_text(content: object) -> str:
    """Extract text from string or OpenAI-compatible typed content blocks."""

    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        pieces = []
        for block in content:
            if isinstance(block, str):
                block_type = None
                piece = block
            elif isinstance(block, dict):
                block_type = block.get("type")
                piece = block.get("text")
            else:
                block_type = getattr(block, "type", None)
                piece = getattr(block, "text", None)
            if block_type in {"analysis", "reasoning"}:
                continue
            if block_type not in {None, "text", "output_text"}:
                raise RuntimeError(f"TianyuAI returned unsupported assistant content type `{block_type}`.")
            if not isinstance(piece, str) or not piece.strip():
                raise RuntimeError("TianyuAI returned an unsupported assistant content block.")
            pieces.append(piece)
        if pieces:
            if len(pieces) > 1:
                try:
                    parsed_pieces = [json.loads(piece) for piece in pieces]
                except json.JSONDecodeError:
                    pass
                else:
                    if all(piece == parsed_pieces[0] for piece in parsed_pieces[1:]):
                        # Some OpenAI-compatible gateways duplicate the same
                        # complete output block. Deduplicate only when every
                        # block independently parses to the identical JSON.
                        return pieces[0]
            combined = "".join(pieces)
            try:
                json.loads(combined)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "TianyuAI returned multiple assistant text blocks that do not form one JSON document: "
                    + repr(pieces)
                ) from exc
            return combined
    raise RuntimeError("TianyuAI returned no structured assistant text.")


def text_conditioned_semantic_json_schema() -> dict[str, Any]:
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


def build_text_conditioned_semantic_parser_prompt(edit_instruction: str) -> str:
    instruction = _string(edit_instruction, "edit_instruction")
    return f"""Text-Conditioned Semantic Progress Parser v1.

You receive exactly two images in this order:
1. Source image.
2. Native Full Edit generated from that Source.

EDIT_INSTRUCTION: {instruction}

Identify only requested, visually grounded edit primitives. For every primitive, identify the edited visual subject,
the changed attribute, its visible Source state, and its visible Native-Full target state. The object may be a local
object or a global visual subject such as the scene, environment, or sky.

Create source_semantic_text and target_semantic_text as a matched pair for contrastive text embedding. They must use
the same grammatical structure, the same object, and the same amount of context; change only the requested attribute.
Each text must directly describe one currently visible semantic state. Do not mention source image, target image,
edited image, before, after, original, final, percentages, strength, stages, or partial/full edit completion. Do not
put preservation requirements into either semantic text. Use exactly one matched text pair per primitive and do not
produce synonyms or prompt ensembles.

Create one directly visual endpoint question with a source_answer for Source and target_answer for Native Full. Keep
preservation requirements only in preserve_constraints. Put ambiguous or ungrounded items in
unresolved_instruction_items rather than guessing. Primitive IDs must be unique lower_snake_case and every weight
must be exactly 1.0. Return only JSON matching the supplied strict schema."""


def parse_text_conditioned_semantic_json(text: str) -> TextConditionedSemanticSpec:
    if not isinstance(text, str):
        raise TypeError("Text-conditioned parser output must be a string.")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Text-conditioned parser output is not valid JSON: {exc.msg}.") from exc
    required = {"edit_instruction", "primitives", "preserve_constraints", "unresolved_instruction_items"}
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("Text-conditioned JSON contains missing or unexpected top-level fields.")
    instruction = _string(payload["edit_instruction"], "edit_instruction")
    raw_primitives = payload["primitives"]
    if not isinstance(raw_primitives, list) or not raw_primitives:
        raise ValueError("`primitives` must contain at least one primitive.")
    primitives = []
    ids: set[str] = set()
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
        source_text = strings["source_semantic_text"]
        target_text = strings["target_semantic_text"]
        for field, value in (("source_semantic_text", source_text), ("target_semantic_text", target_text)):
            if _FORBIDDEN_SEMANTIC_TEXT.search(value):
                raise ValueError(
                    f"`{name}.{field}` contains forbidden endpoint, percentage, strength, or stage wording."
                )
        if _normalized(source_text) == _normalized(target_text):
            raise ValueError("Source and target semantic text must differ.")
        if _normalized(strings["source_state"]) == _normalized(strings["target_state"]):
            raise ValueError("Source and target states must differ.")
        if _normalized(strings["source_answer"]) == _normalized(strings["target_answer"]):
            raise ValueError("Source and target answers must differ.")
        weight = raw["weight"]
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(float(weight)):
            raise ValueError("Primitive weight must be exactly 1.0.")
        if float(weight) != 1.0:
            raise ValueError("Primitive weight must be exactly 1.0.")
        primitives.append(TextConditionedPrimitiveSpec(**strings, weight=1.0))
    return TextConditionedSemanticSpec(
        edit_instruction=instruction,
        primitives=tuple(primitives),
        preserve_constraints=_string_list(payload["preserve_constraints"], "preserve_constraints"),
        unresolved_instruction_items=_string_list(
            payload["unresolved_instruction_items"], "unresolved_instruction_items"
        ),
    )


def make_text_conditioned_semantic_cache_key(
    source_fingerprint: str,
    full_fingerprint: str,
    edit_instruction: str,
    *,
    parser_version: str = TEXT_CONDITIONED_SEMANTIC_PARSER_VERSION,
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
        "cache_schema_version": TEXT_CONDITIONED_SEMANTIC_CACHE_SCHEMA_VERSION,
    }
    import hashlib

    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": TEXT_CONDITIONED_SEMANTIC_CACHE_SCHEMA_VERSION, "entries": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != TEXT_CONDITIONED_SEMANTIC_CACHE_SCHEMA_VERSION or not isinstance(
        payload.get("entries"), dict
    ):
        raise ValueError(f"Text-conditioned semantic cache `{path}` has an unsupported schema.")
    return payload


def save_cached_text_conditioned_parse(cache_path: str | Path, record: TextConditionedParseRecord) -> str:
    if not isinstance(record, TextConditionedParseRecord):
        raise TypeError("`record` must be a TextConditionedParseRecord.")
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
        "cache_hit",
    }
    if not required.issubset(provenance):
        raise ValueError("Parse provenance is incomplete.")
    if any("key" in name.casefold() or "secret" in name.casefold() for name in provenance):
        raise ValueError("Parse provenance must not contain credentials.")
    key = make_text_conditioned_semantic_cache_key(
        provenance["source_fingerprint"],
        provenance["full_fingerprint"],
        provenance["edit_instruction"],
        parser_version=provenance["parser_version"],
        model=provenance["model"],
        provider=provenance["provider"],
        base_url=provenance["base_url"],
    )
    payload = _cache(Path(cache_path))
    payload["entries"][key] = {"spec": asdict(record.spec), "provenance": {**provenance, "cache_hit": False}}
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return key


def load_cached_text_conditioned_parse(
    cache_path: str | Path,
    source_fingerprint: str,
    full_fingerprint: str,
    edit_instruction: str,
    *,
    parser_version: str = TEXT_CONDITIONED_SEMANTIC_PARSER_VERSION,
    model: str = DEFAULT_RELATIVE_ENDPOINT_PARSER_MODEL,
    provider: str = DEFAULT_RELATIVE_ENDPOINT_PARSER_PROVIDER,
    base_url: str = DEFAULT_RELATIVE_ENDPOINT_PARSER_BASE_URL,
) -> TextConditionedParseRecord | None:
    path = Path(cache_path)
    if not path.exists():
        return None
    key = make_text_conditioned_semantic_cache_key(
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
    return TextConditionedParseRecord(
        spec=parse_text_conditioned_semantic_json(json.dumps(entry["spec"])),
        provenance={**entry["provenance"], "cache_hit": True},
    )


class TianyuAITextConditionedSemanticParser:
    """One-call TianyuAI adapter with the independent v7 strict schema."""

    def __init__(
        self, *, model: str | None = None, base_url: str | None = None, api_key: str | None = None, client=None
    ):
        base = TianyuAIRelativeEndpointParser(model=model, base_url=base_url, api_key=api_key, client=client)
        self.model = base.model
        self.provider = base.provider
        self.base_url = base.base_url
        self.client = base.client

    def parse(
        self, source_image: str | Path, full_image: str | Path, edit_instruction: str
    ) -> TextConditionedParseRecord:
        import base64
        import io

        from PIL import Image

        def data_url(path):
            buffer = io.BytesIO()
            Image.open(path).convert("RGB").save(buffer, format="PNG")
            return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

        source_fingerprint = fingerprint_endpoint_image(source_image)
        full_fingerprint = fingerprint_endpoint_image(full_image)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url(source_image)}},
                        {"type": "image_url", "image_url": {"url": data_url(full_image)}},
                        {"type": "text", "text": build_text_conditioned_semantic_parser_prompt(edit_instruction)},
                    ],
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "text_conditioned_semantic_spec",
                    "strict": True,
                    "schema": text_conditioned_semantic_json_schema(),
                },
            },
        )
        try:
            output_text = _assistant_text(response.choices[0].message.content)
        except (AttributeError, IndexError, TypeError) as exc:
            raise RuntimeError("TianyuAI returned no assistant message.") from exc
        spec = parse_text_conditioned_semantic_json(output_text)
        if _normalized(spec.edit_instruction) != _normalized(edit_instruction):
            raise ValueError("Parser output changed the edit instruction.")
        return TextConditionedParseRecord(
            spec=spec,
            provenance={
                "provider": self.provider,
                "base_url": self.base_url,
                "model": self.model,
                "parser_version": TEXT_CONDITIONED_SEMANTIC_PARSER_VERSION,
                "source_fingerprint": source_fingerprint,
                "full_fingerprint": full_fingerprint,
                "edit_instruction": spec.edit_instruction,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "cache_hit": False,
            },
        )


def _vector(value: torch.Tensor, name: str) -> torch.Tensor:
    if not torch.is_tensor(value) or not value.is_floating_point():
        raise TypeError(f"`{name}` must be a floating-point tensor.")
    if value.ndim == 2 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 1 or not value.numel() or not torch.isfinite(value).all():
        raise ValueError(f"`{name}` must be one finite non-empty vector.")
    return value.float()


class TextConditionedSemanticGeometry:
    """Cache fixed endpoint/text features and preserve Candidate image gradients.

    For L2-normalized embeddings, the margin is exactly the image feature dot
    ``(target_text - source_text)``. Source/Full margins provide only an
    instance-relative scale; the sign is never flipped.
    """

    def __init__(
        self,
        encoder: TextFeatureEncoder,
        source_image: torch.Tensor,
        full_image: torch.Tensor,
        source_semantic_text: str,
        target_semantic_text: str,
        *,
        eps: float = 1e-8,
    ):
        if source_image.shape != full_image.shape:
            raise ValueError("Source and Full images must have the same shape.")
        if eps <= 0:
            raise ValueError("`eps` must be positive.")
        self.encoder = encoder
        self.source_image = source_image.detach()
        self.full_image = full_image.detach()
        self.source_semantic_text = _string(source_semantic_text, "source_semantic_text")
        self.target_semantic_text = _string(target_semantic_text, "target_semantic_text")
        self.eps = float(eps)
        with torch.no_grad():
            self.source_text_feature = _vector(encoder.encode_text(self.source_semantic_text), "source_text").detach()
            self.source_text_metadata = dict(getattr(encoder, "last_text_metadata", {}))
            self.target_text_feature = _vector(encoder.encode_text(self.target_semantic_text), "target_text").detach()
            self.target_text_metadata = dict(getattr(encoder, "last_text_metadata", {}))
            self.source_feature = _vector(encoder.encode_image(self.source_image), "source_image_feature").detach()
            self.full_feature = _vector(encoder.encode_image(self.full_image), "full_image_feature").detach()
            self.text_axis = (self.target_text_feature - self.source_text_feature).detach()
            self.text_axis_norm = torch.linalg.vector_norm(self.text_axis).detach()
            self.source_margin = self._margin(self.source_feature).detach()
            self.full_margin = self._margin(self.full_feature).detach()
            self.endpoint_dynamic_range = (self.full_margin - self.source_margin).detach()
        diagnostics = {
            "source_margin": float(self.source_margin),
            "full_margin": float(self.full_margin),
            "endpoint_dynamic_range": float(self.endpoint_dynamic_range),
            "text_axis_norm": float(self.text_axis_norm),
            "finite": bool(
                torch.isfinite(torch.stack((self.source_margin, self.full_margin, self.text_axis_norm))).all().item()
            ),
        }
        if (
            not diagnostics["finite"]
            or float(self.text_axis_norm) <= self.eps
            or float(self.endpoint_dynamic_range) <= self.eps
        ):
            raise TextSemanticEndpointDirectionError(diagnostics)

    def _margin(self, feature: torch.Tensor) -> torch.Tensor:
        return torch.dot(feature, self.target_text_feature.to(feature.device)) - torch.dot(
            feature, self.source_text_feature.to(feature.device)
        )

    def __call__(self, candidate_image: torch.Tensor) -> TextSemanticGeometryOutput:
        feature = _vector(self.encoder.encode_image(candidate_image), "candidate_image_feature")
        source_text = self.source_text_feature.to(feature.device)
        target_text = self.target_text_feature.to(feature.device)
        text_axis = target_text - source_text
        source_feature = self.source_feature.to(feature.device)
        source_similarity = torch.dot(feature, source_text)
        target_similarity = torch.dot(feature, target_text)
        margin = target_similarity - source_similarity
        dynamic_range = self.endpoint_dynamic_range.to(feature.device)
        progress = (margin - self.source_margin.to(feature.device)) / (dynamic_range + self.eps)
        image_delta = feature - source_feature
        image_delta_norm = torch.linalg.vector_norm(image_delta)
        text_axis_norm = torch.linalg.vector_norm(text_axis)
        alignment = torch.dot(image_delta, text_axis) / (image_delta_norm * text_axis_norm + self.eps)
        coefficient = torch.dot(image_delta, text_axis) / (torch.dot(text_axis, text_axis) + self.eps)
        residual = image_delta - coefficient * text_axis
        # ASSUMPTION: normalize the off-direction residual by total image-delta
        # norm so this diagnostic reports the fraction not aligned with text.
        orthogonal_ratio = torch.linalg.vector_norm(residual) / (image_delta_norm + self.eps)
        return TextSemanticGeometryOutput(
            progress=progress,
            semantic_margin=margin,
            source_similarity=source_similarity,
            target_similarity=target_similarity,
            endpoint_dynamic_range=dynamic_range,
            text_axis_norm=text_axis_norm,
            image_delta_text_alignment=alignment,
            orthogonal_ratio=orthogonal_ratio,
        )
