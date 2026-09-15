"""Endpoint-relative, differentiable semantic progress research utilities."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import torch

from .terminal_control import TerminalObjectiveOutput


SEMANTIC_PROGRESS_PARSER_VERSION = "endpoint-relative-semantic-progress-v1"
SEMANTIC_PROGRESS_CACHE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SemanticPrimitiveSpec:
    id: str
    edit_description: str
    question: str
    source_answer: str
    target_answer: str
    weight: float = 1.0


@dataclass(frozen=True)
class SemanticProgressSpec:
    edit_instruction: str
    primitives: tuple[SemanticPrimitiveSpec, ...]
    preserve_constraints: tuple[str, ...]


class TeacherForcedAnswerScorer(Protocol):
    def score_answer(self, image: torch.Tensor, question: str, answer: str) -> torch.Tensor: ...


class EndpointSemanticValidationError(ValueError):
    """Raised when the scorer cannot distinguish the specified endpoint states."""

    def __init__(self, diagnostics: dict[str, Any]):
        self.diagnostics = diagnostics
        super().__init__("Endpoint semantic validation failed; inspect raw source/full answer scores.")


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"`{name}` must be a non-empty string.")
    return value.strip()


def build_semantic_progress_parser_prompt(edit_instruction: str) -> str:
    """Build a provider-neutral two-endpoint semantic parser prompt."""

    instruction = _nonempty_string(edit_instruction, "edit_instruction")
    return f"""Endpoint-Relative Semantic Edit Parser.

You receive two images and one edit instruction:
- IMAGE 1 is the Source image.
- IMAGE 2 is the Native Full Edit produced from that Source.
- EDIT_INSTRUCTION: {instruction}

Identify only semantic attributes that the instruction truly asks to change. Ignore accidental Source/Full
differences unrelated to the instruction. For every edit primitive, output a concise edit description, one visual
question, the visually grounded Source answer, the visually grounded Full-edit target answer, and a positive weight.

Question rules:
- It must be answerable from one candidate image independently.
- It must focus on the requested edit attribute, not overall quality or background preservation.
- It must not contain a strength number or ask for completion percentage.

Answer rules:
- Source and target answers must be short, visually grounded, and semantically distinguishable.
- Do not invent an invisible state.
- Put preservation constraints in `preserve_constraints`, never inside a primitive.

Return JSON only with exactly this schema:
{{
  "edit_instruction": "{instruction}",
  "primitives": [
    {{
      "id": "short_snake_case_id",
      "edit_description": "concise requested change",
      "question": "one directly visual question",
      "source_answer": "short source state",
      "target_answer": "short full-edit state",
      "weight": 1.0
    }}
  ],
  "preserve_constraints": ["short constraint"]
}}"""


def parse_semantic_progress_json(text: str) -> SemanticProgressSpec:
    """Strictly validate provider output without coupling to an online API."""

    if not isinstance(text, str):
        raise TypeError("Semantic progress parser output must be a string.")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Semantic progress parser output is not valid JSON: {exc.msg}.") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "edit_instruction",
        "primitives",
        "preserve_constraints",
    }:
        raise ValueError("Semantic progress JSON contains missing or unexpected top-level fields.")

    instruction = _nonempty_string(payload["edit_instruction"], "edit_instruction")
    raw_primitives = payload["primitives"]
    if not isinstance(raw_primitives, list) or not raw_primitives:
        raise ValueError("`primitives` must contain at least one semantic primitive.")
    primitives = []
    seen_ids = set()
    required = {"id", "edit_description", "question", "source_answer", "target_answer", "weight"}
    for index, raw in enumerate(raw_primitives):
        if not isinstance(raw, dict) or set(raw) != required:
            raise ValueError(f"`primitives[{index}]` contains missing or unexpected fields.")
        primitive_id = _nonempty_string(raw["id"], f"primitives[{index}].id")
        if not re.fullmatch(r"[a-z][a-z0-9_]*", primitive_id):
            raise ValueError(f"`primitives[{index}].id` must be lower snake_case.")
        if primitive_id in seen_ids:
            raise ValueError("Semantic primitive IDs must be unique.")
        seen_ids.add(primitive_id)
        weight = raw["weight"]
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(weight)
            or weight <= 0
        ):
            raise ValueError(f"`primitives[{index}].weight` must be finite and positive.")
        question = _nonempty_string(raw["question"], f"primitives[{index}].question")
        lowered_question = question.lower()
        if "percent" in lowered_question or "%" in question:
            raise ValueError("Semantic questions must not ask for completion percentages.")
        primitives.append(
            SemanticPrimitiveSpec(
                id=primitive_id,
                edit_description=_nonempty_string(raw["edit_description"], f"primitives[{index}].edit_description"),
                question=question,
                source_answer=_nonempty_string(raw["source_answer"], f"primitives[{index}].source_answer"),
                target_answer=_nonempty_string(raw["target_answer"], f"primitives[{index}].target_answer"),
                weight=float(weight),
            )
        )

    raw_constraints = payload["preserve_constraints"]
    if not isinstance(raw_constraints, list):
        raise ValueError("`preserve_constraints` must be a list.")
    constraints = tuple(
        _nonempty_string(value, f"preserve_constraints[{index}]") for index, value in enumerate(raw_constraints)
    )
    if len(set(constraints)) != len(constraints):
        raise ValueError("Preservation constraints must be unique.")
    return SemanticProgressSpec(instruction, tuple(primitives), constraints)


def make_semantic_progress_cache_key(
    source_fingerprint: str,
    full_fingerprint: str,
    edit_instruction: str,
    parser_version: str = SEMANTIC_PROGRESS_PARSER_VERSION,
) -> str:
    """Bind parses to both visual endpoints, the instruction, and parser version."""

    identity = {
        "cache_schema_version": SEMANTIC_PROGRESS_CACHE_SCHEMA_VERSION,
        "source_fingerprint": _nonempty_string(source_fingerprint, "source_fingerprint"),
        "full_fingerprint": _nonempty_string(full_fingerprint, "full_fingerprint"),
        "edit_instruction": " ".join(_nonempty_string(edit_instruction, "edit_instruction").split()),
        "parser_version": _nonempty_string(parser_version, "parser_version"),
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": SEMANTIC_PROGRESS_CACHE_SCHEMA_VERSION, "entries": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read semantic progress cache `{path}`: {exc}.") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("version") != SEMANTIC_PROGRESS_CACHE_SCHEMA_VERSION
        or not isinstance(payload.get("entries"), dict)
    ):
        raise ValueError(f"Semantic progress cache `{path}` has an unsupported schema.")
    return payload


def save_cached_semantic_progress_spec(
    cache_path: str | Path,
    source_fingerprint: str,
    full_fingerprint: str,
    spec: SemanticProgressSpec,
    *,
    parser_version: str = SEMANTIC_PROGRESS_PARSER_VERSION,
) -> None:
    if not isinstance(spec, SemanticProgressSpec):
        raise TypeError("`spec` must be a SemanticProgressSpec.")
    key = make_semantic_progress_cache_key(source_fingerprint, full_fingerprint, spec.edit_instruction, parser_version)
    path = Path(cache_path)
    payload = _read_cache(path)
    payload["entries"][key] = {
        "source_fingerprint": source_fingerprint,
        "full_fingerprint": full_fingerprint,
        "edit_instruction": spec.edit_instruction,
        "parser_version": parser_version,
        "spec": asdict(spec),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_cached_semantic_progress_spec(
    cache_path: str | Path,
    source_fingerprint: str,
    full_fingerprint: str,
    edit_instruction: str,
    *,
    parser_version: str = SEMANTIC_PROGRESS_PARSER_VERSION,
) -> SemanticProgressSpec | None:
    path = Path(cache_path)
    if not path.exists():
        return None
    key = make_semantic_progress_cache_key(source_fingerprint, full_fingerprint, edit_instruction, parser_version)
    entry = _read_cache(path)["entries"].get(key)
    if entry is None:
        return None
    return parse_semantic_progress_json(json.dumps(entry.get("spec")))


def _serialize_scalar(value: torch.Tensor) -> float:
    return value.detach().float().cpu().item()


class EndpointRelativeSemanticProgressReward:
    """Normalize target-vs-source teacher-forced answer contrast between fixed endpoints.

    ASSUMPTION: source-vs-target answer contrast is a usable continuous semantic
    coordinate. This is a new research hypothesis, not a RewardFlow paper claim.
    """

    def __init__(
        self,
        scorer: TeacherForcedAnswerScorer,
        spec: SemanticProgressSpec,
        source_image: torch.Tensor,
        full_image: torch.Tensor,
        *,
        denominator_epsilon: float = 1e-6,
        fail_on_endpoint_validation: bool = True,
    ):
        if not isinstance(spec, SemanticProgressSpec) or not spec.primitives:
            raise ValueError("Semantic progress reward requires at least one validated primitive.")
        if source_image.shape != full_image.shape or source_image.ndim != 4 or source_image.shape[1] != 3:
            raise ValueError("Source and full endpoint images must share shape [B, 3, H, W].")
        if source_image.shape[0] != 1:
            raise ValueError("Endpoint semantic anchor preparation currently requires B=1.")
        if denominator_epsilon <= 0 or not math.isfinite(denominator_epsilon):
            raise ValueError("`denominator_epsilon` must be finite and positive.")
        self.scorer = scorer
        self.spec = spec
        self.denominator_epsilon = float(denominator_epsilon)
        self._anchors: dict[str, dict[str, torch.Tensor]] = {}
        diagnostics = {}
        validation_failed = False
        with torch.no_grad():
            for primitive in spec.primitives:
                source_source = scorer.score_answer(source_image, primitive.question, primitive.source_answer)
                source_target = scorer.score_answer(source_image, primitive.question, primitive.target_answer)
                full_source = scorer.score_answer(full_image, primitive.question, primitive.source_answer)
                full_target = scorer.score_answer(full_image, primitive.question, primitive.target_answer)
                source_contrast = (source_target - source_source).detach()
                full_contrast = (full_target - full_source).detach()
                denominator = (full_contrast - source_contrast).detach()
                finite = all(
                    bool(torch.isfinite(value).all().item())
                    for value in (source_source, source_target, full_source, full_target, denominator)
                )
                source_preference_valid = bool((source_source > source_target).all().item())
                full_preference_valid = bool((full_target > full_source).all().item())
                contrast_order_valid = bool((denominator > self.denominator_epsilon).all().item())
                primitive_valid = finite and source_preference_valid and full_preference_valid and contrast_order_valid
                validation_failed = validation_failed or not primitive_valid
                self._anchors[primitive.id] = {
                    "source_contrast": source_contrast,
                    "full_contrast": full_contrast,
                    "denominator": denominator,
                }
                diagnostics[primitive.id] = {
                    "source_answer_score_source": _serialize_scalar(source_source),
                    "target_answer_score_source": _serialize_scalar(source_target),
                    "source_answer_score_full": _serialize_scalar(full_source),
                    "target_answer_score_full": _serialize_scalar(full_target),
                    "source_contrast": _serialize_scalar(source_contrast),
                    "full_contrast": _serialize_scalar(full_contrast),
                    "contrast_dynamic_range": _serialize_scalar(denominator),
                    "source_preference_valid": source_preference_valid,
                    "full_preference_valid": full_preference_valid,
                    "contrast_order_valid": contrast_order_valid,
                    "valid": primitive_valid,
                }
        self.endpoint_diagnostics = diagnostics
        self.endpoint_semantic_validation_failed = validation_failed
        if validation_failed and fail_on_endpoint_validation:
            raise EndpointSemanticValidationError(diagnostics)

    @property
    def model(self):
        return getattr(self.scorer, "model", None)

    def _target_vector(self, target_strengths: float | Sequence[float] | dict[str, float] | torch.Tensor, reference):
        primitive_ids = [primitive.id for primitive in self.spec.primitives]
        if isinstance(target_strengths, dict):
            if set(target_strengths) != set(primitive_ids):
                raise ValueError("Target-strength mapping must contain exactly every primitive ID.")
            values = [target_strengths[primitive_id] for primitive_id in primitive_ids]
        elif torch.is_tensor(target_strengths) and target_strengths.numel() > 1:
            values = target_strengths.flatten()
        elif isinstance(target_strengths, Sequence) and not isinstance(target_strengths, (str, bytes)):
            values = target_strengths
        else:
            values = [target_strengths] * len(primitive_ids)
        target = torch.as_tensor(values, device=reference.device, dtype=reference.dtype).flatten()
        if target.numel() != len(primitive_ids) or not torch.isfinite(target).all():
            raise ValueError("Target strengths must provide one finite value per primitive.")
        if bool(((target < 0) | (target > 1)).any().item()):
            raise ValueError("Target strengths must lie in [0, 1].")
        return target

    def progress_vector(self, image: torch.Tensor) -> tuple[torch.Tensor, dict[str, dict[str, torch.Tensor]]]:
        if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
            raise ValueError("Candidate semantic scoring currently requires image shape [1, 3, H, W].")
        progresses = []
        diagnostics = {}
        for primitive in self.spec.primitives:
            source_score = self.scorer.score_answer(image, primitive.question, primitive.source_answer)
            target_score = self.scorer.score_answer(image, primitive.question, primitive.target_answer)
            contrast = target_score - source_score
            anchor = self._anchors[primitive.id]
            progress = (contrast - anchor["source_contrast"].to(contrast.device)) / anchor["denominator"].to(
                contrast.device
            )
            progresses.append(progress)
            diagnostics[primitive.id] = {
                "source_answer_score_candidate": source_score,
                "target_answer_score_candidate": target_score,
                "semantic_contrast": contrast,
                "progress_raw": progress,
                "progress_clamped": progress.clamp(0, 1),
            }
        return torch.stack(progresses), diagnostics

    def __call__(
        self,
        image: torch.Tensor,
        target_strengths: float | Sequence[float] | dict[str, float] | torch.Tensor,
    ) -> TerminalObjectiveOutput:
        progress, diagnostics = self.progress_vector(image)
        target = self._target_vector(target_strengths, progress)
        weights = progress.new_tensor([primitive.weight for primitive in self.spec.primitives])
        weights = weights / weights.sum()
        residual = progress - target
        aggregate_progress = (weights * progress).sum()
        aggregate_target = (weights * target).sum()
        return TerminalObjectiveOutput(
            objective_name="semantic_progress",
            loss=(weights * residual.square()).sum(),
            objective_error=(weights * residual.abs()).sum(),
            source_score=progress.new_tensor(0.0),
            full_score=progress.new_tensor(1.0),
            target_score=aggregate_target,
            achieved_score=aggregate_progress,
            diagnostics=diagnostics,
        )
