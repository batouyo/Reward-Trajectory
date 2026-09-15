"""Endpoint-relative ordinal semantic progress research utilities.

The five textual stages are a research hypothesis, not ground-truth
intermediate images and not a component claimed by the RewardFlow paper.
"""

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


ORDINAL_SEMANTIC_PROGRESS_PARSER_VERSION = "ordinal-semantic-progress-v2"
ORDINAL_SEMANTIC_PROGRESS_CACHE_SCHEMA_VERSION = 2
ORDINAL_STAGE_NODES = (0.0, 0.25, 0.5, 0.75, 1.0)
ORDINAL_CHOICE_LABELS = ("A", "B", "C", "D", "E")
_FORBIDDEN_STAGE_PATTERN = re.compile(r"%|\bpercent(?:age)?\b|\bstrength\b", re.IGNORECASE)


@dataclass(frozen=True)
class OrdinalQuestionSpec:
    id: str
    question: str
    stages: tuple[str, str, str, str, str]
    weight: float = 1.0


@dataclass(frozen=True)
class OrdinalSemanticPrimitiveSpec:
    id: str
    edit_description: str
    questions: tuple[OrdinalQuestionSpec, ...]
    weight: float = 1.0


@dataclass(frozen=True)
class OrdinalSemanticProgressSpec:
    edit_instruction: str
    primitives: tuple[OrdinalSemanticPrimitiveSpec, ...]
    preserve_constraints: tuple[str, ...]


class SingleTokenChoiceScorer(Protocol):
    def score_single_token_choices(
        self,
        image: torch.Tensor,
        question: str,
        choices: tuple[str, ...] = ORDINAL_CHOICE_LABELS,
    ) -> torch.Tensor: ...


class EndpointOrdinalValidationError(ValueError):
    """Raised when ordinal endpoints do not anchor at stages zero and four."""

    def __init__(self, diagnostics: dict[str, Any]):
        self.diagnostics = diagnostics
        super().__init__("Ordinal endpoint validation failed; inspect per-question distributions.")


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"`{name}` must be a non-empty string.")
    return value.strip()


def _positive_weight(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"`{name}` must be finite and positive.")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"`{name}` must be finite and positive.")
    return value


def _snake_case_id(value: object, name: str) -> str:
    value = _nonempty_string(value, name)
    if not re.fullmatch(r"[a-z][a-z0-9_]*", value):
        raise ValueError(f"`{name}` must be lower snake_case.")
    return value


def build_ordinal_semantic_progress_parser_prompt(edit_instruction: str) -> str:
    """Build the provider-neutral, two-image ordinal parser prompt."""

    instruction = _nonempty_string(edit_instruction, "edit_instruction")
    return f"""Ordinal Semantic Progress Parser v2.

You receive two images and one edit instruction:
- IMAGE 1 is the Source image.
- IMAGE 2 is the Native Full Edit produced from that Source.
- EDIT_INSTRUCTION: {instruction}

Identify only the visual primitive(s) that the instruction truly changes. For each primitive, write one or more
complementary visual ordinal questions. Every question must have exactly five ordered stage descriptions:
- stage 0 accurately describes the relevant Source state;
- stage 4 accurately describes the relevant Native Full state;
- stages 1, 2, and 3 are visually judgeable transitions in source-to-target order.

Rules:
- Describe only the instructed attribute, not general quality or background preservation.
- Put preservation requirements only in `preserve_constraints`.
- Do not use percentages, numeric strength values, or the word `strength`.
- Do not write empty labels such as "slightly edited" or "moderately edited"; every stage needs visual content.
- Prefer concise, parallel stage descriptions and do not invent unsupported objects or attributes.
- Multiple questions for one primitive must inspect complementary evidence, not paraphrase one another.

Return JSON only with exactly this schema:
{{
  "edit_instruction": "{instruction}",
  "primitives": [
    {{
      "id": "short_snake_case_id",
      "edit_description": "concise requested change",
      "weight": 1.0,
      "questions": [
        {{
          "id": "short_snake_case_id",
          "question": "one directly visual ordinal question",
          "weight": 1.0,
          "stages": [
            "visual source state",
            "visual early transition",
            "visual middle transition",
            "visual late transition",
            "visual full target state"
          ]
        }}
      ]
    }}
  ],
  "preserve_constraints": ["short constraint"]
}}"""


def parse_ordinal_semantic_progress_json(text: str) -> OrdinalSemanticProgressSpec:
    """Strictly validate an offline ordinal spec."""

    if not isinstance(text, str):
        raise TypeError("Ordinal semantic parser output must be a string.")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Ordinal semantic parser output is not valid JSON: {exc.msg}.") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "edit_instruction",
        "primitives",
        "preserve_constraints",
    }:
        raise ValueError("Ordinal semantic JSON contains missing or unexpected top-level fields.")

    instruction = _nonempty_string(payload["edit_instruction"], "edit_instruction")
    raw_primitives = payload["primitives"]
    if not isinstance(raw_primitives, list) or not raw_primitives:
        raise ValueError("`primitives` must contain at least one ordinal primitive.")
    primitives = []
    primitive_ids: set[str] = set()
    question_ids: set[str] = set()
    primitive_fields = {"id", "edit_description", "weight", "questions"}
    question_fields = {"id", "question", "weight", "stages"}
    for primitive_index, raw_primitive in enumerate(raw_primitives):
        prefix = f"primitives[{primitive_index}]"
        if not isinstance(raw_primitive, dict) or set(raw_primitive) != primitive_fields:
            raise ValueError(f"`{prefix}` contains missing or unexpected fields.")
        primitive_id = _snake_case_id(raw_primitive["id"], f"{prefix}.id")
        if primitive_id in primitive_ids:
            raise ValueError("Ordinal primitive IDs must be unique.")
        primitive_ids.add(primitive_id)
        raw_questions = raw_primitive["questions"]
        if not isinstance(raw_questions, list) or not raw_questions:
            raise ValueError(f"`{prefix}.questions` must contain at least one question.")
        questions = []
        for question_index, raw_question in enumerate(raw_questions):
            question_prefix = f"{prefix}.questions[{question_index}]"
            if not isinstance(raw_question, dict) or set(raw_question) != question_fields:
                raise ValueError(f"`{question_prefix}` contains missing or unexpected fields.")
            question_id = _snake_case_id(raw_question["id"], f"{question_prefix}.id")
            if question_id in question_ids:
                raise ValueError("Ordinal question IDs must be globally unique.")
            question_ids.add(question_id)
            question = _nonempty_string(raw_question["question"], f"{question_prefix}.question")
            if _FORBIDDEN_STAGE_PATTERN.search(question):
                raise ValueError("Ordinal questions must not ask about percentage or strength.")
            raw_stages = raw_question["stages"]
            if not isinstance(raw_stages, list) or len(raw_stages) != 5:
                raise ValueError(f"`{question_prefix}.stages` must contain exactly five descriptions.")
            stages = tuple(
                _nonempty_string(stage, f"{question_prefix}.stages[{stage_index}]")
                for stage_index, stage in enumerate(raw_stages)
            )
            if len(set(stages)) != 5:
                raise ValueError("The five ordinal stage descriptions must be unique.")
            if any(_FORBIDDEN_STAGE_PATTERN.search(stage) for stage in stages):
                raise ValueError("Ordinal stages must not contain percentages or strength wording.")
            questions.append(
                OrdinalQuestionSpec(
                    id=question_id,
                    question=question,
                    stages=stages,
                    weight=_positive_weight(raw_question["weight"], f"{question_prefix}.weight"),
                )
            )
        primitives.append(
            OrdinalSemanticPrimitiveSpec(
                id=primitive_id,
                edit_description=_nonempty_string(raw_primitive["edit_description"], f"{prefix}.edit_description"),
                questions=tuple(questions),
                weight=_positive_weight(raw_primitive["weight"], f"{prefix}.weight"),
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
    return OrdinalSemanticProgressSpec(instruction, tuple(primitives), constraints)


def make_ordinal_semantic_progress_cache_key(
    source_fingerprint: str,
    full_fingerprint: str,
    edit_instruction: str,
    parser_version: str = ORDINAL_SEMANTIC_PROGRESS_PARSER_VERSION,
) -> str:
    identity = {
        "cache_schema_version": ORDINAL_SEMANTIC_PROGRESS_CACHE_SCHEMA_VERSION,
        "source_fingerprint": _nonempty_string(source_fingerprint, "source_fingerprint"),
        "full_fingerprint": _nonempty_string(full_fingerprint, "full_fingerprint"),
        "edit_instruction": " ".join(_nonempty_string(edit_instruction, "edit_instruction").split()),
        "parser_version": _nonempty_string(parser_version, "parser_version"),
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _read_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": ORDINAL_SEMANTIC_PROGRESS_CACHE_SCHEMA_VERSION, "entries": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read ordinal semantic progress cache `{path}`: {exc}.") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("version") != ORDINAL_SEMANTIC_PROGRESS_CACHE_SCHEMA_VERSION
        or not isinstance(payload.get("entries"), dict)
    ):
        raise ValueError(f"Ordinal semantic progress cache `{path}` has an unsupported schema.")
    return payload


def save_cached_ordinal_semantic_progress_spec(
    cache_path: str | Path,
    source_fingerprint: str,
    full_fingerprint: str,
    spec: OrdinalSemanticProgressSpec,
    *,
    parser_version: str = ORDINAL_SEMANTIC_PROGRESS_PARSER_VERSION,
) -> None:
    if not isinstance(spec, OrdinalSemanticProgressSpec):
        raise TypeError("`spec` must be an OrdinalSemanticProgressSpec.")
    key = make_ordinal_semantic_progress_cache_key(
        source_fingerprint, full_fingerprint, spec.edit_instruction, parser_version
    )
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


def load_cached_ordinal_semantic_progress_spec(
    cache_path: str | Path,
    source_fingerprint: str,
    full_fingerprint: str,
    edit_instruction: str,
    *,
    parser_version: str = ORDINAL_SEMANTIC_PROGRESS_PARSER_VERSION,
) -> OrdinalSemanticProgressSpec | None:
    path = Path(cache_path)
    if not path.exists():
        return None
    key = make_ordinal_semantic_progress_cache_key(
        source_fingerprint, full_fingerprint, edit_instruction, parser_version
    )
    entry = _read_cache(path)["entries"].get(key)
    if entry is None:
        return None
    return parse_ordinal_semantic_progress_json(json.dumps(entry.get("spec")))


def build_ordinal_choice_prompt(
    question: OrdinalQuestionSpec,
    *,
    stage_to_label: tuple[str, str, str, str, str] = ORDINAL_CHOICE_LABELS,
) -> str:
    """Render five stage semantics under a deterministic label assignment."""

    if set(stage_to_label) != set(ORDINAL_CHOICE_LABELS) or len(stage_to_label) != 5:
        raise ValueError("`stage_to_label` must be a permutation of A, B, C, D, E.")
    label_to_stage = {label: stage for stage, label in enumerate(stage_to_label)}
    options = "\n".join(f"{label}. {question.stages[label_to_stage[label]]}" for label in ORDINAL_CHOICE_LABELS)
    return f"{question.question.strip()}\n\n{options}\n\nAnswer with exactly one letter: A, B, C, D, or E."


def ordinal_choice_distribution(
    choice_logprobs: torch.Tensor,
    *,
    temperature: float = 1.0,
    stage_to_label: tuple[str, str, str, str, str] = ORDINAL_CHOICE_LABELS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return label probabilities, stage probabilities, and ordinal expectation."""

    if choice_logprobs.ndim != 1 or choice_logprobs.numel() != 5:
        raise ValueError("Ordinal choice log-probabilities must have shape [5].")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("`temperature` must be finite and positive.")
    if set(stage_to_label) != set(ORDINAL_CHOICE_LABELS) or len(stage_to_label) != 5:
        raise ValueError("`stage_to_label` must be a permutation of A, B, C, D, E.")
    label_probs = torch.softmax(choice_logprobs.float() / float(temperature), dim=0)
    label_index = {label: index for index, label in enumerate(ORDINAL_CHOICE_LABELS)}
    stage_probs = torch.stack([label_probs[label_index[label]] for label in stage_to_label])
    nodes = stage_probs.new_tensor(ORDINAL_STAGE_NODES)
    return label_probs, stage_probs, (stage_probs * nodes).sum()


def _normalized_entropy(probabilities: torch.Tensor) -> torch.Tensor:
    return -(probabilities * probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()).sum() / math.log(
        probabilities.numel()
    )


def _as_float(value: torch.Tensor) -> float:
    return value.detach().float().cpu().item()


class EndpointRelativeOrdinalSemanticProgressReward:
    """Five-stage ordinal expectation calibrated by fixed Source/Full endpoints.

    ASSUMPTION: Qwen next-token probabilities over audited ordinal stage text
    form a useful continuous semantic coordinate. This is tested, not presumed.
    """

    def __init__(
        self,
        scorer: SingleTokenChoiceScorer,
        spec: OrdinalSemanticProgressSpec,
        source_image: torch.Tensor,
        full_image: torch.Tensor,
        *,
        temperature: float = 1.0,
        denominator_epsilon: float = 1e-6,
        fail_on_endpoint_validation: bool = True,
    ):
        if not isinstance(spec, OrdinalSemanticProgressSpec) or not spec.primitives:
            raise ValueError("Ordinal semantic progress requires at least one validated primitive.")
        if source_image.shape != full_image.shape or source_image.ndim != 4 or source_image.shape[1] != 3:
            raise ValueError("Source and full endpoint images must share shape [B, 3, H, W].")
        if source_image.shape[0] != 1:
            raise ValueError("Ordinal endpoint anchor preparation currently requires B=1.")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("`temperature` must be finite and positive.")
        if not math.isfinite(denominator_epsilon) or denominator_epsilon <= 0:
            raise ValueError("`denominator_epsilon` must be finite and positive.")
        self.scorer = scorer
        self.spec = spec
        self.temperature = float(temperature)
        self.denominator_epsilon = float(denominator_epsilon)
        self._anchors: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
        endpoint_diagnostics: dict[str, Any] = {}
        validation_failed = False
        with torch.no_grad():
            for primitive in spec.primitives:
                primitive_anchors = {}
                primitive_diagnostics = {}
                for question in primitive.questions:
                    source_values = self._raw_question(source_image, question)
                    full_values = self._raw_question(full_image, question)
                    denominator = (full_values["raw_expectation"] - source_values["raw_expectation"]).detach()
                    source_top = int(source_values["stage_probs"].argmax().item())
                    full_top = int(full_values["stage_probs"].argmax().item())
                    finite = all(
                        bool(torch.isfinite(value).all().item())
                        for value in (*source_values.values(), *full_values.values(), denominator)
                    )
                    valid = (
                        finite
                        and source_top == 0
                        and full_top == 4
                        and bool((denominator > self.denominator_epsilon).all().item())
                    )
                    validation_failed = validation_failed or not valid
                    primitive_anchors[question.id] = {
                        "raw_source": source_values["raw_expectation"].detach(),
                        "raw_full": full_values["raw_expectation"].detach(),
                        "denominator": denominator,
                    }
                    primitive_diagnostics[question.id] = {
                        "source_choice_probs": [_as_float(value) for value in source_values["stage_probs"]],
                        "full_choice_probs": [_as_float(value) for value in full_values["stage_probs"]],
                        "source_top_stage": source_top,
                        "full_top_stage": full_top,
                        "source_normalized_entropy": _as_float(source_values["normalized_entropy"]),
                        "full_normalized_entropy": _as_float(full_values["normalized_entropy"]),
                        "raw_source": _as_float(source_values["raw_expectation"]),
                        "raw_full": _as_float(full_values["raw_expectation"]),
                        "dynamic_range": _as_float(denominator),
                        "source_top_stage_valid": source_top == 0,
                        "full_top_stage_valid": full_top == 4,
                        "dynamic_range_valid": bool((denominator > self.denominator_epsilon).all().item()),
                        "valid": valid,
                    }
                self._anchors[primitive.id] = primitive_anchors
                endpoint_diagnostics[primitive.id] = primitive_diagnostics
        self.endpoint_diagnostics = endpoint_diagnostics
        self.endpoint_ordinal_validation_failed = validation_failed
        if validation_failed and fail_on_endpoint_validation:
            raise EndpointOrdinalValidationError(endpoint_diagnostics)

    @property
    def model(self):
        return getattr(self.scorer, "model", None)

    def _raw_question(
        self,
        image: torch.Tensor,
        question: OrdinalQuestionSpec,
        *,
        stage_to_label: tuple[str, str, str, str, str] = ORDINAL_CHOICE_LABELS,
    ) -> dict[str, torch.Tensor]:
        prompt = build_ordinal_choice_prompt(question, stage_to_label=stage_to_label)
        logprobs = self.scorer.score_single_token_choices(image, prompt, ORDINAL_CHOICE_LABELS)
        label_probs, stage_probs, expectation = ordinal_choice_distribution(
            logprobs, temperature=self.temperature, stage_to_label=stage_to_label
        )
        return {
            "choice_logprobs": logprobs,
            "label_probs": label_probs,
            "stage_probs": stage_probs,
            "raw_expectation": expectation,
            "normalized_entropy": _normalized_entropy(stage_probs),
        }

    def question_progress(
        self,
        image: torch.Tensor,
        primitive: OrdinalSemanticPrimitiveSpec,
        question: OrdinalQuestionSpec,
        *,
        stage_to_label: tuple[str, str, str, str, str] = ORDINAL_CHOICE_LABELS,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        values = self._raw_question(image, question, stage_to_label=stage_to_label)
        anchor = self._anchors[primitive.id][question.id]
        progress = (values["raw_expectation"] - anchor["raw_source"].to(image.device)) / anchor["denominator"].to(
            image.device
        )
        top_label_index = int(values["label_probs"].argmax().item())
        top_stage = int(values["stage_probs"].argmax().item())
        diagnostics = {
            "choice_logprobs": values["choice_logprobs"],
            "choice_probs": values["label_probs"],
            "stage_probs": values["stage_probs"],
            "top_choice": ORDINAL_CHOICE_LABELS[top_label_index],
            "top_stage": top_stage,
            "normalized_entropy": values["normalized_entropy"],
            "raw_ordinal_expectation": values["raw_expectation"],
            "endpoint_calibrated_progress_raw": progress,
            "progress_clamped": progress.clamp(0, 1),
            "stage_to_label": stage_to_label,
        }
        return progress, diagnostics

    def _target_vector(
        self,
        target_strengths: float | Sequence[float] | dict[str, float] | torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
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

    def progress_vector(self, image: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
            raise ValueError("Candidate ordinal scoring currently requires image shape [1, 3, H, W].")
        primitive_progresses = []
        diagnostics: dict[str, Any] = {}
        for primitive in self.spec.primitives:
            question_progresses = []
            question_diagnostics = {}
            for question in primitive.questions:
                progress, question_diagnostic = self.question_progress(image, primitive, question)
                question_progresses.append(progress)
                question_diagnostics[question.id] = question_diagnostic
            stacked = torch.stack(question_progresses)
            weights = stacked.new_tensor([question.weight for question in primitive.questions])
            weights = weights / weights.sum()
            primitive_progress = (weights * stacked).sum()
            primitive_progresses.append(primitive_progress)
            diagnostics[primitive.id] = {
                "questions": question_diagnostics,
                "primitive_progress": primitive_progress,
            }
        return torch.stack(primitive_progresses), diagnostics

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
        return TerminalObjectiveOutput(
            objective_name="ordinal_semantic_progress",
            loss=(weights * residual.square()).sum(),
            objective_error=(weights * residual.abs()).sum(),
            source_score=progress.new_tensor(0.0),
            full_score=progress.new_tensor(1.0),
            target_score=(weights * target).sum(),
            achieved_score=(weights * progress).sum(),
            diagnostics=diagnostics,
        )
