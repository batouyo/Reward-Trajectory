"""Semantic-primitives prompt, validation, and provider-independent cache."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class SemanticParseResult:
    short_prompts: list[str]
    question: str
    answer: str


def build_semantic_parser_prompt(edit_instruction: str) -> str:
    """Build the Figure 10 parser prompt; the caller supplies the image to its provider."""

    if not isinstance(edit_instruction, str) or not edit_instruction.strip():
        raise ValueError("`edit_instruction` must be a non-empty string.")
    return f"""Vision-language Editing Assistant.

You are a vision-language assistant. You receive an image and a short edit instruction.
1) Extract short edit prompts: output a compact list of 5-12 atomic, actionable tags/phrases that guide the image edit.
Include:
    - Visible subject descriptors (pose, angle, clothing items) actually present.
    - The edit action(s) and key visual attributes (style, color, size, placement).
    - Constraints to preserve identity, lighting, composition, realism, and continuity.
    - Practical rendering notes (alignment, shadows, reflections, edges).
2) Create exactly one Q&A pair focused on the final edited image's appearance.
    - Ask one question that would most affect the final look (for example style, colorway, size/scale, placement,
      material/finish, or mood/lighting continuity).
    - Give one concise answer based on the image/instruction; if not determinable, answer
      "Unspecified from image."

Rules:
- Output JSON only in the exact schema below--no extra text.
- Keep each short prompt <= 6 words; imperative, neutral wording.
- Do not invent details not visible or implied by the instruction.
- Avoid sensitive inferences (for example ethnicity or health).
- Use American English.

Input:
EDIT_INSTRUCTION: {edit_instruction.strip()}

Output schema (JSON only):
{{
  "short_prompts": ["<tag1>", "<tag2>", "..."],
  "qna": {{
    "question": "<visual-outcome question>",
    "answer": "<concise answer or 'Unspecified from image.'>"
  }}
}}"""


def parse_semantic_parser_json(text: str) -> SemanticParseResult:
    """Parse and strictly validate one semantic-primitives JSON object."""

    if not isinstance(text, str):
        raise TypeError("Semantic parser output must be a string.")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Semantic parser output is not valid JSON: {exc.msg}.") from exc
    if not isinstance(payload, dict) or set(payload) != {"short_prompts", "qna"}:
        raise ValueError("Parser JSON must contain exactly `short_prompts` and `qna`.")

    short_prompts = payload["short_prompts"]
    if not isinstance(short_prompts, list) or not 5 <= len(short_prompts) <= 12:
        raise ValueError("`short_prompts` must be a list containing 5 to 12 entries.")
    validated_prompts = []
    for index, value in enumerate(short_prompts):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"`short_prompts[{index}]` must be a non-empty string.")
        value = value.strip()
        if len(value.split()) > 6:
            raise ValueError(f"`short_prompts[{index}]` exceeds the six-word limit.")
        validated_prompts.append(value)

    qna = payload["qna"]
    if not isinstance(qna, dict) or set(qna) != {"question", "answer"}:
        raise ValueError("`qna` must contain exactly one `question` and one `answer`.")
    question = qna["question"]
    answer = qna["answer"]
    if not isinstance(question, str) or not question.strip():
        raise ValueError("`qna.question` must be a non-empty string.")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("`qna.answer` must be a non-empty string.")
    return SemanticParseResult(validated_prompts, question.strip(), answer.strip())


def _cache_key(edit_instruction: str) -> str:
    # ASSUMPTION: The paper says parses are cached but does not specify a key
    # or file format. A SHA-256 key avoids path-unsafe instruction strings.
    return hashlib.sha256(edit_instruction.strip().encode("utf-8")).hexdigest()


def _read_cache(path: Path) -> dict:
    if not path.exists():
        return {"version": 1, "entries": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read semantic parse cache `{path}`: {exc}.") from exc
    if not isinstance(payload, dict) or payload.get("version") != 1 or not isinstance(payload.get("entries"), dict):
        raise ValueError(f"Semantic parse cache `{path}` has an unsupported schema.")
    return payload


def save_cached_parse(
    cache_path: str | Path,
    edit_instruction: str,
    result: SemanticParseResult,
) -> None:
    """Store one validated parse in an atomic JSON cache."""

    if not isinstance(result, SemanticParseResult):
        raise TypeError("`result` must be a SemanticParseResult.")
    path = Path(cache_path)
    payload = _read_cache(path)
    payload["entries"][_cache_key(edit_instruction)] = {
        "edit_instruction": edit_instruction.strip(),
        "result": asdict(result),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_cached_parse(cache_path: str | Path, edit_instruction: str) -> SemanticParseResult | None:
    """Load a cached parse, returning ``None`` when the instruction is absent."""

    path = Path(cache_path)
    if not path.exists():
        return None
    entry = _read_cache(path)["entries"].get(_cache_key(edit_instruction))
    if entry is None:
        return None
    if entry.get("edit_instruction") != edit_instruction.strip():
        raise ValueError("Semantic parse cache key collision or corrupted instruction entry.")
    result = entry.get("result")
    if not isinstance(result, dict):
        raise ValueError("Cached semantic parse result is malformed.")
    return parse_semantic_parser_json(
        json.dumps(
            {
                "short_prompts": result.get("short_prompts"),
                "qna": {"question": result.get("question"), "answer": result.get("answer")},
            }
        )
    )
