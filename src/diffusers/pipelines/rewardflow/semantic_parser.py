"""Semantic-primitives prompt, validation, and provider-independent cache."""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import asdict, dataclass
from pathlib import Path

import PIL.Image


SEMANTIC_PARSER_VERSION = "rewardflow-figure10-v1"
SEMANTIC_CACHE_SCHEMA_VERSION = 2


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


def _normalize_edit_instruction(edit_instruction: str) -> str:
    if not isinstance(edit_instruction, str) or not edit_instruction.strip():
        raise ValueError("`edit_instruction` must be a non-empty string.")
    return " ".join(edit_instruction.split())


def fingerprint_image(image: object) -> str:
    """Return a deterministic, content-based fingerprint for a PIL image or tensor.

    PIL images are canonicalized to RGB and hashed as dimensions plus raw RGB
    bytes. Tensors are detached, moved to CPU, canonicalized to contiguous
    float32, and hashed with their shape. Tensor device and source dtype do not
    affect the result when their numeric pixel values are identical.
    """

    digest = hashlib.sha256()
    if isinstance(image, PIL.Image.Image):
        rgb = image.convert("RGB")
        digest.update(b"rewardflow-pil-rgb-v1\0")
        digest.update(struct.pack(">II", rgb.width, rgb.height))
        digest.update(rgb.tobytes())
        return digest.hexdigest()

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - diffusers paper mode requires torch
        raise TypeError("Tensor image fingerprinting requires torch.") from exc
    if not torch.is_tensor(image):
        raise TypeError("`image` must be a PIL.Image.Image or torch.Tensor.")
    if image.numel() == 0:
        raise ValueError("Cannot fingerprint an empty image tensor.")

    canonical = image.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if not torch.isfinite(canonical).all():
        raise ValueError("Image tensors must contain only finite values.")
    canonical = canonical.clone()
    canonical[canonical == 0] = 0  # Canonicalize negative zero.
    digest.update(b"rewardflow-tensor-float32-v1\0")
    digest.update(json.dumps(list(canonical.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical.numpy().tobytes(order="C"))
    return digest.hexdigest()


def make_semantic_cache_key(
    edit_instruction: str,
    image_fingerprint: str,
    parser_version: str = SEMANTIC_PARSER_VERSION,
) -> str:
    """Bind a semantic parse cache key to image, instruction, and prompt version."""

    instruction = _normalize_edit_instruction(edit_instruction)
    if not isinstance(image_fingerprint, str) or not image_fingerprint.strip():
        raise ValueError("`image_fingerprint` must be a non-empty string.")
    if not isinstance(parser_version, str) or not parser_version.strip():
        raise ValueError("`parser_version` must be a non-empty string.")

    # ASSUMPTION: The paper says parses are cached but does not specify a key
    # or file format. Canonical JSON makes every semantic input explicit and
    # avoids ambiguous string concatenation.
    identity = {
        "cache_schema_version": SEMANTIC_CACHE_SCHEMA_VERSION,
        "edit_instruction": instruction,
        "image_fingerprint": image_fingerprint.strip(),
        "parser_version": parser_version.strip(),
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_cache(path: Path) -> dict:
    if not path.exists():
        return {"version": SEMANTIC_CACHE_SCHEMA_VERSION, "entries": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read semantic parse cache `{path}`: {exc}.") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("version") != SEMANTIC_CACHE_SCHEMA_VERSION
        or not isinstance(payload.get("entries"), dict)
    ):
        raise ValueError(f"Semantic parse cache `{path}` has an unsupported schema.")
    return payload


def save_cached_parse(
    cache_path: str | Path,
    edit_instruction: str,
    image_fingerprint: str,
    result: SemanticParseResult,
    *,
    parser_version: str = SEMANTIC_PARSER_VERSION,
) -> None:
    """Store one validated parse in an atomic JSON cache."""

    if not isinstance(result, SemanticParseResult):
        raise TypeError("`result` must be a SemanticParseResult.")
    instruction = _normalize_edit_instruction(edit_instruction)
    cache_key = make_semantic_cache_key(instruction, image_fingerprint, parser_version)
    path = Path(cache_path)
    payload = _read_cache(path)
    payload["entries"][cache_key] = {
        "edit_instruction": instruction,
        "image_fingerprint": image_fingerprint.strip(),
        "parser_version": parser_version.strip(),
        "result": asdict(result),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_cached_parse(
    cache_path: str | Path,
    edit_instruction: str,
    image_fingerprint: str,
    *,
    parser_version: str = SEMANTIC_PARSER_VERSION,
) -> SemanticParseResult | None:
    """Load a cached parse, returning ``None`` when the instruction is absent."""

    path = Path(cache_path)
    if not path.exists():
        return None
    instruction = _normalize_edit_instruction(edit_instruction)
    cache_key = make_semantic_cache_key(instruction, image_fingerprint, parser_version)
    entry = _read_cache(path)["entries"].get(cache_key)
    if entry is None:
        return None
    if (
        entry.get("edit_instruction") != instruction
        or entry.get("image_fingerprint") != image_fingerprint.strip()
        or entry.get("parser_version") != parser_version.strip()
    ):
        raise ValueError("Semantic parse cache key collision or corrupted identity fields.")
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
