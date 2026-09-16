"""Minimal generator-native T5 text suppression diagnostic for FLUX.1-Kontext.

This is a single-pair diagnostic, not a paper-faithful contrastive
Difference-of-Means implementation. It contains no reward or velocity control.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class T5PhraseAlignment:
    text: str
    phrase: str
    token_indices: tuple[int, ...]
    token_ids: tuple[int, ...]
    token_pieces: tuple[str, ...]
    offsets: tuple[tuple[int, int], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "phrase": self.phrase,
            "token_indices": list(self.token_indices),
            "token_ids": list(self.token_ids),
            "token_pieces": list(self.token_pieces),
            "offsets": [list(item) for item in self.offsets],
        }


@dataclass(frozen=True)
class T5SteeringDirection:
    direction: torch.Tensor
    raw_direction_norm: float
    normalized_direction_norm: float
    source_token_indices: tuple[int, ...]
    target_token_indices: tuple[int, ...]
    source_token_pieces: tuple[str, ...]
    target_token_pieces: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "raw_norm": self.raw_direction_norm,
            "normalized_norm": self.normalized_direction_norm,
            "finite": bool(torch.isfinite(self.direction).all().item()),
            "source_token_indices": list(self.source_token_indices),
            "target_token_indices": list(self.target_token_indices),
            "source_token_pieces": list(self.source_token_pieces),
            "target_token_pieces": list(self.target_token_pieces),
        }


@dataclass(frozen=True)
class CLIPSteeringDirection:
    direction: torch.Tensor
    raw_direction_norm: float
    normalized_direction_norm: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "raw_norm": self.raw_direction_norm,
            "normalized_norm": self.normalized_direction_norm,
            "finite": bool(torch.isfinite(self.direction).all().item()),
        }


def _field(encoded: Any, name: str) -> Any:
    return encoded.get(name) if isinstance(encoded, dict) else getattr(encoded, name, None)


def _debug(tokenizer: Any, input_ids: torch.Tensor, offsets: Any, text: str, phrase: str) -> str:
    ids = input_ids.detach().cpu().reshape(-1).tolist()
    try:
        pieces = tokenizer.convert_ids_to_tokens(ids)
    except Exception:
        pieces = [str(value) for value in ids]
    try:
        shown_offsets = [tuple(int(value) for value in pair) for pair in offsets] if offsets is not None else []
    except (TypeError, ValueError):
        shown_offsets = [str(offsets)]
    return (
        "T5 phrase alignment failed.\n"
        f"original text: {text!r}\nphrase: {phrase!r}\n"
        f"token ids: {ids}\ndecoded token pieces: {pieces}\noffsets: {shown_offsets}"
    )


def find_t5_phrase_alignment(tokenizer: Any, text: str, phrase: str, max_sequence_length: int) -> T5PhraseAlignment:
    """Find all padded T5 token rows whose offsets overlap ``phrase``."""

    if not isinstance(text, str) or not text:
        raise ValueError("text must be a non-empty string")
    if not isinstance(phrase, str) or not phrase:
        raise ValueError("phrase must be a non-empty string")
    if not isinstance(max_sequence_length, int) or max_sequence_length <= 0:
        raise ValueError("max_sequence_length must be a positive integer")
    try:
        encoded = tokenizer(
            [text],
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
    except Exception as error:
        # Diagnose unsupported offsets with the same native tokenization,
        # rather than silently falling back to a guessed token position.
        try:
            without_offsets = tokenizer(
                [text],
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                return_tensors="pt",
            )
            debug_ids = torch.as_tensor(_field(without_offsets, "input_ids"))
        except Exception:
            debug_ids = torch.empty(0, dtype=torch.long)
        raise ValueError(
            f"Tokenizer does not support offsets: {error}; " + _debug(tokenizer, debug_ids, None, text, phrase)
        ) from error
    input_ids, offsets = _field(encoded, "input_ids"), _field(encoded, "offset_mapping")
    if input_ids is None or offsets is None:
        debug_ids = torch.as_tensor(input_ids) if input_ids is not None else torch.empty(0, dtype=torch.long)
        raise ValueError(_debug(tokenizer, debug_ids, offsets, text, phrase))
    if not torch.is_tensor(input_ids) or input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(_debug(tokenizer, torch.as_tensor(input_ids), offsets, text, phrase))
    if input_ids.shape[1] != max_sequence_length:
        raise ValueError(_debug(tokenizer, input_ids, offsets, text, phrase))
    try:
        normalized_offsets = [tuple(int(value) for value in pair) for pair in offsets[0]]
    except (TypeError, ValueError, IndexError) as error:
        raise ValueError(_debug(tokenizer, input_ids, offsets, text, phrase)) from error
    if len(normalized_offsets) != input_ids.shape[1]:
        raise ValueError(_debug(tokenizer, input_ids, normalized_offsets, text, phrase))
    occurrences = []
    start = text.find(phrase)
    while start >= 0:
        occurrences.append((start, start + len(phrase)))
        start = text.find(phrase, start + 1)
    if not occurrences:
        raise ValueError(_debug(tokenizer, input_ids[0], normalized_offsets, text, phrase))
    indices, selected_offsets = [], []
    for token_index, (token_start, token_end) in enumerate(normalized_offsets):
        if token_start < 0 or token_end < token_start or token_end > len(text):
            raise ValueError(_debug(tokenizer, input_ids[0], normalized_offsets, text, phrase))
        if token_start != token_end and any(token_start < end and token_end > begin for begin, end in occurrences):
            indices.append(token_index)
            selected_offsets.append((token_start, token_end))
    if not indices or any(index < 0 or index >= max_sequence_length for index in indices):
        raise ValueError(_debug(tokenizer, input_ids[0], normalized_offsets, text, phrase))
    token_ids = tuple(int(input_ids[0, index].item()) for index in indices)
    try:
        pieces = tuple(str(item) for item in tokenizer.convert_ids_to_tokens(token_ids))
    except Exception:
        pieces = tuple(str(item) for item in token_ids)
    return T5PhraseAlignment(text, phrase, tuple(indices), token_ids, pieces, tuple(selected_offsets))


def find_t5_phrase_indices(tokenizer: Any, text: str, phrase: str, max_sequence_length: int) -> list[int]:
    return list(find_t5_phrase_alignment(tokenizer, text, phrase, max_sequence_length).token_indices)


def encode_t5_text(pipe: Any, text: str, max_sequence_length: int) -> torch.Tensor:
    """Use the official Kontext T5 helper and no alternate encoder path."""

    if not isinstance(text, str) or not text:
        raise ValueError("text must be a non-empty string")
    if not hasattr(pipe, "_get_t5_prompt_embeds") or not hasattr(pipe, "tokenizer_2"):
        raise TypeError("pipe must expose _get_t5_prompt_embeds and tokenizer_2")
    encoder = getattr(pipe, "text_encoder_2", None)
    embeds = pipe._get_t5_prompt_embeds(
        prompt=text,
        num_images_per_prompt=1,
        max_sequence_length=max_sequence_length,
        device=getattr(pipe, "_execution_device", getattr(encoder, "device", None)),
        dtype=getattr(encoder, "dtype", None),
    )
    if not torch.is_tensor(embeds) or embeds.ndim != 3 or embeds.shape[0] != 1:
        raise ValueError("Native T5 encoding must have shape [1, tokens, hidden]")
    if not torch.isfinite(embeds).all():
        raise ValueError("Native T5 encoding contains non-finite values")
    return embeds


def pool_phrase_representation(
    prompt_embeds: torch.Tensor, phrase_indices: list[int] | tuple[int, ...]
) -> torch.Tensor:
    """Mean-pool selected T5 rows using float32 arithmetic."""

    if not torch.is_tensor(prompt_embeds) or prompt_embeds.ndim != 3 or prompt_embeds.shape[0] != 1:
        raise ValueError("prompt_embeds must have shape [1, tokens, hidden]")
    indices = tuple(int(index) for index in phrase_indices)
    if (
        not indices
        or len(set(indices)) != len(indices)
        or any(index < 0 or index >= prompt_embeds.shape[1] for index in indices)
    ):
        raise ValueError("phrase_indices must contain unique in-range indices")
    pooled = prompt_embeds[:, list(indices), :].float().mean(dim=1).squeeze(0)
    if not torch.isfinite(pooled).all():
        raise ValueError("Pooled phrase representation contains non-finite values")
    return pooled


def build_single_pair_t5_direction(
    pipe: Any,
    source_text: str,
    target_text: str,
    source_phrase: str,
    target_phrase: str,
    max_sequence_length: int,
    base_prompt_embeds: torch.Tensor | None = None,
) -> T5SteeringDirection:
    """Build normalized ``target - source`` using generator-native T5 rows."""

    source_embeds = encode_t5_text(pipe, source_text, max_sequence_length)
    target_embeds = encode_t5_text(pipe, target_text, max_sequence_length)
    if source_embeds.shape[-1] != target_embeds.shape[-1]:
        raise ValueError("Source and target T5 hidden dimensions differ")
    if base_prompt_embeds is not None and (
        not torch.is_tensor(base_prompt_embeds)
        or base_prompt_embeds.ndim != 3
        or base_prompt_embeds.shape[-1] != source_embeds.shape[-1]
    ):
        raise ValueError("T5 direction dimension does not match base prompt_embeds hidden dimension")
    source_alignment = find_t5_phrase_alignment(pipe.tokenizer_2, source_text, source_phrase, max_sequence_length)
    target_alignment = find_t5_phrase_alignment(pipe.tokenizer_2, target_text, target_phrase, max_sequence_length)
    source_representation = pool_phrase_representation(source_embeds, source_alignment.token_indices)
    target_representation = pool_phrase_representation(target_embeds, target_alignment.token_indices)
    raw_direction = target_representation - source_representation
    raw_norm = torch.linalg.vector_norm(raw_direction)
    if not torch.isfinite(raw_direction).all() or not torch.isfinite(raw_norm) or raw_norm <= 0:
        raise ValueError("T5 source-to-target direction is non-finite or zero")
    direction = raw_direction / raw_norm
    normalized_norm = torch.linalg.vector_norm(direction)
    if not torch.isfinite(direction).all() or normalized_norm <= 0:
        raise ValueError("Normalized T5 direction is non-finite or zero")
    return T5SteeringDirection(
        direction,
        float(raw_norm.detach().cpu()),
        float(normalized_norm.detach().cpu()),
        source_alignment.token_indices,
        target_alignment.token_indices,
        source_alignment.token_pieces,
        target_alignment.token_pieces,
    )


def apply_kontext_text_steering(
    base_prompt_embeds: torch.Tensor,
    edit_token_indices: list[int] | tuple[int, ...],
    steering_direction: torch.Tensor,
    factor: float,
) -> torch.Tensor:
    """Return a clone with ``factor * direction`` added to selected rows."""

    if not torch.is_tensor(base_prompt_embeds) or base_prompt_embeds.ndim != 3:
        raise ValueError("base_prompt_embeds must have shape [batch, tokens, hidden]")
    if not torch.is_tensor(steering_direction) or steering_direction.ndim != 1:
        raise ValueError("steering_direction must have shape [hidden]")
    if steering_direction.shape[0] != base_prompt_embeds.shape[-1]:
        raise ValueError("Steering direction dimension does not match prompt hidden dimension")
    if not torch.isfinite(base_prompt_embeds).all() or not torch.isfinite(steering_direction).all():
        raise ValueError("Prompt embeddings and direction must be finite")
    try:
        factor_value = float(factor)
    except (TypeError, ValueError) as error:
        raise ValueError("factor must be finite") from error
    if not torch.isfinite(torch.tensor(factor_value)):
        raise ValueError("factor must be finite")
    indices = tuple(int(index) for index in edit_token_indices)
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("edit_token_indices must contain unique indices")
    if any(index < 0 or index >= base_prompt_embeds.shape[1] for index in indices):
        raise ValueError("edit_token_indices contains an out-of-range index")
    steered = base_prompt_embeds.clone()
    if factor_value == 0:
        return steered
    update = steering_direction.to(device=steered.device, dtype=steered.dtype) * factor_value
    steered[:, list(indices), :] = steered[:, list(indices), :] + update
    if not torch.isfinite(steered).all():
        raise ValueError("Steered prompt embeddings contain non-finite values")
    return steered


def reversion_fraction_to_alpha(reversion_fraction: float, raw_direction_norm: float) -> float:
    """Convert text-space displacement fraction to a negative steering coefficient.

    This fraction describes embedding displacement, never image semantic strength.
    """

    fraction = float(reversion_fraction)
    norm = float(raw_direction_norm)
    if not math.isfinite(fraction) or fraction < 0:
        raise ValueError("reversion_fraction must be finite and nonnegative")
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("raw_direction_norm must be finite and positive")
    return -fraction * norm


def build_single_pair_clip_direction(
    pipe: Any,
    source_text: str,
    target_text: str,
    base_pooled_prompt_embeds: torch.Tensor | None = None,
) -> CLIPSteeringDirection:
    """Build target-minus-source pooled direction with Kontext's own CLIP helper."""

    if not isinstance(source_text, str) or not source_text or not isinstance(target_text, str) or not target_text:
        raise ValueError("CLIP source and target texts must be non-empty strings")
    if not hasattr(pipe, "_get_clip_prompt_embeds"):
        raise TypeError("pipe must expose native _get_clip_prompt_embeds")
    device = getattr(pipe, "_execution_device", None)
    source = pipe._get_clip_prompt_embeds(prompt=source_text, num_images_per_prompt=1, device=device)
    target = pipe._get_clip_prompt_embeds(prompt=target_text, num_images_per_prompt=1, device=device)
    if (
        not torch.is_tensor(source)
        or not torch.is_tensor(target)
        or source.ndim != 2
        or target.ndim != 2
        or source.shape[0] != 1
        or source.shape != target.shape
    ):
        raise ValueError("Native CLIP pooled embeddings must have matching shape [1, hidden]")
    if base_pooled_prompt_embeds is not None and (
        not torch.is_tensor(base_pooled_prompt_embeds)
        or base_pooled_prompt_embeds.ndim != 2
        or base_pooled_prompt_embeds.shape != source.shape
    ):
        raise ValueError("CLIP direction dimension does not match base pooled prompt embeddings")
    raw = target.float().squeeze(0) - source.float().squeeze(0)
    norm = torch.linalg.vector_norm(raw)
    if not torch.isfinite(raw).all() or not torch.isfinite(norm) or norm <= 0:
        raise ValueError("CLIP source-to-target direction is non-finite or zero")
    direction = raw / norm
    normalized_norm = torch.linalg.vector_norm(direction)
    if not torch.isfinite(direction).all() or normalized_norm <= 0:
        raise ValueError("Normalized CLIP direction is non-finite or zero")
    return CLIPSteeringDirection(direction, float(norm.detach().cpu()), float(normalized_norm.detach().cpu()))


def apply_kontext_pooled_steering(
    base_pooled_prompt_embeds: torch.Tensor, steering_direction: torch.Tensor, factor: float
) -> torch.Tensor:
    """Return a clone with a native CLIP pooled direction added; never mutate base."""

    if not torch.is_tensor(base_pooled_prompt_embeds) or base_pooled_prompt_embeds.ndim != 2:
        raise ValueError("base_pooled_prompt_embeds must have shape [batch, hidden]")
    if not torch.is_tensor(steering_direction) or steering_direction.ndim != 1:
        raise ValueError("CLIP steering_direction must have shape [hidden]")
    if base_pooled_prompt_embeds.shape[1] != steering_direction.shape[0]:
        raise ValueError("CLIP steering direction dimension mismatch")
    if not torch.isfinite(base_pooled_prompt_embeds).all() or not torch.isfinite(steering_direction).all():
        raise ValueError("CLIP pooled embeddings and direction must be finite")
    factor_value = float(factor)
    if not math.isfinite(factor_value):
        raise ValueError("CLIP steering factor must be finite")
    steered = base_pooled_prompt_embeds.clone()
    if factor_value == 0:
        return steered
    update = steering_direction.to(device=steered.device, dtype=steered.dtype) * factor_value
    steered = steered + update
    if not torch.isfinite(steered).all():
        raise ValueError("Steered CLIP pooled embeddings contain non-finite values")
    return steered
