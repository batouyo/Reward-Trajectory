"""Auditable utilities for feature-controller and blinded visual evaluation.

Nothing in this module participates in the feature reward or terminal-control
optimization. Pixel diagnostics and the offline visual judge are evaluation
only.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from typing import Sequence

import torch

from .endpoint_comparator_metrics import ordering_diagnostics


TRAINING_STRENGTHS = (0.2, 0.5, 0.8)
DENSE_STRENGTHS = tuple(index / 10 for index in range(11))
BLIND_LABELS = ("A", "B", "C", "D", "E")
BLIND_IDENTITIES = ("source", "best_0.2", "best_0.5", "best_0.8", "native_full")
VISUAL_JUDGE_PERMUTATION_SEED = 20260916
VISUAL_JUDGE_PROMPT_VERSION = "feature-controller-v5-blind-judge-v1"
PRESERVATION_CATEGORIES = (
    "shape",
    "texture",
    "lighting",
    "background",
    "noise_or_artifacts",
    "global_color_drift",
)

SOURCE_INPUT_PROVENANCE = "SOURCE INPUT"
NATIVE_FULL_PROVENANCE = "MODEL-GENERATED NATIVE FULL FLUX-KONTEXT OUTPUT"
FEATURE_CONTROL_PROVENANCE = "MODEL-GENERATED FLUX-KONTEXT FEATURE-CONTROL OUTPUT"
EVALUATION_MASK_PROVENANCE = "EVALUATION ONLY; NOT USED BY REWARD OR CONTROLLER"


def format_strength_tag(strength: float) -> str:
    return f"{float(strength):.1f}".replace(".", "p")


def build_blind_permutations(
    identities: Sequence[str] = BLIND_IDENTITIES,
    *,
    count: int = 6,
    seed: int = VISUAL_JUDGE_PERMUTATION_SEED,
) -> list[dict[str, str]]:
    """Create reproducible unique blind-label mappings before judge calls."""

    identities = tuple(identities)
    if len(identities) != len(BLIND_LABELS) or len(set(identities)) != len(identities):
        raise ValueError("Blind evaluation requires exactly five unique image identities.")
    if not 1 <= count <= math.factorial(len(identities)):
        raise ValueError("Requested blind permutation count is invalid.")
    generator = random.Random(seed)
    seen = set()
    mappings = []
    while len(mappings) < count:
        order = list(identities)
        generator.shuffle(order)
        key = tuple(order)
        if key in seen:
            continue
        seen.add(key)
        mappings.append(dict(zip(BLIND_LABELS, order)))
    return mappings


def visual_judge_json_schema() -> dict[str, object]:
    relation = {"type": "string", "enum": ["closer_to_source", "intermediate", "closer_to_full"]}
    per_image_entry = {
        "type": "object",
        "additionalProperties": False,
        "required": ["endpoint_relation", "semantic_description", "preservation_violations"],
        "properties": {
            "endpoint_relation": relation,
            "semantic_description": {"type": "string", "minLength": 1},
            "preservation_violations": {
                "type": "array",
                "items": {"type": "string", "enum": list(PRESERVATION_CATEGORIES)},
            },
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["ordered_labels", "per_image", "indistinguishable_pairs"],
        "properties": {
            "ordered_labels": {
                "type": "array",
                "minItems": 5,
                "maxItems": 5,
                "items": {"type": "string", "enum": list(BLIND_LABELS)},
            },
            "per_image": {
                "type": "object",
                "additionalProperties": False,
                "required": list(BLIND_LABELS),
                "properties": {label: per_image_entry for label in BLIND_LABELS},
            },
            "indistinguishable_pairs": {
                "type": "array",
                "items": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 2,
                        "items": {"type": "string", "enum": list(BLIND_LABELS)},
                },
            },
        },
    }


def parse_visual_judge_json(text: str) -> dict[str, object]:
    """Strictly validate one blind judge response without repairing output."""

    if not isinstance(text, str):
        raise TypeError("Visual judge output must be JSON text.")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"Visual judge output is not valid JSON: {error.msg}.") from error
    if not isinstance(payload, dict) or set(payload) != {"ordered_labels", "per_image", "indistinguishable_pairs"}:
        raise ValueError("Visual judge output contains missing or unexpected top-level fields.")
    ordered = payload["ordered_labels"]
    if not isinstance(ordered, list) or len(ordered) != 5 or set(ordered) != set(BLIND_LABELS):
        raise ValueError("`ordered_labels` must be a permutation of A-E.")
    per_image = payload["per_image"]
    if not isinstance(per_image, dict) or set(per_image) != set(BLIND_LABELS):
        raise ValueError("`per_image` must contain exactly A-E.")
    allowed_relations = {"closer_to_source", "intermediate", "closer_to_full"}
    for label, item in per_image.items():
        if not isinstance(item, dict) or set(item) != {
            "endpoint_relation",
            "semantic_description",
            "preservation_violations",
        }:
            raise ValueError(f"`per_image.{label}` contains missing or unexpected fields.")
        if item["endpoint_relation"] not in allowed_relations:
            raise ValueError(f"`per_image.{label}.endpoint_relation` is invalid.")
        if not isinstance(item["semantic_description"], str) or not item["semantic_description"].strip():
            raise ValueError(f"`per_image.{label}.semantic_description` must be non-empty.")
        violations = item["preservation_violations"]
        if (
            not isinstance(violations, list)
            or len(violations) != len(set(violations))
            or any(value not in PRESERVATION_CATEGORIES for value in violations)
        ):
            raise ValueError(f"`per_image.{label}.preservation_violations` is invalid.")
    pairs = payload["indistinguishable_pairs"]
    if not isinstance(pairs, list):
        raise ValueError("`indistinguishable_pairs` must be a list.")
    normalized_pairs = set()
    for pair in pairs:
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or len(set(pair)) != 2
            or any(x not in BLIND_LABELS for x in pair)
        ):
            raise ValueError("Every indistinguishable pair must contain two distinct labels from A-E.")
        normalized = tuple(sorted(pair))
        if normalized in normalized_pairs:
            raise ValueError("Indistinguishable pairs must be unique.")
        normalized_pairs.add(normalized)
    return payload


def remap_visual_judgment(payload: dict[str, object], mapping: dict[str, str]) -> dict[str, object]:
    if set(mapping) != set(BLIND_LABELS) or len(set(mapping.values())) != 5:
        raise ValueError("Blind mapping must map A-E to five unique identities.")
    return {
        "ordered_identities": [mapping[label] for label in payload["ordered_labels"]],
        "per_identity": {mapping[label]: item for label, item in payload["per_image"].items()},
        "indistinguishable_identity_pairs": [
            sorted((mapping[left], mapping[right])) for left, right in payload["indistinguishable_pairs"]
        ],
    }


def summarize_visual_judgments(trials: Sequence[dict[str, object]]) -> dict[str, object]:
    """Apply the preregistered 5/6 ordering and majority-relation gates."""

    if len(trials) != 6:
        raise ValueError("Formal visual evaluation requires exactly six completed trials.")
    expected = list(BLIND_IDENTITIES)
    exact_order_count = sum(trial["ordered_identities"] == expected for trial in trials)
    relation_counts = {identity: Counter() for identity in BLIND_IDENTITIES}
    violation_counts = {identity: Counter() for identity in BLIND_IDENTITIES}
    pair_counts = Counter()
    rank_positions = {identity: [] for identity in BLIND_IDENTITIES}
    for trial in trials:
        for rank, identity in enumerate(trial["ordered_identities"]):
            rank_positions[identity].append(rank)
        for identity, item in trial["per_identity"].items():
            relation_counts[identity][item["endpoint_relation"]] += 1
            violation_counts[identity].update(item["preservation_violations"])
        pair_counts.update(tuple(pair) for pair in trial["indistinguishable_identity_pairs"])
    relation_gate = {
        "best_0.2_closer_to_source_majority": relation_counts["best_0.2"]["closer_to_source"] >= 4,
        "best_0.5_intermediate_majority": relation_counts["best_0.5"]["intermediate"] >= 4,
        "best_0.8_closer_to_full_majority": relation_counts["best_0.8"]["closer_to_full"] >= 4,
    }
    adjacent_indistinguishable = {
        "best_0.2_vs_best_0.5": pair_counts[tuple(sorted(("best_0.2", "best_0.5")))],
        "best_0.5_vs_best_0.8": pair_counts[tuple(sorted(("best_0.5", "best_0.8")))],
    }
    indistinguishable_gate = all(count <= 3 for count in adjacent_indistinguishable.values())
    majority_violations = {
        identity: sorted(category for category, count in counts.items() if count >= 4)
        for identity, counts in violation_counts.items()
    }
    reward_hacking = any(majority_violations[identity] for identity in ("best_0.2", "best_0.5", "best_0.8"))
    ordering_gate = exact_order_count >= 5
    primary_gate = ordering_gate and all(relation_gate.values()) and indistinguishable_gate
    return {
        "expected_order": expected,
        "exact_order_count": exact_order_count,
        "required_exact_order_count": 5,
        "ordering_gate": ordering_gate,
        "relation_counts": {identity: dict(counts) for identity, counts in relation_counts.items()},
        "relation_gate": relation_gate,
        "adjacent_indistinguishable_counts": adjacent_indistinguishable,
        "indistinguishable_gate": indistinguishable_gate,
        "mean_rank_position_zero_based": {
            identity: sum(positions) / len(positions) for identity, positions in rank_positions.items()
        },
        "preservation_violation_counts": {identity: dict(counts) for identity, counts in violation_counts.items()},
        "majority_preservation_violations": majority_violations,
        "reward_hacking": reward_hacking,
        "primary_visual_ordering_gate": primary_gate,
        "visual_weak_mid_strong": "PASS" if primary_gate and not reward_hacking else "FAIL",
    }


def endpoint_difference_evaluation_mask(
    source: torch.Tensor,
    full: torch.Tensor,
    *,
    top_fraction: float = 0.25,
) -> torch.Tensor:
    """Select an exact top pixel fraction using Source/Full only."""

    if source.shape != full.shape or source.ndim != 4 or source.shape[0] != 1 or source.shape[1] != 3:
        raise ValueError("Endpoint mask requires matching batch-one RGB images.")
    if not 0 < top_fraction <= 1:
        raise ValueError("`top_fraction` must lie in (0, 1].")
    difference = (full.detach().float() - source.detach().float()).abs().mean(dim=1, keepdim=True)
    flat = difference.flatten(1)
    count = max(1, math.ceil(flat.shape[1] * top_fraction))
    indices = torch.argsort(flat, dim=1, descending=True, stable=True)[:, :count]
    mask = torch.zeros_like(flat)
    mask.scatter_(1, indices, 1.0)
    return mask.reshape_as(difference).detach()


def endpoint_pixel_diagnostics(
    image: torch.Tensor,
    source: torch.Tensor,
    full: torch.Tensor,
    edit_mask: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> dict[str, torch.Tensor]:
    """Compute evaluation-only endpoint errors and unclamped edit-axis projection."""

    if image.shape != source.shape or image.shape != full.shape:
        raise ValueError("Image and endpoints must share shape.")
    if edit_mask.shape != (image.shape[0], 1, image.shape[2], image.shape[3]):
        raise ValueError("Edit mask must have shape [B, 1, H, W].")
    mask = edit_mask.to(device=image.device, dtype=image.dtype).expand_as(image)
    preserve = 1 - mask

    def region_mse(left, right, weight):
        return ((left - right).float().square() * weight.float()).sum() / weight.float().sum().clamp_min(eps)

    delta = (full - source).float()
    centered = (image - source).float()
    numerator = (centered * delta * mask.float()).sum()
    denominator = (delta.square() * mask.float()).sum().clamp_min(eps)
    return {
        "edit_mse_to_source": region_mse(image, source, mask),
        "edit_mse_to_full": region_mse(image, full, mask),
        "preserve_mse_to_source": region_mse(image, source, preserve),
        "preserve_mse_to_full": region_mse(image, full, preserve),
        "endpoint_pixel_axis_projection": numerator / denominator,
    }


def dense_feature_curve_diagnostics(
    strengths: Sequence[float],
    coordinates: Sequence[float],
    *,
    sudden_jump_threshold: float = 0.2,
) -> dict[str, object]:
    if len(strengths) != len(coordinates) or len(strengths) < 2:
        raise ValueError("Dense strengths and coordinates must align.")
    if any(left >= right for left, right in zip(strengths, strengths[1:])):
        raise ValueError("Dense strengths must be strictly increasing.")
    ordering = ordering_diagnostics(coordinates)
    deviations = [abs(float(coordinate) - float(strength)) for strength, coordinate in zip(strengths, coordinates)]
    jumps = [float(right) - float(left) for left, right in zip(coordinates, coordinates[1:])]
    # ASSUMPTION: the brief does not define "sudden" numerically. A change
    # larger than 0.2 between adjacent 0.1 requests is reported as a jump.
    return {
        **ordering,
        "maximum_coordinate_deviation": max(deviations),
        "mean_coordinate_deviation": sum(deviations) / len(deviations),
        "adjacent_coordinate_changes": jumps,
        "sudden_jump_threshold": sudden_jump_threshold,
        "sudden_jump_indices": [index for index, jump in enumerate(jumps) if abs(jump) > sudden_jump_threshold],
    }
