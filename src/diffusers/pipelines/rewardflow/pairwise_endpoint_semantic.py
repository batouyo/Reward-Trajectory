"""Independent two-image endpoint-affinity reward for research bake-offs.

This comparator is not part of RewardFlow. It avoids direct Source/Full A/B
competition by scoring the same affirmative statement for Candidate/Source and
Candidate/Full pairs, then symmetrizing image order.
"""

from __future__ import annotations

import math
from typing import Any, Protocol, Sequence

import torch

from .relative_endpoint_parser import RelativeEndpointPrimitiveSpec, RelativeEndpointSemanticSpec
from .terminal_control import TerminalObjectiveOutput


class MultiImageAnswerScorer(Protocol):
    def score_multi_image_answer(
        self,
        images: Sequence[torch.Tensor],
        question: str,
        answer: str,
        *,
        margin: float = 0.0,
        lambda_margin: float = 0.0,
    ) -> torch.Tensor: ...


class PairwiseEndpointValidationError(ValueError):
    def __init__(self, diagnostics: dict[str, Any]):
        self.diagnostics = diagnostics
        super().__init__("Pairwise endpoint affinity validation failed; inspect endpoint diagnostics.")


def build_pairwise_endpoint_affinity_prompt(comparison_focus: str) -> str:
    """Build the identical question used for Source and Full reference pairs."""

    if not isinstance(comparison_focus, str) or not comparison_focus.strip():
        raise ValueError("`comparison_focus` must be a non-empty string.")
    return f"""You are comparing two images from the same image-editing trajectory.

Focus ONLY on:
{comparison_focus.strip()}

Ignore unrelated differences and preservation attributes.

Determine whether the two images show a similar visual semantic state for the focused attribute.

Evaluate the pair itself, not editing percentage or completion strength."""


def _number(value: torch.Tensor) -> float:
    return float(value.detach().float().cpu())


class PairwiseEndpointSemanticReward:
    """Symmetric, independently scored Source/Full endpoint affinity margin."""

    def __init__(
        self,
        scorer: MultiImageAnswerScorer,
        spec: RelativeEndpointSemanticSpec,
        source_image: torch.Tensor,
        full_image: torch.Tensor,
        *,
        statements: dict[str, str],
        denominator_epsilon: float = 1e-6,
        fail_on_endpoint_validation: bool = True,
    ):
        if not isinstance(spec, RelativeEndpointSemanticSpec) or not spec.primitives:
            raise ValueError("Pairwise endpoint reward requires a validated semantic spec.")
        if source_image.shape != full_image.shape or source_image.ndim != 4 or source_image.shape[1] != 3:
            raise ValueError("Source and full images must share shape [B, 3, H, W].")
        if source_image.shape[0] != 1:
            raise ValueError("Pairwise endpoint reward currently supports endpoint batch size one.")
        if not math.isfinite(denominator_epsilon) or denominator_epsilon <= 0:
            raise ValueError("`denominator_epsilon` must be finite and positive.")
        primitive_ids = {primitive.id for primitive in spec.primitives}
        if set(statements) != primitive_ids:
            raise ValueError("`statements` must contain exactly one fixed affirmative statement per primitive ID.")
        if any(not isinstance(value, str) or not value.strip() for value in statements.values()):
            raise ValueError("Every pairwise affirmative statement must be non-empty.")

        self.scorer = scorer
        self.spec = spec
        self.source_image = source_image.detach()
        self.full_image = full_image.detach()
        self.statements = {key: value.strip() for key, value in statements.items()}
        self.denominator_epsilon = float(denominator_epsilon)
        self._anchors: dict[str, dict[str, torch.Tensor]] = {}
        diagnostics: dict[str, Any] = {}
        failed = False
        with torch.no_grad():
            for primitive in spec.primitives:
                source = self.primitive_affinities(self.source_image, primitive)
                full = self.primitive_affinities(self.full_image, primitive)
                dynamic_range = (full["raw_margin"] - source["raw_margin"]).detach()
                finite = all(torch.isfinite(value).all().item() for value in (*source.values(), *full.values()))
                range_valid = bool((dynamic_range > self.denominator_epsilon).item())
                valid = finite and range_valid
                failed = failed or not valid
                self._anchors[primitive.id] = {
                    "source_margin": source["raw_margin"].detach(),
                    "full_margin": full["raw_margin"].detach(),
                    "dynamic_range": dynamic_range,
                }
                diagnostics[primitive.id] = {
                    "source": {name: _number(value) for name, value in source.items()},
                    "full": {name: _number(value) for name, value in full.items()},
                    "dynamic_range": _number(dynamic_range),
                    "finite": finite,
                    "dynamic_range_valid": range_valid,
                    "valid": valid,
                }
        self.endpoint_diagnostics = diagnostics
        self.endpoint_validation_failed = failed
        if failed and fail_on_endpoint_validation:
            raise PairwiseEndpointValidationError(diagnostics)

    @property
    def model(self):
        return getattr(self.scorer, "model", None)

    def primitive_affinities(
        self,
        candidate_image: torch.Tensor,
        primitive: RelativeEndpointPrimitiveSpec,
    ) -> dict[str, torch.Tensor]:
        if candidate_image.ndim != 4 or candidate_image.shape[0] != 1 or candidate_image.shape[1] != 3:
            raise ValueError("Candidate image must have shape [1, 3, H, W].")
        question = build_pairwise_endpoint_affinity_prompt(primitive.comparison_focus)
        answer = self.statements[primitive.id]
        candidate_source = self.scorer.score_multi_image_answer(
            (candidate_image, self.source_image), question, answer, margin=0.0, lambda_margin=0.0
        )
        source_candidate = self.scorer.score_multi_image_answer(
            (self.source_image, candidate_image), question, answer, margin=0.0, lambda_margin=0.0
        )
        candidate_full = self.scorer.score_multi_image_answer(
            (candidate_image, self.full_image), question, answer, margin=0.0, lambda_margin=0.0
        )
        full_candidate = self.scorer.score_multi_image_answer(
            (self.full_image, candidate_image), question, answer, margin=0.0, lambda_margin=0.0
        )
        source_affinity = 0.5 * (candidate_source + source_candidate)
        full_affinity = 0.5 * (candidate_full + full_candidate)
        candidate_first_margin = candidate_full - candidate_source
        reference_first_margin = full_candidate - source_candidate
        return {
            "candidate_source_affinity": candidate_source,
            "source_candidate_affinity": source_candidate,
            "candidate_full_affinity": candidate_full,
            "full_candidate_affinity": full_candidate,
            "source_affinity": source_affinity,
            "full_affinity": full_affinity,
            "candidate_first_margin": candidate_first_margin,
            "reference_first_margin": reference_first_margin,
            "raw_margin": full_affinity - source_affinity,
            "order_bias": 0.5 * (candidate_first_margin - reference_first_margin),
        }

    def coordinate_vector(self, image: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        coordinates = []
        diagnostics = {}
        for primitive in self.spec.primitives:
            values = self.primitive_affinities(image, primitive)
            anchor = self._anchors[primitive.id]
            coordinate = (values["raw_margin"] - anchor["source_margin"].to(image.device)) / anchor[
                "dynamic_range"
            ].to(image.device)
            coordinates.append(coordinate)
            diagnostics[primitive.id] = {
                **values,
                "source_margin": anchor["source_margin"].to(image.device),
                "full_margin": anchor["full_margin"].to(image.device),
                "dynamic_range": anchor["dynamic_range"].to(image.device),
                "relative_coordinate": coordinate,
                "coordinate_clamped_for_logging_only": coordinate.clamp(0, 1),
            }
        return torch.stack(coordinates), diagnostics

    def _target(self, target_strengths, reference: torch.Tensor) -> torch.Tensor:
        ids = [primitive.id for primitive in self.spec.primitives]
        if isinstance(target_strengths, dict):
            if set(target_strengths) != set(ids):
                raise ValueError("Target mapping must contain exactly every primitive ID.")
            values = [target_strengths[primitive_id] for primitive_id in ids]
        elif torch.is_tensor(target_strengths) and target_strengths.numel() > 1:
            values = target_strengths.flatten()
        elif isinstance(target_strengths, Sequence) and not isinstance(target_strengths, (str, bytes)):
            values = target_strengths
        else:
            values = [target_strengths] * len(ids)
        target = torch.as_tensor(values, device=reference.device, dtype=reference.dtype).flatten()
        if target.numel() != len(ids) or not torch.isfinite(target).all():
            raise ValueError("Provide one finite target per primitive.")
        if bool(((target < 0) | (target > 1)).any().item()):
            raise ValueError("Target strengths must lie in [0, 1].")
        return target

    def __call__(self, image: torch.Tensor, target_strengths) -> TerminalObjectiveOutput:
        coordinates, diagnostics = self.coordinate_vector(image)
        target = self._target(target_strengths, coordinates)
        residual = coordinates - target
        weights = coordinates.new_tensor([primitive.weight for primitive in self.spec.primitives])
        weights = weights / weights.sum()
        source = torch.stack([self._anchors[p.id]["source_margin"].to(image.device) for p in self.spec.primitives])
        full = torch.stack([self._anchors[p.id]["full_margin"].to(image.device) for p in self.spec.primitives])
        target_margin = source + target * (full - source)
        for index, primitive in enumerate(self.spec.primitives):
            diagnostics[primitive.id].update(
                {
                    "target_margin": target_margin[index],
                    "normalized_residual": residual[index],
                    "semantic_loss": residual[index].square(),
                }
            )
        return TerminalObjectiveOutput(
            objective_name="pairwise_endpoint_affinity_margin",
            loss=(weights * residual.square()).sum(),
            objective_error=(weights * residual.abs()).sum(),
            source_score=(weights * source).sum(),
            full_score=(weights * full).sum(),
            target_score=(weights * target_margin).sum(),
            achieved_score=(weights * coordinates).sum(),
            diagnostics=diagnostics,
        )
