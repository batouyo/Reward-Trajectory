"""Focus-conditioned Qwen feature-distance diagnostic for endpoint geometry.

This is a research baseline, not a RewardFlow paper component and not a claim
of human semantic strength.
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence

import torch
import torch.nn.functional as F

from .relative_endpoint_parser import RelativeEndpointPrimitiveSpec, RelativeEndpointSemanticSpec
from .terminal_control import TerminalObjectiveOutput


class FocusConditionedFeatureScorer(Protocol):
    def focus_conditioned_representation(self, image: torch.Tensor, prompt: str) -> torch.Tensor: ...


def build_focus_conditioned_feature_prompt(comparison_focus: str) -> str:
    if not isinstance(comparison_focus, str) or not comparison_focus.strip():
        raise ValueError("`comparison_focus` must be a non-empty string.")
    return f"Focus only on: {comparison_focus.strip()}. Represent the visible semantic state of this attribute."


class FeatureEndpointDistanceReward:
    """Cosine-distance-ratio baseline over frozen Qwen multimodal hidden states.

    The registered representation is the L2-normalized final-layer hidden state
    at the final prompt token. Source and Full features are cached under
    ``torch.no_grad()``; Candidate extraction remains differentiable.
    """

    def __init__(
        self,
        scorer: FocusConditionedFeatureScorer,
        spec: RelativeEndpointSemanticSpec,
        source_image: torch.Tensor,
        full_image: torch.Tensor,
        *,
        denominator_epsilon: float = 1e-8,
    ):
        if not isinstance(spec, RelativeEndpointSemanticSpec) or not spec.primitives:
            raise ValueError("Feature endpoint reward requires a validated semantic spec.")
        if source_image.shape != full_image.shape or source_image.ndim != 4 or source_image.shape[1] != 3:
            raise ValueError("Source and full images must share shape [B, 3, H, W].")
        if denominator_epsilon <= 0:
            raise ValueError("`denominator_epsilon` must be positive.")
        self.scorer = scorer
        self.spec = spec
        self.source_image = source_image.detach()
        self.full_image = full_image.detach()
        self.denominator_epsilon = float(denominator_epsilon)
        self._endpoint_features: dict[str, dict[str, torch.Tensor]] = {}
        with torch.no_grad():
            for primitive in spec.primitives:
                prompt = build_focus_conditioned_feature_prompt(primitive.comparison_focus)
                source = F.normalize(scorer.focus_conditioned_representation(self.source_image, prompt).float(), dim=0)
                full = F.normalize(scorer.focus_conditioned_representation(self.full_image, prompt).float(), dim=0)
                self._endpoint_features[primitive.id] = {"source": source.detach(), "full": full.detach()}

    @property
    def model(self):
        return getattr(self.scorer, "model", None)

    def primitive_distances(
        self,
        candidate_image: torch.Tensor,
        primitive: RelativeEndpointPrimitiveSpec,
    ) -> dict[str, torch.Tensor]:
        prompt = build_focus_conditioned_feature_prompt(primitive.comparison_focus)
        candidate = F.normalize(self.scorer.focus_conditioned_representation(candidate_image, prompt).float(), dim=0)
        source = self._endpoint_features[primitive.id]["source"].to(candidate.device)
        full = self._endpoint_features[primitive.id]["full"].to(candidate.device)
        cosine_source = 1 - torch.dot(candidate, source)
        cosine_full = 1 - torch.dot(candidate, full)
        cosine_ratio = cosine_source / (cosine_source + cosine_full + self.denominator_epsilon)
        euclidean_source = torch.linalg.vector_norm(candidate - source)
        euclidean_full = torch.linalg.vector_norm(candidate - full)
        euclidean_ratio = euclidean_source / (euclidean_source + euclidean_full + self.denominator_epsilon)
        axis = full - source
        axis_norm = torch.linalg.vector_norm(axis)
        unit_axis = axis / axis_norm.clamp_min(self.denominator_epsilon)
        projection = torch.dot(candidate - source, unit_axis) / (axis_norm + self.denominator_epsilon)
        return {
            "cosine_distance_source": cosine_source,
            "cosine_distance_full": cosine_full,
            "cosine_distance_ratio": cosine_ratio,
            "euclidean_distance_source": euclidean_source,
            "euclidean_distance_full": euclidean_full,
            "euclidean_distance_ratio": euclidean_ratio,
            "axis_projection": projection,
        }

    def coordinate_vector(self, image: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        coordinates = []
        diagnostics = {}
        for primitive in self.spec.primitives:
            values = self.primitive_distances(image, primitive)
            coordinates.append(values["cosine_distance_ratio"])
            diagnostics[primitive.id] = values
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
        for index, primitive in enumerate(self.spec.primitives):
            diagnostics[primitive.id].update(
                {
                    "target_coordinate": target[index],
                    "normalized_residual": residual[index],
                    "semantic_loss": residual[index].square(),
                }
            )
        zero = coordinates.new_zeros(())
        one = coordinates.new_ones(())
        return TerminalObjectiveOutput(
            objective_name="focus_conditioned_feature_distance_ratio",
            loss=(weights * residual.square()).sum(),
            objective_error=(weights * residual.abs()).sum(),
            source_score=zero,
            full_score=one,
            target_score=(weights * target).sum(),
            achieved_score=(weights * coordinates).sum(),
            diagnostics=diagnostics,
        )
