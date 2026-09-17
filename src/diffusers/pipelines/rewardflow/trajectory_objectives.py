"""Image-trajectory objective for RewardSlider V1 independent controls.

CLIP endpoint-axis values are used solely in an adjacent ranking objective;
they are never treated as calibrated semantic percentages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

import torch
import torch.nn.functional as F

from .coupled_terminal_control import (
    CoupledControlPrior,
    adjacent_ranking_loss,
    coarse_anchor_indices,
    coarse_pairwise_ranking_loss,
    control_band_loss,
    control_energy_loss,
    control_no_jump_loss,
    endpoint_progress_batch,
    fine_jump_loss,
    per_interval_semantic_coverage_loss,
    scalar_direction_residual_diagnostics,
    semantic_coverage_loss,
    spatial_prior_loss,
    triangle_deficit_loss,
    weighted_source_preservation,
)


class BatchedImageFeatureEncoder(Protocol):
    def encode_image(self, image: torch.Tensor) -> torch.Tensor: ...

    def encode_images(self, images: torch.Tensor) -> torch.Tensor: ...


class DreamSimDistance(Protocol):
    def distance(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True)
class RewardSliderV1LossWeights:
    """Smoke-test defaults, all intentionally explicit rather than paper values."""

    semantic_order: float = 1.0
    semantic_coverage: float = 1.0
    semantic_pairwise_order: float = 1.0
    fine_jump: float = 1.0
    coarse_gap: float = 1.0
    second: float = 0.25
    preserve: float = 1.0
    ctrl: float = 0.0
    band: float = 0.01
    spatial: float = 0.05
    energy: float = 0.0


@dataclass
class RewardSliderV1ObjectiveOutput:
    total: torch.Tensor
    components: dict[str, torch.Tensor]
    progress: torch.Tensor
    adjacent_semantic_gaps: torch.Tensor
    coarse_anchor_indices: tuple[int, ...]
    coarse_semantic_gaps: torch.Tensor
    dreamsim_gaps: torch.Tensor
    dreamsim_gap_fractions: torch.Tensor
    dreamsim_worst_jump: torch.Tensor
    dreamsim_worst_jump_index: torch.Tensor
    coarse_dreamsim_gaps: torch.Tensor
    coarse_dreamsim_fractions: torch.Tensor
    triangle_raw_deficits: torch.Tensor
    triangle_normalized_deficits: torch.Tensor
    control_diagnostics: list[dict[str, torch.Tensor]]
    scalar_direction_diagnostics: list[dict[str, torch.Tensor]]


class CoupledTrajectoryObjective:
    """Cache fixed endpoints and evaluate all V1 terms on K candidate images."""

    def __init__(
        self,
        *,
        feature_encoder: BatchedImageFeatureEncoder,
        dreamsim: DreamSimDistance,
        source_image: torch.Tensor,
        native_full_image: torch.Tensor,
        prior: CoupledControlPrior,
        token_height: int,
        token_width: int,
        weights: RewardSliderV1LossWeights = RewardSliderV1LossWeights(),
        order_margin: float = 0.0,
        sem_min_fraction: float = 0.05,
        sem_max_fraction: float = 0.55,
        fine_max_jump_fraction: float = 0.55,
        coarse_min_fraction: float = 0.05,
        coarse_max_fraction: float = 0.55,
    ):
        if source_image.shape != native_full_image.shape or source_image.shape[0] != 1:
            raise ValueError("Fixed source and native-full endpoints must be matching [1,3,H,W] tensors.")
        if token_height * token_width != prior.relevance[0].shape[1]:
            raise ValueError("Token grid does not match the cached native relevance tokens.")
        self.feature_encoder = feature_encoder
        self.dreamsim = dreamsim
        self.source_image = source_image.detach()
        self.native_full_image = native_full_image.detach()
        self.prior = prior
        self.token_height, self.token_width = token_height, token_width
        self.weights = weights
        self.order_margin = order_margin
        self.sem_min_fraction, self.sem_max_fraction = sem_min_fraction, sem_max_fraction
        self.fine_max_jump_fraction = fine_max_jump_fraction
        self.coarse_min_fraction, self.coarse_max_fraction = coarse_min_fraction, coarse_max_fraction
        with torch.no_grad():
            self.source_feature = feature_encoder.encode_image(self.source_image).detach()
            self.full_feature = feature_encoder.encode_image(self.native_full_image).detach()
        cache_endpoints = getattr(dreamsim, "cache_endpoint_embeddings", None)
        if callable(cache_endpoints):
            cache_endpoints(self.source_image, self.native_full_image)
        # Mean relevance across controlled timesteps is a fixed image-space
        # preservation prior, not a generated image segmentation mask.
        token_map = torch.stack([item[0] for item in prior.relevance]).mean(0).reshape(1, 1, token_height, token_width)
        self.relevance_image = F.interpolate(
            token_map, size=source_image.shape[-2:], mode="bilinear", align_corners=False
        ).detach()

    @staticmethod
    def _pair_distances(images: torch.Tensor, distance: DreamSimDistance, offset: int) -> torch.Tensor:
        return distance.distance(images[:-offset], images[offset:])

    def _weight(self, weights: RewardSliderV1LossWeights | Mapping[str, float] | None, name: str) -> float:
        weights = self.weights if weights is None else weights
        return float(weights[name]) if isinstance(weights, Mapping) else float(getattr(weights, name))

    def image_total(self, components: dict[str, torch.Tensor], weights=None) -> torch.Tensor:
        """Image-only objective used by Pass A of exact two-pass VJP."""

        return sum(
            self._weight(weights, name) * components[key]
            for name, key in (
                ("semantic_order", "semantic_order"),
                ("semantic_coverage", "semantic_coverage"),
                ("semantic_pairwise_order", "semantic_pairwise_order"),
                ("fine_jump", "fine_jump"),
                ("coarse_gap", "coarse_gap"),
                ("second", "second"),
                ("preserve", "preserve"),
            )
        )

    def control_total(
        self, controls: Sequence[torch.Tensor], weights=None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], list[dict[str, torch.Tensor]]]:
        """Control-only terms, evaluated after the image reward graph is freed."""

        band, diagnostics = control_band_loss(controls, self.prior.directions, self.prior.relevance)
        components = {
            "control_no_jump": control_no_jump_loss(controls),
            "band": band,
            "spatial": spatial_prior_loss(controls, self.prior.relevance),
            "energy": control_energy_loss(controls),
        }
        total = sum(
            self._weight(weights, name) * components[key]
            for name, key in (
                ("ctrl", "control_no_jump"),
                ("band", "band"),
                ("spatial", "spatial"),
                ("energy", "energy"),
            )
        )
        return total, components, diagnostics

    def total_from_components(self, components: dict[str, torch.Tensor], weights=None) -> torch.Tensor:
        """Combine raw components with explicit effective weights."""

        return self.image_total(components, weights) + sum(
            self._weight(weights, name) * components[key]
            for name, key in (
                ("ctrl", "control_no_jump"),
                ("band", "band"),
                ("spatial", "spatial"),
                ("energy", "energy"),
            )
        )

    def __call__(self, candidates: torch.Tensor, controls: Sequence[torch.Tensor]) -> RewardSliderV1ObjectiveOutput:
        if candidates.ndim != 4 or candidates.shape[0] < 1 or candidates.shape[1:] != self.source_image.shape[1:]:
            raise ValueError("Candidates must have shape [K,3,H,W] matching source endpoints.")
        if len(controls) != len(self.prior.directions):
            raise ValueError("One coupled control tensor is required per cached controlled timestep.")
        features = self.feature_encoder.encode_images(candidates)
        progress = torch.cat(
            (
                features.new_zeros(1),
                endpoint_progress_batch(
                    self.source_feature.to(features.device), features, self.full_feature.to(features.device)
                ),
                features.new_ones(1),
            )
        )
        anchor_indices = coarse_anchor_indices(progress.numel())
        semantic_order = adjacent_ranking_loss(progress, margin=self.order_margin)
        semantic_pairwise_order = coarse_pairwise_ranking_loss(progress, anchor_indices=anchor_indices)
        semantic_coverage, semantic_collapse, semantic_jump, coarse_semantic_gaps = per_interval_semantic_coverage_loss(
            progress,
            anchor_indices=anchor_indices,
            min_fraction=self.sem_min_fraction,
            max_fraction=self.sem_max_fraction,
        )
        trajectory = torch.cat(
            (self.source_image.to(candidates), candidates, self.native_full_image.to(candidates)), dim=0
        )
        gaps = self._pair_distances(trajectory, self.dreamsim, 1)
        fine_jump, gap_fractions, worst_jump = fine_jump_loss(gaps, max_fraction=self.fine_max_jump_fraction)
        coarse_trajectory = trajectory[list(anchor_indices)]
        coarse_gaps = self._pair_distances(coarse_trajectory, self.dreamsim, 1)
        coarse_gap, coarse_collapse, coarse_jump, coarse_fractions = semantic_coverage_loss(
            torch.cat((coarse_gaps.new_zeros(1), coarse_gaps.cumsum(0))),
            min_fraction=self.coarse_min_fraction,
            max_fraction=self.coarse_max_fraction,
        )
        coarse_skip = self._pair_distances(coarse_trajectory, self.dreamsim, 2)
        second, raw_deficits, normalized_deficits = triangle_deficit_loss(coarse_gaps, coarse_skip)
        preserve = weighted_source_preservation(candidates, self.source_image, self.relevance_image.to(candidates))
        control_total, control_components, diagnostics = self.control_total(controls)
        components = {
            "semantic_order": semantic_order,
            "semantic_coverage": semantic_coverage,
            "semantic_pairwise_order": semantic_pairwise_order,
            "semantic_collapse": semantic_collapse,
            "semantic_jump": semantic_jump,
            "fine_jump": fine_jump,
            "coarse_gap": coarse_gap,
            "coarse_collapse": coarse_collapse,
            "coarse_jump": coarse_jump,
            "second": second,
            "preserve": preserve,
            **control_components,
        }
        total = self.total_from_components(components)
        return RewardSliderV1ObjectiveOutput(
            total=total,
            components=components,
            progress=progress,
            adjacent_semantic_gaps=progress[1:] - progress[:-1],
            coarse_anchor_indices=anchor_indices,
            coarse_semantic_gaps=coarse_semantic_gaps,
            dreamsim_gaps=gaps,
            dreamsim_gap_fractions=gap_fractions,
            dreamsim_worst_jump=worst_jump,
            dreamsim_worst_jump_index=gap_fractions.argmax(),
            coarse_dreamsim_gaps=coarse_gaps,
            coarse_dreamsim_fractions=coarse_fractions,
            triangle_raw_deficits=raw_deficits,
            triangle_normalized_deficits=normalized_deficits,
            control_diagnostics=diagnostics,
            scalar_direction_diagnostics=scalar_direction_residual_diagnostics(controls, self.prior.directions),
        )
