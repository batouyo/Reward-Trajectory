"""Image-trajectory objective for RewardSlider V1 independent controls.

CLIP endpoint-axis values are used solely in an adjacent ranking objective;
they are never treated as calibrated semantic percentages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import torch
import torch.nn.functional as F

from .coupled_terminal_control import (
    CoupledControlPrior,
    adjacent_ranking_loss,
    control_band_loss,
    control_energy_loss,
    control_smoothness_loss,
    endpoint_progress_batch,
    gap_bound_loss,
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

    rank: float = 1.0
    gap: float = 1.0
    second: float = 0.25
    preserve: float = 1.0
    ctrl: float = 0.05
    band: float = 0.05
    spatial: float = 0.05
    energy: float = 1e-4


@dataclass
class RewardSliderV1ObjectiveOutput:
    total: torch.Tensor
    components: dict[str, torch.Tensor]
    progress: torch.Tensor
    adjacent_semantic_gaps: torch.Tensor
    dreamsim_gaps: torch.Tensor
    triangle_deficits: torch.Tensor
    control_diagnostics: list[dict[str, torch.Tensor]]


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
        rank_margin: float = 0.01,
        min_gap_ratio: float = 0.25,
        max_gap_ratio: float = 2.0,
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
        self.rank_margin = rank_margin
        self.min_gap_ratio, self.max_gap_ratio = min_gap_ratio, max_gap_ratio
        with torch.no_grad():
            self.source_feature = feature_encoder.encode_image(self.source_image).detach()
            self.full_feature = feature_encoder.encode_image(self.native_full_image).detach()
        # Mean relevance across controlled timesteps is a fixed image-space
        # preservation prior, not a generated image segmentation mask.
        token_map = torch.stack([item[0] for item in prior.relevance]).mean(0).reshape(1, 1, token_height, token_width)
        self.relevance_image = F.interpolate(token_map, size=source_image.shape[-2:], mode="bilinear", align_corners=False).detach()

    @staticmethod
    def _pair_distances(images: torch.Tensor, distance: DreamSimDistance, offset: int) -> torch.Tensor:
        return distance.distance(images[:-offset], images[offset:])

    def __call__(self, candidates: torch.Tensor, controls: Sequence[torch.Tensor]) -> RewardSliderV1ObjectiveOutput:
        if candidates.ndim != 4 or candidates.shape[0] < 1 or candidates.shape[1:] != self.source_image.shape[1:]:
            raise ValueError("Candidates must have shape [K,3,H,W] matching source endpoints.")
        if len(controls) != len(self.prior.directions):
            raise ValueError("One coupled control tensor is required per cached controlled timestep.")
        features = self.feature_encoder.encode_images(candidates)
        progress = torch.cat(
            (
                features.new_zeros(1),
                endpoint_progress_batch(self.source_feature.to(features.device), features, self.full_feature.to(features.device)),
                features.new_ones(1),
            )
        )
        rank = adjacent_ranking_loss(progress, margin=self.rank_margin)
        trajectory = torch.cat((self.source_image.to(candidates), candidates, self.native_full_image.to(candidates)), dim=0)
        gaps = self._pair_distances(trajectory, self.dreamsim, 1)
        source_full_distance = self.dreamsim.distance(self.source_image.to(candidates), self.native_full_image.to(candidates)).reshape(())
        gap, collapse, jump = gap_bound_loss(
            gaps,
            source_full_distance,
            min_ratio=self.min_gap_ratio,
            max_ratio=self.max_gap_ratio,
        )
        skip = self._pair_distances(trajectory, self.dreamsim, 2)
        second, deficits = triangle_deficit_loss(gaps, skip, source_full_distance)
        preserve = weighted_source_preservation(candidates, self.source_image, self.relevance_image.to(candidates))
        band, diagnostics = control_band_loss(controls, self.prior.directions, self.prior.relevance)
        spatial = spatial_prior_loss(controls, self.prior.relevance)
        smooth = control_smoothness_loss(controls, self.prior.directions)
        energy = control_energy_loss(controls)
        components = {
            "rank": rank,
            "gap": gap,
            "gap_collapse": collapse,
            "gap_jump": jump,
            "second": second,
            "preserve": preserve,
            "control_smoothness": smooth,
            "band": band,
            "spatial": spatial,
            "energy": energy,
        }
        total = sum(getattr(self.weights, name) * components[key] for name, key in (
            ("rank", "rank"), ("gap", "gap"), ("second", "second"), ("preserve", "preserve"),
            ("ctrl", "control_smoothness"), ("band", "band"), ("spatial", "spatial"), ("energy", "energy"),
        ))
        return RewardSliderV1ObjectiveOutput(
            total=total,
            components=components,
            progress=progress,
            adjacent_semantic_gaps=progress[1:] - progress[:-1],
            dreamsim_gaps=gaps,
            triangle_deficits=deficits,
            control_diagnostics=diagnostics,
        )
