"""Differentiable preservation objectives for RewardSlider V2."""

from __future__ import annotations

from typing import Protocol, Sequence

import torch
import torch.nn.functional as F

from .coupled_terminal_control import weighted_source_preservation


class TensorImageDistance(Protocol):
    def distance(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor: ...

def build_image_space_relevance(
    relevance_per_step: Sequence[torch.Tensor],
    *,
    token_height: int,
    token_width: int,
    target_height: int,
    target_width: int,
) -> torch.Tensor:
    """Aggregate token relevance and reshape the FLUX grid into image space."""

    if not relevance_per_step:
        raise ValueError("At least one per-step relevance map is required.")
    if token_height < 1 or token_width < 1 or target_height < 1 or target_width < 1:
        raise ValueError("Token and target dimensions must be positive.")
    token_count = token_height * token_width
    token_maps = []
    for relevance in relevance_per_step:
        if relevance.ndim == 1:
            token_map = relevance
        elif relevance.ndim == 2 and relevance.shape[0] >= 1:
            token_map = relevance[0]
        else:
            raise ValueError("Each relevance map must have shape [tokens] or [batch, tokens].")
        if token_map.shape[0] != token_count:
            raise ValueError("Token relevance length must equal token_height * token_width.")
        token_maps.append(token_map)
    token_relevance = torch.stack(token_maps, dim=0).mean(dim=0)
    grid = token_relevance.reshape(1, 1, token_height, token_width)
    return F.interpolate(
        grid.float(), size=(target_height, target_width), mode="bilinear", align_corners=False
    ).detach()

def masked_lpips_preservation_loss(
    candidate: torch.Tensor,
    source: torch.Tensor,
    relevance: torch.Tensor,
    distance: TensorImageDistance,
) -> torch.Tensor:
    """Measure LPIPS preservation after removing edit-region influence.

    ``relevance`` is a soft image map in ``[0, 1]``.  The source replacement is
    detached so the objective can only send image gradients through the
    preserved outside region; no hard candidate mask is applied.
    """

    if candidate.ndim != 4 or source.ndim != 4 or source.shape[1:] != candidate.shape[1:]:
        raise ValueError("Candidate and source must be [batch, channels, height, width] with matching image shape.")
    if source.shape[0] not in (1, candidate.shape[0]):
        raise ValueError("Source batch must be one or match the candidate batch.")
    if relevance.ndim != 4 or relevance.shape[1] != 1 or relevance.shape[-2:] != candidate.shape[-2:]:
        raise ValueError("Relevance must have shape [1 or batch, 1, height, width].")
    if relevance.shape[0] not in (1, candidate.shape[0]):
        raise ValueError("Relevance batch must be one or match the candidate batch.")
    source_value = source.detach().to(device=candidate.device, dtype=candidate.dtype)
    if source_value.shape[0] == 1 and candidate.shape[0] != 1:
        source_value = source_value.expand(candidate.shape[0], -1, -1, -1)
    relevance_value = relevance.detach().to(device=candidate.device, dtype=candidate.dtype).clamp(0, 1)
    if relevance_value.shape[0] == 1 and candidate.shape[0] != 1:
        relevance_value = relevance_value.expand(candidate.shape[0], -1, -1, -1)
    preserve_image = (1 - relevance_value) * candidate + relevance_value * source_value
    values = distance.distance(preserve_image, source_value)
    if not torch.is_tensor(values) or not torch.isfinite(values).all():
        raise ValueError("The tensor image distance must return finite tensor values.")
    return values.float().reshape(-1).mean()


def weighted_l1_preservation_loss(
    candidate: torch.Tensor,
    source: torch.Tensor,
    relevance: torch.Tensor,
) -> torch.Tensor:
    """Retain V1's weighted L1 preservation as a diagnostic/fallback."""

    return weighted_source_preservation(candidate, source, relevance)
