"""Encoder-agnostic Source-to-Full embedding geometry.

This module defines a research diagnostic. Its uncalibrated coordinate must
not be interpreted as a perceptual percentage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


class ImageFeatureEncoder(Protocol):
    def encode_image(self, image: torch.Tensor) -> torch.Tensor: ...


@dataclass
class EndpointGeometryOutput:
    progress: torch.Tensor
    off_axis: torch.Tensor
    source_distance: torch.Tensor
    full_distance: torch.Tensor
    axis_norm: torch.Tensor
    cosine_to_source: torch.Tensor
    cosine_to_full: torch.Tensor
    cosine_distance_source: torch.Tensor
    cosine_distance_full: torch.Tensor


def _feature_vector(feature: torch.Tensor, name: str) -> torch.Tensor:
    if not torch.is_tensor(feature) or not feature.is_floating_point():
        raise TypeError(f"`{name}` must be a floating-point tensor.")
    if feature.ndim == 2 and feature.shape[0] == 1:
        feature = feature[0]
    if feature.ndim != 1 or feature.numel() == 0:
        raise ValueError(f"`{name}` must be a non-empty vector or shape [1, D].")
    if not torch.isfinite(feature).all():
        raise ValueError(f"`{name}` must be finite.")
    return feature.float()


def endpoint_axis_geometry(
    source_feature: torch.Tensor,
    candidate_feature: torch.Tensor,
    full_feature: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> EndpointGeometryOutput:
    """Project Candidate onto the normalized Source-to-Full endpoint axis.

    The projection is deliberately not clamped. Values below zero and above
    one are valid extrapolation diagnostics.
    """

    if eps <= 0:
        raise ValueError("`eps` must be positive.")
    # Encoders own L2 normalization. Keeping this utility literal to the
    # supplied f_s/f_c/f_f avoids silently changing the registered geometry.
    source = _feature_vector(source_feature, "source_feature")
    candidate = _feature_vector(candidate_feature, "candidate_feature")
    full = _feature_vector(full_feature, "full_feature")
    if source.shape != candidate.shape or source.shape != full.shape:
        raise ValueError("Source, Candidate, and Full features must have the same shape.")

    axis = full - source
    axis_squared = torch.dot(axis, axis)
    axis_norm = torch.sqrt(axis_squared)
    if float(axis_norm.detach()) <= eps:
        raise ValueError("Source and Full features define a degenerate endpoint axis.")

    candidate_delta = candidate - source
    progress = torch.dot(candidate_delta, axis) / (axis_squared + eps)
    residual = candidate_delta - progress * axis
    off_axis = torch.linalg.vector_norm(residual) / (axis_norm + eps)
    cosine_to_source = torch.dot(candidate, source)
    cosine_to_full = torch.dot(candidate, full)
    return EndpointGeometryOutput(
        progress=progress,
        off_axis=off_axis,
        source_distance=torch.linalg.vector_norm(candidate - source),
        full_distance=torch.linalg.vector_norm(candidate - full),
        axis_norm=axis_norm,
        cosine_to_source=cosine_to_source,
        cosine_to_full=cosine_to_full,
        cosine_distance_source=1 - cosine_to_source,
        cosine_distance_full=1 - cosine_to_full,
    )


class CachedEndpointEmbeddingGeometry:
    """Cache fixed endpoint features while preserving Candidate gradients."""

    def __init__(
        self,
        encoder: ImageFeatureEncoder,
        source_image: torch.Tensor,
        full_image: torch.Tensor,
        *,
        eps: float = 1e-8,
    ):
        if source_image.shape != full_image.shape:
            raise ValueError("Source and Full images must have the same shape.")
        self.encoder = encoder
        self.source_image = source_image.detach()
        self.full_image = full_image.detach()
        self.eps = float(eps)
        with torch.no_grad():
            self.source_feature = encoder.encode_image(self.source_image).detach()
            self.full_feature = encoder.encode_image(self.full_image).detach()
        endpoint_axis_geometry(self.source_feature, self.source_feature, self.full_feature, eps=self.eps)

    def __call__(self, candidate_image: torch.Tensor) -> EndpointGeometryOutput:
        candidate_feature = self.encoder.encode_image(candidate_image)
        return endpoint_axis_geometry(
            self.source_feature.to(candidate_feature.device),
            candidate_feature,
            self.full_feature.to(candidate_feature.device),
            eps=self.eps,
        )
