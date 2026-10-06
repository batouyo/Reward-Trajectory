"""Instance-specific progress projection between source and native full-edit features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import os
import warnings

import torch
import torch.nn.functional as F


@dataclass
class ProgressValues:
    raw_progress: torch.Tensor
    clamped_progress: torch.Tensor
    drift: torch.Tensor
    anchor_norm: torch.Tensor
    candidate_feature_norm: torch.Tensor
    warning: str | None = None

    def diagnostics(self) -> dict[str, float | str | None]:
        return {
            "raw_progress": float(self.raw_progress.detach().mean().cpu()),
            "clamped_progress": float(self.clamped_progress.detach().mean().cpu()),
            "drift": float(self.drift.detach().mean().cpu()),
            "anchor_norm": float(self.anchor_norm.detach().mean().cpu()),
            "candidate_feature_norm": float(self.candidate_feature_norm.detach().mean().cpu()),
            "warning": self.warning,
        }


def project_progress_features(
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    candidate_features: torch.Tensor,
    *,
    eps: float = 1e-8,
    danger_threshold: float = 1e-7,
    warning_threshold: float = 1e-4,
) -> ProgressValues:
    """Project candidate movement onto the source-to-native-target feature axis."""
    if source_features.ndim < 2 or target_features.ndim < 2 or candidate_features.ndim < 2:
        raise ValueError("feature tensors must include batch and feature dimensions")
    source = source_features.detach().float()
    target = target_features.detach().float()
    candidate = candidate_features.float()
    if source.shape[-1] != target.shape[-1] or source.shape[-1] != candidate.shape[-1]:
        raise ValueError("source, target, and candidate feature dimensions must match")
    if source.shape[0] != 1 or target.shape[0] != 1:
        raise ValueError("progress anchors must each describe one image")
    direction = target - source
    direction_norm_sq = direction.square().sum(dim=-1, keepdim=True)
    anchor_norm = direction_norm_sq.sqrt().reshape(())
    norm_value = float(anchor_norm.detach().cpu())
    if not torch.isfinite(anchor_norm):
        raise FloatingPointError("source-to-target progress anchor norm is non-finite")
    if norm_value <= danger_threshold:
        raise ValueError(
            f"native full-edit progress anchor is degenerate (norm={norm_value:.3e})"
        )
    warning = None
    if norm_value < warning_threshold:
        warning = (
            f"native full-edit progress anchor is near-degenerate (norm={norm_value:.3e})"
        )
        warnings.warn(warning, RuntimeWarning, stacklevel=2)

    displacement = candidate - source
    raw = (displacement * direction).sum(dim=-1) / (
        direction_norm_sq.reshape(()) + eps
    )
    orthogonal = displacement - raw.unsqueeze(-1) * direction
    drift = orthogonal.square().sum(dim=-1) / (
        direction_norm_sq.reshape(()) + eps
    )
    return ProgressValues(
        raw_progress=raw,
        clamped_progress=raw.clamp(0.0, 1.0),
        drift=drift,
        anchor_norm=anchor_norm,
        candidate_feature_norm=candidate.norm(dim=-1),
        warning=warning,
    )


class ProgressEstimator:
    """Frozen SigLIP or DINOv2 image encoder with differentiable candidate path."""

    DEFAULT_SIGLIP_PATH = "/data15/hyp/weight/reward_models/siglip-so400m-patch14-384"
    DEFAULT_DINO_PATH = "/data15/hyp/weight/dinov2-large"

    def __init__(
        self,
        backbone: str = "siglip",
        *,
        device: torch.device | str = "cuda:0",
        cache_dir: str = "/data15/hyp/weight",
        model_path: str | None = None,
        encoder: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        if backbone not in {"siglip", "dino"}:
            raise ValueError("backbone must be 'siglip' or 'dino'")
        self.backbone = backbone
        self.device = torch.device(device)
        self.cache_dir = cache_dir
        self.model_path = model_path or (
            self.DEFAULT_SIGLIP_PATH if backbone == "siglip" else self.DEFAULT_DINO_PATH
        )
        self._encoder = encoder
        self._reward = None
        self.source_features: torch.Tensor | None = None
        self.target_features: torch.Tensor | None = None
        self.anchor_diagnostics: dict[str, object] | None = None

    def _load_encoder(self) -> None:
        if self._encoder is not None:
            return
        if self.backbone == "siglip":
            from rewardflow_calibration.rewards.semantic import SemanticReward
            reward = SemanticReward(
                device=self.device,
                model_name=self.model_path if os.path.exists(self.model_path)
                else "google/siglip-so400m-patch14-384",
                cache_dir=self.cache_dir,
            )
            model = reward.model["model"]
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)

            def encode(image: torch.Tensor) -> torch.Tensor:
                pixels = reward._preprocess_image_siglip(image.to(self.device))
                features = model.get_image_features(pixel_values=pixels)
                return F.normalize(features.float(), p=2, dim=-1)
            self._reward = reward
            self._encoder = encode
            return

        from rewardflow_calibration.rewards.spatial_layout import SpatialLayoutReward
        reward = SpatialLayoutReward(
            device=self.device,
            model_name=self.model_path if os.path.exists(self.model_path)
            else "facebook/dinov2-large",
            cache_dir=self.cache_dir,
        )
        model = reward.model
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        def encode(image: torch.Tensor) -> torch.Tensor:
            pixels = reward._preprocess_image(image.to(self.device), target_size=518)
            outputs = model(pixel_values=pixels, return_dict=True)
            # Use the normalized mean of patch tokens; omit the CLS token.
            patches = outputs.last_hidden_state[:, 1:, :].float()
            return F.normalize(patches.mean(dim=1), p=2, dim=-1)
        self._reward = reward
        self._encoder = encode

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        self._load_encoder()
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("images must have shape [batch, 3, height, width]")
        features = self._encoder(image.to(self.device))
        if not torch.isfinite(features).all():
            raise FloatingPointError("progress encoder returned non-finite features")
        return features

    def set_anchors(self, source: torch.Tensor, native_full: torch.Tensor) -> dict[str, object]:
        """Encode and detach source/native-full anchors exactly once."""
        with torch.no_grad():
            self.source_features = self.encode(source).detach()
            self.target_features = self.encode(native_full).detach()
            src = project_progress_features(
                self.source_features, self.target_features, self.source_features
            )
            tgt = project_progress_features(
                self.source_features, self.target_features, self.target_features
            )
        self.anchor_diagnostics = {
            "anchor_norm": float(src.anchor_norm.cpu()),
            "source_progress": float(src.raw_progress.mean().cpu()),
            "target_progress": float(tgt.raw_progress.mean().cpu()),
            "source_progress_error": abs(float(src.raw_progress.mean().cpu())),
            "target_progress_error": abs(float(tgt.raw_progress.mean().cpu()) - 1.0),
            "backbone": self.backbone,
            "dino_representation": (
                "normalized mean of patch tokens, CLS excluded"
                if self.backbone == "dino" else None
            ),
            "warning": src.warning or tgt.warning,
        }
        return self.anchor_diagnostics

    def __call__(self, candidate: torch.Tensor) -> ProgressValues:
        if self.source_features is None or self.target_features is None:
            raise RuntimeError("call set_anchors(source, native_full) before estimating progress")
        # Deliberately keep autograd enabled on this candidate image path.
        candidate_features = self.encode(candidate)
        return project_progress_features(
            self.source_features, self.target_features, candidate_features
        )
