"""Differentiable reward terms for masked backward trajectory control."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class InvalidSemanticAnchorError(ValueError):
    """Raised when native full edit is not semantically stronger than source."""

    code = "invalid_semantic_anchor"

    def __init__(self, diagnostics: dict[str, float]):
        super().__init__(
            "native full edit semantic score must exceed source by semantic_anchor_min_gap"
        )
        self.diagnostics = diagnostics


@dataclass(frozen=True)
class BackwardRewardConfig:
    semantic_floor_fraction: float = 0.5
    semantic_anchor_min_gap: float = 0.02
    source_weight: float = 1.0
    semantic_weight: float = 1.0
    keep_weight: float = 1.0

    def validate(self) -> None:
        if not 0.0 <= self.semantic_floor_fraction <= 1.0:
            raise ValueError("semantic_floor_fraction must be in [0, 1]")
        if self.semantic_anchor_min_gap < 0:
            raise ValueError("semantic_anchor_min_gap must be non-negative")
        if min(self.source_weight, self.semantic_weight, self.keep_weight) < 0:
            raise ValueError("reward weights must be non-negative")


@dataclass
class BackwardRewardValues:
    total: torch.Tensor
    source_loss: torch.Tensor
    semantic_gate_loss: torch.Tensor
    keep_loss: torch.Tensor
    semantic_score: torch.Tensor

    def diagnostics(self) -> dict[str, float]:
        return {
            "guide_total": float(self.total.detach().cpu()),
            "source_loss": float(self.source_loss.detach().cpu()),
            "semantic_gate_loss": float(self.semantic_gate_loss.detach().cpu()),
            "keep_loss": float(self.keep_loss.detach().cpu()),
            "semantic_score": float(self.semantic_score.detach().cpu()),
        }


def semantic_floor_from_anchors(
    source_score: float | torch.Tensor,
    full_score: float | torch.Tensor,
    fraction: float,
    *,
    min_gap: float = 0.0,
) -> float:
    """Return an instance-relative floor or reject a degenerate semantic anchor."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")
    source = float(torch.as_tensor(source_score).detach().cpu())
    full = float(torch.as_tensor(full_score).detach().cpu())
    gap = full - source
    if not torch.isfinite(torch.tensor([source, full])).all() or gap <= min_gap:
        raise InvalidSemanticAnchorError({
            "semantic_source": source,
            "semantic_full": full,
            "semantic_gap": gap,
            "semantic_anchor_min_gap": float(min_gap),
        })
    return source + fraction * gap


def semantic_hinge_loss(score: torch.Tensor, floor: float | torch.Tensor) -> torch.Tensor:
    floor_tensor = torch.as_tensor(floor, device=score.device, dtype=score.dtype)
    return F.relu(floor_tensor - score).square().mean()


def source_attraction_loss(candidate_features: torch.Tensor, source_features: torch.Tensor) -> torch.Tensor:
    candidate = F.normalize(candidate_features.float(), p=2, dim=-1)
    source = F.normalize(source_features.detach().float(), p=2, dim=-1)
    return (1.0 - (candidate * source).sum(dim=-1)).mean()


def keep_region_l1(candidate: torch.Tensor, source: torch.Tensor, keep_mask: torch.Tensor) -> torch.Tensor:
    if candidate.shape != source.shape or candidate.ndim != 4 or candidate.shape[1] != 3:
        raise ValueError("candidate and source must have matching [B, 3, H, W] shapes")
    if keep_mask.ndim != 4 or keep_mask.shape[0] not in (1, candidate.shape[0]):
        raise ValueError("keep_mask must have shape [1|B, 1, H, W]")
    if keep_mask.shape[1] != 1 or keep_mask.shape[-2:] != candidate.shape[-2:]:
        raise ValueError("keep_mask spatial dimensions must match image and have one channel")
    mask = keep_mask.detach().to(device=candidate.device, dtype=candidate.dtype).clamp(0, 1)
    mass = mask.sum() * candidate.shape[1]
    if float(mass.detach().cpu()) <= 1e-12:
        return candidate.sum() * 0.0
    return ((candidate - source.detach()).abs() * mask).sum() / mass


def _feature_tensor(output: object) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    for name in ("image_embeds", "text_embeds", "pooler_output", "last_hidden_state"):
        value = getattr(output, name, None)
        if isinstance(value, torch.Tensor):
            if name == "last_hidden_state":
                return value[:, 0]
            return value
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"encoder output {type(output).__name__} does not contain a feature tensor")


class BackwardReward:
    """Frozen SigLIP/DINO encoders with gradients retained to candidate pixels."""

    DEFAULT_SIGLIP_PATH = "/data15/hyp/weight/reward_models/siglip-so400m-patch14-384"
    DEFAULT_DINO_PATH = "/data15/hyp/weight/dinov2-large"

    def __init__(
        self,
        prompt: str,
        *,
        device: torch.device | str = "cuda:0",
        config: BackwardRewardConfig | None = None,
        siglip_model_path: str | None = None,
        dino_model_path: str | None = None,
        cache_dir: str = "/data15/hyp/weight",
        semantic_image_encoder: Callable[[torch.Tensor], torch.Tensor] | nn.Module | None = None,
        semantic_text_encoder: Callable[[str], torch.Tensor] | None = None,
        dino_image_encoder: Callable[[torch.Tensor], torch.Tensor] | nn.Module | None = None,
        dreamsim_distance: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.prompt = prompt
        self.device = torch.device(device)
        self.config = config or BackwardRewardConfig()
        self.config.validate()
        self.siglip_model_path = siglip_model_path or self.DEFAULT_SIGLIP_PATH
        self.dino_model_path = dino_model_path or self.DEFAULT_DINO_PATH
        self.cache_dir = cache_dir
        self.semantic_image_encoder = semantic_image_encoder
        self.semantic_text_encoder = semantic_text_encoder
        self.dino_image_encoder = dino_image_encoder
        self.dreamsim_distance = dreamsim_distance
        self._semantic_reward = None
        self._dino_estimator = None
        self._dreamsim = None
        self.target_text_features: torch.Tensor | None = None
        self.source_dino_features: torch.Tensor | None = None
        self.semantic_source: float | None = None
        self.semantic_full: float | None = None
        self.semantic_gap: float | None = None
        self.semantic_floor: float | None = None
        self.anchor_diagnostics: dict[str, object] = {}

    @staticmethod
    def _freeze(encoder: object) -> None:
        if isinstance(encoder, nn.Module):
            encoder.eval()
            for parameter in encoder.parameters():
                parameter.requires_grad_(False)

    def _load_encoders(self) -> None:
        if self.semantic_image_encoder is None or self.semantic_text_encoder is None:
            from rewardflow_calibration.rewards.semantic import SemanticReward
            reward = SemanticReward(
                device=self.device,
                model_name=self.siglip_model_path if os.path.exists(self.siglip_model_path)
                else "google/siglip-so400m-patch14-384",
                cache_dir=self.cache_dir,
            )
            model = reward.model["model"]
            processor = reward.model["processor"]
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)

            def image_encoder(image: torch.Tensor) -> torch.Tensor:
                pixels = reward._preprocess_image_siglip(image.to(self.device))
                return _feature_tensor(model.get_image_features(pixel_values=pixels))

            def text_encoder(text: str) -> torch.Tensor:
                inputs = processor(
                    text=text, padding="max_length", truncation=True,
                    return_tensors="pt", max_length=64,
                )
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                return _feature_tensor(model.get_text_features(**inputs))

            self.semantic_image_encoder = image_encoder
            self.semantic_text_encoder = text_encoder
            self._semantic_reward = reward
        if self.dino_image_encoder is None:
            from .progress_reward import ProgressEstimator
            estimator = ProgressEstimator(
                "dino", device=self.device, cache_dir=self.cache_dir,
                model_path=self.dino_model_path,
            )
            self._dino_estimator = estimator
            self.dino_image_encoder = estimator.encode
        if self.dreamsim_distance is None:
            from rewardflow_calibration.metrics.dreamsim import DreamSimDistance
            self._dreamsim = DreamSimDistance(self.device)
            self.dreamsim_distance = self._dreamsim.distance
        self._freeze(self.semantic_image_encoder)
        self._freeze(self.dino_image_encoder)

    def _semantic_features(self, image: torch.Tensor) -> torch.Tensor:
        assert self.semantic_image_encoder is not None
        return F.normalize(_feature_tensor(self.semantic_image_encoder(image)).float(), p=2, dim=-1)

    def semantic_score(self, image: torch.Tensor) -> torch.Tensor:
        if self.target_text_features is None:
            raise RuntimeError("set_anchors must be called before semantic scoring")
        candidate = self._semantic_features(image)
        target = F.normalize(self.target_text_features.detach().float(), p=2, dim=-1)
        return (candidate * target).sum(dim=-1).mean()

    def set_anchors(
        self,
        source: torch.Tensor,
        native_full: torch.Tensor,
        native_proxy: torch.Tensor,
    ) -> dict[str, object]:
        self._load_encoders()
        assert self.semantic_text_encoder is not None
        assert self.dino_image_encoder is not None
        assert self.dreamsim_distance is not None
        with torch.no_grad():
            text = _feature_tensor(self.semantic_text_encoder(self.prompt)).to(self.device).float()
            self.target_text_features = F.normalize(text, p=2, dim=-1).detach()
            source_sem = float(self.semantic_score(source).detach().cpu())
            full_sem = float(self.semantic_score(native_full).detach().cpu())
            proxy_sem = float(self.semantic_score(native_proxy).detach().cpu())
            self.semantic_source, self.semantic_full = source_sem, full_sem
            self.semantic_gap = full_sem - source_sem
            try:
                self.semantic_floor = semantic_floor_from_anchors(
                    source_sem, full_sem, self.config.semantic_floor_fraction,
                    min_gap=self.config.semantic_anchor_min_gap,
                )
            except InvalidSemanticAnchorError as exc:
                self.anchor_diagnostics = {
                    **exc.diagnostics,
                    "semantic_floor_fraction": self.config.semantic_floor_fraction,
                    "status": InvalidSemanticAnchorError.code,
                }
                raise
            self.source_dino_features = F.normalize(
                _feature_tensor(self.dino_image_encoder(source)).float(), p=2, dim=-1
            ).detach()
            dreamsim_source_full = float(torch.as_tensor(
                self.dreamsim_distance(source, native_full)
            ).float().mean().cpu())
            dreamsim_proxy_full = float(torch.as_tensor(
                self.dreamsim_distance(native_proxy, native_full)
            ).float().mean().cpu())
            dreamsim_source_proxy = float(torch.as_tensor(
                self.dreamsim_distance(source, native_proxy)
            ).float().mean().cpu())
        self.anchor_diagnostics = {
            "semantic_source": source_sem,
            "semantic_full": full_sem,
            "semantic_gap": self.semantic_gap,
            "semantic_floor": self.semantic_floor,
            "semantic_floor_fraction": self.config.semantic_floor_fraction,
            "semantic_anchor_min_gap": self.config.semantic_anchor_min_gap,
            "native_proxy_semantic_score": proxy_sem,
            "dreamsim_source_native_full": dreamsim_source_full,
            "dreamsim_native_proxy_native_full": dreamsim_proxy_full,
            "dreamsim_source_native_proxy": dreamsim_source_proxy,
            "status": "valid",
        }
        return self.anchor_diagnostics

    def evaluate(
        self,
        candidate: torch.Tensor,
        source: torch.Tensor,
        keep_mask: torch.Tensor,
    ) -> BackwardRewardValues:
        if self.semantic_floor is None or self.source_dino_features is None:
            raise RuntimeError("valid anchors are required before evaluating candidates")
        assert self.dino_image_encoder is not None
        semantic = self.semantic_score(candidate)
        candidate_dino = F.normalize(
            _feature_tensor(self.dino_image_encoder(candidate)).float(), p=2, dim=-1
        )
        source_loss = source_attraction_loss(candidate_dino, self.source_dino_features)
        sem_loss = semantic_hinge_loss(semantic, self.semantic_floor)
        keep_loss = keep_region_l1(candidate, source, keep_mask)
        total = (
            self.config.source_weight * source_loss
            + self.config.semantic_weight * sem_loss
            + self.config.keep_weight * keep_loss
        )
        return BackwardRewardValues(total, source_loss, sem_loss, keep_loss, semantic)

    def dreamsim(self, first: torch.Tensor, second: torch.Tensor) -> float:
        self._load_encoders()
        assert self.dreamsim_distance is not None
        with torch.no_grad():
            value = self.dreamsim_distance(first.detach(), second.detach())
        return float(torch.as_tensor(value).float().mean().cpu())
