"""Differentiable adapters for the calibration project's existing rewards."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from rewardflow_calibration.optimization.reward_plugin import GoalRewardBundle
from rewardflow_calibration.rewards.face_identity import FaceIdentityReward
from rewardflow_calibration.rewards.perception import DreamSimReward
from rewardflow_calibration.rewards.semantic import SemanticReward
from rewardflow_calibration.rewards.spatial_layout import SpatialLayoutReward


class SigLIPEditScore:
    """Gradient-preserving image path through the existing frozen SigLIP model."""

    def __init__(self, reward: SemanticReward, target_prompt: str) -> None:
        reward.prepare_text(target_prompt)
        self.reward = reward
        self.model = reward.model["model"]
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.eval()
        self.text_features = reward._encode_text(target_prompt).detach()

    def __call__(self, image: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        del source
        image_input = self.reward._preprocess_image_siglip(image)
        image_features = self.model.get_image_features(pixel_values=image_input)
        image_features = F.normalize(image_features.float(), p=2, dim=-1)
        text_features = self.text_features.to(image_features)
        return (image_features * text_features).sum(dim=-1).mean()


class DINOv2LayoutLoss:
    """Dense source-layout feature loss with a frozen, differentiable encoder path."""

    def __init__(self, reward: SpatialLayoutReward) -> None:
        self.reward = reward
        self.model = reward.model
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.eval()
        self._source_features: torch.Tensor | None = None

    def _features(self, image: torch.Tensor) -> torch.Tensor:
        pixels = self.reward._preprocess_image(image, target_size=518)
        outputs = self.model(pixel_values=pixels, return_dict=True)
        hidden = outputs.last_hidden_state
        # DINOv2's first token is CLS; compare corresponding spatial patch tokens.
        return F.normalize(hidden[:, 1:].float(), p=2, dim=-1)

    def __call__(self, image: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        if self._source_features is None:
            with torch.no_grad():
                self._source_features = self._features(source).detach()
        source_features = self._source_features.to(device=image.device)
        edited_features = self._features(image)
        if edited_features.shape != source_features.shape:
            raise ValueError(
                "DINOv2 source/output patch shapes differ; use identical source and output sizes"
            )
        cosine = (edited_features * source_features).sum(dim=-1)
        return (1.0 - cosine).mean()


class DifferentiableDreamSimDistance:
    """DreamSim distance with gradients preserved to the generated image.

    The project's generic DreamSimReward converts tensors to PIL and wraps
    inference in ``no_grad`` for evaluation. Here the same frozen model is
    called directly, with its PIL Resize+ToTensor preprocessing mirrored by a
    differentiable bicubic tensor resize.
    """

    def __init__(self, reward: DreamSimReward) -> None:
        model_info = reward.model
        self.model = model_info["model"]
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.eval()

    @staticmethod
    def _resize(image: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            image.float(), size=(224, 224), mode="bicubic", align_corners=False,
            antialias=True,
        ).clamp(0.0, 1.0)

    def __call__(self, image: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        return self.model(self._resize(source), self._resize(image)).float().mean()


class CompositePreservationLoss:
    """DreamSim perceptual preservation plus dense DINOv2 structure loss."""

    def __init__(
        self,
        dreamsim_loss: DifferentiableDreamSimDistance,
        dino_loss: DINOv2LayoutLoss,
        structure_weight: float = 1.0,
    ) -> None:
        if structure_weight < 0:
            raise ValueError("structure_weight must be non-negative")
        self.dreamsim_loss = dreamsim_loss
        self.dino_loss = dino_loss
        self.structure_weight = float(structure_weight)

    def __call__(self, image: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        return self.dreamsim_loss(image, source) + self.structure_weight * self.dino_loss(image, source)


def build_default_goal_rewards(
    *,
    device: torch.device | str,
    target_prompt: str,
    cache_dir: str = "/data15/hyp/weight",
    dinov2_path: str = "/data15/hyp/weight/dinov2-large",
    siglip_path: str = "/data15/hyp/weight/reward_models/siglip-so400m-patch14-384",
    structure_weight: float = 1.0,
) -> tuple[GoalRewardBundle, FaceIdentityReward | None]:
    """Build SigLIP edit scoring and DreamSim+DINOv2 preservation rewards.

    FaceIdentityReward is returned only as a non-differentiable report metric;
    its InsightFace/ONNX path cannot provide a gradient to the generated image.
    """
    import os

    semantic = SemanticReward(
        device=device,
        model_name=siglip_path if os.path.exists(siglip_path) else "google/siglip-so400m-patch14-384",
        cache_dir=cache_dir,
    )
    edit_score = SigLIPEditScore(semantic, target_prompt)

    layout = SpatialLayoutReward(
        device=device,
        model_name=dinov2_path if os.path.exists(dinov2_path) else "facebook/dinov2-vitb14",
        cache_dir=cache_dir,
    )
    preservation_loss = DINOv2LayoutLoss(layout)
    dreamsim_reward = DreamSimReward(device=device, cache_dir=cache_dir)
    dreamsim_loss = DifferentiableDreamSimDistance(dreamsim_reward)
    preservation_loss = CompositePreservationLoss(
        dreamsim_loss, preservation_loss, structure_weight=structure_weight
    )

    try:
        face_metric = FaceIdentityReward(device=device, cache_dir=cache_dir)
        _ = face_metric.model
    except Exception:
        face_metric = None

    return GoalRewardBundle(edit_score, preservation_loss), face_metric


@torch.no_grad()
def face_identity_similarity(
    reward: FaceIdentityReward | None,
    source: torch.Tensor,
    generated: torch.Tensor,
) -> float | None:
    """Evaluate InsightFace similarity after optimization (diagnostic only)."""
    if reward is None:
        return None
    source_embedding = reward._extract_face_embedding(source).reshape(1, -1)
    generated_embedding = reward._extract_face_embedding(generated).reshape(1, -1)
    return float(F.cosine_similarity(source_embedding, generated_embedding, dim=-1).mean())
