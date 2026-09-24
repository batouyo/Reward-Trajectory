"""Base reward function interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class RewardFn(Protocol):
    """Protocol for reward functions.

    A reward function takes an image and returns a scalar reward.
    Higher is better.
    """

    def __call__(self, image: torch.Tensor, prompt: str | list[str]) -> torch.Tensor:
        """Compute reward for an image.

        Args:
            image: Image tensor [B, 3, H, W] in [0, 1]
            prompt: Text prompt for the edit

        Returns:
            Scalar reward tensor (higher is better)
        """
        ...


@dataclass
class RewardOutput:
    """Output from a reward computation."""

    total: torch.Tensor
    components: dict[str, torch.Tensor]
    metadata: dict | None = None

    @property
    def scalar(self) -> float:
        """Get the total reward as a Python float."""
        return float(self.total.detach().cpu())

    def __repr__(self) -> str:
        component_str = ", ".join(f"{k}={v:.4f}" for k, v in self.components.items())
        return f"RewardOutput(total={self.scalar:.4f}, {component_str})"


class BaseReward(ABC):
    """Abstract base class for reward functions.

    Provides common functionality for reward models.
    """

    def __init__(self, device: str | torch.device = "cuda:0"):
        self.device = torch.device(device)
        self._model = None

    @abstractmethod
    def __call__(self, image: torch.Tensor, prompt: str | list[str]) -> torch.Tensor:
        """Compute the reward for an image."""
        pass

    @property
    def model(self):
        """Lazily loaded model."""
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _check_availability(self) -> bool:
        """Check if the required dependencies are available."""
        try:
            import insightface  # noqa: F401
            return True
        except ImportError:
            return False

    @abstractmethod
    def _load_model(self):
        """Load the underlying model."""
        pass

    def _preprocess_image(
        self,
        image: torch.Tensor,
        target_size: int,
        mean: tuple[float, ...] | None = None,
        std: tuple[float, ...] | None = None,
    ) -> torch.Tensor:
        """Preprocess image for model input.

        Args:
            image: Image tensor [B, 3, H, W] in [0, 1]
            target_size: Target size for shortest edge
            mean: Normalization mean (default: ImageNet)
            std: Normalization std (default: ImageNet)

        Returns:
            Preprocessed tensor [B, 3, target_size, target_size]
        """
        import torch.nn.functional as F

        if mean is None:
            mean = (0.485, 0.456, 0.406)
        if std is None:
            std = (0.229, 0.224, 0.225)

        # Resize to target size (shortest edge)
        _, _, h, w = image.shape
        scale = target_size / min(h, w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        image = F.interpolate(
            image, size=(new_h, new_w), mode="bicubic", align_corners=False
        )

        # Normalize
        mean_t = (
            torch.tensor(mean, device=image.device, dtype=image.dtype)
            .view(1, 3, 1, 1)
        )
        std_t = (
            torch.tensor(std, device=image.device, dtype=image.dtype)
            .view(1, 3, 1, 1)
        )
        return (image - mean_t) / std_t

    def maybe_onload(self) -> None:
        """Load model to device if not already loaded."""
        _ = self.model

    def maybe_offload(self) -> None:
        """Offload model from GPU to save memory (optional)."""
        pass
