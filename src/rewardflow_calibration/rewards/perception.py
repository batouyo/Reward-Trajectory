"""Perception reward using DreamSim for perceptual distance.

This reward ensures that the edited image maintains perceptual quality
by computing DreamSim distance between source and edited.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from .base import BaseReward


class DreamSimReward(BaseReward):
    """Perceptual quality reward using DreamSim.

    DreamSim computes a perceptual distance that correlates better with
    human perception than pixel-level metrics.

    This reward returns NEGATIVE distance (higher = closer to source = better).
    """

    def __init__(
        self,
        device: str | torch.device = "cuda:0",
        cache_dir: str | None = None,
    ):
        """
        Args:
            device: Device to run the model on
            cache_dir: Directory to cache model weights
        """
        super().__init__(device)
        self.cache_dir = cache_dir
        self._source_image: torch.Tensor | None = None

    def _load_model(self):
        """Load DreamSim model."""
        try:
            from dreamsim import dreamsim
        except ImportError:
            raise ImportError(
                "DreamSimReward requires dreamsim. "
                "Install with: pip install dreamsim"
            )

        model, preprocess = dreamsim(
            pretrained=True,
            device=str(self.device),
        )
        model.eval()
        return {"model": model, "preprocess": preprocess}

    @staticmethod
    def _to_pil(image: torch.Tensor) -> Image.Image:
        """Convert tensor to PIL Image."""
        if image.dim() == 4:
            image = image[0]  # Take first in batch
        value = image.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(np.rint(value * 255).astype(np.uint8), mode="RGB")

    def register_source_image(self, image: torch.Tensor) -> None:
        """Register source image for perceptual comparison.

        Args:
            image: Source image [1, 3, H, W] in [0, 1]
        """
        self._source_image = image.detach()

    def __call__(self, image: torch.Tensor, prompt: str | list[str]) -> torch.Tensor:
        """Compute perceptual similarity reward.

        Args:
            image: Edited image [B, 3, H, W] in [0, 1]
            prompt: Text prompt (not used)

        Returns:
            Scalar reward (higher = more similar to source)
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)

        if self._source_image is None:
            # Register as source
            self.register_source_image(image)
            return torch.tensor(1.0, device=image.device, dtype=image.dtype)

        # Get model and preprocess
        model_info = self.model
        preprocess = model_info["preprocess"]
        model = model_info["model"]

        # Preprocess images
        source_pil = self._to_pil(self._source_image)
        edited_pil = self._to_pil(image)

        source_prep = preprocess(source_pil).to(self.device)
        edited_prep = preprocess(edited_pil).to(self.device)

        # Compute distance
        with torch.no_grad():
            distance = model(source_prep, edited_prep)

        # Return NEGATIVE distance as reward (higher = closer = better)
        return -distance.float()


class LPIPSReward(BaseReward):
    """LPIPS-based perceptual reward as an alternative to DreamSim.

    Uses LPIPS (Learned Perceptual Image Patch Similarity) to compute
    perceptual distance.
    """

    def __init__(
        self,
        device: str | torch.device = "cuda:0",
        net: str = "vgg",
    ):
        """
        Args:
            device: Device to run the model on
            net: Network backbone ('vgg', 'alex', 'squeeze')
        """
        super().__init__(device)
        self.net = net
        self._source_image: torch.Tensor | None = None

    def _load_model(self):
        """Load LPIPS model."""
        try:
            import lpips
        except ImportError:
            raise ImportError(
                "LPIPSReward requires lpips. "
                "Install with: pip install lpips"
            )

        loss_fn = lpips.LPIPS(net=self.net)
        loss_fn = loss_fn.to(self.device)
        loss_fn.eval()
        return loss_fn

    def register_source_image(self, image: torch.Tensor) -> None:
        """Register source image for perceptual comparison."""
        self._source_image = image.detach()

    def __call__(self, image: torch.Tensor, prompt: str | list[str]) -> torch.Tensor:
        """Compute LPIPS-based perceptual reward.

        Args:
            image: Edited image [B, 3, H, W] in [-1, 1]
            prompt: Text prompt (not used)

        Returns:
            Scalar reward (higher = more similar to source)
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)

        if self._source_image is None:
            self.register_source_image(image)
            return torch.tensor(1.0, device=image.device, dtype=image.dtype)

        # LPIPS expects [-1, 1] range
        source = self._source_image
        if source.min() >= 0:
            source = source * 2 - 1
        if image.min() >= 0:
            image = image * 2 - 1

        # Compute LPIPS distance
        loss_fn = self.model
        with torch.no_grad():
            distance = loss_fn(source, image).squeeze()

        # Return negative distance as reward
        return -distance.float()
