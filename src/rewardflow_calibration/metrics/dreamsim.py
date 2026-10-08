"""DreamSim metric adapter used by activation calibration."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image


class DreamSimDistance:
    metric_name = "DreamSim"

    def __init__(self, device: torch.device | str, *, cache_dir: str | None = None):
        try:
            from dreamsim import dreamsim
        except ImportError as exc:
            raise ImportError("DreamSim is required for activation calibration") from exc
        self.device = torch.device(device)
        kwargs = {"pretrained": True, "device": str(self.device)}
        if cache_dir is not None:
            kwargs["cache_dir"] = cache_dir
        self.model, self.preprocess = dreamsim(**kwargs)
        self.model.eval()

    @staticmethod
    def _to_pil(image: torch.Tensor) -> Image.Image:
        value = image.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(np.rint(value * 255).astype(np.uint8), mode="RGB")

    def distance(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        if first.ndim != 4 or second.shape != first.shape or first.shape[1] != 3:
            raise ValueError("DreamSim inputs must have matching [B, 3, H, W] shapes")
        first_batch = torch.cat([self.preprocess(self._to_pil(item)) for item in first], dim=0).to(self.device)
        second_batch = torch.cat([self.preprocess(self._to_pil(item)) for item in second], dim=0).to(self.device)
        with torch.no_grad():
            return self.model(first_batch, second_batch).float()
