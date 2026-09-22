"""Image conversion and VeloEdit working-resolution helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def working_size(image: Image.Image, max_area: int = 1024 * 1024, multiple: int = 16) -> tuple[int, int]:
    scale = (max_area / (image.height * image.width)) ** 0.5
    height = max(multiple, round(image.height * scale) // multiple * multiple)
    width = max(multiple, round(image.width * scale) // multiple * multiple)
    return height, width


def image_tensor(image: Image.Image, device: torch.device | str = "cpu") -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32) / 255.0


def save_tensor_image(image: torch.Tensor, path: str | Path) -> None:
    value = image.detach().float().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    Image.fromarray(np.rint(value * 255).astype(np.uint8), mode="RGB").save(path)
