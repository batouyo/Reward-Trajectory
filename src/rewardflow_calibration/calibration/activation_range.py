"""DreamSim valid-range detection against the input source image."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image


DEFAULT_ACTIVATION_CANDIDATES = (0.25, 0.5, 0.75, 1.0)


def normalize_alpha(beta: torch.Tensor | float, alpha_start: float) -> torch.Tensor | float:
    if not 0.0 <= float(alpha_start) <= 1.0:
        raise ValueError("alpha_start must be in [0, 1]")
    if torch.is_tensor(beta):
        if not torch.isfinite(beta).all() or torch.any((beta < 0) | (beta > 1)):
            raise ValueError("beta values must be in [0, 1]")
        return alpha_start + beta * (1.0 - alpha_start)
    beta = float(beta)
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be in [0, 1]")
    return alpha_start + beta * (1.0 - alpha_start)


@dataclass(frozen=True)
class ActivationRangeConfig:
    activation_distance_threshold: float = 0.001
    alpha_resolution: float = 0.05
    candidate_alphas: tuple[float, ...] = DEFAULT_ACTIVATION_CANDIDATES

    def validate(self) -> None:
        if not np.isfinite(self.activation_distance_threshold) or self.activation_distance_threshold < 0:
            raise ValueError("activation_distance_threshold must be finite and non-negative")
        if not 0 < self.alpha_resolution <= 1:
            raise ValueError("alpha_resolution must be in (0, 1]")
        if len(self.candidate_alphas) < 1 or self.candidate_alphas[-1] != 1.0:
            raise ValueError("candidate_alphas must include alpha=1")
        if any(not 0 <= value <= 1 for value in self.candidate_alphas):
            raise ValueError("candidate_alphas must be in [0, 1]")
        if any(right <= left for left, right in zip(self.candidate_alphas[:-1], self.candidate_alphas[1:])):
            raise ValueError("candidate_alphas must be strictly increasing")


class ActivationRangeDetector:
    """Find the first alpha whose source-image distance clears the threshold."""

    def __init__(self, distance_metric: Any, config: ActivationRangeConfig | None = None):
        self.distance_metric = distance_metric
        self.config = config or ActivationRangeConfig()
        self.config.validate()

    @staticmethod
    def _save_image(image: torch.Tensor, path: Path) -> None:
        if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
            raise ValueError("image must have shape [1, 3, H, W]")
        value = image.detach().float().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
        Image.fromarray(np.rint(value * 255).astype(np.uint8), mode="RGB").save(path)

    def detect(
        self,
        *,
        source_image: torch.Tensor,
        rollout: Callable[[float], torch.Tensor],
        output_dir: str | Path,
        inference_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if source_image.ndim != 4 or source_image.shape[:2] != (1, 3):
            raise ValueError("source_image must have shape [1, 3, H, W]")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.config.validate()
        reference = source_image.detach().float()
        reference_path = output_dir / "source.png"
        self._save_image(reference, reference_path)
        results: dict[float, dict[str, Any]] = {}

        def probe(alpha: float, stage: str) -> dict[str, Any]:
            key = round(float(alpha), 10)
            if key in results:
                return results[key]
            with torch.no_grad():
                image = rollout(key)
                if image.shape != source_image.shape:
                    raise ValueError("rollout image shape must match source_image")
                distance = self.distance_metric.distance(reference, image.float())
                value = float(distance.reshape(-1).mean().detach().cpu())
            if not np.isfinite(value):
                raise ValueError(f"distance is not finite at alpha={key}")
            path = output_dir / f"alpha_{key:.3f}.png"
            self._save_image(image, path)
            results[key] = {"alpha": key, "distance": value, "image_path": str(path), "stage": stage}
            return results[key]

        coarse = [probe(alpha, "coarse") for alpha in self.config.candidate_alphas]
        valid_index = next(
            (index for index, row in enumerate(coarse) if row["distance"] > self.config.activation_distance_threshold),
            None,
        )
        if valid_index is None:
            alpha_start = 1.0
            activation_found = False
        else:
            activation_found = True
            low = 0.0 if valid_index == 0 else float(coarse[valid_index - 1]["alpha"])
            high = float(coarse[valid_index]["alpha"])
            while high - low >= self.config.alpha_resolution:
                midpoint = (low + high) / 2.0
                if probe(midpoint, "refine")["distance"] > self.config.activation_distance_threshold:
                    high = midpoint
                else:
                    low = midpoint
            alpha_start = high

        probes = [results[key] for key in sorted(results)]
        return {
            "alpha_start": float(alpha_start),
            "activation_found": activation_found,
            "distance_metric": getattr(self.distance_metric, "metric_name", type(self.distance_metric).__name__),
            "reference_image_path": str(reference_path),
            "activation_distance_threshold": self.config.activation_distance_threshold,
            "alpha_resolution": self.config.alpha_resolution,
            "candidate_alphas": list(self.config.candidate_alphas),
            "probe_results": probes,
            "distance_curve": [{"alpha": row["alpha"], "distance": row["distance"]} for row in probes],
            "inference_config": inference_config or {},
            "search_config": asdict(self.config),
        }
