"""Perceptual activation-range detection for VeloEdit-style strength control."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image


DEFAULT_ACTIVATION_CANDIDATES = (0.0, 0.25, 0.5, 0.75, 1.0)


def normalize_alpha(beta: torch.Tensor | float, alpha_start: float) -> torch.Tensor | float:
    """Map optimizer coordinates in [0, 1] onto the active alpha interval."""
    if not 0.0 <= float(alpha_start) <= 1.0:
        raise ValueError("alpha_start must be in [0, 1].")
    if torch.is_tensor(beta):
        if not torch.isfinite(beta).all() or torch.any(beta < 0) or torch.any(beta > 1):
            raise ValueError("beta values must be finite and in [0, 1].")
        return alpha_start + beta * (1.0 - alpha_start)
    beta_value = float(beta)
    if not 0.0 <= beta_value <= 1.0:
        raise ValueError("beta must be in [0, 1].")
    return alpha_start + beta_value * (1.0 - alpha_start)


@dataclass(frozen=True)
class ActivationRangeConfig:
    """Threshold and search settings for perceptual activation detection."""

    activation_distance_threshold: float = 0.065
    alpha_resolution: float = 0.05
    candidate_alphas: tuple[float, ...] = DEFAULT_ACTIVATION_CANDIDATES

    def validate(self) -> None:
        if not np.isfinite(self.activation_distance_threshold) or self.activation_distance_threshold < 0:
            raise ValueError("activation_distance_threshold must be finite and non-negative.")
        if not np.isfinite(self.alpha_resolution) or not 0 < self.alpha_resolution <= 1:
            raise ValueError("alpha_resolution must be in (0, 1].")
        candidates = self.candidate_alphas
        if len(candidates) < 2 or candidates[0] != 0.0 or candidates[-1] != 1.0:
            raise ValueError("candidate_alphas must include 0 and 1 endpoints.")
        if any(not np.isfinite(value) or not 0 <= value <= 1 for value in candidates):
            raise ValueError("candidate_alphas must be finite values in [0, 1].")
        if any(right <= left for left, right in zip(candidates[:-1], candidates[1:])):
            raise ValueError("candidate_alphas must be strictly increasing.")


class ActivationRangeDetector:
    """Find the first alpha whose image clears a perceptual distance threshold.

    rollout must call the project's existing inference path and return an image
    tensor in [0, 1], shaped [1, 3, H, W]. Metric inputs use LPIPS' [-1, 1]
    range. Latent, prompt embedding, and inference config are retained as
    explicit experiment context; model execution stays in the rollout callback.
    """

    def __init__(self, distance_metric: Any, config: ActivationRangeConfig | None = None):
        self.distance_metric = distance_metric
        self.config = config or ActivationRangeConfig()
        self.config.validate()

    @staticmethod
    def _save_image(image: torch.Tensor, path: Path) -> None:
        if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
            raise ValueError("rollout must return an image tensor shaped [1, 3, H, W].")
        value = image.detach().float().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
        Image.fromarray(np.rint(value * 255).astype(np.uint8), mode="RGB").save(path)

    def detect(
        self,
        *,
        source_image: torch.Tensor,
        rollout: Callable[[float], torch.Tensor],
        output_dir: str | Path,
        source_latent: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        inference_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run coarse probes and bisect the first invalid/valid interval."""
        if source_image.ndim != 4 or source_image.shape[:2] != (1, 3):
            raise ValueError("source_image must have shape [1, 3, H, W].")
        self.config.validate()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        source_metric = source_image.detach().float() * 2.0 - 1.0
        results: dict[float, dict[str, Any]] = {}

        def probe(alpha: float, stage: str) -> dict[str, Any]:
            key = round(float(alpha), 10)
            if key in results:
                return results[key]
            with torch.no_grad():
                image = rollout(key)
                if image.shape != source_image.shape:
                    raise ValueError("rollout image shape must match source_image.")
                distance = self.distance_metric.distance(image.float() * 2.0 - 1.0, source_metric)
                distance_value = float(distance.reshape(-1).mean().detach().cpu())
            if not np.isfinite(distance_value):
                raise ValueError(f"Perceptual distance is not finite at alpha={key}.")
            image_path = output_dir / f"alpha_{key:.3f}.png"
            self._save_image(image, image_path)
            row = {"alpha": key, "distance": distance_value, "image_path": str(image_path), "stage": stage}
            results[key] = row
            return row

        coarse = [probe(alpha, "coarse") for alpha in self.config.candidate_alphas]
        valid_index = next(
            (index for index, row in enumerate(coarse)
             if row["distance"] > self.config.activation_distance_threshold),
            None,
        )
        activation_found = valid_index is not None
        if not activation_found:
            alpha_start = 1.0
        elif valid_index == 0:
            alpha_start = float(coarse[0]["alpha"])
        else:
            low = float(coarse[valid_index - 1]["alpha"])
            high = float(coarse[valid_index]["alpha"])
            # Local bisection assumes distance rises across this first bracket.
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
            "activation_distance_threshold": self.config.activation_distance_threshold,
            "alpha_resolution": self.config.alpha_resolution,
            "candidate_alphas": list(self.config.candidate_alphas),
            "probe_results": probes,
            "distance_curve": [{"alpha": row["alpha"], "distance": row["distance"]} for row in probes],
            "source_latent_shape": None if source_latent is None else list(source_latent.shape),
            "prompt_embedding_shape": None if prompt_embeds is None else list(prompt_embeds.shape),
            "inference_config": inference_config or {},
            "search_config": asdict(self.config),
            "monotonicity_assumption": "local threshold crossing between first valid coarse point and predecessor",
        }
