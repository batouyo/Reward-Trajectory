"""Official DreamSim adapter with a tensor-only candidate-image path.

The public package's recommended preprocessing accepts PIL images. Candidate
images in RewardSlider V1 instead stay as torch tensors and use the equivalent
224px RGB bicubic resize, so ``distance`` retains the image gradient.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image


class DreamSimUnavailableError(RuntimeError):
    """Raised when the official DreamSim package or checkpoints are unavailable."""


class DreamSimAdapter:
    """Frozen official DreamSim metric with differentiable tensor preprocessing."""

    image_size = 224

    def __init__(self, *, model_path: str | None = None, device: torch.device | str | None = None):
        try:
            from dreamsim import dreamsim
        except ImportError as error:  # pragma: no cover - optional dependency
            raise DreamSimUnavailableError(
                "RewardSlider V1 requires the official `dreamsim` package; no substitute metric is used."
            ) from error
        if model_path is not None and not Path(model_path).is_dir():
            raise DreamSimUnavailableError(f"DreamSim checkpoint directory does not exist: {model_path}")
        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        try:
            self.model, self.official_preprocess = dreamsim(
                pretrained=True,
                device=self.device,
                cache_dir=model_path or "./models",
            )
        except Exception as error:  # pragma: no cover - external weights
            raise DreamSimUnavailableError(f"Official DreamSim could not load from {model_path!r}: {error}") from error
        self.model.eval().requires_grad_(False)
        for parameter in self.model.parameters():
            parameter.grad = None
        self._cached_endpoint_embeddings: dict[str, torch.Tensor] = {}

    @staticmethod
    def _validate_image(image: torch.Tensor, name: str) -> None:
        if not torch.is_tensor(image) or image.ndim != 4 or image.shape[0] < 1 or image.shape[1] != 3:
            raise ValueError(f"`{name}` must have shape [B, 3, H, W].")
        if not image.is_floating_point():
            raise TypeError(f"`{name}` must be floating point RGB values in [0,1].")
        detached = image.detach()
        if not torch.isfinite(detached).all() or bool((detached < 0).any()) or bool((detached > 1).any()):
            raise ValueError(f"`{name}` must be finite and lie in [0,1].")

    def preprocess_tensor(self, image: torch.Tensor) -> torch.Tensor:
        """Torch-native official-shape preprocessing without image detachment."""

        self._validate_image(image, "image")
        return F.interpolate(
            image.to(device=self.device, dtype=torch.float32),
            size=(self.image_size, self.image_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )

    def embed(self, image: torch.Tensor) -> torch.Tensor:
        """Return model embeddings while preserving candidate-image gradients."""

        return self.model.embed(self.preprocess_tensor(image))

    def distance(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        """Return one official DreamSim distance per aligned image pair."""

        self._validate_image(first, "first")
        self._validate_image(second, "second")
        if first.shape != second.shape:
            raise ValueError("DreamSim pair batches must have identical shapes.")
        distance = self.model(self.preprocess_tensor(first), self.preprocess_tensor(second))
        if distance.shape != (first.shape[0],):
            raise RuntimeError("Official DreamSim did not return one distance per image pair.")
        return distance.to(first.device)

    def cache_endpoint_embeddings(self, source: torch.Tensor, native_full: torch.Tensor) -> None:
        """Cache fixed endpoint embeddings only; candidate images are never cached."""

        with torch.no_grad():
            self._cached_endpoint_embeddings = {
                "source": self.embed(source).detach(),
                "native_full": self.embed(native_full).detach(),
            }

    def endpoint_distance(self) -> torch.Tensor:
        if not self._cached_endpoint_embeddings:
            raise RuntimeError("Call `cache_endpoint_embeddings` before requesting endpoint distance.")
        source = self._cached_endpoint_embeddings["source"]
        full = self._cached_endpoint_embeddings["native_full"]
        return 1 - F.cosine_similarity(source, full, dim=-1)

    def fidelity_audit(self, first: torch.Tensor, second: torch.Tensor) -> dict[str, float]:
        """Compare fixed-image PIL reference preprocessing to the tensor path.

        PIL conversion is restricted to this no-grad audit of fixed images.
        ``distance`` itself never uses it.
        """

        self._validate_image(first, "first")
        self._validate_image(second, "second")
        if first.shape[0] != 1 or second.shape[0] != 1:
            raise ValueError("DreamSim fidelity audit accepts one fixed image per side.")
        with torch.no_grad():
            def to_pil(image: torch.Tensor) -> Image.Image:
                array = image.detach().cpu()[0].mul(255).round().byte().permute(1, 2, 0).numpy()
                return Image.fromarray(array, mode="RGB")

            official = self.model(
                self.official_preprocess(to_pil(first)).to(self.device),
                self.official_preprocess(to_pil(second)).to(self.device),
            ).reshape(())
            differentiable = self.distance(first, second).reshape(())
        return {
            "official_distance": float(official.cpu()),
            "differentiable_distance": float(differentiable.cpu()),
            "absolute_difference": float((official - differentiable).abs().cpu()),
        }

    @classmethod
    def availability_reason(cls, *, model_path: str | None = None) -> str | None:
        try:
            instance = cls(model_path=model_path, device="cpu")
            del instance
        except DreamSimUnavailableError as error:
            return str(error)
        return None
