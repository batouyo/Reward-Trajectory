"""Frozen differentiable image encoders for endpoint-geometry research."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import torch
import torch.nn.functional as F
from PIL import Image

from .endpoint_feature_distance import build_focus_conditioned_feature_prompt


class ImageFeatureScorer(Protocol):
    def encode_image(self, image: torch.Tensor) -> torch.Tensor: ...


class ImageTextFeatureScorer(ImageFeatureScorer, Protocol):
    def encode_text(self, text: str) -> torch.Tensor: ...


def _feature_tensor(output) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    for name in ("image_embeds", "pooler_output"):
        value = getattr(output, name, None)
        if torch.is_tensor(value):
            return value
    raise TypeError("The model image-feature API did not return a tensor.")


def _size_pair(value, *, shortest_edge: bool = False) -> tuple[int, int] | int:
    if isinstance(value, int):
        return value if shortest_edge else (value, value)
    if not isinstance(value, dict):
        raise NotImplementedError(f"Unsupported image processor size: {value!r}.")
    if shortest_edge and "shortest_edge" in value:
        return int(value["shortest_edge"])
    if "height" in value and "width" in value:
        return int(value["height"]), int(value["width"])
    raise NotImplementedError(f"Unsupported image processor size mapping: {value!r}.")


def _pil_cubic_kernel(distance: torch.Tensor) -> torch.Tensor:
    """Keys cubic kernel with a=-0.5, matching Pillow's BICUBIC filter."""

    distance = distance.abs()
    near = ((1.5 * distance - 2.5) * distance * distance + 1) * (distance < 1)
    far = (((-0.5 * distance + 2.5) * distance - 4) * distance + 2) * ((distance >= 1) & (distance < 2))
    return near + far


def _pil_bicubic_weights(input_size: int, output_size: int, reference: torch.Tensor) -> torch.Tensor:
    scale = input_size / output_size
    filter_scale = max(scale, 1.0)
    output_positions = (torch.arange(output_size, device=reference.device, dtype=torch.float32) + 0.5) * scale
    input_positions = torch.arange(input_size, device=reference.device, dtype=torch.float32) + 0.5
    distances = (input_positions[None, :] - output_positions[:, None]) / filter_scale
    weights = _pil_cubic_kernel(distances)
    return (weights / weights.sum(dim=1, keepdim=True)).to(reference.dtype)


def _pil_bicubic_resize(image: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Torch-native separable approximation of Pillow BICUBIC resizing."""

    output_height, output_width = size
    height_weights = _pil_bicubic_weights(image.shape[-2], output_height, image)
    width_weights = _pil_bicubic_weights(image.shape[-1], output_width, image)
    resized_width = torch.matmul(image, width_weights.transpose(0, 1))
    width_quantized = resized_width.clamp(0, 1).mul(255).round().div(255)
    resized_width = resized_width + (width_quantized - resized_width).detach()
    return torch.einsum("oh,bchw->bcow", height_weights, resized_width)


class _ProjectedImageFeatureScorer:
    model_class_name = ""

    def __init__(
        self,
        model_id: str,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        local_files_only: bool | None = None,
        revision: str | None = None,
        model=None,
        image_processor=None,
        tokenizer=None,
    ):
        if dtype != torch.float32:
            raise ValueError("The v6 formal geometry bake-off requires FP32 image encoders.")
        if (model is None) != (image_processor is None):
            raise ValueError("Inject both `model` and `image_processor`, or neither.")
        if model is None:
            try:
                import transformers
                from transformers import AutoImageProcessor, AutoTokenizer
            except Exception as error:  # pragma: no cover - optional dependency
                raise ImportError("Image feature scorers require transformers.") from error
            model_class = getattr(transformers, self.model_class_name)
            kwargs = {"torch_dtype": dtype}
            if local_files_only is not None:
                kwargs["local_files_only"] = local_files_only
            if revision is not None:
                kwargs["revision"] = revision
            model = model_class.from_pretrained(model_id, **kwargs)
            processor_kwargs = {key: value for key, value in kwargs.items() if key != "torch_dtype"}
            image_processor = AutoImageProcessor.from_pretrained(model_id, use_fast=False, **processor_kwargs)
            tokenizer = AutoTokenizer.from_pretrained(model_id, **processor_kwargs)
        self.model_id = str(model_id)
        self.revision = revision
        self.model = model
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.last_text_metadata = None
        self.model.to(device=device, dtype=dtype)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        self._validate_processor()

    def _validate_processor(self) -> None:
        processor = self.image_processor
        required = ("do_resize", "size", "do_rescale", "rescale_factor", "do_normalize", "image_mean", "image_std")
        missing = [name for name in required if not hasattr(processor, name)]
        if missing:
            raise NotImplementedError("Processor is missing required fields: " + ", ".join(missing))
        if not processor.do_resize or not processor.do_rescale or not processor.do_normalize:
            raise NotImplementedError("v6 only supports processors with resize, rescale, and normalize enabled.")

    def _model_device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _resize(self, image: torch.Tensor) -> torch.Tensor:
        processor = self.image_processor
        size = processor.size
        if isinstance(size, dict) and "shortest_edge" in size:
            target = int(size["shortest_edge"])
            height, width = image.shape[-2:]
            if height <= width:
                new_height = target
                new_width = int(target * width / height)
            else:
                new_width = target
                new_height = int(target * height / width)
            output_size = (new_height, new_width)
        else:
            output_size = _size_pair(size)
        resized = _pil_bicubic_resize(image, output_size).clamp(0, 1)
        # ASSUMPTION: The checkpoint's slow reference processor resizes an
        # 8-bit PIL image and therefore quantizes the resized pixels. A
        # torch-native straight-through round matches that forward contract
        # while retaining an identity gradient for image optimization.
        quantized = resized.mul(255).round().div(255)
        return resized + (quantized - resized).detach()

    @staticmethod
    def _center_crop(image: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        crop_height, crop_width = size
        height, width = image.shape[-2:]
        if crop_height > height or crop_width > width:
            pad_height = max(crop_height - height, 0)
            pad_width = max(crop_width - width, 0)
            image = F.pad(
                image,
                (pad_width // 2, pad_width - pad_width // 2, pad_height // 2, pad_height - pad_height // 2),
            )
            height, width = image.shape[-2:]
        top = (height - crop_height) // 2
        left = (width - crop_width) // 2
        return image[..., top : top + crop_height, left : left + crop_width]

    def preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
            raise ValueError("Expected one RGB image with shape [1, 3, H, W].")
        if not image.is_floating_point():
            raise TypeError("Image input must be floating point in [0, 1].")
        image = self._resize(image.float().clamp(0, 1))
        if bool(getattr(self.image_processor, "do_center_crop", False)):
            image = self._center_crop(image, _size_pair(self.image_processor.crop_size))
        scale = float(self.image_processor.rescale_factor) * 255.0
        image = image * scale
        mean = image.new_tensor(self.image_processor.image_mean).view(1, 3, 1, 1)
        std = image.new_tensor(self.image_processor.image_std).view(1, 3, 1, 1)
        return (image - mean) / std

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        pixels = self.preprocess_image(image).to(self._model_device(), dtype=torch.float32)
        features = _feature_tensor(self.model.get_image_features(pixel_values=pixels))
        if features.ndim != 2 or features.shape[0] != 1:
            raise ValueError("Projected image features must have shape [1, D].")
        return F.normalize(features[0].float(), dim=0).to(image.device)

    def _text_max_length(self) -> int:
        text_config = getattr(getattr(self.model, "config", None), "text_config", None)
        configured = getattr(text_config, "max_position_embeddings", None)
        if isinstance(configured, int) and configured > 0:
            return configured
        tokenizer_limit = getattr(self.tokenizer, "model_max_length", None)
        if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 1_000_000:
            return tokenizer_limit
        raise ValueError("Cannot determine the checkpoint's text maximum length.")

    def encode_text(self, text: str) -> torch.Tensor:
        """Return the model's official projected, normalized text feature.

        Text is fixed experiment context, so this method intentionally runs
        without autograd. Candidate-image gradients remain enabled in
        :meth:`encode_image`.
        """

        if self.tokenizer is None:
            raise RuntimeError("`encode_text` requires the checkpoint tokenizer.")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Semantic text must be a non-empty string.")
        max_length = self._text_max_length()
        raw = self.tokenizer(text, add_special_tokens=True, truncation=False)
        raw_ids = raw["input_ids"]
        raw_length = len(raw_ids[0]) if raw_ids and isinstance(raw_ids[0], list) else len(raw_ids)
        tokenized = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        model_inputs = {
            key: value.to(self._model_device())
            for key, value in tokenized.items()
            if key in {"input_ids", "attention_mask"} and torch.is_tensor(value)
        }
        with torch.no_grad():
            features = self.model.get_text_features(**model_inputs)
        if features.ndim != 2 or features.shape[0] != 1:
            raise ValueError("Projected text features must have shape [1, D].")
        effective_length = int(model_inputs.get("attention_mask", model_inputs["input_ids"].new_ones(1)).sum())
        self.last_text_metadata = {
            "raw_tokenized_length": raw_length,
            "effective_tokenized_length": effective_length,
            "model_max_length": max_length,
            "truncated": raw_length > max_length,
        }
        return F.normalize(features[0].float(), dim=0).detach()

    def processor_fidelity(self, image: torch.Tensor) -> dict[str, object]:
        if image.ndim == 3:
            image = image.unsqueeze(0)
        quantized = image.detach().clamp(0, 1).mul(255).round().div(255)
        array = quantized[0].mul(255).byte().permute(1, 2, 0).cpu().numpy()
        pil_image = Image.fromarray(array, mode="RGB")
        official = self.image_processor(images=pil_image, return_tensors="pt").pixel_values.float()
        differentiable = self.preprocess_image(quantized).detach().cpu().float()
        if official.shape != differentiable.shape:
            raise RuntimeError(
                f"Official and differentiable pixel shapes differ: {official.shape} vs {differentiable.shape}."
            )
        with torch.no_grad():
            device = self._model_device()
            official_embedding = F.normalize(
                _feature_tensor(self.model.get_image_features(pixel_values=official.to(device))).float(), dim=-1
            ).cpu()
            differentiable_embedding = F.normalize(
                _feature_tensor(self.model.get_image_features(pixel_values=differentiable.to(device))).float(), dim=-1
            ).cpu()
        return {
            "official_pixel_shape": list(official.shape),
            "differentiable_pixel_shape": list(differentiable.shape),
            "pixel_cosine": float(
                F.cosine_similarity(official.flatten()[None], differentiable.flatten()[None]).item()
            ),
            "pixel_mse": float((official - differentiable).square().mean().item()),
            "pixel_max_abs_difference": float((official - differentiable).abs().max().item()),
            "embedding_cosine": float(F.cosine_similarity(official_embedding, differentiable_embedding).item()),
            "embedding_mse": float((official_embedding - differentiable_embedding).square().mean().item()),
            "engineering_gate_embedding_cosine_at_least_0p999": bool(
                F.cosine_similarity(official_embedding, differentiable_embedding).item() >= 0.999
            ),
        }

    def metadata(self) -> dict[str, object]:
        config = getattr(self.model, "config", None)
        native_size = getattr(self.image_processor, "crop_size", None) or self.image_processor.size
        return {
            "model_id": self.model_id,
            "local_path": str(Path(self.model_id).resolve()) if Path(self.model_id).exists() else None,
            "revision": self.revision or getattr(config, "_commit_hash", None) or "unavailable_local_snapshot",
            "parameter_count": sum(parameter.numel() for parameter in self.model.parameters()),
            "dtype": str(next(self.model.parameters()).dtype).removeprefix("torch."),
            "native_image_resolution": native_size,
            "parameters_frozen": all(not parameter.requires_grad for parameter in self.model.parameters()),
        }


class CLIPImageFeatureScorer(_ProjectedImageFeatureScorer):
    model_class_name = "CLIPModel"


class SigLIPImageFeatureScorer(_ProjectedImageFeatureScorer):
    model_class_name = "SiglipModel"


class QwenHiddenFeatureScorer:
    """Adapter over the existing Qwen focus-conditioned representation."""

    def __init__(self, scorer, comparison_focus: str):
        self.scorer = scorer
        self.comparison_focus = comparison_focus.strip()
        if not self.comparison_focus:
            raise ValueError("`comparison_focus` must be non-empty.")
        self.prompt = build_focus_conditioned_feature_prompt(self.comparison_focus)
        self.model = scorer.model

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.scorer.focus_conditioned_representation(image, self.prompt)
