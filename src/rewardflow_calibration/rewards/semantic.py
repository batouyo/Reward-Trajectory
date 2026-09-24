"""Semantic alignment reward using SigLIP.

This reward ensures the edited image aligns with the target text description.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import BaseReward


class SemanticReward(BaseReward):
    """Semantic alignment reward using SigLIP.

    SigLIP provides text-image similarity scores that are:
    - Good at semantic alignment (is this image "old man"?)
    - Robust to specific instantiation details

    This is used to ensure the edit is going in the right semantic direction.
    """

    def __init__(
        self,
        device: str | torch.device = "cuda:0",
        model_name: str = "google/siglip-so400m-patch14-384",
        cache_dir: str | None = None,
    ):
        """
        Args:
            device: Device to run the model on
            model_name: SigLIP model variant
            cache_dir: Directory to cache model weights
        """
        super().__init__(device)
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._cached_text_features: dict[str, torch.Tensor] = {}

    def _load_model(self):
        """Load SigLIP model and tokenizer."""
        try:
            from transformers import SiglipModel, SiglipProcessor
        except ImportError:
            raise ImportError(
                "SemanticReward requires transformers. "
                "Install with: pip install transformers"
            )

        import os

        # Check if model_name is a local path
        if self.model_name and os.path.exists(self.model_name):
            model = SiglipModel.from_pretrained(
                self.model_name,
                local_files_only=True,
            )
            processor = SiglipProcessor.from_pretrained(
                self.model_name,
                local_files_only=True,
            )
        else:
            model = SiglipModel.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                local_files_only=self.cache_dir is not None,
            )
            processor = SiglipProcessor.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                local_files_only=self.cache_dir is not None,
            )
        model = model.to(self.device)
        model.eval()

        return {"model": model, "processor": processor}

    def prepare_text(self, prompt: str | list[str]) -> None:
        """Cache text embeddings for a prompt.

        Call this before rollout to precompute text features.

        Args:
            prompt: Target text description
        """
        if isinstance(prompt, list):
            prompt_str = "|".join(prompt)
        else:
            prompt_str = prompt

        if prompt_str not in self._cached_text_features:
            text_features = self._encode_text(prompt)
            self._cached_text_features[prompt_str] = text_features.detach()

    def _encode_text(self, prompt: str | list[str]) -> torch.Tensor:
        """Encode text to embeddings.

        Args:
            prompt: Text prompt(s)

        Returns:
            Normalized text features [B, dim]
        """
        processor = self.model["processor"]
        model = self.model["model"]

        # Tokenize
        inputs = processor(
            text=prompt,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=64,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Encode
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)

        # Normalize
        text_features = F.normalize(text_features, p=2, dim=-1)
        return text_features

    def _preprocess_image_siglip(self, image: torch.Tensor) -> torch.Tensor:
        """Preprocess image for SigLIP.

        SigLIP expects specific normalization.

        Args:
            image: Image tensor [B, 3, H, W] in [0, 1]

        Returns:
            Preprocessed tensor
        """
        processor = self.model["processor"]
        # Use processor to preprocess
        # But we want to keep gradients, so we do it manually

        # SigLIP normalization
        mean = torch.tensor([0.5, 0.5, 0.5], device=image.device).view(1, 3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5], device=image.device).view(1, 3, 1, 1)

        # Resize to model input size
        # SigLIP-SO400M uses 384x384
        image = F.interpolate(
            image, size=(384, 384), mode="bicubic", align_corners=False
        )

        # Normalize to [-1, 1]
        image = (image - mean) / std

        return image

    def __call__(self, image: torch.Tensor, prompt: str | list[str]) -> torch.Tensor:
        """Compute semantic similarity reward.

        Args:
            image: Edited image [B, 3, H, W] in [0, 1]
            prompt: Target text description

        Returns:
            Scalar reward (higher = better semantic alignment)
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)

        # Encode text
        if isinstance(prompt, list):
            prompt_str = "|".join(prompt)
        else:
            prompt_str = prompt

        if prompt_str in self._cached_text_features:
            text_features = self._cached_text_features[prompt_str]
            if text_features.device != image.device:
                text_features = text_features.to(image.device)
        else:
            text_features = self._encode_text(prompt)
            self._cached_text_features[prompt_str] = text_features

        # Preprocess image
        image_input = self._preprocess_image_siglip(image)

        # Encode image
        model = self.model["model"]
        with torch.no_grad():
            image_features = model.get_image_features(pixel_values=image_input)

        # Normalize
        image_features = F.normalize(image_features, p=2, dim=-1)

        # Compute similarity
        similarity = (image_features * text_features).sum(dim=-1)

        return similarity.mean()

    def compute_reward_with_target(
        self,
        image: torch.Tensor,
        target_prompt: str,
        source_prompt: str | None = None,
    ) -> torch.Tensor:
        """Compute reward with explicit target and optional source prompt.

        This can be used to compute "edit direction" rewards.

        Args:
            image: Edited image [B, 3, H, W]
            target_prompt: Target description (e.g., "an old man")
            source_prompt: Source description (e.g., "a young man")

        Returns:
            Semantic reward
        """
        # Encode both prompts
        target_features = self._encode_text(target_prompt)
        reward = self(image, target_prompt)

        if source_prompt is not None:
            source_features = self._encode_text(source_prompt)

            # Compute similarity to both
            image_input = self._preprocess_image_siglip(image)
            model = self.model["model"]
            with torch.no_grad():
                image_features = model.get_image_features(pixel_values=image_input)
            image_features = F.normalize(image_features, p=2, dim=-1)

            # Relative similarity (how much closer to target vs source)
            sim_target = (image_features * target_features).sum(dim=-1)
            sim_source = (image_features * source_features).sum(dim=-1)

            # Reward = similarity to target minus similarity to source
            # Positive = closer to target, negative = closer to source
            reward = sim_target - sim_source

        return reward.mean()


class CLIPSemanticReward(BaseReward):
    """CLIP-based semantic reward as a fallback.

    Uses OpenAI CLIP instead of SigLIP.
    """

    def __init__(
        self,
        device: str | torch.device = "cuda:0",
        model_name: str = "openai/clip-vit-large-patch14-336",
        cache_dir: str | None = None,
    ):
        super().__init__(device)
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._cached_text_features: dict[str, torch.Tensor] = {}
        self._source_image: torch.Tensor | None = None

    def _load_model(self):
        """Load CLIP model."""
        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError:
            raise ImportError(
                "CLIPSemanticReward requires transformers. "
                "Install with: pip install transformers"
            )

        import os

        # Check if model_name is a local path
        if self.model_name and os.path.exists(self.model_name):
            model = CLIPModel.from_pretrained(
                self.model_name,
                local_files_only=True,
            )
            processor = CLIPProcessor.from_pretrained(
                self.model_name,
                local_files_only=True,
            )
        else:
            model = CLIPModel.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                local_files_only=self.cache_dir is not None,
            )
            processor = CLIPProcessor.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                local_files_only=self.cache_dir is not None,
            )
        model = model.to(self.device)
        model.eval()

        return {"model": model, "processor": processor}

    def register_source_image(self, image: torch.Tensor) -> None:
        """Register source image for identity comparison."""
        self._source_image = image.detach()

    def get_source_embedding(self) -> torch.Tensor | None:
        """Get the stored source image for identity comparison."""
        return self._source_image

    def __call__(self, image: torch.Tensor, prompt: str | list[str]) -> torch.Tensor:
        """Compute CLIP-based semantic similarity."""
        if image.dim() == 3:
            image = image.unsqueeze(0)

        processor = self.model["processor"]
        model = self.model["model"]

        # Encode text
        text_inputs = processor(
            text=prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

        with torch.no_grad():
            text_features = model.get_text_features(**text_inputs)
            text_features = F.normalize(text_features, p=2, dim=-1)

        # Encode image
        image_inputs = processor(
            images=image,
            return_tensors="pt",
        )
        # CLIPProcessor returns a dict with 'pixel_values'
        pixel_values = image_inputs.get("pixel_values")
        if pixel_values is not None:
            pixel_values = pixel_values.to(self.device)
            with torch.no_grad():
                image_features = model.get_image_features(pixel_values=pixel_values)
        else:
            # Fallback for different processor versions
            image_np = (image[0].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
            from PIL import Image as PILImage
            pil_img = PILImage.fromarray(image_np)
            image_inputs = processor(images=pil_img, return_tensors="pt")
            pixel_values = image_inputs["pixel_values"].to(self.device)
            with torch.no_grad():
                image_features = model.get_image_features(pixel_values=pixel_values)

        image_features = F.normalize(image_features, p=2, dim=-1)

        # Similarity
        similarity = (image_features * text_features).sum(dim=-1)
        return similarity.mean()
