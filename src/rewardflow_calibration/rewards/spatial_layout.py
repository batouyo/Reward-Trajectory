"""Spatial layout preservation reward using DINOv2 features.

This reward penalizes global spatial shifts (like the "cabin moves up 2px" failure)
while allowing local texture changes within regions.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import BaseReward


class SpatialLayoutReward(BaseReward):
    """Spatial layout preservation reward using DINOv2.

    Uses DINOv2 features to detect global spatial composition changes.
    This is specifically designed to catch failures like:
    - "Scene shifts up/down by a few pixels"
    - "Object moves from bottom to center"
    - "Camera angle changes"

    The reward computes spatial correlation between source and edited images.
    Low correlation = spatial drift = high penalty.
    """

    def __init__(
        self,
        device: str | torch.device = "cuda:0",
        model_name: str = "facebook/dinov2-vitb14",
        cache_dir: str | None = None,
    ):
        """
        Args:
            device: Device to run the model on
            model_name: DINOv2 model variant
            cache_dir: Directory to cache model weights
        """
        super().__init__(device)
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._source_features: dict[int, torch.Tensor] = {}

    def _load_model(self):
        """Load DINOv2 model."""
        try:
            from transformers import AutoModel
        except ImportError:
            raise ImportError(
                "SpatialLayoutReward requires transformers. "
                "Install with: pip install transformers"
            )

        import os

        # Check if model_name is a local path
        if self.model_name and os.path.exists(self.model_name):
            model = AutoModel.from_pretrained(
                self.model_name,
                local_files_only=True,
            )
        else:
            model = AutoModel.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                local_files_only=self.cache_dir is not None,
            )
        model = model.to(self.device)
        model.eval()
        return model

    def register_source_image(self, image: torch.Tensor) -> None:
        """Register source image for layout comparison.

        Args:
            image: Source image [1, 3, H, W] in [0, 1]
        """
        with torch.no_grad():
            features = self._extract_spatial_features(image)
        img_hash = hash(image.cpu().numpy().tobytes()[:1000])
        self._source_features[img_hash] = features.detach()

    def get_source_features(self, image: torch.Tensor) -> torch.Tensor | None:
        """Get stored source features for an image."""
        img_hash = hash(image.cpu().numpy().tobytes()[:1000])
        return self._source_features.get(img_hash)

    def _extract_spatial_features(self, image: torch.Tensor) -> torch.Tensor:
        """Extract spatial feature map from an image.

        Args:
            image: Image tensor [B, 3, H, W] in [0, 1]

        Returns:
            Feature map tensor [B, H*W, hidden_dim] (spatial tokens)
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)

        # Preprocess for DINOv2
        # DINOv2 expects 224x224 input with ImageNet normalization
        image_input = self._preprocess_image(image, target_size=518)  # ViT-B/14 uses 518

        # Get model
        model = self.model

        # Extract features with attention mask
        with torch.no_grad():
            # Use get_features=True to get intermediate features
            outputs = model(
                pixel_values=image_input,
                return_dict=True,
            )

            # Get the feature map (last hidden state)
            if hasattr(outputs, "last_hidden_state"):
                features = outputs.last_hidden_state
            elif hasattr(outputs, "hidden_states") and outputs.hidden_states:
                features = outputs.hidden_states[-1]
            else:
                # Try direct output
                features = outputs[0]

        # features: [B, seq_len, hidden_dim]
        return features

    def _compute_spatial_correlation(
        self,
        features_src: torch.Tensor,
        features_edit: torch.Tensor,
    ) -> torch.Tensor:
        """Compute spatial correlation between two feature maps.

        Args:
            features_src: Source features [1, seq_len, dim]
            features_edit: Edited features [1, seq_len, dim]

        Returns:
            Correlation score (higher = more spatially aligned)
        """
        # Normalize features
        features_src = F.normalize(features_src, p=2, dim=-1)
        features_edit = F.normalize(features_edit, p=2, dim=-1)

        # Cosine similarity matrix
        # [1, seq_len, dim] @ [1, dim, seq_len] -> [1, seq_len, seq_len]
        sim_matrix = torch.bmm(features_src, features_edit.transpose(1, 2))

        # Extract diagonal (matching positions)
        batch_size = sim_matrix.shape[0]
        seq_len = sim_matrix.shape[1]

        # Get diagonal elements (correspondence at same spatial position)
        diag_indices = torch.arange(seq_len, device=sim_matrix.device)
        batch_indices = torch.arange(batch_size, device=sim_matrix.device)

        # [B, seq_len]
        correspondence = sim_matrix[:, diag_indices, diag_indices]

        return correspondence.mean(dim=-1)

    def _compute_feature_distance(
        self,
        features_src: torch.Tensor,
        features_edit: torch.Tensor,
    ) -> torch.Tensor:
        """Compute average feature distance.

        This is more robust to small translations than correlation.

        Args:
            features_src: Source features [1, seq_len, dim]
            features_edit: Edited features [1, seq_len, dim]

        Returns:
            Distance (lower = more similar)
        """
        # Normalize
        features_src = F.normalize(features_src, p=2, dim=-1)
        features_edit = F.normalize(features_edit, p=2, dim=-1)

        # Euclidean distance per position
        diff = features_src - features_edit
        distance = diff.pow(2).sum(dim=-1).mean()

        return distance

    def __call__(self, image: torch.Tensor, prompt: str | list[str]) -> torch.Tensor:
        """Compute spatial layout similarity reward.

        Args:
            image: Edited image [B, 3, H, W] in [0, 1]
            prompt: Text prompt (not used)

        Returns:
            Scalar reward (higher = better spatial alignment)
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)

        source_features = self.get_source_features(image)
        if source_features is None:
            # Register this as source
            self.register_source_image(image)
            return torch.tensor(1.0, device=image.device, dtype=image.dtype)

        # Extract features from current image
        with torch.no_grad():
            edit_features = self._extract_spatial_features(image)

        # Compute correlation
        correlation = self._compute_spatial_correlation(source_features, edit_features)

        return correlation.mean()

    def compute_penalty(
        self,
        image: torch.Tensor,
        source_image: torch.Tensor,
        intensity: float = 1.0,
    ) -> torch.Tensor:
        """Compute spatial layout penalty for gradient optimization.

        Args:
            image: Edited image [1, 3, H, W]
            source_image: Source image [1, 3, H, W]
            intensity: Scaling factor

        Returns:
            Penalty tensor (lower = better)
        """
        if self.get_source_features(source_image) is None:
            self.register_source_image(source_image)

        with torch.no_grad():
            edit_features = self._extract_spatial_features(image)
        src_features = self.get_source_features(source_image)

        # Compute distance
        distance = self._compute_feature_distance(src_features, edit_features)

        return intensity * distance

    def get_spatial_shift(
        self,
        image: torch.Tensor,
        source_image: torch.Tensor,
    ) -> dict[str, float]:
        """Detect spatial shift direction and magnitude.

        Useful for debugging which direction the scene is shifting.

        Returns:
            Dict with shift_x, shift_y, confidence
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if source_image.dim() == 3:
            source_image = source_image.unsqueeze(0)

        with torch.no_grad():
            src_features = self._extract_spatial_features(source_image)
            edit_features = self._extract_spatial_features(image)

        # Normalize
        src_features = F.normalize(src_features, p=2, dim=-1)
        edit_features = F.normalize(edit_features, p=2, dim=-1)

        # Compute cross-correlation to find shift
        # This is a simplified version - full optical flow would be more accurate
        # but this catches the common "whole scene shifts" case

        # Flatten spatial dimensions
        B, L, D = src_features.shape
        seq_h = int(L ** 0.5)  # Assume square spatial grid
        seq_w = seq_h

        src_spatial = src_features.reshape(B, seq_h, seq_w, D).permute(0, 3, 1, 2)
        edit_spatial = edit_features.reshape(B, seq_h, seq_w, D).permute(0, 3, 1, 2)

        # Simple shift detection: compute center of mass shift
        src_sum = src_spatial.abs().sum(dim=1)  # [B, H, W]
        edit_sum = edit_spatial.abs().sum(dim=1)

        # Weighted center of mass
        coords_y, coords_x = torch.meshgrid(
            torch.arange(seq_h, device=src_features.device),
            torch.arange(seq_w, device=src_features.device),
            indexing="ij",
        )

        src_mass = src_sum[0]
        edit_mass = edit_sum[0]

        src_mass = src_mass / (src_mass.sum() + 1e-8)
        edit_mass = edit_mass / (edit_mass.sum() + 1e-8)

        src_cx = (coords_x.float() * src_mass).sum()
        src_cy = (coords_y.float() * src_mass).sum()

        edit_cx = (coords_x.float() * edit_mass).sum()
        edit_cy = (coords_y.float() * edit_mass).sum()

        return {
            "shift_x": float((edit_cx - src_cx).cpu()),
            "shift_y": float((edit_cy - src_cy).cpu()),
            "confidence": float(
                F.cosine_similarity(
                    src_spatial[0].flatten().unsqueeze(0),
                    edit_spatial[0].flatten().unsqueeze(0),
                ).cpu()
            ),
        }
