"""Composite reward function combining multiple reward signals.

This module provides a unified interface for combining:
- Face identity preservation
- Spatial layout preservation
- Semantic alignment

The composite reward can be used for gradient-based trajectory optimization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from .base import BaseReward, RewardOutput

if TYPE_CHECKING:
    from .face_identity import FaceIdentityReward
    from .spatial_layout import SpatialLayoutReward
    from .semantic import SemanticReward


@dataclass
class RewardWeights:
    """Weights for different reward components.

    These weights control the relative importance of each reward signal.
    """

    # Semantic alignment (SigLIP) - ensures edit goes in right direction
    semantic: float = 1.0

    # Face identity preservation
    identity: float = 1.0

    # Spatial layout preservation
    layout: float = 1.0

    # KL divergence anchor (latent space)
    kl: float = 0.1

    def __post_init__(self):
        """Validate weights."""
        for name, value in vars(self).items():
            if value < 0:
                raise ValueError(f"Weight {name} must be non-negative, got {value}")


@dataclass
class CompositeRewardConfig:
    """Configuration for composite reward behavior."""

    # Whether to use gradient-based optimization
    use_gradient: bool = True

    # Gradient step size for reward-based latent updates
    reward_lr: float = 0.01

    # Threshold below which rewards are not applied
    reward_threshold: float = 0.5

    # Alpha threshold above which reward guidance activates
    # (low alpha = reference preserved, high alpha = reward needed)
    reward_alpha_threshold: float = 0.4

    # How often to apply reward guidance (every N steps)
    reward_update_interval: int = 2

    # Whether to cache source embeddings
    cache_source: bool = True

    # Normalize rewards to [0, 1] range
    normalize_rewards: bool = True


class CompositeReward:
    """Combines multiple reward functions into one.

    The composite reward computes:
    - Semantic reward: alignment with target description
    - Identity reward: face structure preservation
    - Layout reward: spatial composition preservation
    - KL reward: latent space proximity to source

    Usage:
        reward_fn = CompositeReward(
            semantic_reward=SemanticReward(),
            identity_reward=FaceIdentityReward(),
            layout_reward=SpatialLayoutReward(),
        )

        # Register source image
        reward_fn.register_source(source_image)

        # Compute reward
        result = reward_fn(edited_image, target_prompt)
        print(result.total)  # Scalar reward
        print(result.components)  # Per-component rewards

        # Get gradient for optimization
        if result.total < threshold:
            gradient = reward_fn.compute_gradient(edited_image, target_prompt)
            edited_image = edited_image - lr * gradient
    """

    def __init__(
        self,
        semantic_reward: SemanticReward | None = None,
        identity_reward: FaceIdentityReward | None = None,
        layout_reward: SpatialLayoutReward | None = None,
        weights: RewardWeights | None = None,
        config: CompositeRewardConfig | None = None,
        device: str | torch.device = "cuda:0",
    ):
        """
        Args:
            semantic_reward: SigLIP-based semantic alignment reward
            identity_reward: Face embedding-based identity reward
            layout_reward: DINOv2-based spatial layout reward
            weights: Component weights
            config: Behavior configuration
            device: Device for computations
        """
        self.device = torch.device(device)
        self.semantic_reward = semantic_reward
        self.identity_reward = identity_reward
        self.layout_reward = layout_reward
        self.weights = weights or RewardWeights()
        self.config = config or CompositeRewardConfig()

        self._source_image: torch.Tensor | None = None
        self._source_latent: torch.Tensor | None = None
        self._source_prompt: str | None = None
        self._initialized = False

    def register_source(
        self,
        source_image: torch.Tensor,
        source_latent: torch.Tensor | None = None,
        source_prompt: str | None = None,
    ) -> None:
        """Register source image for reward computation.

        This must be called before computing rewards.

        Args:
            source_image: Source image [1, 3, H, W] in [0, 1]
            source_latent: Source latent [1, seq, channels] (optional, for KL reward)
            source_prompt: Source text prompt
        """
        if source_image.dim() == 3:
            source_image = source_image.unsqueeze(0)

        self._source_image = source_image.detach()
        self._source_latent = source_latent.detach() if source_latent is not None else None
        self._source_prompt = source_prompt

        # Register with individual reward models
        if self.identity_reward is not None and self.config.cache_source:
            self.identity_reward.register_source_image(source_image)

        if self.layout_reward is not None and self.config.cache_source:
            self.layout_reward.register_source_image(source_image)

        if self.semantic_reward is not None and source_prompt is not None:
            self.semantic_reward.prepare_text(source_prompt)

        self._initialized = True

    @property
    def is_initialized(self) -> bool:
        """Check if source has been registered."""
        return self._initialized

    def _compute_component_rewards(
        self,
        image: torch.Tensor,
        target_prompt: str,
    ) -> dict[str, torch.Tensor]:
        """Compute all component rewards.

        Args:
            image: Edited image [B, 3, H, W]
            target_prompt: Target text description

        Returns:
            Dict of component rewards
        """
        rewards = {}

        # Semantic reward
        if self.semantic_reward is not None:
            rewards["semantic"] = self.semantic_reward(image, target_prompt)

        # Identity reward
        if self.identity_reward is not None:
            rewards["identity"] = self.identity_reward(image, target_prompt)

        # Layout reward
        if self.layout_reward is not None:
            rewards["layout"] = self.layout_reward(image, target_prompt)

        return rewards

    def _compute_kl_reward(
        self,
        latent: torch.Tensor,
        source_latent: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute KL divergence reward in latent space.

        Args:
            latent: Current latent [B, seq, channels]
            source_latent: Source latent (uses stored if None)

        Returns:
            KL reward (higher = closer to source)
        """
        if source_latent is None:
            source_latent = self._source_latent
        if source_latent is None:
            return torch.tensor(0.0, device=latent.device)

        # Negative squared distance (higher = closer)
        distance = (latent - source_latent).pow(2).mean()
        return -distance

    def maybe_onload(self) -> None:
        """Load all reward models to device."""
        if self.semantic_reward is not None:
            self.semantic_reward.maybe_onload()
        if self.identity_reward is not None:
            self.identity_reward.maybe_onload()
        if self.layout_reward is not None:
            self.layout_reward.maybe_onload()

    def maybe_offload(self) -> None:
        """Offload all reward models from device to save memory."""
        if self.semantic_reward is not None:
            self.semantic_reward.maybe_offload()
        if self.identity_reward is not None:
            self.identity_reward.maybe_offload()
        if self.layout_reward is not None:
            self.layout_reward.maybe_offload()

    def __call__(
        self,
        image: torch.Tensor,
        target_prompt: str,
        latent: torch.Tensor | None = None,
    ) -> RewardOutput:
        """Compute composite reward.

        Args:
            image: Edited image [B, 3, H, W] in [0, 1]
            target_prompt: Target text description
            latent: Current latent for KL reward (optional)

        Returns:
            RewardOutput with total and component rewards
        """
        if not self._initialized:
            raise RuntimeError(
                "CompositeReward not initialized. Call register_source() first."
            )

        if image.dim() == 3:
            image = image.unsqueeze(0)

        # Compute component rewards
        components = self._compute_component_rewards(image, target_prompt)

        # Compute KL reward
        kl_reward = self._compute_kl_reward(latent) if latent is not None else None

        # Combine with weights
        total = torch.tensor(0.0, device=image.device, dtype=image.dtype)

        weighted_components = {}
        for name, reward in components.items():
            weight = getattr(self.weights, name, 0.0)
            weighted = weight * reward
            weighted_components[name] = weighted
            total = total + weighted

        if kl_reward is not None and self.weights.kl > 0:
            weighted_kl = self.weights.kl * kl_reward
            weighted_components["kl"] = weighted_kl
            total = total + weighted_kl

        return RewardOutput(
            total=total,
            components=weighted_components,
            metadata={"raw_rewards": components} if components else None,
        )

    def get_identity_reward(self, image: torch.Tensor) -> float:
        """Get the raw identity similarity (0-1) for failure detection.

        Returns:
            Identity cosine similarity, or -1 if no identity reward configured
        """
        if self.identity_reward is None:
            return -1.0
        raw_rewards = self._compute_component_rewards(image, "")
        if "identity" in raw_rewards:
            return float(raw_rewards["identity"].detach().cpu())
        return -1.0

    def get_source_image(self) -> torch.Tensor | None:
        """Get the registered source image for perceptual loss computation.

        Returns the source image tensor. Caller should move to appropriate device.
        """
        return self._source_image

    def compute_gradient(
        self,
        image: torch.Tensor,
        target_prompt: str,
        latent: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Compute gradient of reward with respect to latent.

        This computes d(reward)/d(latent) using backpropagation through:
        latent -> VAE decode -> reward model -> reward

        Args:
            image: Current decoded image [1, 3, H, W]
            target_prompt: Target text description
            latent: Current latent [1, seq, channels]

        Returns:
            Gradient tensor [1, seq, channels] or None if gradient unavailable
        """
        if latent is None:
            return None

        # Set up for gradient computation
        latent_var = latent.detach().requires_grad_(True)

        # Note: Full gradient computation requires VAE decode in the graph
        # This is a simplified version - in practice you'd use the actual
        # VAE from the pipeline

        # For now, return gradient computed from reward components
        # In the full implementation, this would go through VAE decode

        return None  # Placeholder - actual implementation needs VAE

    def compute_latent_update(
        self,
        latent: torch.Tensor,
        gradient: torch.Tensor,
        step_size: float | None = None,
    ) -> torch.Tensor:
        """Compute latent update from gradient.

        Args:
            latent: Current latent [B, seq, channels]
            gradient: Reward gradient [B, seq, channels]
            step_size: Step size (uses config if None)

        Returns:
            Updated latent
        """
        if step_size is None:
            step_size = self.config.reward_lr

        # Normalize gradient to prevent explosions
        grad_norm = gradient.float().norm()
        if grad_norm > 0:
            gradient = gradient / (grad_norm + 1e-8)

        # Gradient ASCENT (we want to maximize reward)
        # So we add gradient to latent
        updated = latent + step_size * gradient.to(latent.dtype)

        return updated.detach()

    def should_apply_reward(
        self,
        alpha: float,
        current_step: int,
        total_steps: int,
    ) -> bool:
        """Determine if reward guidance should be applied.

        Args:
            alpha: Current alpha value (0=preserve, 1=full edit)
            current_step: Current denoising step
            total_steps: Total number of denoising steps

        Returns:
            True if reward guidance should be applied
        """
        if alpha < self.config.reward_alpha_threshold:
            return False

        if current_step % self.config.reward_update_interval != 0:
            return False

        return True

    def check_reward_threshold(
        self,
        reward_output: RewardOutput,
    ) -> bool:
        """Check if reward is below threshold (needs correction).

        Args:
            reward_output: Output from __call__

        Returns:
            True if reward is below threshold
        """
        return reward_output.scalar < self.config.reward_threshold

    def get_component_diagnostics(
        self,
        image: torch.Tensor,
        target_prompt: str,
    ) -> dict[str, dict]:
        """Get detailed diagnostics for each reward component.

        Useful for debugging which reward is causing issues.

        Args:
            image: Edited image
            target_prompt: Target prompt

        Returns:
            Dict of component diagnostics
        """
        diagnostics = {}

        # Raw reward values
        components = self._compute_component_rewards(image, target_prompt)
        diagnostics["raw_rewards"] = {
            name: float(val.detach().cpu()) for name, val in components.items()
        }

        # Weighted values
        diagnostics["weighted_rewards"] = {
            name: float(val.detach().cpu())
            for name, val in self(image, target_prompt).components.items()
        }

        # Individual model diagnostics
        if self.identity_reward is not None:
            try:
                shift = self.identity_reward._extract_face_embedding(image)
                diagnostics["identity"] = {
                    "embedding_norm": float(shift.norm().cpu()),
                    "face_detected": True,
                }
            except Exception:
                diagnostics["identity"] = {"face_detected": False}

        if self.layout_reward is not None and self._source_image is not None:
            try:
                shift_info = self.layout_reward.get_spatial_shift(
                    image, self._source_image
                )
                diagnostics["layout"] = shift_info
            except Exception as e:
                diagnostics["layout"] = {"error": str(e)}

        return diagnostics


class IntensityAwareReward:
    """Reward that adjusts weights based on edit intensity (alpha).

    This is specifically designed for the VeloEdit trajectory where:
    - Low alpha: Reference preserved, minimal reward needed
    - High alpha: More edit freedom, identity/layout protection critical
    """

    def __init__(
        self,
        composite_reward: CompositeReward,
        intensity_scale: float = 1.5,
    ):
        """
        Args:
            composite_reward: Base composite reward
            intensity_scale: How much to scale weights at high intensity
        """
        self.base_reward = composite_reward
        self.intensity_scale = intensity_scale

    def get_weights_for_alpha(
        self,
        alpha: float,
    ) -> RewardWeights:
        """Get reward weights adjusted for current alpha.

        At low alpha: lower identity/layout weights (reference preserved)
        At high alpha: higher identity/layout weights (protection needed)

        Args:
            alpha: Current alpha [0, 1]

        Returns:
            Adjusted weights
        """
        base = self.base_reward.weights

        # Scale factor increases with alpha
        scale = 1.0 + (alpha ** self.intensity_scale - alpha) * self.intensity_scale

        return RewardWeights(
            semantic=base.semantic,  # Semantic alignment stays constant
            identity=base.identity * scale,  # Identity protection increases
            layout=base.layout * scale,  # Layout protection increases
            kl=base.kl * scale,  # KL anchor weakens (less reference)
        )

    def __call__(
        self,
        image: torch.Tensor,
        target_prompt: str,
        alpha: float,
        latent: torch.Tensor | None = None,
    ) -> RewardOutput:
        """Compute intensity-adjusted reward.

        Args:
            image: Edited image
            target_prompt: Target prompt
            alpha: Current edit intensity [0, 1]
            latent: Current latent

        Returns:
            Adjusted reward output
        """
        # Temporarily adjust weights
        original_weights = self.base_reward.weights
        self.base_reward.weights = self.get_weights_for_alpha(alpha)

        result = self.base_reward(image, target_prompt, latent)

        # Restore original weights
        self.base_reward.weights = original_weights

        return result

    def should_correct(
        self,
        reward_output: RewardOutput,
        alpha: float,
    ) -> bool:
        """Determine if correction is needed based on reward and alpha.

        Args:
            reward_output: Current reward output
            alpha: Current alpha

        Returns:
            True if correction should be applied
        """
        # Higher alpha = lower threshold for correction
        threshold = self.base_reward.config.reward_threshold * (2.0 - alpha)

        return reward_output.scalar < threshold
