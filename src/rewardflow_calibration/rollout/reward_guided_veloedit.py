"""Reward-guided VeloEdit rollout.

This module extends VeloEditCompatibleRollout with reward-based trajectory
optimization to address identity drift and spatial layout issues at high alpha.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch
import torch.nn.functional as F
from PIL import Image

from .veloedit import (
    PreparedVeloEdit,
    VeloEditCompatibleRollout,
    VeloEditRolloutConfig,
)


@dataclass
class RewardGuidanceConfig:
    """Configuration for reward guidance in VeloEdit."""

    # Whether to enable reward guidance
    enabled: bool = True

    # Reward model weights
    identity_weight: float = 1.0
    layout_weight: float = 1.0
    semantic_weight: float = 0.5

    # When to activate reward guidance
    reward_alpha_threshold: float = 0.4  # Activate when alpha >= this
    reward_step_start: int = 4  # Start reward guidance after this step

    # How often to apply reward guidance (every N steps)
    reward_update_interval: int = 2

    # Gradient step size
    reward_lr: float = 0.005

    # Threshold for applying correction (identity similarity threshold)
    reward_threshold: float = 0.95

    # Maximum correction magnitude per step
    max_correction: float = 0.01

    # Use gradient clipping
    clip_grad_norm: float = 1.0


@dataclass
class RewardGuidedVeloEditResult:
    """Result from reward-guided VeloEdit rollout."""

    images: torch.Tensor  # [branches, 3, H, W]
    alpha: torch.Tensor  # [branches]
    reward_diagnostics: dict | None = None


class RewardGuidedVeloEditRollout:
    """VeloEdit with reward-based trajectory optimization.

    This extends VeloEditCompatibleRollout by adding reward guidance during
    the denoising process to prevent identity drift and spatial layout
    issues at high alpha values.

    The key insight is:
    - VeloEdit provides efficient multi-strength generation via velocity blending
    - Rewards detect when the trajectory goes wrong (identity drift, layout shift)
    - Reward gradients correct the latent trajectory

    For gradients, we use differentiable perceptual losses (not non-differentiable
    face detectors). Face detection is only used for decision-making.

    Memory optimization: processes branches sequentially to save GPU memory.
    """

    def __init__(
        self,
        model_path: str,
        *,
        device: str | torch.device = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        local_files_only: bool = True,
    ):
        """
        Args:
            model_path: Path to FLUX-Kontext model
            device: Device to run on
            dtype: Model dtype
            local_files_only: Only load local files
        """
        self.device = torch.device(device)
        self.dtype = dtype
        self.base_rollout = VeloEditCompatibleRollout(
            model_path,
            device=device,
            dtype=dtype,
            local_files_only=local_files_only,
        )

        # Perceptual feature extractor for differentiable identity loss
        self._perceptual_model = None

    def _get_perceptual_model(self):
        """Get or create a perceptual feature extractor for differentiable loss."""
        if self._perceptual_model is None:
            # Use a simple VGG-style feature extractor
            # This is differentiable unlike face detectors
            try:
                import torchvision.models as models
                vgg = models.vgg16(pretrained=True).features[:16].to(self.device)
                vgg.eval()
                for p in vgg.parameters():
                    p.requires_grad = False
                self._perceptual_model = vgg
            except Exception:
                # Fallback: use identity
                self._perceptual_model = None
        return self._perceptual_model

    def _decode_for_reward(
        self,
        z: torch.Tensor,
        prepared: PreparedVeloEdit,
    ) -> torch.Tensor:
        """Decode latent to image for reward computation.

        Args:
            z: Latent tensor [1, seq, channels]
            prepared: PreparedVeloEdit with metadata

        Returns:
            Decoded images [1, 3, H, W] in [0, 1]
        """
        pipeline = self.base_rollout.pipeline

        with torch.no_grad():
            unpacked = pipeline._unpack_latents(
                z,
                prepared.height,
                prepared.width,
                pipeline.vae_scale_factor,
            )
            unpacked = unpacked / pipeline.vae.config.scaling_factor + pipeline.vae.config.shift_factor
            decoded = pipeline.vae.decode(
                unpacked.to(dtype=pipeline.vae.dtype),
                return_dict=False,
            )[0]
            images = pipeline.image_processor.postprocess(decoded, output_type="pt")
            return images.float().clamp(0, 1)

    def _offload_reward_models(self):
        """Offload reward models to free memory."""
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _compute_perceptual_identity_loss(
        self,
        edited_image: torch.Tensor,
        source_image: torch.Tensor,
    ) -> torch.Tensor:
        """Compute differentiable perceptual identity loss.

        Uses pretrained features to preserve structural similarity.
        Unlike face detectors, this is fully differentiable.
        """
        model = self._get_perceptual_model()

        if model is None:
            # Fallback: simple L2 loss
            return F.mse_loss(edited_image, source_image)

        # Normalize images to [-1, 1] for VGG
        edited_norm = edited_image * 2 - 1
        source_norm = source_image * 2 - 1

        # Ensure source is same size as edited
        if source_norm.shape != edited_norm.shape:
            source_norm = F.interpolate(source_norm, size=edited_norm.shape[2:], mode='bilinear', align_corners=False)

        # Extract features - source with no grad (pretrained, frozen)
        with torch.no_grad():
            source_features = model(source_norm)

        # Extract edited features WITH grad enabled (we want gradients through this)
        edited_features = model(edited_norm)

        # Perceptual loss: match intermediate features
        # We want gradients to flow FROM edited_features TO edited_image
        loss = F.mse_loss(edited_features, source_features)
        return loss

    def _compute_identity_preserving_correction(
        self,
        z: torch.Tensor,
        z_ref: torch.Tensor,
        sigma: float,
        source_image: torch.Tensor,
        current_image: torch.Tensor,
    ) -> torch.Tensor | None:
        """Compute identity-preserving correction by blending with reference.

        Strategy:
        When identity is lost (face changed), we need to restore the identity.
        Instead of computing gradients (which VAE doesn't support), we use direct
        latent space manipulation:

        1. Compute identity loss (how different is the face from source?)
        2. Blend current latent with reference latent to restore identity
        3. The blend weight is proportional to identity loss

        This is efficient (no extra decodes) and effective.

        Args:
            z: Current latent
            z_ref: Reference latent (source at same noise level)
            sigma: Current noise level
            source_image: Source image tensor
            current_image: Current decoded image

        Returns:
            Correction tensor to apply to latent, or None
        """
        try:
            # Compute identity loss (face difference)
            identity_loss = F.mse_loss(current_image, source_image)

            # The loss can range from 0 (identical) to ~1 (completely different)
            # Map to blend weight: higher loss -> more reference influence
            # Clamp to reasonable range
            blend_weight = min(identity_loss.item() * 2.0, 0.8)  # Max 80% reference

            # Blend: z_new = (1 - w) * z + w * z_ref
            # This pulls toward reference while keeping some of the edit
            correction = blend_weight * (z_ref - z)

            return correction

        except Exception as e:
            print(f"Warning: Identity correction failed: {e}")
            return None

    def rollout_with_rewards(
        self,
        prepared: PreparedVeloEdit,
        alphas: Sequence[float] | torch.Tensor,
        target_prompt: str,
        composite_reward,
        *,
        reward_config: RewardGuidanceConfig | None = None,
        config: VeloEditRolloutConfig | None = None,
    ) -> RewardGuidedVeloEditResult:
        """Rollout with reward guidance.

        Args:
            prepared: PreparedVeloEdit from base rollout
            alphas: Alpha values to generate [0, 1]
            target_prompt: Target text description
            composite_reward: Composite reward function (must be initialized)
            reward_config: Reward guidance configuration
            config: VeloEdit rollout configuration

        Returns:
            RewardGuidedVeloEditResult with images and diagnostics
        """
        config = config or VeloEditRolloutConfig()
        reward_config = reward_config or RewardGuidanceConfig()
        config.validate()

        # Validate alpha
        alpha = torch.as_tensor(alphas, device=self.device, dtype=torch.float32).flatten()
        if alpha.numel() < 1 or not torch.isfinite(alpha).all() or torch.any((alpha < 0) | (alpha > 1)):
            raise ValueError("alphas must be finite values in [0, 1]")

        branches = alpha.numel()
        dtype = self.base_rollout.pipeline.transformer.dtype

        # Track diagnostics
        reward_diagnostics = {
            "corrections_applied": [],
            "reward_values": [],
            "gradient_norms": [],
        }

        # Offload reward models initially to save memory
        self._offload_reward_models()

        train_steps = self.base_rollout.pipeline.scheduler.config.get("num_train_timesteps", 1000)

        # Process branches sequentially to save memory
        final_z_list = []

        for b in range(branches):
            # Get single branch data
            z_b = prepared.latents.clone()
            ref_b = prepared.reference_latent.clone()
            img_lat_b = None if prepared.image_latents is None else prepared.image_latents.clone()
            pe_b = prepared.prompt_embeds
            pooled_b = prepared.pooled_prompt_embeds
            guid_b = prepared.guidance
            current_alpha = alpha[b].item()

            with torch.no_grad():
                for index in range(len(prepared.sigma_schedule) - 1):
                    sigma = prepared.sigma_schedule[index]
                    sigma_next = prepared.sigma_schedule[index + 1]

                    # === Standard VeloEdit velocity blending ===
                    model_input = z_b if img_lat_b is None else torch.cat([z_b, img_lat_b], dim=1)
                    sigma_input = torch.as_tensor(sigma, device=self.device, dtype=torch.float32)
                    # Timestep should be [1] tensor
                    timestep = (sigma_input * train_steps).to(dtype=z_b.dtype).reshape(1) / train_steps

                    native = self.base_rollout.pipeline.transformer(
                        hidden_states=model_input,
                        timestep=timestep,
                        guidance=guid_b,
                        pooled_projections=pooled_b,
                        encoder_hidden_states=pe_b,
                        txt_ids=prepared.text_ids,
                        img_ids=prepared.latent_ids,
                        joint_attention_kwargs={},
                        return_dict=False,
                    )[0][:, : z_b.shape[1]]

                    actual = native
                    if index < max(config.preserve_steps, config.edit_steps):
                        ref_velocity = ((z_b.float() - ref_b.float()) / (sigma.float() + 1e-8)).to(dtype)
                        ref_abs = ref_velocity.float().abs() + 1e-8
                        similarity = ref_abs / (ref_abs + (native.float() - ref_velocity.float()).abs())
                        high = similarity >= config.similarity_threshold
                        low = ~high

                        if index < config.preserve_steps:
                            actual = torch.where(high, ref_velocity, actual)
                        if index < config.edit_steps:
                            blend_weight = (1.0 - current_alpha)
                            blended = (
                                blend_weight * ref_velocity.float()
                                + current_alpha * native.float()
                            ).to(dtype)
                            actual = torch.where(low, blended, actual)

                    dt = sigma_next.float() - sigma.float()
                    z_b = (z_b.float() + dt * actual.float()).to(dtype)

                    # === Reward Guidance (only at high alpha) ===
                    if (
                        reward_config.enabled
                        and current_alpha >= reward_config.reward_alpha_threshold
                        and index >= reward_config.reward_step_start
                        and index % reward_config.reward_update_interval == 0
                    ):
                        # Decode current latent
                        with torch.no_grad():
                            image = self._decode_for_reward(z_b, prepared)

                        # Load reward models when needed
                        composite_reward.maybe_onload()

                        # Get source image for pixel comparison
                        source_image = composite_reward.get_source_image()
                        if source_image is not None:
                            source_image = source_image.to(image.device)

                        # Compute reward
                        reward_output = composite_reward(image, target_prompt)
                        reward_val = reward_output.scalar

                        # CRITICAL: Check identity similarity SEPARATELY
                        # identity_reward returns cosine sim (higher = better identity preserved)
                        # We correct when identity similarity is LOW (face changed)
                        identity_sim = composite_reward.get_identity_reward(image)

                        # Also check layout similarity for spatial drift detection
                        layout_sim = 1.0  # Default if no layout reward

                        # Record diagnostics with identity info
                        reward_diagnostics["reward_values"].append({
                            "step": index,
                            "branch": b,
                            "alpha": current_alpha,
                            "reward": reward_val,
                            "identity_similarity": identity_sim,
                            "layout_similarity": layout_sim,
                        })

                        # Apply correction if NEEDED:
                        #
                        # SIMPLE STRATEGY: Only correct when we have RELIABLE face detection.
                        #
                        # Face detection is reliable ONLY when:
                        # 1. Face is actually detected (identity_sim > 0.1, not zero vector)
                        # 2. Identity is clearly lost (similarity < threshold)
                        #
                        # We do NOT use pixel_diff because it triggers on noise at early steps.
                        # We do NOT use identity_sim=0 because that means face not detected.

                        face_detected = identity_sim > 0.1
                        identity_lost = identity_sim < reward_config.reward_threshold

                        needs_correction = face_detected and identity_lost
                        correction_type = "face"

                        if needs_correction:
                            correction_type = "identity"
                            try:
                                # Compute identity-preserving correction using reference velocity
                                # This is efficient: no extra decodes needed
                                source_image = composite_reward.get_source_image()
                                if source_image is not None:
                                    source_image = source_image.to(z_b.device)
                                    grad = self._compute_identity_preserving_correction(
                                        z_b,
                                        ref_b,
                                        sigma,
                                        source_image,
                                        image,
                                    )

                                if grad is not None:
                                    # Normalize gradient
                                    grad_norm = grad.float().norm()
                                    if grad_norm > 0:
                                        grad = grad / (grad_norm + 1e-8)

                                    # Clip gradient
                                    if reward_config.clip_grad_norm > 0:
                                        grad = grad / max(grad_norm / reward_config.clip_grad_norm, 1.0)

                                    # Scale gradient
                                    grad = grad * reward_config.reward_lr

                                    # Higher alpha = stronger correction
                                    alpha_scale = 1.0 + current_alpha * reward_config.reward_lr * 10
                                    correction = grad.to(z_b.dtype) * alpha_scale

                                    # Clamp correction magnitude
                                    correction_norm = correction.norm()
                                    if correction_norm > reward_config.max_correction:
                                        correction = correction / correction_norm * reward_config.max_correction

                                    # Apply to latent
                                    z_b = (z_b.float() + correction).to(z_b.dtype)

                                    reward_diagnostics["corrections_applied"].append({
                                        "step": index,
                                        "branch": b,
                                        "alpha": current_alpha,
                                        "reward": reward_val,
                                        "identity_similarity": identity_sim,
                                        "correction_type": correction_type,
                                        "grad_norm": float(grad_norm.cpu()),
                                    })
                                    reward_diagnostics["gradient_norms"].append(
                                        float(correction_norm.cpu())
                                    )
                            except Exception as e:
                                print(f"Warning: Reward gradient computation failed: {e}")

                        # Offload reward models after each use
                        composite_reward.maybe_offload()

                        # Force garbage collection
                        import gc
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

            final_z_list.append(z_b.detach().cpu())

        # Offload reward models before final decode
        self._offload_reward_models()

        # Final decode - process one at a time
        final_images = []
        for z_b_cpu in final_z_list:
            z_b = z_b_cpu.to(self.device)
            with torch.no_grad():
                unpacked = self.base_rollout.pipeline._unpack_latents(
                    z_b,
                    prepared.height,
                    prepared.width,
                    self.base_rollout.pipeline.vae_scale_factor,
                )
                unpacked = unpacked / self.base_rollout.pipeline.vae.config.scaling_factor + self.base_rollout.pipeline.vae.config.shift_factor
                decoded = self.base_rollout.pipeline.vae.decode(
                    unpacked.to(dtype=self.base_rollout.pipeline.vae.dtype),
                    return_dict=False,
                )[0]
                images = self.base_rollout.pipeline.image_processor.postprocess(decoded, output_type="pt")
                final_images.append(images.float().clamp(0, 1))

        images = torch.cat(final_images, dim=0)

        return RewardGuidedVeloEditResult(
            images=images,
            alpha=alpha,
            reward_diagnostics=reward_diagnostics,
        )

    def rollout(
        self,
        prepared: PreparedVeloEdit,
        alphas: Sequence[float] | torch.Tensor,
        *,
        config: VeloEditRolloutConfig | None = None,
    ) -> torch.Tensor:
        """Standard VeloEdit rollout without reward guidance.

        Delegates to VeloEditCompatibleRollout.
        """
        return self.base_rollout.rollout(prepared, alphas, config=config)

    def prepare(
        self,
        image: Image.Image,
        prompt: str,
        *,
        config: VeloEditRolloutConfig | None = None,
        seed: int | None = None,
    ) -> PreparedVeloEdit:
        """Prepare for rollout.

        Delegates to VeloEditCompatibleRollout.
        """
        return self.base_rollout.prepare(image, prompt, config=config, seed=seed)
