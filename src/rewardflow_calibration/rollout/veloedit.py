"""Deterministic VeloEdit-compatible FLUX-Kontext rollout."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.checkpoint import checkpoint

from diffusers import FluxKontextPipeline
from diffusers.pipelines.flux.pipeline_flux_kontext import calculate_shift, retrieve_timesteps


@dataclass(frozen=True)
class VeloEditRolloutConfig:
    steps: int = 15
    seed: int = 42
    guidance_scale: float = 2.5
    first_step_align_steps: int = 4
    preserve_steps: int = 4
    edit_steps: int = 4
    similarity_threshold: float = 0.8
    max_area: int = 1024 * 1024

    def validate(self) -> None:
        if self.steps < 1:
            raise ValueError("steps must be positive")
        if self.first_step_align_steps < 0:
            raise ValueError("first_step_align_steps must be non-negative")
        if self.preserve_steps < 0 or self.edit_steps < 0:
            raise ValueError("intervention steps must be non-negative")
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in [0, 1]")


@dataclass
class PreparedVeloEdit:
    working_image: Image.Image
    original_size: tuple[int, int]
    height: int
    width: int
    latents: torch.Tensor
    reference_latent: torch.Tensor
    prompt_embeds: torch.Tensor
    pooled_prompt_embeds: torch.Tensor
    text_ids: torch.Tensor
    latent_ids: torch.Tensor
    image_latents: torch.Tensor | None
    guidance: torch.Tensor | None
    sigma_schedule: torch.Tensor


def align_first_step_to_reference_step(
    sigma_schedule: torch.Tensor,
    reference_schedule: torch.Tensor,
    reference_steps: int,
    atol: float = 1e-6,
) -> torch.Tensor:
    """Use the same first-transition replacement as VeloEdit."""
    requested_steps = len(sigma_schedule) - 1
    if (
        reference_steps <= 0
        or requested_steps <= reference_steps
        or len(sigma_schedule) < 2
        or len(reference_schedule) < 2
    ):
        return sigma_schedule
    current = sigma_schedule.detach().float()
    reference = reference_schedule.detach().float()
    target = current[0] - (reference[0] - reference[1])
    if target >= current[0] or target <= current[-1]:
        return sigma_schedule
    tail = current[1:] < (target - atol)
    return torch.cat(
        [sigma_schedule[:1], target.to(sigma_schedule).reshape(1), sigma_schedule[1:][tail]],
        dim=0,
    )


class VeloEditCompatibleRollout:
    """Prepare FLUX-Kontext once and roll out arbitrary alpha branches."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str | torch.device = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        local_files_only: bool = True,
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.pipeline = FluxKontextPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            local_files_only=local_files_only,
        )
        self.pipeline.to(self.device)
        self.pipeline.set_progress_bar_config(disable=True)

    def prepare(
        self,
        image: Image.Image,
        prompt: str,
        *,
        config: VeloEditRolloutConfig | None = None,
        seed: int | None = None,
    ) -> PreparedVeloEdit:
        config = config or VeloEditRolloutConfig()
        config.validate()
        if seed is None:
            seed = config.seed

        original_width, original_height = image.size
        scale = (config.max_area / (original_height * original_width)) ** 0.5
        width = max(16, round(original_width * scale))
        height = max(16, round(original_height * scale))
        multiple = self.pipeline.vae_scale_factor * 2
        width = width // multiple * multiple
        height = height // multiple * multiple
        working_image = self.pipeline.image_processor.resize(image, height, width)

        prompt_embeds, pooled_prompt_embeds, text_ids = self.pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            prompt_embeds=None,
            pooled_prompt_embeds=None,
            device=self.device,
            num_images_per_prompt=1,
            max_sequence_length=512,
            lora_scale=None,
        )
        processed = self.pipeline.image_processor.preprocess(working_image, height, width)
        channels = self.pipeline.transformer.config.in_channels // 4
        processed = processed.to(device=self.device, dtype=prompt_embeds.dtype)
        generator = torch.Generator(device="cpu").manual_seed(int(seed))

        with torch.no_grad():
            latents, image_latents, latent_ids, image_ids = self.pipeline.prepare_latents(
                processed,
                1,
                channels,
                height,
                width,
                prompt_embeds.dtype,
                self.device,
                generator,
                None,
            )
        if image_ids is not None:
            latent_ids = torch.cat([latent_ids, image_ids], dim=0)
        transformer_dtype = self.pipeline.transformer.dtype
        latents = latents.to(dtype=transformer_dtype)
        if image_latents is not None:
            image_latents = image_latents.to(dtype=transformer_dtype)
        reference_latent = image_latents.clone() if image_latents is not None else latents.clone()

        raw_sigmas = np.linspace(1.0, 1 / config.steps, config.steps)
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.pipeline.scheduler.config.get("base_image_seq_len", 256),
            self.pipeline.scheduler.config.get("max_image_seq_len", 4096),
            self.pipeline.scheduler.config.get("base_shift", 0.5),
            self.pipeline.scheduler.config.get("max_shift", 1.15),
        )
        retrieve_timesteps(self.pipeline.scheduler, config.steps, self.device, sigmas=raw_sigmas, mu=mu)
        sigma_schedule = self.pipeline.scheduler.sigmas.float().clone()
        if config.first_step_align_steps > 0 and config.steps > config.first_step_align_steps:
            reference_sigmas = np.linspace(1.0, 1 / config.first_step_align_steps, config.first_step_align_steps)
            retrieve_timesteps(
                self.pipeline.scheduler,
                config.first_step_align_steps,
                self.device,
                sigmas=reference_sigmas,
                mu=mu,
            )
            reference_schedule = self.pipeline.scheduler.sigmas.float().clone()
            sigma_schedule = align_first_step_to_reference_step(
                sigma_schedule,
                reference_schedule,
                config.first_step_align_steps,
            )

        if self.pipeline.transformer.config.guidance_embeds:
            guidance = torch.full(
                [1], config.guidance_scale, device=self.device, dtype=torch.float32
            )
        else:
            guidance = None
        return PreparedVeloEdit(
            working_image=working_image,
            original_size=(original_width, original_height),
            height=height,
            width=width,
            latents=latents,
            reference_latent=reference_latent,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            text_ids=text_ids,
            latent_ids=latent_ids,
            image_latents=image_latents,
            guidance=guidance,
            sigma_schedule=sigma_schedule,
        )

    def rollout(
        self,
        prepared: PreparedVeloEdit,
        alphas: Sequence[float] | torch.Tensor,
        *,
        config: VeloEditRolloutConfig | None = None,
        goal_residual: torch.Tensor | None = None,
        early_stop_steps: int | None = None,
        velocity_trace: list[dict[str, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        """Return decoded images, optionally adding early ``V_goal`` velocities.

        A residual has shape ``[goal_steps, latent_tokens, latent_channels]``.
        It is added after the existing VeloEdit alpha intervention. Passing a
        residual enables gradients through the frozen transformer, remaining
        rollout steps, and VAE decoder; transformer and VAE forwards are
        activation-checkpointed to reduce memory use. ``early_stop_steps`` is
        an optional proxy path: after that many denoising steps, decode the
        rectified-flow clean-latent estimate ``z_t - sigma_t * v_t`` instead
        of completing the trajectory. The default full-rollout path is
        unchanged.
        """
        config = config or VeloEditRolloutConfig()
        config.validate()
        alpha = torch.as_tensor(alphas, device=self.device, dtype=torch.float32).flatten()
        if alpha.numel() < 1 or not torch.isfinite(alpha).all() or torch.any((alpha < 0) | (alpha > 1)):
            raise ValueError("alphas must be finite values in [0, 1]")
        total_steps = len(prepared.sigma_schedule) - 1
        if early_stop_steps is not None and not 1 <= early_stop_steps <= total_steps:
            raise ValueError(f"early_stop_steps must be in [1, {total_steps}]")

        branches = alpha.numel()
        dtype = self.pipeline.transformer.dtype
        z = prepared.latents.expand(branches, -1, -1).clone()
        reference = prepared.reference_latent.expand(branches, -1, -1)
        image_latents = None if prepared.image_latents is None else prepared.image_latents.expand(branches, -1, -1)
        prompt_embeds = prepared.prompt_embeds.expand(branches, -1, -1)
        pooled = prepared.pooled_prompt_embeds.expand(branches, -1)
        guidance = None if prepared.guidance is None else prepared.guidance.expand(branches)
        train_steps = self.pipeline.scheduler.config.get("num_train_timesteps", 1000)
        decode_latents = z

        if goal_residual is not None:
            if goal_residual.ndim != 3 or goal_residual.shape[1:] != prepared.latents.shape[1:]:
                raise ValueError(
                    "goal_residual must have shape [goal_steps, latent_tokens, latent_channels]"
                )
            if goal_residual.shape[0] > len(prepared.sigma_schedule) - 1:
                raise ValueError("goal_residual has more steps than the rollout")
            if early_stop_steps is not None and goal_residual.shape[0] > early_stop_steps:
                raise ValueError("early_stop_steps must cover every goal_residual step")
            if branches != 1:
                raise ValueError("goal_residual optimization supports one fixed-alpha branch")

        grad_enabled = goal_residual is not None and torch.is_grad_enabled()
        with torch.set_grad_enabled(grad_enabled):
            for index in range(len(prepared.sigma_schedule) - 1):
                sigma = prepared.sigma_schedule[index]
                sigma_next = prepared.sigma_schedule[index + 1]
                model_input = z if image_latents is None else torch.cat([z, image_latents], dim=1)
                sigma_input = torch.as_tensor(sigma, device=self.device, dtype=torch.float32)
                timestep = (sigma_input * train_steps).expand(branches).to(dtype=z.dtype) / train_steps
                def predict_velocity(hidden_states: torch.Tensor) -> torch.Tensor:
                    return self.pipeline.transformer(
                        hidden_states=hidden_states,
                        timestep=timestep,
                        guidance=guidance,
                        pooled_projections=pooled,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=prepared.text_ids,
                        img_ids=prepared.latent_ids,
                        joint_attention_kwargs={},
                        return_dict=False,
                    )[0][:, : z.shape[1]]

                if grad_enabled:
                    native = checkpoint(predict_velocity, model_input, use_reentrant=False)
                else:
                    native = predict_velocity(model_input)

                actual = native
                edit_direction = None
                if index < max(config.preserve_steps, config.edit_steps):
                    ref_velocity = ((z.float() - reference.float()) / (sigma.float() + 1e-8)).to(dtype)
                    ref_abs = ref_velocity.float().abs() + 1e-8
                    similarity = ref_abs / (ref_abs + (native.float() - ref_velocity.float()).abs())
                    high = similarity >= config.similarity_threshold
                    low = ~high
                    if index < config.preserve_steps:
                        actual = torch.where(high, ref_velocity, actual)
                    if index < config.edit_steps:
                        blend_weight = (1.0 - alpha).view(branches, 1, 1)
                        blended = (blend_weight * ref_velocity.float() + alpha.view(branches, 1, 1) * native.float()).to(dtype)
                        actual = torch.where(low, blended, actual)
                        edit_direction = (
                            (native.float() - ref_velocity.float()) * low.to(dtype=torch.float32)
                        ).detach()

                if velocity_trace is not None and edit_direction is not None:
                    velocity_trace.append({
                        "step": torch.tensor(index),
                        "edit_direction": edit_direction,
                    })

                if goal_residual is not None and index < goal_residual.shape[0]:
                    actual = actual.float() + goal_residual[index].unsqueeze(0).float()

                if early_stop_steps is not None and index + 1 == early_stop_steps:
                    # In this sigma convention, z_sigma = z_clean + sigma*v.
                    # Use the early clean-state prediction as a differentiable
                    # proxy, not as a replacement for the final rollout.
                    decode_latents = (z.float() - sigma.float() * actual.float()).to(dtype)
                    break

                dt = sigma_next.float() - sigma.float()
                z = (z.float() + dt * actual.float()).to(dtype)

            if early_stop_steps is None:
                decode_latents = z
            unpacked = self.pipeline._unpack_latents(decode_latents, prepared.height, prepared.width, self.pipeline.vae_scale_factor)
            unpacked = unpacked / self.pipeline.vae.config.scaling_factor + self.pipeline.vae.config.shift_factor
            if grad_enabled:
                decoded = checkpoint(
                    lambda value: self.pipeline.vae.decode(
                        value.to(dtype=self.pipeline.vae.dtype), return_dict=False
                    )[0],
                    unpacked,
                    use_reentrant=False,
                )
            else:
                decoded = self.pipeline.vae.decode(unpacked.to(dtype=self.pipeline.vae.dtype), return_dict=False)[0]
            images = self.pipeline.image_processor.postprocess(decoded, output_type="pt")
            return images.float().clamp(0, 1)
