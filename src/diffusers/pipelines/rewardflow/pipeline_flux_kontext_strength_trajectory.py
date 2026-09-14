"""Coupled strength trajectories on the repository's native FLUX.1-Kontext pipeline."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch

from ...image_processor import PipelineImageInput
from ...utils import is_torch_xla_available, logging
from ..flux.pipeline_flux_kontext import (
    PREFERRED_KONTEXT_RESOLUTIONS,
    FluxKontextPipeline,
    calculate_shift,
    retrieve_timesteps,
)
from .paper_components import (
    flow_step_size,
    freeze_module_parameters,
    paper_euler_update,
    paper_gamma_schedule,
    predict_clean_latent,
)
from .pipeline_output import StrengthTrajectoryPipelineOutput
from .strength_trajectory import (
    StrengthBranchLayout,
    StrengthRewardContext,
    StrengthRewardFn,
    StrengthRewardGuidance,
    StrengthTrajectoryConfig,
    expand_for_strengths,
    expand_shared_initial_latents,
    is_strength_reward_step,
    make_flat_strength_tensor,
    make_strength_branch_layout,
    max_shared_noise_difference,
    sample_base_langevin_noise,
    unflatten_strength_branches,
    validate_strength_reward_context,
    validate_trajectory_mode,
)


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class FluxKontextStrengthTrajectoryPipeline(FluxKontextPipeline):
    """Opt-in coupled-strength research sampler over native Kontext conditioning.

    When trajectory mode is disabled, ``__call__`` delegates directly to
    :class:`FluxKontextPipeline`. The enabled path preserves the official
    prompt, image, guidance, true-CFG, and IP-Adapter preparation semantics.
    """

    def _freeze_kontext_trajectory_modules(self, strength_reward: StrengthRewardFn | None = None) -> int:
        """Freeze inference weights without disabling gradients for latent inputs."""

        modules = [
            self.transformer,
            self.vae,
            self.text_encoder,
            self.text_encoder_2,
            getattr(self, "image_encoder", None),
            strength_reward,
        ]
        return sum(freeze_module_parameters(module) for module in modules)

    def _prepare_strength_reward_guidance(
        self,
        strength_reward: StrengthRewardFn | None,
        strength_reward_context: StrengthRewardContext,
        *,
        lambda_strength_reward: float,
        device: torch.device,
        base_batch_size: int,
        num_strengths: int,
    ) -> StrengthRewardGuidance | None:
        """Prepare fixed endpoint context only after all inference modules are frozen."""

        if strength_reward is None or lambda_strength_reward <= 0:
            return None
        guidance = StrengthRewardGuidance(strength_reward)
        guidance.prepare_context(
            strength_reward_context,
            device=device,
            base_batch_size=base_batch_size,
            num_strengths=num_strengths,
        )
        return guidance

    @staticmethod
    def _expand_ip_adapter_embeds(
        image_embeds: list[torch.Tensor] | None, num_strengths: int
    ) -> list[torch.Tensor] | None:
        """Expand official IP-Adapter outputs, whose first dimension is batch."""

        if image_embeds is None:
            return None
        return [expand_for_strengths(embed, num_strengths) for embed in image_embeds]

    @staticmethod
    def _materialize_kontext_strength_batch(
        num_strengths: int,
        latents: torch.Tensor,
        image_latents: torch.Tensor | None,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        guidance: torch.Tensor | None,
        negative_prompt_embeds: torch.Tensor | None,
        negative_pooled_prompt_embeds: torch.Tensor | None,
        image_embeds: list[torch.Tensor] | None,
        negative_image_embeds: list[torch.Tensor] | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        list[torch.Tensor] | None,
        list[torch.Tensor] | None,
    ]:
        """Materialize only Kontext tensors with a documented batch-first dimension.

        Official local shape contract:
        - sampling/source latents: ``[B, image_tokens, channels]``;
        - T5 prompt embeddings: ``[B, text_tokens, channels]``;
        - CLIP pooled embeddings: ``[B, channels]``;
        - guidance: ``[B]``;
        - IP-Adapter entries: batch-first tensors.

        ``text_ids`` and the combined sampling/source ``image_ids`` are shared
        ``[sequence, 3]`` tensors in the official pipeline and intentionally do
        not appear here: expanding them would change Kontext position semantics.
        """

        return (
            expand_shared_initial_latents(latents, num_strengths),
            expand_for_strengths(image_latents, num_strengths),
            expand_for_strengths(prompt_embeds, num_strengths),
            expand_for_strengths(pooled_prompt_embeds, num_strengths),
            expand_for_strengths(guidance, num_strengths),
            expand_for_strengths(negative_prompt_embeds, num_strengths),
            expand_for_strengths(negative_pooled_prompt_embeds, num_strengths),
            FluxKontextStrengthTrajectoryPipeline._expand_ip_adapter_embeds(image_embeds, num_strengths),
            FluxKontextStrengthTrajectoryPipeline._expand_ip_adapter_embeds(negative_image_embeds, num_strengths),
        )

    def _predict_kontext_velocity(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        *,
        image_latents: torch.Tensor | None,
        image_ids: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        text_ids: torch.Tensor,
        guidance: torch.Tensor | None,
        do_true_cfg: bool,
        true_cfg_scale: float,
        negative_prompt_embeds: torch.Tensor | None,
        negative_pooled_prompt_embeds: torch.Tensor | None,
        negative_text_ids: torch.Tensor | None,
        image_embeds: list[torch.Tensor] | None,
        negative_image_embeds: list[torch.Tensor] | None,
    ) -> torch.Tensor:
        """Match the local official Kontext Transformer forward exactly."""

        if image_embeds is not None:
            self._joint_attention_kwargs["ip_adapter_image_embeds"] = image_embeds

        # Source tokens are fixed conditioning, concatenated after sampling
        # tokens. Only the sampling prefix is returned as velocity.
        latent_model_input = latents
        if image_latents is not None:
            latent_model_input = torch.cat([latents, image_latents], dim=1)
        expanded_timestep = timestep.expand(latents.shape[0]).to(latents.dtype)
        noise_pred = self.transformer(
            hidden_states=latent_model_input,
            timestep=expanded_timestep / 1000,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=image_ids,
            joint_attention_kwargs=self.joint_attention_kwargs,
            return_dict=False,
        )[0]
        noise_pred = noise_pred[:, : latents.size(1)]

        if do_true_cfg:
            if negative_prompt_embeds is None or negative_pooled_prompt_embeds is None or negative_text_ids is None:
                raise RuntimeError("True CFG requires complete negative text conditioning.")
            if negative_image_embeds is not None:
                self._joint_attention_kwargs["ip_adapter_image_embeds"] = negative_image_embeds
            neg_noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=expanded_timestep / 1000,
                guidance=guidance,
                pooled_projections=negative_pooled_prompt_embeds,
                encoder_hidden_states=negative_prompt_embeds,
                txt_ids=negative_text_ids,
                img_ids=image_ids,
                joint_attention_kwargs=self.joint_attention_kwargs,
                return_dict=False,
            )[0]
            neg_noise_pred = neg_noise_pred[:, : latents.size(1)]
            noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

        return noise_pred

    def _decode_kontext_clean_latent_for_reward(
        self, clean_latent: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        """Decode packed Kontext clean latents to differentiable ``[N,3,H,W]`` images in ``[0,1]``."""

        unpacked = self._unpack_latents(clean_latent, height, width, self.vae_scale_factor)
        unpacked = unpacked / self.vae.config.scaling_factor + self.vae.config.shift_factor
        decoded = self.vae.decode(unpacked, return_dict=False)[0]
        # VaeImageProcessor's tensor postprocess uses this same denormalization,
        # but the reward path stays explicitly torch-native to preserve autograd.
        return (decoded / 2 + 0.5).clamp(0, 1)

    @staticmethod
    def _select_batch_tensor(tensor: torch.Tensor | None, indices: torch.Tensor) -> torch.Tensor | None:
        return None if tensor is None else tensor.index_select(0, indices)

    @staticmethod
    def _select_ip_adapter_embeds(
        image_embeds: list[torch.Tensor] | None, indices: torch.Tensor
    ) -> list[torch.Tensor] | None:
        if image_embeds is None:
            return None
        return [embed.index_select(0, indices) for embed in image_embeds]

    def _kontext_strength_trajectory_chunk(
        self,
        *,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        image_latents: torch.Tensor | None,
        image_ids: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        text_ids: torch.Tensor,
        guidance: torch.Tensor | None,
        do_true_cfg: bool,
        true_cfg_scale: float,
        negative_prompt_embeds: torch.Tensor | None,
        negative_pooled_prompt_embeds: torch.Tensor | None,
        negative_text_ids: torch.Tensor | None,
        image_embeds: list[torch.Tensor] | None,
        negative_image_embeds: list[torch.Tensor] | None,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        langevin_noise: torch.Tensor,
        trajectory_config: StrengthTrajectoryConfig,
        target_strengths: torch.Tensor | None,
        strength_reward_guidance: StrengthRewardGuidance | None,
        strength_reward_context: StrengthRewardContext,
        branch_layout: StrengthBranchLayout | None,
        reward_active: bool,
        height: int,
        width: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Advance one Kontext branch chunk, enabling autograd only for reward-active work."""

        forward_kwargs = {
            "image_latents": image_latents,
            "image_ids": image_ids,
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "text_ids": text_ids,
            "guidance": guidance,
            "do_true_cfg": do_true_cfg,
            "true_cfg_scale": true_cfg_scale,
            "negative_prompt_embeds": negative_prompt_embeds,
            "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds,
            "negative_text_ids": negative_text_ids,
            "image_embeds": image_embeds,
            "negative_image_embeds": negative_image_embeds,
        }
        if not reward_active:
            with torch.no_grad():
                velocity = self._predict_kontext_velocity(latents, timestep, **forward_kwargs)
                updated = paper_euler_update(
                    latents,
                    velocity,
                    sigma,
                    sigma_next,
                    langevin_noise=langevin_noise,
                )
            zero_norms = torch.zeros(latents.shape[0], device=latents.device, dtype=torch.float32)
            return updated, None, zero_norms

        if strength_reward_guidance is None or target_strengths is None or branch_layout is None:
            raise RuntimeError("An active strength reward step requires guidance, targets, and branch layout.")

        latent_var = latents.detach().requires_grad_(True)
        with torch.enable_grad():
            velocity = self._predict_kontext_velocity(latent_var, timestep, **forward_kwargs)
            clean_pred = predict_clean_latent(latent_var, velocity, sigma)
            clean_image = self._decode_kontext_clean_latent_for_reward(clean_pred, height, width)
            branch_rewards = strength_reward_guidance.compute(
                image=clean_image,
                target_strength=target_strengths,
                context=strength_reward_context,
                layout=branch_layout,
            )
            if not branch_rewards.requires_grad:
                raise RuntimeError("Strength reward is not differentiable with respect to the clean image.")
            reward_grad = torch.autograd.grad(branch_rewards.sum(), latent_var, allow_unused=True)[0]
            if reward_grad is None:
                raise RuntimeError(
                    "Strength reward gradient did not reach the current latent through Kontext and the VAE."
                )
            if not torch.isfinite(reward_grad).all():
                raise RuntimeError("Strength reward gradient contains non-finite values.")
            reward_drift = trajectory_config.lambda_strength_reward * reward_grad

        updated = paper_euler_update(
            latent_var.detach(),
            velocity.detach(),
            sigma,
            sigma_next,
            reward_drift=reward_drift.detach(),
            langevin_noise=langevin_noise,
        )
        reward_grad_norms = torch.linalg.vector_norm(reward_grad.detach().float().flatten(start_dim=1), dim=1)
        return updated, branch_rewards.detach(), reward_grad_norms

    def _kontext_strength_trajectory_step(
        self,
        *,
        latents: torch.Tensor,
        step_index: int,
        timestep: torch.Tensor,
        image_latents: torch.Tensor | None,
        image_ids: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        text_ids: torch.Tensor,
        guidance: torch.Tensor | None,
        do_true_cfg: bool,
        true_cfg_scale: float,
        negative_prompt_embeds: torch.Tensor | None,
        negative_pooled_prompt_embeds: torch.Tensor | None,
        negative_text_ids: torch.Tensor | None,
        image_embeds: list[torch.Tensor] | None,
        negative_image_embeds: list[torch.Tensor] | None,
        trajectory_config: StrengthTrajectoryConfig,
        target_strengths: torch.Tensor,
        strength_reward_guidance: StrengthRewardGuidance | None,
        strength_reward_context: StrengthRewardContext,
        base_batch_size: int,
        num_strengths: int,
        generator: torch.Generator | list[torch.Generator] | None,
        reward_active: bool,
        branches_materialized: bool,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Run one exact Kontext step with one base-level RNG draw outside all chunks."""

        if step_index + 1 >= len(self.scheduler.sigmas):
            raise IndexError("Trajectory sampling requires both sigma[i] and sigma[i + 1].")
        expected_batch = base_batch_size * num_strengths if branches_materialized else base_batch_size
        if latents.shape[0] != expected_batch:
            raise ValueError(f"Expected effective trajectory batch {expected_batch}, got {latents.shape[0]}.")
        if reward_active and not branches_materialized:
            raise RuntimeError("Strength branches must be materialized before a reward-active step.")

        sigma = self.scheduler.sigmas[step_index].to(device=latents.device)
        sigma_next = self.scheduler.sigmas[step_index + 1].to(device=latents.device)
        eta = flow_step_size(sigma, sigma_next, latents.float())
        gamma = torch.zeros((), device=latents.device, dtype=latents.dtype)
        base_noise = torch.zeros((base_batch_size, *latents.shape[1:]), device=latents.device, dtype=latents.dtype)
        if trajectory_config.use_shared_sde_noise:
            sigma_start = self.scheduler.sigmas[0].to(device=latents.device)
            gamma = paper_gamma_schedule(
                sigma,
                sigma_start,
                gamma_min=trajectory_config.gamma_min,
                gamma_max=trajectory_config.gamma_max,
                rho=trajectory_config.gamma_rho,
            )
            base_noise = sample_base_langevin_noise(
                latents,
                gamma,
                eta,
                base_batch_size=base_batch_size,
                generator=generator,
            )

        effective_batch = latents.shape[0]
        flat_indices = torch.arange(effective_batch, device=latents.device)
        base_indices = (
            torch.div(flat_indices, num_strengths, rounding_mode="floor") if branches_materialized else flat_indices
        )
        chunk_size = trajectory_config.branch_chunk_size or effective_batch
        updated_chunks = []
        reward_chunks = []
        reward_grad_norm_chunks = []

        for start in range(0, effective_batch, chunk_size):
            chunk_indices = flat_indices[start : start + chunk_size]
            chunk_base_indices = base_indices[start : start + chunk_size]
            layout = (
                make_strength_branch_layout(chunk_indices, base_batch_size, num_strengths)
                if branches_materialized
                else None
            )
            updated_chunk, reward_chunk, reward_grad_norm_chunk = self._kontext_strength_trajectory_chunk(
                latents=latents.index_select(0, chunk_indices),
                timestep=timestep,
                image_latents=self._select_batch_tensor(image_latents, chunk_indices),
                image_ids=image_ids,
                prompt_embeds=prompt_embeds.index_select(0, chunk_indices),
                pooled_prompt_embeds=pooled_prompt_embeds.index_select(0, chunk_indices),
                text_ids=text_ids,
                guidance=self._select_batch_tensor(guidance, chunk_indices),
                do_true_cfg=do_true_cfg,
                true_cfg_scale=true_cfg_scale,
                negative_prompt_embeds=self._select_batch_tensor(negative_prompt_embeds, chunk_indices),
                negative_pooled_prompt_embeds=self._select_batch_tensor(negative_pooled_prompt_embeds, chunk_indices),
                negative_text_ids=negative_text_ids,
                image_embeds=self._select_ip_adapter_embeds(image_embeds, chunk_indices),
                negative_image_embeds=self._select_ip_adapter_embeds(negative_image_embeds, chunk_indices),
                sigma=sigma,
                sigma_next=sigma_next,
                langevin_noise=base_noise.index_select(0, chunk_base_indices),
                trajectory_config=trajectory_config,
                target_strengths=(target_strengths.index_select(0, chunk_indices) if reward_active else None),
                strength_reward_guidance=strength_reward_guidance,
                strength_reward_context=strength_reward_context,
                branch_layout=layout,
                reward_active=reward_active,
                height=height,
                width=width,
            )
            updated_chunks.append(updated_chunk)
            reward_grad_norm_chunks.append(reward_grad_norm_chunk)
            if reward_chunk is not None:
                reward_chunks.append(reward_chunk)

        updated = torch.cat(updated_chunks, dim=0)
        branch_rewards = torch.cat(reward_chunks, dim=0) if reward_chunks else None
        reward_grad_norms = torch.cat(reward_grad_norm_chunks, dim=0)

        if trajectory_config.collect_trace:
            if branches_materialized:
                grouped_latents = unflatten_strength_branches(updated.detach(), base_batch_size, num_strengths)
                branch_latent_norms = torch.linalg.vector_norm(grouped_latents.float().flatten(start_dim=2), dim=2)
                branch_reward_grad_norms = reward_grad_norms.reshape(base_batch_size, num_strengths)
            else:
                base_latent_norms = torch.linalg.vector_norm(updated.detach().float().flatten(start_dim=1), dim=1)
                branch_latent_norms = base_latent_norms[:, None].expand(-1, num_strengths)
                branch_reward_grad_norms = torch.zeros(
                    (base_batch_size, num_strengths), device=latents.device, dtype=torch.float32
                )
            virtual_shared_noise = base_noise.repeat_interleave(num_strengths, dim=0)
            self.last_strength_trajectory_trace.append(
                {
                    "step": step_index,
                    "timestep": float(timestep.detach().cpu()),
                    "sigma": float(sigma.detach().cpu()),
                    "sigma_next": float(sigma_next.detach().cpu()),
                    "eta": float((sigma - sigma_next).detach().cpu()),
                    "gamma": float(gamma.detach().cpu()),
                    "strengths": list(trajectory_config.strengths),
                    "reward_active": reward_active,
                    "branches_materialized": branches_materialized,
                    "effective_model_batch": effective_batch,
                    "branch_chunk_size": trajectory_config.branch_chunk_size,
                    "num_chunks": len(updated_chunks),
                    "branch_rewards": (
                        None
                        if branch_rewards is None
                        else unflatten_strength_branches(branch_rewards, base_batch_size, num_strengths).cpu().tolist()
                    ),
                    "branch_reward_grad_norms": branch_reward_grad_norms.cpu().tolist(),
                    "branch_latent_norms": branch_latent_norms.cpu().tolist(),
                    "shared_noise_norm": float(torch.linalg.vector_norm(virtual_shared_noise.float()).cpu()),
                    "max_shared_noise_difference": float(
                        max_shared_noise_difference(virtual_shared_noise, base_batch_size, num_strengths).cpu()
                    ),
                }
            )
        return updated

    def __call__(
        self,
        image: PipelineImageInput | None = None,
        prompt: str | list[str] = None,
        prompt_2: str | list[str] | None = None,
        negative_prompt: str | list[str] = None,
        negative_prompt_2: str | list[str] | None = None,
        true_cfg_scale: float = 1.0,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 28,
        sigmas: list[float] | None = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: int | None = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.FloatTensor | None = None,
        prompt_embeds: torch.FloatTensor | None = None,
        pooled_prompt_embeds: torch.FloatTensor | None = None,
        ip_adapter_image: PipelineImageInput | None = None,
        ip_adapter_image_embeds: list[torch.Tensor] | None = None,
        negative_ip_adapter_image: PipelineImageInput | None = None,
        negative_ip_adapter_image_embeds: list[torch.Tensor] | None = None,
        negative_prompt_embeds: torch.FloatTensor | None = None,
        negative_pooled_prompt_embeds: torch.FloatTensor | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end: Callable[[int, int], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        max_sequence_length: int = 512,
        max_area: int = 1024**2,
        _auto_resize: bool = True,
        trajectory_config: StrengthTrajectoryConfig | dict[str, Any] | None = None,
        strength_reward: StrengthRewardFn | None = None,
        strength_reward_context: StrengthRewardContext | dict[str, Any] | None = None,
    ):
        """Run native Kontext or the explicitly enabled coupled trajectory path."""

        if trajectory_config is None:
            return super().__call__(
                image=image,
                prompt=prompt,
                prompt_2=prompt_2,
                negative_prompt=negative_prompt,
                negative_prompt_2=negative_prompt_2,
                true_cfg_scale=true_cfg_scale,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                sigmas=sigmas,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images_per_prompt,
                generator=generator,
                latents=latents,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                ip_adapter_image=ip_adapter_image,
                ip_adapter_image_embeds=ip_adapter_image_embeds,
                negative_ip_adapter_image=negative_ip_adapter_image,
                negative_ip_adapter_image_embeds=negative_ip_adapter_image_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                output_type=output_type,
                return_dict=return_dict,
                joint_attention_kwargs=joint_attention_kwargs,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                max_sequence_length=max_sequence_length,
                max_area=max_area,
                _auto_resize=_auto_resize,
            )
        if isinstance(trajectory_config, dict):
            trajectory_config = StrengthTrajectoryConfig(**trajectory_config)
        elif not isinstance(trajectory_config, StrengthTrajectoryConfig):
            raise TypeError("`trajectory_config` must be a StrengthTrajectoryConfig, dict, or None.")
        if not trajectory_config.enabled:
            return super().__call__(
                image=image,
                prompt=prompt,
                prompt_2=prompt_2,
                negative_prompt=negative_prompt,
                negative_prompt_2=negative_prompt_2,
                true_cfg_scale=true_cfg_scale,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                sigmas=sigmas,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images_per_prompt,
                generator=generator,
                latents=latents,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                ip_adapter_image=ip_adapter_image,
                ip_adapter_image_embeds=ip_adapter_image_embeds,
                negative_ip_adapter_image=negative_ip_adapter_image,
                negative_ip_adapter_image_embeds=negative_ip_adapter_image_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                output_type=output_type,
                return_dict=return_dict,
                joint_attention_kwargs=joint_attention_kwargs,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                max_sequence_length=max_sequence_length,
                max_area=max_area,
                _auto_resize=_auto_resize,
            )

        validate_trajectory_mode(
            trajectory_config,
            paper_enabled=False,
            legacy_reward_guidance=False,
            strength_reward=strength_reward,
        )
        if bool(getattr(self.scheduler.config, "stochastic_sampling", False)):
            raise ValueError(
                "Kontext trajectory mode requires deterministic FlowMatch Euler scheduler semantics; "
                "scheduler.config.stochastic_sampling=True is not equivalent to the explicit trajectory update."
            )
        context_was_provided = strength_reward_context is not None
        if strength_reward_context is None:
            strength_reward_context = StrengthRewardContext(prompt=prompt)
        elif isinstance(strength_reward_context, dict):
            strength_reward_context = StrengthRewardContext(**strength_reward_context)
        elif not isinstance(strength_reward_context, StrengthRewardContext):
            raise TypeError("`strength_reward_context` must be a StrengthRewardContext, dict, or None.")

        # Freeze all model and reward parameters before any fixed endpoint
        # context can be encoded. This still permits gradients with respect to
        # reward-active current latents.
        self._freeze_kontext_trajectory_modules(strength_reward)
        self.last_strength_trajectory_trace = []

        with torch.no_grad():
            height = height or self.default_sample_size * self.vae_scale_factor
            width = width or self.default_sample_size * self.vae_scale_factor
            original_height, original_width = height, width
            aspect_ratio = width / height
            width = round((max_area * aspect_ratio) ** 0.5)
            height = round((max_area / aspect_ratio) ** 0.5)
            multiple_of = self.vae_scale_factor * 2
            width = width // multiple_of * multiple_of
            height = height // multiple_of * multiple_of
            if height != original_height or width != original_width:
                logger.warning(
                    f"Generation `height` and `width` have been adjusted to {height} and {width} to fit the model requirements."
                )

            self.check_inputs(
                prompt,
                prompt_2,
                height,
                width,
                negative_prompt=negative_prompt,
                negative_prompt_2=negative_prompt_2,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                max_sequence_length=max_sequence_length,
            )
            self._guidance_scale = guidance_scale
            self._joint_attention_kwargs = joint_attention_kwargs
            self._current_timestep = None
            self._interrupt = False

            if isinstance(prompt, str):
                batch_size = 1
            elif isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = prompt_embeds.shape[0]
            base_batch_size = batch_size * num_images_per_prompt
            device = self._execution_device
            lora_scale = (
                self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
            )
            has_neg_prompt = negative_prompt is not None or (
                negative_prompt_embeds is not None and negative_pooled_prompt_embeds is not None
            )
            do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
            prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(
                prompt=prompt,
                prompt_2=prompt_2,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                lora_scale=lora_scale,
            )
            negative_text_ids = None
            if do_true_cfg:
                negative_prompt_embeds, negative_pooled_prompt_embeds, negative_text_ids = self.encode_prompt(
                    prompt=negative_prompt,
                    prompt_2=negative_prompt_2,
                    prompt_embeds=negative_prompt_embeds,
                    pooled_prompt_embeds=negative_pooled_prompt_embeds,
                    device=device,
                    num_images_per_prompt=num_images_per_prompt,
                    max_sequence_length=max_sequence_length,
                    lora_scale=lora_scale,
                )

            if image is not None and not (isinstance(image, torch.Tensor) and image.size(1) == self.latent_channels):
                img = image[0] if isinstance(image, list) else image
                image_height, image_width = self.image_processor.get_default_height_width(img)
                image_aspect_ratio = image_width / image_height
                if _auto_resize:
                    _, image_width, image_height = min(
                        (abs(image_aspect_ratio - w / h), w, h) for w, h in PREFERRED_KONTEXT_RESOLUTIONS
                    )
                image_width = image_width // multiple_of * multiple_of
                image_height = image_height // multiple_of * multiple_of
                image = self.image_processor.resize(image, image_height, image_width)
                image = self.image_processor.preprocess(image, image_height, image_width)

            num_channels_latents = self.transformer.config.in_channels // 4
            latents, image_latents, image_ids, source_image_ids = self.prepare_latents(
                image,
                base_batch_size,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
            )
            if latents.shape[0] != base_batch_size:
                raise ValueError(
                    f"Trajectory input latents must use base batch B={base_batch_size}, got {latents.shape[0]}."
                )
            if source_image_ids is not None:
                image_ids = torch.cat([image_ids, source_image_ids], dim=0)

            schedule_sigmas = (
                np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
            )
            image_seq_len = latents.shape[1]
            mu = calculate_shift(
                image_seq_len,
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 4096),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.15),
            )
            timesteps, num_inference_steps = retrieve_timesteps(
                self.scheduler,
                num_inference_steps,
                device,
                sigmas=schedule_sigmas,
                mu=mu,
            )
            num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
            self._num_timesteps = len(timesteps)

            if self.transformer.config.guidance_embeds:
                guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
                guidance = guidance.expand(latents.shape[0])
            else:
                guidance = None

            if (ip_adapter_image is not None or ip_adapter_image_embeds is not None) and (
                negative_ip_adapter_image is None and negative_ip_adapter_image_embeds is None
            ):
                negative_ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
                negative_ip_adapter_image = [
                    negative_ip_adapter_image
                ] * self.transformer.encoder_hid_proj.num_ip_adapters
            elif (ip_adapter_image is None and ip_adapter_image_embeds is None) and (
                negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None
            ):
                ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
                ip_adapter_image = [ip_adapter_image] * self.transformer.encoder_hid_proj.num_ip_adapters

            if self.joint_attention_kwargs is None:
                self._joint_attention_kwargs = {}
            image_embeds = None
            negative_image_embeds = None
            if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
                image_embeds = self.prepare_ip_adapter_image_embeds(
                    ip_adapter_image,
                    ip_adapter_image_embeds,
                    device,
                    base_batch_size,
                )
            if negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None:
                negative_image_embeds = self.prepare_ip_adapter_image_embeds(
                    negative_ip_adapter_image,
                    negative_ip_adapter_image_embeds,
                    device,
                    base_batch_size,
                )

            # The official pipeline runs all conditioning under global
            # no-grad. The trajectory path later re-enables autograd only for
            # current sampling latents, so caller-supplied conditioning must
            # also be made explicit constants here.
            image_latents = None if image_latents is None else image_latents.detach()
            prompt_embeds = prompt_embeds.detach()
            pooled_prompt_embeds = pooled_prompt_embeds.detach()
            guidance = None if guidance is None else guidance.detach()
            negative_prompt_embeds = None if negative_prompt_embeds is None else negative_prompt_embeds.detach()
            negative_pooled_prompt_embeds = (
                None if negative_pooled_prompt_embeds is None else negative_pooled_prompt_embeds.detach()
            )
            image_embeds = None if image_embeds is None else [embed.detach() for embed in image_embeds]
            negative_image_embeds = (
                None if negative_image_embeds is None else [embed.detach() for embed in negative_image_embeds]
            )

            if not context_was_provided and isinstance(prompt, list):
                strength_reward_context = StrengthRewardContext(
                    prompt=[item for item in prompt for _ in range(num_images_per_prompt)]
                )
            validate_strength_reward_context(strength_reward_context, base_batch_size)
            strength_reward_context = StrengthRewardContext(
                prompt=strength_reward_context.prompt,
                source_endpoint=(
                    None
                    if strength_reward_context.source_endpoint is None
                    else strength_reward_context.source_endpoint.detach()
                ),
                target_endpoint=(
                    None
                    if strength_reward_context.target_endpoint is None
                    else strength_reward_context.target_endpoint.detach()
                ),
                metadata=strength_reward_context.metadata,
            )

        num_strengths = len(trajectory_config.strengths)
        flat_strengths = make_flat_strength_tensor(
            trajectory_config.strengths,
            base_batch_size,
            device=latents.device,
        )
        strength_reward_guidance = self._prepare_strength_reward_guidance(
            strength_reward,
            strength_reward_context,
            lambda_strength_reward=trajectory_config.lambda_strength_reward,
            device=latents.device,
            base_batch_size=base_batch_size,
            num_strengths=num_strengths,
        )
        lazy_branch_materialization = trajectory_config.lazy_branch_materialization
        if callback_on_step_end is not None and lazy_branch_materialization:
            logger.warning_once(
                "A trajectory callback requires a stable branch layout, so lazy materialization is disabled."
            )
            lazy_branch_materialization = False

        branches_materialized = False
        if not lazy_branch_materialization:
            (
                latents,
                image_latents,
                prompt_embeds,
                pooled_prompt_embeds,
                guidance,
                negative_prompt_embeds,
                negative_pooled_prompt_embeds,
                image_embeds,
                negative_image_embeds,
            ) = self._materialize_kontext_strength_batch(
                num_strengths,
                latents,
                image_latents,
                prompt_embeds,
                pooled_prompt_embeds,
                guidance,
                negative_prompt_embeds if do_true_cfg else None,
                negative_pooled_prompt_embeds if do_true_cfg else None,
                image_embeds,
                negative_image_embeds,
            )
            branches_materialized = True

        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                self._current_timestep = t
                reward_active = is_strength_reward_step(
                    i,
                    trajectory_config,
                    has_reward=strength_reward_guidance is not None,
                )
                if reward_active and not branches_materialized:
                    (
                        latents,
                        image_latents,
                        prompt_embeds,
                        pooled_prompt_embeds,
                        guidance,
                        negative_prompt_embeds,
                        negative_pooled_prompt_embeds,
                        image_embeds,
                        negative_image_embeds,
                    ) = self._materialize_kontext_strength_batch(
                        num_strengths,
                        latents,
                        image_latents,
                        prompt_embeds,
                        pooled_prompt_embeds,
                        guidance,
                        negative_prompt_embeds if do_true_cfg else None,
                        negative_pooled_prompt_embeds if do_true_cfg else None,
                        image_embeds,
                        negative_image_embeds,
                    )
                    branches_materialized = True

                latents_dtype = latents.dtype
                latents = self._kontext_strength_trajectory_step(
                    latents=latents,
                    step_index=i,
                    timestep=t,
                    image_latents=image_latents,
                    image_ids=image_ids,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    text_ids=text_ids,
                    guidance=guidance,
                    do_true_cfg=do_true_cfg,
                    true_cfg_scale=true_cfg_scale,
                    negative_prompt_embeds=negative_prompt_embeds if do_true_cfg else None,
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds if do_true_cfg else None,
                    negative_text_ids=negative_text_ids if do_true_cfg else None,
                    image_embeds=image_embeds,
                    negative_image_embeds=negative_image_embeds,
                    trajectory_config=trajectory_config,
                    target_strengths=flat_strengths,
                    strength_reward_guidance=strength_reward_guidance,
                    strength_reward_context=strength_reward_context,
                    base_batch_size=base_batch_size,
                    num_strengths=num_strengths,
                    generator=generator,
                    reward_active=reward_active,
                    branches_materialized=branches_materialized,
                    height=height,
                    width=width,
                )
                if latents.dtype != latents_dtype and torch.backends.mps.is_available():
                    latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for name in callback_on_step_end_tensor_inputs:
                        callback_kwargs[name] = locals()[name]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None
        if not branches_materialized:
            latents = expand_shared_initial_latents(latents, num_strengths)

        with torch.no_grad():
            if output_type == "latent":
                output_images = latents
            else:
                decoded_latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
                decoded_latents = decoded_latents / self.vae.config.scaling_factor + self.vae.config.shift_factor
                output_images = self.vae.decode(decoded_latents, return_dict=False)[0]
                output_images = self.image_processor.postprocess(output_images, output_type=output_type)

        if strength_reward_guidance is not None:
            strength_reward_guidance.clear_context_cache()
            strength_reward_guidance.maybe_offload()
        self.maybe_free_model_hooks()

        if not return_dict:
            return output_images, tuple(trajectory_config.strengths)
        return StrengthTrajectoryPipelineOutput(
            images=output_images,
            strengths=tuple(trajectory_config.strengths),
            base_batch_size=base_batch_size,
            num_strengths=num_strengths,
        )
