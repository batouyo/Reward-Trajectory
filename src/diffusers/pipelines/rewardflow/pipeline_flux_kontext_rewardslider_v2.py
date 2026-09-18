"""Standalone FLUX-Kontext RewardSlider V2 pipeline path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from .pipeline_flux_kontext_terminal_control import FluxKontextTerminalControlPipeline, KontextTerminalControlInputs
from .rewardslider_v2_unroll import RewardSliderV2UnrollOutput, unroll_rewardslider_v2
from .strength_trajectory import expand_shared_initial_latents


@dataclass(frozen=True)
class RewardSliderV2Inputs:
    native: KontextTerminalControlInputs
    num_branches: int
    forward_kwargs: dict[str, Any]


class FluxKontextRewardSliderV2Pipeline(FluxKontextTerminalControlPipeline):
    """V2 facade over the existing frozen Kontext preparation and decoder."""

    def prepare_rewardslider_v2_inputs(self, *, num_branches: int, **kwargs) -> RewardSliderV2Inputs:
        if num_branches < 1:
            raise ValueError("`num_branches` must be positive.")
        native = self.prepare_terminal_control_inputs(**kwargs)
        f = native.forward_kwargs
        values = self._materialize_kontext_strength_batch(
            num_branches,
            native.initial_latent,
            f["image_latents"],
            f["prompt_embeds"],
            f["pooled_prompt_embeds"],
            f["guidance"],
            f["negative_prompt_embeds"],
            f["negative_pooled_prompt_embeds"],
            f["image_embeds"],
            f["negative_image_embeds"],
        )
        _, image_latents, prompt_embeds, pooled_prompt_embeds, guidance, neg_prompt, neg_pooled, image_embeds, neg_image_embeds = values
        forward_kwargs = dict(f)
        forward_kwargs.update(
            image_latents=image_latents,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            guidance=guidance,
            negative_prompt_embeds=neg_prompt,
            negative_pooled_prompt_embeds=neg_pooled,
            image_embeds=image_embeds,
            negative_image_embeds=neg_image_embeds,
        )
        return RewardSliderV2Inputs(native, num_branches, forward_kwargs)

    def unroll_rewardslider_v2_controls(
        self,
        inputs: RewardSliderV2Inputs,
        alphas: torch.Tensor | float,
        v_goals: Sequence[torch.Tensor],
        *,
        control_steps: int = 4,
        use_checkpointing: bool = True,
    ) -> RewardSliderV2UnrollOutput:
        if torch.is_tensor(alphas) and alphas.ndim == 1 and alphas.shape[0] != inputs.num_branches:
            raise ValueError("One alpha value is required per branch.")

        def velocity_fn(latent: torch.Tensor, timestep: torch.Tensor, step_index: int) -> torch.Tensor:
            del step_index
            return self._predict_kontext_velocity(latent, timestep, **inputs.forward_kwargs)

        return unroll_rewardslider_v2(
            inputs.native.initial_latent,
            inputs.native.timesteps,
            inputs.native.sigmas,
            velocity_fn,
            alphas,
            v_goals,
            source_clean_latent=inputs.native.source_clean_latent,
            control_steps=control_steps,
            use_checkpointing=use_checkpointing,
        )

    def decode_rewardslider_v2_terminal(self, latent: torch.Tensor, inputs: RewardSliderV2Inputs) -> torch.Tensor:
        return self.decode_terminal_latent(latent, inputs.native)
