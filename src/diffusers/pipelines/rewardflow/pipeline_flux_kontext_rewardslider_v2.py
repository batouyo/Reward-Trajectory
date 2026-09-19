"""Standalone FLUX-Kontext RewardSlider V2 pipeline path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch

from .pipeline_flux_kontext_terminal_control import FluxKontextTerminalControlPipeline, KontextTerminalControlInputs
from .rewardslider_v2_unroll import RewardSliderV2UnrollOutput, unroll_rewardslider_v2



def validate_v2_control_steps(control_steps: int) -> int:
    if control_steps != 4:
        raise ValueError("RewardSlider V2 requires exactly four controlled timesteps.")
    return control_steps

def predict_kontext_velocity_branchwise(
    predict_fn: Callable[..., torch.Tensor], latent: torch.Tensor, **kwargs: Any
) -> torch.Tensor:
    """Evaluate frozen Kontext dynamics one branch at a time (B=1)."""
    if latent.ndim < 1 or latent.shape[0] < 1:
        raise ValueError("`latent` must have a non-empty batch dimension.")
    outputs = []
    batch_size = latent.shape[0]
    for branch_index in range(batch_size):
        branch_kwargs = {}
        for name, value in kwargs.items():
            if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == batch_size:
                branch_kwargs[name] = value[branch_index : branch_index + 1]
            else:
                branch_kwargs[name] = value
        output = predict_fn(latent[branch_index : branch_index + 1], **branch_kwargs)
        if output.ndim < 1 or output.shape[0] != 1:
            raise ValueError("Branchwise prediction must return a B=1 tensor for every branch.")
        outputs.append(output)
    return torch.cat(outputs, dim=0)
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
            num_branches, native.initial_latent, f["image_latents"], f["prompt_embeds"],
            f["pooled_prompt_embeds"], f["guidance"], f["negative_prompt_embeds"],
            f["negative_pooled_prompt_embeds"], f["image_embeds"], f["negative_image_embeds"]
        )
        _, image_latents, prompt_embeds, pooled_prompt_embeds, guidance, neg_prompt, neg_pooled, image_embeds, neg_image_embeds = values
        forward_kwargs = dict(f)
        forward_kwargs.update(image_latents=image_latents, prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds,
                              guidance=guidance, negative_prompt_embeds=neg_prompt, negative_pooled_prompt_embeds=neg_pooled,
                              image_embeds=image_embeds, negative_image_embeds=neg_image_embeds)
        return RewardSliderV2Inputs(native, num_branches, forward_kwargs)
    def rematerialize_rewardslider_v2_inputs(self, inputs: RewardSliderV2Inputs, *, num_branches: int) -> RewardSliderV2Inputs:
        if num_branches < 1:
            raise ValueError("`num_branches` must be positive.")
        f = inputs.native.forward_kwargs
        values = self._materialize_kontext_strength_batch(
            num_branches, inputs.native.initial_latent, f["image_latents"], f["prompt_embeds"],
            f["pooled_prompt_embeds"], f["guidance"], f["negative_prompt_embeds"],
            f["negative_pooled_prompt_embeds"], f["image_embeds"], f["negative_image_embeds"]
        )
        _, image_latents, prompt_embeds, pooled_prompt_embeds, guidance, neg_prompt, neg_pooled, image_embeds, neg_image_embeds = values
        forward_kwargs = dict(f)
        forward_kwargs.update(image_latents=image_latents, prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds,
                              guidance=guidance, negative_prompt_embeds=neg_prompt, negative_pooled_prompt_embeds=neg_pooled,
                              image_embeds=image_embeds, negative_image_embeds=neg_image_embeds)
        return RewardSliderV2Inputs(inputs.native, num_branches, forward_kwargs)

    def unroll_rewardslider_v2_controls(self, inputs: RewardSliderV2Inputs, alphas: torch.Tensor | float,
                                        v_goals: Sequence[torch.Tensor], *, control_steps: int = 4,
                                        use_checkpointing: bool = True, branchwise: bool = True) -> RewardSliderV2UnrollOutput:
        validate_v2_control_steps(control_steps)
        if torch.is_tensor(alphas) and alphas.ndim == 1 and alphas.shape[0] != inputs.num_branches:
            raise ValueError("One alpha value is required per branch.")

        def velocity_fn(latent: torch.Tensor, timestep: torch.Tensor, step_index: int) -> torch.Tensor:
            del step_index
            model_dtype = next(self.transformer.parameters()).dtype
            predict = lambda branch_latent, **branch_kwargs: self._predict_kontext_velocity(
                branch_latent.to(dtype=model_dtype), timestep, **branch_kwargs
            )
            if branchwise:
                predicted = predict_kontext_velocity_branchwise(predict, latent, **inputs.forward_kwargs)
            else:
                predicted = self._predict_kontext_velocity(latent.to(dtype=model_dtype), timestep, **inputs.forward_kwargs)
            return predicted.to(dtype=latent.dtype)

        return unroll_rewardslider_v2(inputs.native.initial_latent, inputs.native.timesteps, inputs.native.sigmas,
                                      velocity_fn, alphas, v_goals,
                                      source_clean_latent=inputs.native.source_clean_latent,
                                      control_steps=control_steps, use_checkpointing=use_checkpointing)

    def decode_rewardslider_v2_terminal(self, latent: torch.Tensor, inputs: RewardSliderV2Inputs) -> torch.Tensor:
        vae_dtype = next(self.vae.parameters()).dtype
        return self.decode_terminal_latent(latent.to(dtype=vae_dtype), inputs.native)
