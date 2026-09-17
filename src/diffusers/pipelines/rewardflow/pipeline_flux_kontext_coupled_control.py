"""RewardSlider V1 batched independent-control path for FLUX.1-Kontext.

Unlike the legacy terminal controller, this class only prepares frozen native
context and runs K independently parameterized middle trajectories.  It does
not alter legacy, RewardFlow-paper, or shared-direction behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from .coupled_terminal_control import CoupledControlPrior, make_coupled_prior, unroll_coupled_velocity_controls
from .pipeline_flux_kontext_terminal_control import FluxKontextTerminalControlPipeline, KontextTerminalControlInputs
from .strength_trajectory import expand_shared_initial_latents
from .terminal_control import source_restoring_velocity


@dataclass(frozen=True)
class CoupledKontextControlInputs:
    """One fixed native endpoint plus K-batched official Kontext conditioning."""

    native: KontextTerminalControlInputs
    num_branches: int
    initial_latent: torch.Tensor
    forward_kwargs: dict[str, Any]
    prior: CoupledControlPrior


class FluxKontextCoupledControlPipeline(FluxKontextTerminalControlPipeline):
    """Research-only RewardSlider V1 preparation and batched full unroll."""

    def prepare_coupled_control_inputs(
        self,
        *,
        num_branches: int,
        control_steps: int,
        **kwargs,
    ) -> CoupledKontextControlInputs:
        if num_branches < 1:
            raise ValueError("`num_branches` must be positive.")
        native = self.prepare_terminal_control_inputs(**kwargs)
        if not 1 <= control_steps <= len(native.timesteps):
            raise ValueError("`control_steps` must lie within the captured native schedule.")
        # Native trajectory is computed once and only used as a detached prior.
        _, native_unroll = self.prepare_velocity_edit_masks(native, control_steps=control_steps, mode="none")
        directions = tuple(
            source_restoring_velocity(state, native.source_clean_latent, native.sigmas[index]).detach()
            - velocity.detach()
            for index, (state, velocity) in enumerate(
                zip(native_unroll.states[:control_steps], native_unroll.native_control_velocities)
            )
        )
        # velocity_edit_score(v_edit, v_keep) is the norm of this same D.
        scores = tuple(direction.float().square().mean(dim=-1).sqrt().detach() for direction in directions)
        prior = make_coupled_prior(directions, scores)
        f = native.forward_kwargs
        (
            initial,
            image_latents,
            prompt_embeds,
            pooled_prompt_embeds,
            guidance,
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
            image_embeds,
            negative_image_embeds,
        ) = self._materialize_kontext_strength_batch(
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
        # _materialize... is the repository's tested batch contract.  Keep
        # image/text position ids shared because they are sequence semantics.
        forward_kwargs = dict(f)
        forward_kwargs.update(
            {
                "image_latents": image_latents,
                "prompt_embeds": prompt_embeds,
                "pooled_prompt_embeds": pooled_prompt_embeds,
                "guidance": guidance,
                "negative_prompt_embeds": negative_prompt_embeds,
                "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds,
                "image_embeds": image_embeds,
                "negative_image_embeds": negative_image_embeds,
            }
        )
        return CoupledKontextControlInputs(
            native=native,
            num_branches=num_branches,
            initial_latent=initial,
            forward_kwargs=forward_kwargs,
            prior=prior,
        )

    def unroll_coupled_controls(
        self,
        inputs: CoupledKontextControlInputs,
        controls: Sequence[torch.Tensor],
        *,
        use_checkpointing: bool = True,
    ):
        if not controls or controls[0].shape[0] != inputs.num_branches:
            raise ValueError("Controls must use the input's K branch dimension.")

        def velocity_fn(latent: torch.Tensor, timestep: torch.Tensor, step_index: int) -> torch.Tensor:
            del step_index
            return self._predict_kontext_velocity(latent, timestep, **inputs.forward_kwargs)

        return unroll_coupled_velocity_controls(
            inputs.native.initial_latent,
            inputs.native.timesteps,
            inputs.native.sigmas,
            velocity_fn,
            controls,
            use_checkpointing=use_checkpointing,
        )

    def decode_coupled_terminal_latent(
        self, latent: torch.Tensor, inputs: CoupledKontextControlInputs
    ) -> torch.Tensor:
        return self.decode_terminal_latent(latent, inputs.native)

    def unroll_coupled_control_microbatch(
        self,
        inputs: CoupledKontextControlInputs,
        controls: Sequence[torch.Tensor],
        branch_slice: slice,
        *,
        use_checkpointing: bool,
    ):
        """Replay an exact subset of independent branches for branchwise VJP."""

        indices = torch.arange(inputs.num_branches, device=inputs.initial_latent.device)[branch_slice]
        if indices.numel() < 1:
            raise ValueError("VJP microbatch must contain at least one branch.")
        kwargs = {}
        for name, value in inputs.forward_kwargs.items():
            if torch.is_tensor(value) and value.ndim and value.shape[0] == inputs.num_branches:
                kwargs[name] = value.index_select(0, indices)
            elif isinstance(value, list):
                kwargs[name] = [
                    item.index_select(0, indices) if item.shape[0] == inputs.num_branches else item for item in value
                ]
            else:
                kwargs[name] = value

        def velocity_fn(latent: torch.Tensor, timestep: torch.Tensor, step_index: int) -> torch.Tensor:
            del step_index
            return self._predict_kontext_velocity(latent, timestep, **kwargs)

        return unroll_coupled_velocity_controls(
            inputs.native.initial_latent,
            inputs.native.timesteps,
            inputs.native.sigmas,
            velocity_fn,
            [control.index_select(0, indices) for control in controls],
            use_checkpointing=use_checkpointing,
        )

    @staticmethod
    def expand_native_to_branches(latent: torch.Tensor, num_branches: int) -> torch.Tensor:
        """Expose the tested B-major/K-minor repeat used for parity checks."""

        return expand_shared_initial_latents(latent, num_branches)
