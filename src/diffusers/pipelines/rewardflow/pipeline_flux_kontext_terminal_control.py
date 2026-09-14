"""Independent long-horizon terminal-control path for FLUX.1-Kontext."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from ...image_processor import PipelineImageInput
from .pipeline_flux_kontext_strength_trajectory import FluxKontextStrengthTrajectoryPipeline
from .strength_trajectory import StrengthTrajectoryConfig
from .terminal_control import TerminalControlUnrollOutput, unroll_terminal_velocity_controls


@dataclass
class KontextTerminalControlInputs:
    """Fixed native Kontext trajectory state captured from one K=1 run."""

    initial_latent: torch.Tensor
    native_final_latent: torch.Tensor
    timesteps: tuple[torch.Tensor, ...]
    sigmas: torch.Tensor
    forward_kwargs: dict[str, Any]
    height: int
    width: int


def _detach_constant(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach()
    if isinstance(value, list):
        return [_detach_constant(item) for item in value]
    return value


class FluxKontextTerminalControlPipeline(FluxKontextStrengthTrajectoryPipeline):
    """Research-only terminal control over frozen native Kontext dynamics.

    Preparation intentionally reuses the already parity-tested K=1 strength
    trajectory setup. The inherited local per-step reward behavior is not
    changed; this class captures its fixed conditioning and then performs a
    separate differentiable unroll.
    """

    _TERMINAL_FORWARD_KEYS = (
        "image_latents",
        "image_ids",
        "prompt_embeds",
        "pooled_prompt_embeds",
        "text_ids",
        "guidance",
        "do_true_cfg",
        "true_cfg_scale",
        "negative_prompt_embeds",
        "negative_pooled_prompt_embeds",
        "negative_text_ids",
        "image_embeds",
        "negative_image_embeds",
    )

    def _kontext_strength_trajectory_step(self, **kwargs) -> torch.Tensor:
        if getattr(self, "_terminal_capture_active", False):
            if not self._terminal_capture_steps:
                self._terminal_capture_initial_latent = kwargs["latents"].detach().clone()
                self._terminal_capture_forward_kwargs = {
                    key: _detach_constant(kwargs[key]) for key in self._TERMINAL_FORWARD_KEYS
                }
            self._terminal_capture_steps.append(kwargs["timestep"].detach().clone())
        return super()._kontext_strength_trajectory_step(**kwargs)

    def prepare_terminal_control_inputs(
        self,
        *,
        image: PipelineImageInput,
        prompt: str,
        height: int,
        width: int,
        num_inference_steps: int,
        guidance_scale: float,
        generator: torch.Generator,
        latents: torch.Tensor | None = None,
        sigmas: list[float] | None = None,
        max_sequence_length: int = 512,
    ) -> KontextTerminalControlInputs:
        """Capture one fixed initial latent, conditioning set, and native endpoint."""

        if image is None:
            raise ValueError("Terminal Kontext control requires exactly one source image.")
        if not isinstance(prompt, str):
            raise TypeError("Terminal Kontext control currently supports one string prompt and B=1.")
        if num_inference_steps < 1:
            raise ValueError("`num_inference_steps` must be positive.")
        if bool(getattr(self.scheduler.config, "stochastic_sampling", False)):
            raise ValueError(
                "Terminal control supports deterministic FlowMatch Euler only; SDE is intentionally absent."
            )

        # Freeze before native preparation. Frozen weights still preserve input
        # Jacobians in the later differentiable unroll.
        self._freeze_kontext_trajectory_modules(None)
        self._terminal_capture_active = True
        self._terminal_capture_steps = []
        self._terminal_capture_initial_latent = None
        self._terminal_capture_forward_kwargs = None
        try:
            output = super().__call__(
                image=image,
                prompt=prompt,
                height=height,
                width=width,
                max_area=height * width,
                _auto_resize=False,
                num_inference_steps=num_inference_steps,
                sigmas=sigmas,
                guidance_scale=guidance_scale,
                num_images_per_prompt=1,
                generator=generator,
                latents=latents,
                output_type="latent",
                return_dict=True,
                max_sequence_length=max_sequence_length,
                trajectory_config=StrengthTrajectoryConfig(
                    enabled=True,
                    strengths=(0.5,),
                    lambda_strength_reward=0.0,
                    use_shared_sde_noise=False,
                    lazy_branch_materialization=True,
                ),
            )
        finally:
            self._terminal_capture_active = False

        if self._terminal_capture_initial_latent is None or self._terminal_capture_forward_kwargs is None:
            raise RuntimeError("Kontext preparation did not capture the initial denoising state.")
        if self._terminal_capture_initial_latent.shape[0] != 1:
            raise ValueError("Terminal Kontext control currently supports only B=1.")
        if len(self._terminal_capture_steps) != num_inference_steps:
            raise RuntimeError("Captured timestep count does not match the native Kontext run.")

        scheduler_sigmas = self.scheduler.sigmas[: num_inference_steps + 1].detach().clone()
        if scheduler_sigmas.numel() != num_inference_steps + 1:
            raise RuntimeError("Scheduler did not expose one more sigma than denoising timesteps.")
        return KontextTerminalControlInputs(
            initial_latent=self._terminal_capture_initial_latent,
            native_final_latent=output.images.detach().clone(),
            timesteps=tuple(self._terminal_capture_steps),
            sigmas=scheduler_sigmas,
            forward_kwargs=self._terminal_capture_forward_kwargs,
            height=height,
            width=width,
        )

    def unroll_terminal_controls(
        self,
        inputs: KontextTerminalControlInputs,
        controls: Sequence[torch.Tensor] = (),
        *,
        use_checkpointing: bool = True,
    ) -> TerminalControlUnrollOutput:
        """Run a full differentiable trajectory without detaching future states."""

        if len(controls) > len(inputs.timesteps):
            raise ValueError("Control prefix cannot exceed the denoising trajectory length.")

        def velocity_fn(latent: torch.Tensor, timestep: torch.Tensor, step_index: int) -> torch.Tensor:
            del step_index
            return self._predict_kontext_velocity(latent, timestep, **inputs.forward_kwargs)

        return unroll_terminal_velocity_controls(
            inputs.initial_latent,
            inputs.timesteps,
            inputs.sigmas,
            velocity_fn,
            controls,
            use_checkpointing=use_checkpointing,
        )

    def decode_terminal_latent(
        self,
        latent: torch.Tensor,
        inputs: KontextTerminalControlInputs,
    ) -> torch.Tensor:
        """Differentiably decode a final latent; no image detach occurs here."""

        return self._decode_kontext_clean_latent_for_reward(latent, inputs.height, inputs.width)
