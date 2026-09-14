"""Independent long-horizon terminal-control path for FLUX.1-Kontext."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from ...image_processor import PipelineImageInput
from .pipeline_flux_kontext_strength_trajectory import FluxKontextStrengthTrajectoryPipeline
from .strength_trajectory import StrengthTrajectoryConfig
from .terminal_control import (
    TerminalControlUnrollOutput,
    VelocityEditMasks,
    build_velocity_edit_masks,
    initialize_velocity_controls,
    unroll_terminal_velocity_controls,
)


@dataclass
class KontextTerminalControlInputs:
    """Fixed native Kontext trajectory state captured from one K=1 run."""

    initial_latent: torch.Tensor
    native_final_latent: torch.Tensor
    timesteps: tuple[torch.Tensor, ...]
    sigmas: torch.Tensor
    forward_kwargs: dict[str, Any]
    source_clean_latent: torch.Tensor
    sampling_token_height: int
    sampling_token_width: int
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

    @staticmethod
    def _validated_token_grid(image_ids: torch.Tensor, *, expected_plane: int, name: str) -> tuple[int, int]:
        """Validate official FLUX row-major spatial IDs and return their grid shape."""

        if image_ids.ndim != 2 or image_ids.shape[1] != 3:
            raise ValueError(f"{name} position IDs must have shape [tokens, 3].")
        ids = image_ids.detach().float().cpu()
        if not torch.equal(ids[:, 0], torch.full_like(ids[:, 0], expected_plane)):
            raise ValueError(f"{name} position IDs do not have the expected Kontext plane marker.")
        rows = torch.unique(ids[:, 1], sorted=True)
        columns = torch.unique(ids[:, 2], sorted=True)
        if not torch.equal(rows, torch.arange(rows.numel(), dtype=rows.dtype)) or not torch.equal(
            columns, torch.arange(columns.numel(), dtype=columns.dtype)
        ):
            raise ValueError(f"{name} spatial position IDs must be contiguous and start at zero.")
        if rows.numel() * columns.numel() != ids.shape[0]:
            raise ValueError(f"{name} position IDs do not describe one dense rectangular token grid.")
        expected = torch.stack(torch.meshgrid(rows, columns, indexing="ij"), dim=-1).reshape(-1, 2)
        if not torch.equal(ids[:, 1:], expected):
            raise ValueError(f"{name} position IDs are not in official FLUX row-major spatial order.")
        return int(rows.numel()), int(columns.numel())

    def _align_source_clean_latent(
        self,
        sampling_latent: torch.Tensor,
        source_latent: torch.Tensor,
        combined_image_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, int, int]:
        """Align packed source tokens to sampling tokens using official FLUX semantics."""

        if sampling_latent.ndim != 3 or source_latent.ndim != 3:
            raise ValueError("Sampling and source packed latents must have shape [B, tokens, channels].")
        if sampling_latent.shape[0] != source_latent.shape[0] or sampling_latent.shape[2] != source_latent.shape[2]:
            raise ValueError("Sampling/source packed latent batch and channel dimensions must match.")
        sampling_tokens = sampling_latent.shape[1]
        source_tokens = source_latent.shape[1]
        if combined_image_ids.shape[0] != sampling_tokens + source_tokens:
            raise ValueError("Combined Kontext position IDs do not match sampling plus source token counts.")
        sampling_ids = combined_image_ids[:sampling_tokens]
        source_ids = combined_image_ids[sampling_tokens:]
        sampling_height, sampling_width = self._validated_token_grid(sampling_ids, expected_plane=0, name="sampling")
        source_height, source_width = self._validated_token_grid(source_ids, expected_plane=1, name="source")

        if (sampling_height, sampling_width) == (source_height, source_width):
            if not torch.equal(sampling_ids[:, 1:].detach().cpu(), source_ids[:, 1:].detach().cpu()):
                raise ValueError("Equal-size source and sampling grids have different spatial position IDs.")
            return source_latent.detach().clone(), sampling_height, sampling_width

        source_pixel_height = source_height * 2 * self.vae_scale_factor
        source_pixel_width = source_width * 2 * self.vae_scale_factor
        unpacked = self._unpack_latents(source_latent, source_pixel_height, source_pixel_width, self.vae_scale_factor)
        aligned = F.interpolate(
            unpacked.float(), size=(sampling_height * 2, sampling_width * 2), mode="bilinear", align_corners=False
        ).to(source_latent.dtype)
        aligned = self._pack_latents(
            aligned,
            aligned.shape[0],
            aligned.shape[1],
            aligned.shape[2],
            aligned.shape[3],
        )
        if aligned.shape != sampling_latent.shape:
            raise RuntimeError("Official source unpack/interpolate/repack did not produce the sampling latent shape.")
        return aligned.detach().clone(), sampling_height, sampling_width

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
        source_latent = self._terminal_capture_forward_kwargs.get("image_latents")
        combined_image_ids = self._terminal_capture_forward_kwargs.get("image_ids")
        if source_latent is None or combined_image_ids is None:
            raise RuntimeError("Kontext source latent and position IDs are required for terminal control.")
        source_clean_latent, token_height, token_width = self._align_source_clean_latent(
            self._terminal_capture_initial_latent, source_latent, combined_image_ids
        )
        return KontextTerminalControlInputs(
            initial_latent=self._terminal_capture_initial_latent,
            native_final_latent=output.images.detach().clone(),
            timesteps=tuple(self._terminal_capture_steps),
            sigmas=scheduler_sigmas,
            forward_kwargs=self._terminal_capture_forward_kwargs,
            source_clean_latent=source_clean_latent,
            sampling_token_height=token_height,
            sampling_token_width=token_width,
            height=height,
            width=width,
        )

    def prepare_velocity_edit_masks(
        self,
        inputs: KontextTerminalControlInputs,
        *,
        control_steps: int,
        mode: str,
        topk_fraction: float = 0.25,
    ) -> tuple[VelocityEditMasks, TerminalControlUnrollOutput]:
        """Compute fixed masks once from the zero-control native trajectory."""

        if not 1 <= control_steps <= len(inputs.timesteps):
            raise ValueError("Control steps must lie within the captured trajectory.")
        zero_controls = initialize_velocity_controls(inputs.initial_latent, control_steps)
        with torch.no_grad():
            native = self.unroll_terminal_controls(inputs, zero_controls, use_checkpointing=False)
            masks = build_velocity_edit_masks(
                native.states[:control_steps],
                native.native_control_velocities,
                inputs.source_clean_latent,
                inputs.sigmas[:control_steps],
                mode=mode,
                topk_fraction=topk_fraction,
            )
        return masks, native

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
