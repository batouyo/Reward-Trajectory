from dataclasses import dataclass

import numpy as np
import PIL.Image

from ...utils import BaseOutput


@dataclass
class FluxRewardFlowPipelineOutput(BaseOutput):
    """
    Output class for FluxRewardFlow image generation pipelines.

    Args:
        images (`list[PIL.Image.Image]` or `torch.Tensor` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array or torch tensor of shape `(batch_size,
            height, width, num_channels)`. PIL images or numpy array present the denoised images of the diffusion
            pipeline. Torch tensors can represent either the denoised images or the intermediate latents ready to be
            passed to the decoder.
    """

    images: list[PIL.Image.Image, np.ndarray]


@dataclass
class StrengthTrajectoryPipelineOutput(BaseOutput):
    """Flat B-major/K-minor images plus explicit trajectory metadata."""

    images: list[PIL.Image.Image] | np.ndarray
    strengths: tuple[float, ...]
    base_batch_size: int
    num_strengths: int

    @property
    def grouped_images(self) -> list[list[PIL.Image.Image | np.ndarray]]:
        """Expose outputs as ``[base sample][strength]`` without copying them."""

        from .strength_trajectory import group_trajectory_images

        return group_trajectory_images(self.images, self.base_batch_size, self.num_strengths)
