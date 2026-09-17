"""Exact branchwise vector-Jacobian products for terminal image rewards.

Pass A obtains the image-space gradient without retaining a generator graph.
Pass B replays frozen generator dynamics by branch microbatch and applies that
gradient as ``grad_tensors``.  This is the chain rule, not a surrogate.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


ImageGenerator = Callable[[slice, bool], torch.Tensor]
ImageLoss = Callable[[torch.Tensor], torch.Tensor]


def exact_two_pass_image_vjp(
    *,
    num_branches: int,
    microbatch_size: int,
    generate: ImageGenerator,
    image_loss: ImageLoss,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Accumulate exact ``d image_loss / d controls`` through branch replay.

    ``generate(slice, False)`` must replay the corresponding branch images
    under no-grad; ``generate(slice, True)`` must replay the same function
    with the normal autograd graph.  Caller-owned control gradients are
    accumulated by ``Tensor.backward`` and any control-only loss can be added
    afterwards without a generator graph.
    """

    if num_branches < 1 or not 1 <= microbatch_size <= num_branches:
        raise ValueError("Microbatch size must lie in [1, num_branches].")
    with torch.no_grad():
        detached_images = generate(slice(0, num_branches), False).detach()
    reward_images = detached_images.requires_grad_(True)
    reward = image_loss(reward_images)
    if reward.ndim != 0:
        raise ValueError("`image_loss` must return a scalar.")
    image_grad = torch.autograd.grad(reward, reward_images)[0].detach()
    if image_grad.shape != detached_images.shape:
        raise RuntimeError("Image reward gradient has the wrong shape.")
    for start in range(0, num_branches, microbatch_size):
        end = min(start + microbatch_size, num_branches)
        replay = generate(slice(start, end), True)
        torch.autograd.backward(replay, grad_tensors=image_grad[start:end])
    return reward.detach(), image_grad
