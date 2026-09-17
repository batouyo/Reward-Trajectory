from __future__ import annotations

import pytest
import torch

from diffusers.pipelines.rewardflow.two_pass_vjp import exact_two_pass_image_vjp


@pytest.mark.parametrize("microbatch_size", [1, 2, 3])
def test_exact_two_pass_vjp_matches_one_pass_for_coupled_toy_generator(microbatch_size):
    torch.manual_seed(0)
    controls = torch.nn.Parameter(torch.randn(3, 2, 4))

    def generator(branches: slice, grad_enabled: bool):
        del grad_enabled
        # Each branch depends nonlinearly on all of its own high-dimensional U;
        # this mirrors independent controls without making a scalar-D shortcut.
        return torch.tanh(controls[branches].mean(dim=1)).reshape(-1, 1, 2, 2)

    def image_loss(images):
        coupled = images.mean(dim=0).square().sum()
        return images.square().mean() + 0.3 * coupled

    controls.grad = None
    one_pass = image_loss(generator(slice(0, 3), True))
    one_pass.backward()
    expected = controls.grad.detach().clone()
    controls.grad = None
    reward, gradient = exact_two_pass_image_vjp(
        num_branches=3,
        microbatch_size=microbatch_size,
        generate=generator,
        image_loss=image_loss,
    )
    assert torch.isfinite(reward) and torch.isfinite(gradient).all()
    torch.testing.assert_close(controls.grad, expected, rtol=1e-6, atol=1e-7)
