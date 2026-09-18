import torch
from torch import nn

from diffusers.pipelines.rewardflow.rewardslider_v2_lpips import (
    LPIPSDistance,
    lpips_trajectory_stats,
    lpips_uniform_kl,
)


def test_uniform_distances_have_zero_kl():
    torch.testing.assert_close(lpips_uniform_kl(torch.ones(4)), torch.tensor(0.0), atol=1e-7, rtol=0)


def test_stall_then_jump_has_large_kl_and_correct_worst_interval():
    stats = lpips_trajectory_stats(torch.tensor([0.01, 0.01, 0.01, 1.0]))
    assert stats.kl_uniform > 0.15
    assert stats.worst_interval.item() == 3
    assert stats.max_normalized_gap > 0.9


def test_kl_matches_independent_manual_formula():
    distances = torch.tensor([1.0, 2.0, 3.0])
    probabilities = distances / distances.sum()
    expected = (probabilities * (probabilities / (1 / 3)).log()).sum()
    torch.testing.assert_close(lpips_uniform_kl(distances), expected)


class _MockLPIPS(nn.Module):
    def forward(self, first, second):
        return (first - second).square().mean(dim=(1, 2, 3), keepdim=True).sqrt()


def test_tensor_native_lpips_adapter_preserves_image_gradient():
    images = torch.randn(5, 3, 4, 4, requires_grad=True)
    distance = LPIPSDistance(model=_MockLPIPS())
    stats = distance.trajectory(images)
    stats.kl_uniform.backward()
    assert images.grad is not None
    assert torch.isfinite(images.grad).all()
    assert images.grad.abs().sum() > 0
