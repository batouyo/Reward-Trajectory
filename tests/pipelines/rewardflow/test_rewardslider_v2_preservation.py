import torch

from diffusers.pipelines.rewardflow.rewardslider_v2_preservation import (
    build_image_space_relevance,
    masked_lpips_preservation_loss,
)


class _Distance:
    def distance(self, first, second):
        return (first - second).square().mean(dim=(1, 2, 3)).sqrt()
def test_image_relevance_uses_two_dimensional_token_grid():
    relevance = [torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])]
    image_map = build_image_space_relevance(
        relevance, token_height=2, token_width=3, target_height=8, target_width=12
    )
    assert image_map.shape == (1, 1, 8, 12)
    assert image_map[:, :, :4, :4].mean() > image_map[:, :, 4:, :].mean()
    assert image_map[:, :, :4, :4].mean() > image_map[:, :, :4, 4:].mean()
def test_image_relevance_averages_all_controlled_timesteps_before_resize():
    maps = [
        torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]),
    ]
    expected = torch.tensor([[[[0.25, 0.25, 0.25], [0.25, 0.0, 0.0]]]])
    image_map = build_image_space_relevance(
        maps, token_height=2, token_width=3, target_height=2, target_width=3
    )
    torch.testing.assert_close(image_map, expected)


def test_masked_preservation_keeps_candidate_gradient():
    source = torch.zeros(1, 3, 4, 4)
    candidate = torch.ones(2, 3, 4, 4, requires_grad=True)
    relevance = torch.zeros(1, 1, 4, 4)
    loss = masked_lpips_preservation_loss(candidate, source, relevance, _Distance())
    loss.backward()
    assert torch.isfinite(loss)
    assert candidate.grad is not None and torch.isfinite(candidate.grad).all() and candidate.grad.abs().sum() > 0


def test_edit_region_is_replaced_by_detached_source_while_outside_is_penalized():
    source = torch.zeros(1, 3, 4, 4)
    candidate = torch.ones(1, 3, 4, 4, requires_grad=True)
    relevance = torch.zeros(1, 1, 4, 4)
    relevance[:, :, :2] = 1
    distance = _Distance()
    loss = masked_lpips_preservation_loss(candidate, source, relevance, distance)
    expected = torch.sqrt(torch.tensor(0.5))
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert candidate.grad[:, :, :2].abs().sum() == 0
    assert candidate.grad[:, :, 2:].abs().sum() > 0


def test_candidate_batch_broadcasts_source_and_relevance():
    source = torch.zeros(1, 3, 2, 2)
    candidate = torch.ones(3, 3, 2, 2, requires_grad=True)
    relevance = torch.zeros(1, 1, 2, 2)
    loss = masked_lpips_preservation_loss(candidate, source, relevance, _Distance())
    assert loss.ndim == 0
