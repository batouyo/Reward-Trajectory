import torch

from diffusers.pipelines.rewardflow.rewardslider_v2_preservation import masked_lpips_preservation_loss


class _Distance:
    def distance(self, first, second):
        return (first - second).square().mean(dim=(1, 2, 3)).sqrt()


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
