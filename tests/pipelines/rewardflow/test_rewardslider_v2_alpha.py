import torch

from diffusers.pipelines.rewardflow.rewardslider_v2_alpha import OrderedAlphaParameterization


def test_random_alpha_is_strictly_ordered_and_seed_reproducible():
    first = OrderedAlphaParameterization.random(num_interior=3, seed=17)
    second = OrderedAlphaParameterization.random(num_interior=3, seed=17)
    torch.testing.assert_close(first.alphas.detach(), second.alphas.detach())
    assert torch.all(first.alphas[1:] > first.alphas[:-1])
    assert first.alphas[0] == 0 and first.alphas[-1] == 1


def test_alpha_logits_receive_finite_nonzero_gradient():
    parameterization = OrderedAlphaParameterization.random(num_interior=3, seed=3)
    loss = (parameterization.alphas[1:-1] ** 2).sum()
    loss.backward()
    gradient = parameterization.interval_logits.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_optimizer_step_preserves_strict_order_without_repair():
    parameterization = OrderedAlphaParameterization.random(num_interior=3, seed=5)
    optimizer = torch.optim.SGD(parameterization.parameters(), lr=0.5)
    (parameterization.alphas[1:-1].square().sum()).backward()
    optimizer.step()
    assert torch.all(parameterization.alphas[1:] > parameterization.alphas[:-1])
    assert parameterization.alphas[0] == 0 and parameterization.alphas[-1] == 1


def test_explicit_alpha_round_trips_through_gap_logits():
    expected = torch.tensor([0.0, 0.1, 0.55, 0.9, 1.0])
    parameterization = OrderedAlphaParameterization.from_alphas(expected)
    torch.testing.assert_close(parameterization.alphas.detach(), expected, atol=1e-6, rtol=1e-6)


def test_only_interval_logits_are_optimizer_parameters():
    parameterization = OrderedAlphaParameterization.random(num_interior=3, seed=7)
    parameters = list(parameterization.parameters())
    assert parameters == [parameterization.interval_logits]
    assert parameterization.alphas[0].item() == 0.0
    assert parameterization.alphas[-1].item() == 1.0


def test_ordered_alpha_endpoints_remain_exact_after_logit_updates():
    for seed in range(100):
        parameterization = OrderedAlphaParameterization.random(num_interior=3, seed=seed)
        with torch.no_grad():
            parameterization.interval_logits.add_(torch.randn_like(parameterization.interval_logits) * 2)
        alphas = parameterization.alphas
        assert alphas[0].item() == 0.0
        assert alphas[-1].item() == 1.0
        assert torch.all(alphas[1:] > alphas[:-1])
