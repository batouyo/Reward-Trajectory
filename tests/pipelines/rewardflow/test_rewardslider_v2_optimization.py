import torch

from diffusers.pipelines.rewardflow.rewardslider_v2_optimization import coordinate_search_alphas


def test_coordinate_search_accepts_only_non_worsening_real_evaluations():
    initial = torch.tensor([0.0, 0.2, 0.7, 1.0])

    def evaluate(alphas):
        return ((alphas[1:-1] - torch.tensor([0.4, 0.8])).square()).sum()

    result = coordinate_search_alphas(
        initial,
        evaluate,
        initial_delta=0.2,
        min_delta=0.05,
        margin=0.01,
    )
    assert result.kl <= float(evaluate(initial))
    assert result.accepted_steps > 0
    assert torch.all(result.alphas[1:] > result.alphas[:-1])
    assert result.alphas[0] == 0 and result.alphas[-1] == 1
    assert result.alphas[1] >= 0.01
    assert result.alphas[-2] <= 0.99


def test_coordinate_search_does_not_accept_a_bad_candidate():
    initial = torch.tensor([0.0, 0.4, 0.8, 1.0])

    def evaluate(alphas):
        return ((alphas[1:-1] - torch.tensor([0.4, 0.8])).square()).sum()

    result = coordinate_search_alphas(initial, evaluate, initial_delta=0.2, min_delta=0.2)
    torch.testing.assert_close(result.alphas, initial)
    torch.testing.assert_close(result.kl, evaluate(initial))


def test_hybrid_acceptance_restores_parameters_when_post_step_kl_worsens():
    from diffusers.pipelines.rewardflow.rewardslider_v2_optimization import hybrid_acceptance_step
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.Adam([parameter], lr=1.0)
    before = parameter.detach().clone()
    result = hybrid_acceptance_step(
        parameter,
        optimizer,
        current_kl=torch.tensor(0.1),
        step=lambda: parameter.data.add_(1.0),
        evaluate=lambda: torch.tensor(0.5),
        tolerance=0.0,
    )
    assert not result.accepted
    torch.testing.assert_close(parameter, before)
    assert result.rejected_steps == 1



def test_local_insert_line_search_balances_the_new_interval():
    from diffusers.pipelines.rewardflow.rewardslider_v2_optimization import local_insert_line_search
    alphas = torch.tensor([0.0, 0.2, 0.8, 1.0])
    def evaluate(candidate):
        value = candidate[2]
        left = value - 0.2
        right = 0.8 - value
        return left.abs(), right.abs(), (left.abs() - right.abs()).square()
    result = local_insert_line_search(alphas, interval=1, evaluate=evaluate)
    torch.testing.assert_close(result.alpha, torch.tensor(0.5))
    assert result.balance_ratio < 0.01
