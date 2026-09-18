import pytest
import torch

from diffusers.pipelines.rewardflow.rewardslider_v2_topology import (
    TopologyManager,
    rebuild_topology_optimization_state,
    rebuild_topology_optimizer,
)


def _goals(interior=3):
    return tuple(torch.arange(interior * 2, dtype=torch.float32).reshape(interior, 2, 1) for _ in range(4))


def test_insert_uses_worst_interval_midpoint_and_preserves_order():
    manager = TopologyManager(max_nodes=10)
    alphas = torch.tensor([0.0, 0.2, 0.4, 0.9, 1.0])
    new_alphas, new_goals, event = manager.insert_node(alphas, _goals(), torch.tensor([0.1, 0.1, 0.8, 0.1]))
    torch.testing.assert_close(new_alphas, torch.tensor([0.0, 0.2, 0.4, 0.65, 0.9, 1.0]))
    assert new_goals[0].shape[0] == 4
    assert torch.equal(new_goals[0][2], torch.zeros_like(new_goals[0][2]))
    assert event.old_nodes == 5 and event.new_nodes == 6 and event.affected_interval == 2


def test_insert_respects_max_nodes_and_prune_minimum():
    manager = TopologyManager(max_nodes=5, min_nodes=3)
    with pytest.raises(ValueError):
        manager.insert_node(torch.linspace(0, 1, 5), _goals(), torch.ones(4))
    with pytest.raises(ValueError):
        manager.prune_node(torch.tensor([0.0, 0.5, 1.0]), _goals(1), 0)


def test_virtual_removal_accepts_redundant_node_and_rejects_stall_node():
    manager = TopologyManager(max_nodes=10)
    images = torch.tensor([0.0, 0.333, 0.334, 0.667, 1.0]).reshape(5, 1)
    distance = lambda first, second: (first - second).abs().mean()
    redundant = manager.virtual_removal_kl(images, 1, distance)
    assert redundant <= 0.15
    before = manager.trajectory_kl(images, distance)
    stall_images = torch.tensor([0.0, 0.1, 0.2, 0.9, 1.0]).reshape(5, 1)
    stall_before = manager.trajectory_kl(stall_images, distance)
    assert manager.virtual_removal_kl(stall_images, 2, distance) > stall_before


def test_optimizer_rebuild_contains_only_current_parameters():
    alpha = torch.nn.Parameter(torch.zeros(4))
    goals = _goals()
    optimizer = rebuild_topology_optimizer([alpha], goals, alpha_lr=0.01, vgoal_lr=0.01)
    current = {parameter for group in optimizer.param_groups for parameter in group["params"]}
    assert current == {alpha, *goals}


def test_topology_rebuild_replaces_all_stale_parameters_and_scheduler_references():
    manager = TopologyManager(max_nodes=10)
    alphas, goals, _ = manager.insert_node(
        torch.tensor([0.0, 0.2, 0.4, 0.9, 1.0]), _goals(), torch.tensor([0.1, 0.1, 0.8, 0.1])
    )
    state = rebuild_topology_optimization_state(alphas, goals, alpha_lr=0.01, vgoal_lr=0.02)
    current = {parameter for group in state.alpha_optimizer.param_groups for parameter in group["params"]}
    current.update(parameter for group in state.vgoal_optimizer.param_groups for parameter in group["params"])
    assert current == {state.alpha_parameterization.interval_logits, *state.v_goals}
    assert all(parameter not in current for parameter in goals)
    torch.testing.assert_close(state.alpha_parameterization.alphas, alphas)
    assert state.scheduler.alpha_parameterization is state.alpha_parameterization
    assert state.scheduler.v_goal_parameters == tuple(state.v_goals)
