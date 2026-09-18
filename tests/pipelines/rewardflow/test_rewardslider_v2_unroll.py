import torch

from diffusers.pipelines.rewardflow.rewardslider_v2_unroll import unroll_rewardslider_v2


def _native_velocity(latent, timestep, step_index):
    del timestep
    return (0.1 + 0.03 * step_index) * latent + 0.01


def test_alpha_one_zero_goal_matches_native_trajectory_exactly():
    initial = torch.tensor([[[0.2, -0.1], [0.4, 0.3]]])
    timesteps = [torch.tensor(3.0), torch.tensor(2.0), torch.tensor(1.0), torch.tensor(0.0), torch.tensor(-1.0)]
    sigmas = torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2, 0.0])
    alphas = torch.ones(2)
    goals = [torch.zeros(2, 2, 2) for _ in range(4)]
    actual = unroll_rewardslider_v2(initial, timesteps, sigmas, _native_velocity, alphas, goals, use_checkpointing=False)
    expected = unroll_rewardslider_v2(initial, timesteps, sigmas, _native_velocity, alphas, goals, control_steps=0, use_checkpointing=False)
    torch.testing.assert_close(actual.final_latent, expected.final_latent, rtol=0, atol=0)

def test_fp32_zero_goal_preserves_half_precision_native_parity():
    initial = torch.tensor([[[0.2, -0.1]]], dtype=torch.bfloat16)
    timesteps = [torch.tensor(1.0), torch.tensor(0.0), torch.tensor(-1.0)]
    sigmas = torch.tensor([1.0, 0.7, 0.3, 0.0])
    alphas = torch.ones(1, dtype=torch.float32)
    goals = [torch.zeros(1, 1, 2, dtype=torch.float32) for _ in range(2)]
    actual = unroll_rewardslider_v2(initial, timesteps, sigmas, _native_velocity, alphas, goals, control_steps=2, use_checkpointing=False)
    expected = unroll_rewardslider_v2(initial, timesteps, sigmas, _native_velocity, alphas, goals, control_steps=0, use_checkpointing=False)
    torch.testing.assert_close(actual.final_latent, expected.final_latent, rtol=0, atol=0)


def test_only_first_four_steps_use_v2_scaffold():
    initial = torch.zeros(1, 1, 1)
    timesteps = [torch.tensor(float(i)) for i in range(6)]
    sigmas = torch.tensor([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    alphas = torch.zeros(1)
    goals = [torch.ones(1, 1, 1) for _ in range(4)]
    result = unroll_rewardslider_v2(initial, timesteps, sigmas, _native_velocity, alphas, goals, use_checkpointing=False)
    assert len(result.controlled_step_indices) == 4
    torch.testing.assert_close(result.actual_velocities[4], result.native_edit_velocities[4])
    torch.testing.assert_close(result.actual_velocities[5], result.native_edit_velocities[5])
    assert not torch.equal(result.actual_velocities[3], result.native_edit_velocities[3])


def test_terminal_loss_reaches_alpha_and_every_controlled_goal():
    initial = torch.tensor([[[0.2, 0.1]]])
    timesteps = [torch.tensor(float(i)) for i in range(6)]
    sigmas = torch.tensor([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    alphas = torch.tensor([0.4], requires_grad=True)
    goals = [torch.nn.Parameter(torch.full((1, 1, 2), 0.01 * (i + 1))) for i in range(4)]
    result = unroll_rewardslider_v2(initial, timesteps, sigmas, _native_velocity, alphas, goals, use_checkpointing=False)
    result.final_latent.square().mean().backward()
    assert alphas.grad is not None and torch.isfinite(alphas.grad).all() and alphas.grad.abs().sum() > 0
    for goal in goals:
        assert goal.grad is not None and torch.isfinite(goal.grad).all() and goal.grad.abs().sum() > 0


def test_frozen_native_model_receives_no_gradient_while_inputs_do():
    model = torch.nn.Linear(2, 2, bias=False)
    model.requires_grad_(False)
    initial = torch.ones(1, 1, 2)
    timesteps = [torch.tensor(float(i)) for i in range(5)]
    sigmas = torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2, 0.0])
    alphas = torch.tensor([0.7], requires_grad=True)
    goals = [torch.nn.Parameter(torch.zeros(1, 1, 2)) for _ in range(4)]
    def velocity(latent, timestep, step_index):
        del timestep, step_index
        return model(latent)
    result = unroll_rewardslider_v2(initial, timesteps, sigmas, velocity, alphas, goals, use_checkpointing=False)
    result.final_latent.sum().backward()
    assert alphas.grad is not None
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(goal.grad is not None for goal in goals)
