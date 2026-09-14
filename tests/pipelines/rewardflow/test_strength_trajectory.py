from types import SimpleNamespace

import pytest
import torch

from diffusers.pipelines.rewardflow.pipeline_output import StrengthTrajectoryPipelineOutput
from diffusers.pipelines.rewardflow.pipeline_rewardflow_flux import FluxRewardFlowPipeline
from diffusers.pipelines.rewardflow.strength_trajectory import (
    StrengthRewardContext,
    StrengthRewardGuidance,
    StrengthTrajectoryConfig,
    expand_for_strengths,
    expand_shared_initial_latents,
    flatten_strength_branches,
    group_trajectory_images,
    make_flat_strength_tensor,
    max_shared_noise_difference,
    sample_shared_langevin_noise,
    unflatten_strength_branches,
    validate_trajectory_mode,
)


class ToyMeanIntensityStrengthReward:
    """Infrastructure-only toy reward with no image-editing meaning."""

    def __call__(self, *, image, target_strength, context):
        progress = image.flatten(1).mean(dim=1)
        return -(progress - target_strength).square()


def _config(**kwargs):
    return StrengthTrajectoryConfig(enabled=True, use_shared_sde_noise=False, **kwargs)


@pytest.mark.parametrize("strength", [0.0, 1.0, -0.1, 1.1])
def test_strength_config_requires_strict_interior_strengths(strength):
    with pytest.raises(ValueError, match="strictly inside"):
        _config(strengths=(strength,)).validate()


@pytest.mark.parametrize("strengths", [(0.4, 0.2), (0.2, 0.2)])
def test_strength_config_requires_sorted_unique_strengths(strengths):
    with pytest.raises(ValueError, match="strictly increasing"):
        _config(strengths=strengths).validate()


def test_strength_config_requires_explicit_shared_noise_parameters():
    with pytest.raises(ValueError, match="gamma_min"):
        StrengthTrajectoryConfig(enabled=True, use_shared_sde_noise=True).validate()


def test_expand_for_strengths_uses_b_major_k_minor_order():
    base = torch.tensor([[10], [20]])

    expanded = expand_for_strengths(base, 3)

    assert torch.equal(expanded, torch.tensor([[10], [10], [10], [20], [20], [20]]))


def test_strength_flatten_unflatten_roundtrip():
    branches = torch.arange(2 * 3 * 4).reshape(2, 3, 4)

    flat = flatten_strength_branches(branches)
    restored = unflatten_strength_branches(flat, base_batch_size=2, num_strengths=3)

    assert torch.equal(restored, branches)


def test_shared_initial_latents_are_identical_across_strengths():
    base = torch.randn(2, 5, 4)
    expanded = expand_shared_initial_latents(base, 4)
    grouped = unflatten_strength_branches(expanded, 2, 4)

    assert torch.equal(grouped[:, 0], base)
    assert torch.equal(grouped, grouped[:, :1].expand_as(grouped))


def test_changing_num_strengths_does_not_change_base_initial_latent():
    def sample_and_expand(num_strengths):
        generator = torch.Generator().manual_seed(7)
        base = torch.randn(2, 5, 4, generator=generator)
        return unflatten_strength_branches(expand_shared_initial_latents(base, num_strengths), 2, num_strengths)

    branches_k2 = sample_and_expand(2)
    branches_k5 = sample_and_expand(5)

    assert torch.equal(branches_k2[:, 0], branches_k5[:, 0])


def test_make_flat_strength_tensor_is_b_major_k_minor():
    actual = make_flat_strength_tensor((0.2, 0.4, 0.8), 2, device="cpu", dtype=torch.float32)

    torch.testing.assert_close(actual, torch.tensor([0.2, 0.4, 0.8, 0.2, 0.4, 0.8]))


def test_shared_langevin_noise_is_bitwise_identical_across_strengths():
    noise = sample_shared_langevin_noise(
        torch.zeros(2 * 4, 3, 2),
        gamma=0.2,
        eta=0.1,
        base_batch_size=2,
        num_strengths=4,
        generator=torch.Generator().manual_seed(7),
    )
    grouped = unflatten_strength_branches(noise, 2, 4)

    assert torch.equal(grouped, grouped[:, :1].expand_as(grouped))


def test_shared_noise_differs_between_base_samples():
    noise = sample_shared_langevin_noise(
        torch.zeros(2 * 3, 8),
        gamma=0.2,
        eta=0.1,
        base_batch_size=2,
        num_strengths=3,
        generator=torch.Generator().manual_seed(11),
    )
    grouped = unflatten_strength_branches(noise, 2, 3)

    assert not torch.equal(grouped[0, 0], grouped[1, 0])


def test_shared_noise_generator_consumption_is_invariant_to_num_strengths():
    def draw(num_strengths):
        sample = torch.zeros(2 * num_strengths, 5)
        flat = sample_shared_langevin_noise(
            sample,
            gamma=0.2,
            eta=0.1,
            base_batch_size=2,
            num_strengths=num_strengths,
            generator=torch.Generator().manual_seed(19),
        )
        return unflatten_strength_branches(flat, 2, num_strengths)[:, 0]

    assert torch.equal(draw(2), draw(5))


def test_shared_noise_generator_list_must_match_base_batch():
    generators = [torch.Generator().manual_seed(index) for index in range(6)]

    with pytest.raises(ValueError, match="one generator per base sample"):
        sample_shared_langevin_noise(
            torch.zeros(6, 2),
            gamma=0.2,
            eta=0.1,
            base_batch_size=2,
            num_strengths=3,
            generator=generators,
        )


@pytest.mark.parametrize(
    "invalid_output",
    [
        torch.tensor(1.0),
        torch.zeros(2),
        torch.zeros(3),
        torch.zeros(6, 2),
    ],
)
def test_strength_reward_requires_one_scalar_per_branch(invalid_output):
    class InvalidReward:
        def __call__(self, *, image, target_strength, context):
            return invalid_output.to(image.device)

    guidance = StrengthRewardGuidance(InvalidReward())
    with pytest.raises(ValueError, match="one scalar per flat branch"):
        guidance.compute(
            image=torch.zeros(6, 3, 2, 2),
            target_strength=torch.tensor([0.2, 0.4, 0.8, 0.2, 0.4, 0.8]),
            context=StrengthRewardContext(),
        )


def test_toy_strength_reward_produces_strength_specific_gradients():
    image = torch.full((2, 3, 2, 2), 0.5, requires_grad=True)
    targets = torch.tensor([0.2, 0.8])
    rewards = ToyMeanIntensityStrengthReward()(
        image=image,
        target_strength=targets,
        context=StrengthRewardContext(),
    )

    grad = torch.autograd.grad(rewards.sum(), image)[0]

    assert not torch.equal(grad[0], grad[1])
    assert torch.sign(grad[0].mean()) != torch.sign(grad[1].mean())


def test_strength_reward_sum_does_not_rescale_gradient_with_k():
    def first_branch_gradient(num_strengths):
        image = torch.zeros(num_strengths, 3, 2, 2, requires_grad=True)
        targets = torch.linspace(0.2, 0.8, num_strengths)
        rewards = ToyMeanIntensityStrengthReward()(
            image=image,
            target_strength=targets,
            context=StrengthRewardContext(),
        )
        return torch.autograd.grad(rewards.sum(), image)[0][0]

    torch.testing.assert_close(first_branch_gradient(2), first_branch_gradient(5))


def test_strength_reward_gradient_reaches_original_latent():
    latent = torch.tensor([[[[0.1]]], [[[0.1]]]], requires_grad=True)
    clean_prediction = latent - 0.25 * (0.5 * latent)
    decoded = clean_prediction.repeat(1, 3, 2, 2).sigmoid()
    guidance = StrengthRewardGuidance(ToyMeanIntensityStrengthReward())
    rewards = guidance.compute(
        image=decoded,
        target_strength=torch.tensor([0.2, 0.8]),
        context=StrengthRewardContext(prompt="toy"),
    )

    grad = torch.autograd.grad(rewards.sum(), latent)[0]

    assert grad is not None
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0


def test_strength_reward_guidance_routes_without_detach():
    if not torch.cuda.is_available():
        pytest.skip("Cross-device strength reward routing requires CUDA.")
    source_device = torch.device("cuda:0")
    reward_device = torch.device("cuda:1" if torch.cuda.device_count() >= 2 else "cpu")

    class RoutedReward(ToyMeanIntensityStrengthReward):
        def __init__(self):
            self.model = torch.nn.Linear(1, 1, bias=False)

        def maybe_onload(self):
            self.model.to(reward_device)

        def __call__(self, *, image, target_strength, context):
            assert image.device == reward_device
            progress = self.model(image.flatten(1).mean(dim=1, keepdim=True)).squeeze(1)
            return -(progress - target_strength).square()

    image = torch.rand(2, 3, 4, 4, device=source_device, requires_grad=True)
    guidance = StrengthRewardGuidance(RoutedReward())
    rewards = guidance.compute(
        image=image,
        target_strength=torch.tensor([0.2, 0.8], device=source_device),
        context=StrengthRewardContext(),
    )
    grad = torch.autograd.grad(rewards.sum(), image)[0]

    assert rewards.device == source_device
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0


def _trajectory_step(pipe, latents, config, strengths, generator, guidance=None):
    return FluxRewardFlowPipeline._strength_trajectory_step(
        pipe,
        latents=latents,
        step_index=len(pipe.last_strength_trajectory_trace),
        timestep=torch.full((latents.shape[0],), 500.0),
        latent_ids=torch.empty(latents.shape[0], latents.shape[1], 4),
        prompt_embeds=torch.empty(latents.shape[0], 1, 1),
        text_ids=torch.empty(latents.shape[0], 1, 4),
        image_latents=None,
        image_latent_ids=None,
        negative_prompt_embeds=None,
        negative_text_ids=None,
        guidance_scale=1.0,
        trajectory_config=config,
        target_strengths=strengths,
        strength_reward_guidance=guidance,
        strength_reward_context=StrengthRewardContext(prompt="toy"),
        base_batch_size=1,
        num_strengths=len(config.strengths),
        generator=generator,
    )


def test_without_strength_reward_all_branches_remain_identical():
    config = StrengthTrajectoryConfig(
        enabled=True,
        strengths=(0.2, 0.5, 0.8),
        use_shared_sde_noise=True,
        gamma_min=0.1,
        gamma_max=0.3,
        gamma_rho=1.0,
        collect_trace=True,
    )
    pipe = SimpleNamespace(
        scheduler=SimpleNamespace(sigmas=torch.tensor([0.8, 0.5, 0.2])),
        _predict_velocity=lambda latent, *args, **kwargs: 0.25 * latent,
        last_strength_trajectory_trace=[],
    )
    latents = expand_shared_initial_latents(torch.randn(1, 2, 3), 3)
    strengths = make_flat_strength_tensor(config.strengths, 1, device="cpu", dtype=latents.dtype)
    generator = torch.Generator().manual_seed(123)

    latents = _trajectory_step(pipe, latents, config, strengths, generator)
    latents = _trajectory_step(pipe, latents, config, strengths, generator)

    torch.testing.assert_close(latents[0], latents[1], rtol=0, atol=0)
    torch.testing.assert_close(latents[0], latents[2], rtol=0, atol=0)
    assert all(trace["max_shared_noise_difference"] == 0 for trace in pipe.last_strength_trajectory_trace)


def test_strength_branch_permutation_only_permutes_outputs():
    base_latent = torch.full((1, 2, 3), 0.1)

    def run(strengths):
        config = StrengthTrajectoryConfig(
            enabled=True,
            strengths=tuple(strengths),
            lambda_strength_reward=0.4,
            use_shared_sde_noise=False,
        )
        pipe = SimpleNamespace(
            scheduler=SimpleNamespace(sigmas=torch.tensor([0.8, 0.5])),
            _predict_velocity=lambda latent, *args, **kwargs: 0.1 * latent,
            _decode_clean_latent_for_reward=lambda clean, latent_ids: clean.unsqueeze(-1).sigmoid(),
            last_strength_trajectory_trace=[],
        )
        flat = torch.tensor(strengths)
        return _trajectory_step(
            pipe,
            expand_shared_initial_latents(base_latent, 2),
            config,
            flat,
            None,
            StrengthRewardGuidance(ToyMeanIntensityStrengthReward()),
        )

    forward = run((0.2, 0.8))
    reverse = run((0.8, 0.2))

    torch.testing.assert_close(forward[0], reverse[1])
    torch.testing.assert_close(forward[1], reverse[0])


def test_trajectory_mode_is_mutually_exclusive_with_paper_mode():
    config = _config()
    with pytest.raises(ValueError, match="paper-faithful"):
        validate_trajectory_mode(config, paper_enabled=True, legacy_reward_guidance=False, strength_reward=None)


def test_trajectory_mode_is_mutually_exclusive_with_legacy_reward_guidance():
    config = _config()
    with pytest.raises(ValueError, match="legacy"):
        validate_trajectory_mode(config, paper_enabled=False, legacy_reward_guidance=True, strength_reward=None)


def test_trajectory_trace_records_zero_shared_noise_difference():
    config = StrengthTrajectoryConfig(
        enabled=True,
        strengths=(0.2, 0.8),
        use_shared_sde_noise=True,
        gamma_min=0.1,
        gamma_max=0.2,
        gamma_rho=1.0,
        collect_trace=True,
    )
    pipe = SimpleNamespace(
        scheduler=SimpleNamespace(sigmas=torch.tensor([0.8, 0.5])),
        _predict_velocity=lambda latent, *args, **kwargs: 0.1 * latent,
        last_strength_trajectory_trace=[],
    )
    latents = expand_shared_initial_latents(torch.randn(1, 2, 3), 2)
    strengths = torch.tensor(config.strengths)

    _trajectory_step(pipe, latents, config, strengths, torch.Generator().manual_seed(4))

    assert pipe.last_strength_trajectory_trace[0]["max_shared_noise_difference"] == 0


def test_trajectory_output_groups_b_major_k_minor_images():
    images = ["b0s0", "b0s1", "b1s0", "b1s1"]
    output = StrengthTrajectoryPipelineOutput(
        images=images,
        strengths=(0.2, 0.8),
        base_batch_size=2,
        num_strengths=2,
    )

    assert group_trajectory_images(images, 2, 2) == [["b0s0", "b0s1"], ["b1s0", "b1s1"]]
    assert output.grouped_images == [["b0s0", "b0s1"], ["b1s0", "b1s1"]]


def test_max_shared_noise_difference_detects_non_shared_increment():
    noise = torch.tensor([[1.0], [1.0], [2.0], [3.0]])

    assert max_shared_noise_difference(noise, 2, 2) == 1
