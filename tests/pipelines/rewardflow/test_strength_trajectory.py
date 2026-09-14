from types import SimpleNamespace

import pytest
import torch

from diffusers.pipelines.rewardflow.pipeline_output import StrengthTrajectoryPipelineOutput
from diffusers.pipelines.rewardflow.pipeline_rewardflow_flux import FluxRewardFlowPipeline
from diffusers.pipelines.rewardflow.strength_trajectory import (
    StrengthRewardContext,
    StrengthRewardGuidance,
    StrengthTrajectoryConfig,
    align_strength_reward_context,
    assert_branch_local_reward,
    expand_for_strengths,
    expand_shared_initial_latents,
    flatten_strength_branches,
    group_trajectory_images,
    is_strength_reward_step,
    make_flat_strength_tensor,
    make_strength_branch_layout,
    max_shared_noise_difference,
    sample_shared_langevin_noise,
    unflatten_strength_branches,
    validate_strength_reward_context,
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


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("reward_start_step", -1, "non-negative"),
        ("reward_every_n_steps", 0, "at least 1"),
        ("branch_chunk_size", 0, "positive"),
    ],
)
def test_strength_config_validates_reward_schedule_and_chunk_size(field, value, message):
    config = _config(**{field: value})

    with pytest.raises(ValueError, match=message):
        config.validate()


def test_strength_reward_step_schedule():
    config = _config(lambda_strength_reward=1.0, reward_start_step=2, reward_every_n_steps=2)

    assert [is_strength_reward_step(step, config, has_reward=True) for step in range(7)] == [
        False,
        False,
        True,
        False,
        True,
        False,
        True,
    ]
    assert not is_strength_reward_step(2, config, has_reward=False)
    config.lambda_strength_reward = 0
    assert not is_strength_reward_step(2, config, has_reward=True)


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


def test_target_strength_is_float32_even_when_latents_are_bfloat16():
    strengths = make_flat_strength_tensor((0.2, 0.8), 1, device="cpu", dtype=torch.bfloat16)

    assert strengths.dtype == torch.float32


@pytest.mark.parametrize("endpoint", ([[[1.0]]], ["image"], object()))
def test_endpoint_context_requires_tensor(endpoint):
    context = StrengthRewardContext(source_endpoint=endpoint)

    with pytest.raises(TypeError, match="torch.Tensor"):
        validate_strength_reward_context(context, base_batch_size=1)


def test_non_tensor_endpoint_is_not_silently_dropped():
    context = StrengthRewardContext(target_endpoint="not-a-tensor")
    layout = make_strength_branch_layout(torch.arange(2), 1, 2)

    with pytest.raises(TypeError, match="torch.Tensor"):
        align_strength_reward_context(context, layout, "cpu")


def test_endpoint_context_requires_base_batch_shape():
    context = StrengthRewardContext(source_endpoint=torch.zeros(6, 3, 2, 2))

    with pytest.raises(ValueError, match=r"B=2"):
        validate_strength_reward_context(context, base_batch_size=2)

    with pytest.raises(ValueError, match=r"\[B, C, H, W\]"):
        validate_strength_reward_context(StrengthRewardContext(source_endpoint=torch.zeros(2, 3)), base_batch_size=2)


def test_endpoint_alignment_matches_b_major_k_minor():
    source = torch.tensor([[[[1.0]]], [[[2.0]]]])
    target = torch.tensor([[[[10.0]]], [[[20.0]]]])
    context = StrengthRewardContext(source_endpoint=source, target_endpoint=target)
    layout = make_strength_branch_layout(torch.arange(6), base_batch_size=2, num_strengths=3)

    aligned = align_strength_reward_context(context, layout, "cpu")

    assert aligned.source_endpoint[:, 0, 0, 0].tolist() == [1, 1, 1, 2, 2, 2]
    assert aligned.target_endpoint[:, 0, 0, 0].tolist() == [10, 10, 10, 20, 20, 20]
    assert aligned.base_indices.tolist() == [0, 0, 0, 1, 1, 1]
    assert aligned.strength_indices.tolist() == [0, 1, 2, 0, 1, 2]


def test_prompt_list_is_base_batch_and_aligned_to_branches():
    context = StrengthRewardContext(prompt=["prompt-A", "prompt-B"])
    layout = make_strength_branch_layout(torch.arange(6), base_batch_size=2, num_strengths=3)

    aligned = align_strength_reward_context(context, layout, "cpu")

    assert aligned.prompt == ["prompt-A", "prompt-A", "prompt-A", "prompt-B", "prompt-B", "prompt-B"]
    with pytest.raises(ValueError, match="length B=2"):
        validate_strength_reward_context(StrengthRewardContext(prompt=["A"]), base_batch_size=2)


def test_chunk_endpoint_alignment_uses_explicit_flat_indices():
    context = StrengthRewardContext(source_endpoint=torch.tensor([[[[1.0]]], [[[2.0]]]]))
    layout = make_strength_branch_layout(torch.tensor([2, 3, 5]), base_batch_size=2, num_strengths=3)

    aligned = align_strength_reward_context(context, layout, "cpu")

    assert aligned.source_endpoint[:, 0, 0, 0].tolist() == [1, 2, 2]
    assert aligned.base_indices.tolist() == [0, 1, 1]
    assert aligned.strength_indices.tolist() == [2, 0, 2]


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
            layout=make_strength_branch_layout(torch.arange(6), 2, 3),
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


def test_branch_local_reward_helper_accepts_toy_reward():
    image = torch.rand(3, 3, 2, 2, requires_grad=True)
    guidance = StrengthRewardGuidance(ToyMeanIntensityStrengthReward())

    assert_branch_local_reward(
        guidance,
        image=image,
        target_strength=torch.tensor([0.2, 0.5, 0.8]),
        context=StrengthRewardContext(),
        layout=make_strength_branch_layout(torch.arange(3), 1, 3),
    )


def test_branch_local_reward_helper_rejects_coupled_reward():
    class CoupledReward:
        def __call__(self, *, image, target_strength, context):
            shared = image.flatten(1).mean()
            local = image.flatten(1).mean(dim=1)
            return local + shared - target_strength

    image = torch.rand(3, 3, 2, 2, requires_grad=True)
    guidance = StrengthRewardGuidance(CoupledReward())

    with pytest.raises(AssertionError, match="another branch"):
        assert_branch_local_reward(
            guidance,
            image=image,
            target_strength=torch.tensor([0.2, 0.5, 0.8]),
            context=StrengthRewardContext(),
            layout=make_strength_branch_layout(torch.arange(3), 1, 3),
        )


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
        layout=make_strength_branch_layout(torch.arange(2), 1, 2),
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
        layout=make_strength_branch_layout(torch.arange(2, device=source_device), 1, 2),
    )
    grad = torch.autograd.grad(rewards.sum(), image)[0]

    assert rewards.device == source_device
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0


def _trajectory_step(pipe, latents, config, strengths, generator, guidance=None):
    if not hasattr(pipe, "_strength_trajectory_chunk"):
        pipe._strength_trajectory_chunk = lambda **kwargs: FluxRewardFlowPipeline._strength_trajectory_chunk(
            pipe, **kwargs
        )
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
        reward_active=guidance is not None,
        branches_materialized=True,
    )


def _run_chunked_toy_step(chunk_size, seed=123):
    base_batch_size = 2
    num_strengths = 3
    config = StrengthTrajectoryConfig(
        enabled=True,
        strengths=(0.2, 0.5, 0.8),
        lambda_strength_reward=0.4,
        use_shared_sde_noise=True,
        gamma_min=0.1,
        gamma_max=0.2,
        gamma_rho=1.0,
        branch_chunk_size=chunk_size,
        collect_trace=True,
    )
    pipe = SimpleNamespace(
        scheduler=SimpleNamespace(sigmas=torch.tensor([0.8, 0.5])),
        _predict_velocity=lambda latent, *args, **kwargs: 0.1 * latent,
        _decode_clean_latent_for_reward=lambda clean, latent_ids: clean.unsqueeze(-1).sigmoid(),
        last_strength_trajectory_trace=[],
    )
    pipe._strength_trajectory_chunk = lambda **kwargs: FluxRewardFlowPipeline._strength_trajectory_chunk(
        pipe, **kwargs
    )
    base = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3) / 20
    latents = expand_shared_initial_latents(base, num_strengths)
    strengths = make_flat_strength_tensor(config.strengths, base_batch_size, device="cpu", dtype=torch.bfloat16)
    generator = torch.Generator().manual_seed(seed)
    updated = FluxRewardFlowPipeline._strength_trajectory_step(
        pipe,
        latents=latents,
        step_index=0,
        timestep=torch.full((base_batch_size * num_strengths,), 500.0),
        latent_ids=torch.empty(base_batch_size * num_strengths, 2, 4),
        prompt_embeds=torch.empty(base_batch_size * num_strengths, 1, 1),
        text_ids=torch.empty(base_batch_size * num_strengths, 1, 4),
        image_latents=None,
        image_latent_ids=None,
        negative_prompt_embeds=None,
        negative_text_ids=None,
        guidance_scale=1.0,
        trajectory_config=config,
        target_strengths=strengths,
        strength_reward_guidance=StrengthRewardGuidance(ToyMeanIntensityStrengthReward()),
        strength_reward_context=StrengthRewardContext(prompt=["A", "B"]),
        base_batch_size=base_batch_size,
        num_strengths=num_strengths,
        generator=generator,
        reward_active=True,
        branches_materialized=True,
    )
    return updated, torch.rand(4, generator=generator), pipe.last_strength_trajectory_trace[0]


def test_no_reward_step_does_not_enable_input_grad_graph():
    seen_requires_grad = []

    def predict_velocity(latent, *args, **kwargs):
        seen_requires_grad.append(latent.requires_grad)
        return 0.25 * latent

    config = StrengthTrajectoryConfig(enabled=True, strengths=(0.2, 0.8), use_shared_sde_noise=False)
    pipe = SimpleNamespace(
        scheduler=SimpleNamespace(sigmas=torch.tensor([0.8, 0.5])),
        _predict_velocity=predict_velocity,
        last_strength_trajectory_trace=[],
    )
    latents = expand_shared_initial_latents(torch.randn(1, 2, 3), 2)
    strengths = torch.tensor(config.strengths)

    updated = _trajectory_step(pipe, latents, config, strengths, None)

    assert seen_requires_grad == [False]
    assert not updated.requires_grad


@pytest.mark.parametrize("chunk_size", [1, 2])
def test_chunked_and_unchunked_toy_step_match(chunk_size):
    expected, _, _ = _run_chunked_toy_step(None)
    actual, _, _ = _run_chunked_toy_step(chunk_size)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("chunk_size", [1, 2, 3])
def test_chunk_size_does_not_change_rng(chunk_size):
    _, expected_next, _ = _run_chunked_toy_step(None)
    _, actual_next, _ = _run_chunked_toy_step(chunk_size)

    torch.testing.assert_close(actual_next, expected_next, rtol=0, atol=0)


def test_chunking_preserves_b_major_k_minor_order():
    expected, _, _ = _run_chunked_toy_step(None)
    actual, _, _ = _run_chunked_toy_step(2)

    torch.testing.assert_close(unflatten_strength_branches(actual, 2, 3), unflatten_strength_branches(expected, 2, 3))


def test_chunking_preserves_branch_reward_gradients():
    _, _, expected_trace = _run_chunked_toy_step(None)
    _, _, actual_trace = _run_chunked_toy_step(1)

    assert actual_trace["branch_reward_grad_norms"] == expected_trace["branch_reward_grad_norms"]


def _run_lazy_schedule(*, lazy, lambda_reward=0.4, reward_start_step=2, seed=91):
    base_batch_size = 2
    num_strengths = 3
    config = StrengthTrajectoryConfig(
        enabled=True,
        strengths=(0.2, 0.5, 0.8),
        lambda_strength_reward=lambda_reward,
        reward_start_step=reward_start_step,
        use_shared_sde_noise=True,
        gamma_min=0.1,
        gamma_max=0.2,
        gamma_rho=1.0,
        lazy_branch_materialization=lazy,
        collect_trace=True,
    )
    model_batches = []

    def predict_velocity(latent, *args, **kwargs):
        model_batches.append(latent.shape[0])
        return 0.1 * latent

    pipe = SimpleNamespace(
        scheduler=SimpleNamespace(sigmas=torch.tensor([0.9, 0.7, 0.5, 0.3, 0.1])),
        _predict_velocity=predict_velocity,
        _decode_clean_latent_for_reward=lambda clean, latent_ids: clean.unsqueeze(-1).sigmoid(),
        last_strength_trajectory_trace=[],
    )
    pipe._strength_trajectory_chunk = lambda **kwargs: FluxRewardFlowPipeline._strength_trajectory_chunk(
        pipe, **kwargs
    )
    latents = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3) / 20
    latent_ids = torch.zeros(2, 2, 4)
    prompt_embeds = torch.zeros(2, 1, 1)
    text_ids = torch.zeros(2, 1, 4)
    image_latents = image_latent_ids = negative_prompt_embeds = negative_text_ids = None
    materialized = False
    if not lazy:
        (
            latents,
            latent_ids,
            prompt_embeds,
            text_ids,
            image_latents,
            image_latent_ids,
            negative_prompt_embeds,
            negative_text_ids,
        ) = FluxRewardFlowPipeline._materialize_strength_trajectory_batch(
            num_strengths,
            latents,
            latent_ids,
            prompt_embeds,
            text_ids,
            image_latents,
            image_latent_ids,
            negative_prompt_embeds,
            negative_text_ids,
        )
        materialized = True

    guidance = StrengthRewardGuidance(ToyMeanIntensityStrengthReward()) if lambda_reward > 0 else None
    context = StrengthRewardContext(prompt=["A", "B"])
    strengths = make_flat_strength_tensor(config.strengths, base_batch_size, device="cpu")
    generator = torch.Generator().manual_seed(seed)
    for step in range(4):
        reward_active = is_strength_reward_step(step, config, has_reward=guidance is not None)
        if reward_active and not materialized:
            (
                latents,
                latent_ids,
                prompt_embeds,
                text_ids,
                image_latents,
                image_latent_ids,
                negative_prompt_embeds,
                negative_text_ids,
            ) = FluxRewardFlowPipeline._materialize_strength_trajectory_batch(
                num_strengths,
                latents,
                latent_ids,
                prompt_embeds,
                text_ids,
                image_latents,
                image_latent_ids,
                negative_prompt_embeds,
                negative_text_ids,
            )
            materialized = True
        latents = FluxRewardFlowPipeline._strength_trajectory_step(
            pipe,
            latents=latents,
            step_index=step,
            timestep=torch.full((latents.shape[0],), 500.0),
            latent_ids=latent_ids,
            prompt_embeds=prompt_embeds,
            text_ids=text_ids,
            image_latents=image_latents,
            image_latent_ids=image_latent_ids,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_text_ids=negative_text_ids,
            guidance_scale=1.0,
            trajectory_config=config,
            target_strengths=strengths,
            strength_reward_guidance=guidance,
            strength_reward_context=context,
            base_batch_size=base_batch_size,
            num_strengths=num_strengths,
            generator=generator,
            reward_active=reward_active,
            branches_materialized=materialized,
        )
    if not materialized:
        latents = expand_shared_initial_latents(latents, num_strengths)
    return latents, torch.rand(4, generator=generator), model_batches, pipe.last_strength_trajectory_trace


def test_lazy_and_eager_are_identical_before_first_reward():
    lazy, _, _, _ = _run_lazy_schedule(lazy=True)
    eager, _, _, _ = _run_lazy_schedule(lazy=False)

    torch.testing.assert_close(lazy, eager, rtol=0, atol=0)


def test_lazy_and_eager_consume_same_rng():
    _, lazy_next, _, _ = _run_lazy_schedule(lazy=True)
    _, eager_next, _, _ = _run_lazy_schedule(lazy=False)

    torch.testing.assert_close(lazy_next, eager_next, rtol=0, atol=0)


def test_no_reward_lazy_path_runs_only_base_batch():
    final, _, batches, trace = _run_lazy_schedule(lazy=True, lambda_reward=0.0)

    assert batches == [2, 2, 2, 2]
    assert final.shape[0] == 6
    assert all(item["effective_model_batch"] == 2 for item in trace)
    assert all(not item["branches_materialized"] for item in trace)


def test_lazy_materialization_removes_redundant_prefix_compute():
    _, _, lazy_batches, lazy_trace = _run_lazy_schedule(lazy=True)
    _, _, eager_batches, eager_trace = _run_lazy_schedule(lazy=False)

    assert lazy_batches == [2, 2, 6, 6]
    assert eager_batches == [6, 6, 6, 6]
    assert [item["reward_active"] for item in lazy_trace] == [False, False, True, True]
    assert [item["branches_materialized"] for item in lazy_trace] == [False, False, True, True]
    assert all(item["branches_materialized"] for item in eager_trace)


def test_strength_reward_context_prepare_called_once_per_pipeline_call():
    class PreparingReward(ToyMeanIntensityStrengthReward):
        def __init__(self):
            self.prepare_calls = 0
            self.clear_calls = 0

        def prepare_context(self, context, *, device, base_batch_size, num_strengths):
            self.prepare_calls += 1

        def clear_context_cache(self):
            self.clear_calls += 1

    reward = PreparingReward()
    guidance = StrengthRewardGuidance(reward)
    context = StrengthRewardContext(prompt=["A", "B"])
    guidance.prepare_context(context, device="cpu", base_batch_size=2, num_strengths=2)
    layout = make_strength_branch_layout(torch.arange(4), 2, 2)
    for _ in range(50):
        guidance.compute(
            image=torch.rand(4, 3, 2, 2),
            target_strength=torch.tensor([0.2, 0.8, 0.2, 0.8]),
            context=context,
            layout=layout,
        )
    guidance.clear_context_cache()

    assert reward.prepare_calls == 1
    assert reward.clear_calls == 1


class PreparationProbeReward:
    def __init__(self):
        self.model = torch.nn.Linear(1, 1, bias=False)
        torch.nn.init.ones_(self.model.weight)
        self.prepare_grad_enabled = None
        self.parameters_frozen_during_prepare = None
        self.prepared_endpoint_requires_grad = None
        self.cached_feature = None
        self.compute_grad_enabled = None

    def prepare_context(self, context, *, device, base_batch_size, num_strengths):
        self.prepare_grad_enabled = torch.is_grad_enabled()
        self.parameters_frozen_during_prepare = all(
            not parameter.requires_grad for parameter in self.model.parameters()
        )
        self.prepared_endpoint_requires_grad = context.source_endpoint.requires_grad
        endpoint_value = context.source_endpoint.flatten(1).mean(dim=1, keepdim=True)
        self.cached_feature = self.model(endpoint_value)

    def __call__(self, *, image, target_strength, context):
        self.compute_grad_enabled = torch.is_grad_enabled()
        progress = self.model(image.flatten(1).mean(dim=1, keepdim=True)).squeeze(1)
        return -(progress - target_strength).square()


def _prepare_probe_guidance():
    reward = PreparationProbeReward()
    pipe = SimpleNamespace(
        transformer=torch.nn.Linear(1, 1),
        vae=torch.nn.Linear(1, 1),
        text_encoder=torch.nn.Linear(1, 1),
    )
    pipe._freeze_trajectory_inference_modules = (
        lambda strength_reward: FluxRewardFlowPipeline._freeze_trajectory_inference_modules(pipe, strength_reward)
    )
    context = StrengthRewardContext(source_endpoint=torch.ones(1, 1, 2, 2, requires_grad=True))
    guidance = FluxRewardFlowPipeline._prepare_strength_reward_guidance(
        pipe,
        reward,
        context,
        lambda_strength_reward=1.0,
        device=torch.device("cpu"),
        base_batch_size=1,
        num_strengths=2,
    )
    return reward, guidance, context


def test_strength_reward_is_frozen_before_context_preparation():
    reward, _, _ = _prepare_probe_guidance()

    assert reward.parameters_frozen_during_prepare is True
    assert all(not parameter.requires_grad for parameter in reward.model.parameters())


def test_strength_reward_context_preparation_disables_autograd():
    reward, _, _ = _prepare_probe_guidance()

    assert reward.prepare_grad_enabled is False
    assert reward.prepared_endpoint_requires_grad is False
    assert reward.cached_feature.requires_grad is False


def test_strength_reward_compute_remains_differentiable_after_context_preparation():
    reward, guidance, context = _prepare_probe_guidance()
    image = torch.rand(2, 3, 2, 2, requires_grad=True)
    rewards = guidance.compute(
        image=image,
        target_strength=torch.tensor([0.2, 0.8]),
        context=context,
        layout=make_strength_branch_layout(torch.arange(2), 1, 2),
    )

    gradient = torch.autograd.grad(rewards.sum(), image)[0]

    assert reward.compute_grad_enabled is True
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


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
