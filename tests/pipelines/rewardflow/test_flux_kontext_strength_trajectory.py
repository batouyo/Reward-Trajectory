import os
from types import SimpleNamespace

import pytest
import torch

from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.flux.pipeline_flux_kontext import FluxKontextPipeline
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.rewardflow.paper_components import paper_euler_update, predict_clean_latent
from diffusers.pipelines.rewardflow.pipeline_flux_kontext_strength_trajectory import (
    FluxKontextStrengthTrajectoryPipeline,
)
from diffusers.pipelines.rewardflow.strength_trajectory import (
    StrengthRewardContext,
    StrengthRewardGuidance,
    StrengthTrajectoryConfig,
    expand_shared_initial_latents,
    make_flat_strength_tensor,
)


class ToyReward:
    def __call__(self, *, image, target_strength, context):
        del context
        return -(image.float().mean(dim=(1, 2, 3)) - target_strength).square()


class _Progress:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def update(self):
        pass


class _ToyTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.125))
        self.config = SimpleNamespace(in_channels=16, guidance_embeds=True)
        self.seen_batches = []

    def forward(self, *, hidden_states, pooled_projections, **kwargs):
        del kwargs
        self.seen_batches.append(hidden_states.shape[0])
        conditioning = pooled_projections[:, :1, None].expand_as(hidden_states)
        return (self.weight * hidden_states + 0.01 * conditioning,)


class _ToyVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.config = SimpleNamespace(scaling_factor=1.0, shift_factor=0.0)


class ToyKontextPipeline(FluxKontextStrengthTrajectoryPipeline):
    """Small deterministic harness exercising both complete denoising loops."""

    def __init__(self):
        DiffusionPipeline.__init__(self)
        self.register_to_config()
        self.transformer = _ToyTransformer()
        self.vae = _ToyVAE()
        self.text_encoder = torch.nn.Linear(1, 1)
        self.text_encoder_2 = torch.nn.Linear(1, 1)
        self.image_encoder = None
        self.scheduler = FlowMatchEulerDiscreteScheduler()
        self.vae_scale_factor = 1
        self.latent_channels = 4
        self.default_sample_size = 16

    @property
    def _execution_device(self):
        return torch.device("cpu")

    def check_inputs(self, *args, **kwargs):
        pass

    def encode_prompt(
        self,
        prompt,
        prompt_2,
        device,
        num_images_per_prompt,
        prompt_embeds,
        pooled_prompt_embeds,
        max_sequence_length,
        lora_scale,
    ):
        del prompt_2, max_sequence_length, lora_scale
        batch_size = len(prompt) if isinstance(prompt, list) else 1
        batch_size *= num_images_per_prompt
        if prompt_embeds is None:
            prompt_embeds = torch.arange(batch_size * 8, dtype=torch.float32, device=device).reshape(batch_size, 2, 4)
            pooled_prompt_embeds = torch.arange(batch_size * 4, dtype=torch.float32, device=device).reshape(
                batch_size, 4
            )
        return prompt_embeds, pooled_prompt_embeds, torch.zeros(2, 3, device=device)

    def prepare_latents(
        self,
        image,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents,
    ):
        del image, num_channels_latents, height, width, generator
        if latents is None:
            raise AssertionError("The toy parity harness requires explicit latents.")
        source = torch.full((batch_size, 2, 4), 0.25, dtype=dtype, device=device)
        return latents.to(device=device, dtype=dtype), source, torch.zeros(3, 3), torch.ones(2, 3)

    def progress_bar(self, total):
        del total
        return _Progress()

    def maybe_free_model_hooks(self):
        pass


def _bind_step_helpers(pipe):
    pipe._select_batch_tensor = FluxKontextStrengthTrajectoryPipeline._select_batch_tensor
    pipe._select_ip_adapter_embeds = FluxKontextStrengthTrajectoryPipeline._select_ip_adapter_embeds
    pipe._kontext_strength_trajectory_chunk = lambda **kwargs: (
        FluxKontextStrengthTrajectoryPipeline._kontext_strength_trajectory_chunk(pipe, **kwargs)
    )
    return pipe


def _run_toy_step(*, num_strengths=3, chunk_size=None, seed=123, reward=False, materialized=True):
    base_batch_size = 2
    strengths = tuple((index + 1) / (num_strengths + 1) for index in range(num_strengths))
    config = StrengthTrajectoryConfig(
        enabled=True,
        strengths=strengths,
        lambda_strength_reward=0.3 if reward else 0.0,
        use_shared_sde_noise=True,
        gamma_min=0.1,
        gamma_max=0.2,
        gamma_rho=1.0,
        branch_chunk_size=chunk_size,
        collect_trace=True,
    )
    pipe = _bind_step_helpers(
        SimpleNamespace(
            scheduler=SimpleNamespace(sigmas=torch.tensor([0.8, 0.5])),
            last_strength_trajectory_trace=[],
            _predict_kontext_velocity=lambda latent, timestep, **kwargs: 0.1 * latent,
            _decode_kontext_clean_latent_for_reward=lambda clean, height, width: clean.transpose(1, 2)
            .reshape(clean.shape[0], 3, 2, 2)
            .sigmoid(),
        )
    )
    base = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3) / 30
    latents = expand_shared_initial_latents(base, num_strengths) if materialized else base
    effective_batch = latents.shape[0]

    def batch_condition(condition_width):
        return torch.arange(base_batch_size * condition_width, dtype=torch.float32).reshape(
            base_batch_size, condition_width
        )

    prompt_embeds = batch_condition(2)[:, None, :]
    pooled_prompt_embeds = batch_condition(2)
    image_latents = base[:, :1]
    guidance = torch.ones(base_batch_size)
    if materialized:
        (
            _,
            image_latents,
            prompt_embeds,
            pooled_prompt_embeds,
            guidance,
            _,
            _,
            _,
            _,
        ) = FluxKontextStrengthTrajectoryPipeline._materialize_kontext_strength_batch(
            num_strengths,
            base,
            image_latents,
            prompt_embeds,
            pooled_prompt_embeds,
            guidance,
            None,
            None,
            None,
            None,
        )
    generator = torch.Generator().manual_seed(seed)
    reward_guidance = StrengthRewardGuidance(ToyReward()) if reward else None
    updated = FluxKontextStrengthTrajectoryPipeline._kontext_strength_trajectory_step(
        pipe,
        latents=latents,
        step_index=0,
        timestep=torch.tensor(500.0),
        image_latents=image_latents,
        image_ids=torch.zeros(5, 3),
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        text_ids=torch.zeros(1, 3),
        guidance=guidance,
        do_true_cfg=False,
        true_cfg_scale=1.0,
        negative_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        negative_text_ids=None,
        image_embeds=None,
        negative_image_embeds=None,
        trajectory_config=config,
        target_strengths=make_flat_strength_tensor(strengths, base_batch_size, device="cpu"),
        strength_reward_guidance=reward_guidance,
        strength_reward_context=StrengthRewardContext(prompt=["A", "B"]),
        base_batch_size=base_batch_size,
        num_strengths=num_strengths,
        generator=generator,
        reward_active=reward,
        branches_materialized=materialized,
        height=2,
        width=2,
    )
    return updated, torch.rand(4, generator=generator), pipe.last_strength_trajectory_trace[0], effective_batch


def test_disabled_trajectory_directly_delegates_to_official_pipeline(monkeypatch):
    sentinel = object()
    calls = []

    def official_call(self, **kwargs):
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(FluxKontextPipeline, "__call__", official_call)
    pipe = object.__new__(FluxKontextStrengthTrajectoryPipeline)

    assert pipe(prompt="edit", trajectory_config=None) is sentinel
    assert pipe(prompt="edit", trajectory_config={"enabled": False}) is sentinel
    assert [call["prompt"] for call in calls] == ["edit", "edit"]


def test_kontext_materialization_is_b_major_k_minor_and_ids_stay_shared():
    batch_values = torch.tensor([[1.0], [2.0]])
    ip_values = [torch.tensor([[[10.0]], [[20.0]]])]
    outputs = FluxKontextStrengthTrajectoryPipeline._materialize_kontext_strength_batch(
        3,
        batch_values[:, None, :],
        batch_values[:, None, :],
        batch_values[:, None, :],
        batch_values,
        batch_values[:, 0],
        batch_values[:, None, :],
        batch_values,
        ip_values,
        ip_values,
    )
    expected = torch.tensor([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    for tensor in outputs[:7]:
        assert torch.equal(tensor.reshape(6, -1)[:, 0], expected)
    assert torch.equal(outputs[7][0].reshape(6, -1)[:, 0], expected * 10)

    shared_image_ids = torch.randn(9, 3)
    shared_text_ids = torch.randn(5, 3)
    assert shared_image_ids.shape == (9, 3)
    assert shared_text_ids.shape == (5, 3)


def test_kontext_velocity_matches_official_source_injection_and_true_cfg():
    calls = []

    def transformer(**kwargs):
        calls.append(kwargs)
        batch, tokens, channels = kwargs["hidden_states"].shape
        value = kwargs["pooled_projections"][:, :1, None]
        return (value.expand(batch, tokens, channels),)

    joint_attention_kwargs = {}
    pipe = SimpleNamespace(
        transformer=transformer,
        _joint_attention_kwargs=joint_attention_kwargs,
        joint_attention_kwargs=joint_attention_kwargs,
    )
    latents = torch.zeros(2, 3, 4)
    source = torch.ones(2, 2, 4)
    image_ids = torch.randn(5, 3)
    text_ids = torch.randn(6, 3)
    negative_text_ids = torch.randn(6, 3)
    velocity = FluxKontextStrengthTrajectoryPipeline._predict_kontext_velocity(
        pipe,
        latents,
        torch.tensor(500.0),
        image_latents=source,
        image_ids=image_ids,
        prompt_embeds=torch.zeros(2, 6, 8),
        pooled_prompt_embeds=torch.full((2, 4), 3.0),
        text_ids=text_ids,
        guidance=torch.full((2,), 2.5),
        do_true_cfg=True,
        true_cfg_scale=2.0,
        negative_prompt_embeds=torch.ones(2, 6, 8),
        negative_pooled_prompt_embeds=torch.full((2, 4), 1.0),
        negative_text_ids=negative_text_ids,
        image_embeds=[torch.full((2, 1, 2), 7.0)],
        negative_image_embeds=[torch.full((2, 1, 2), -7.0)],
    )

    assert torch.equal(velocity, torch.full_like(latents, 5.0))
    assert len(calls) == 2
    assert calls[0]["hidden_states"].shape == (2, 5, 4)
    assert calls[0]["img_ids"] is image_ids
    assert calls[0]["txt_ids"] is text_ids
    assert calls[1]["txt_ids"] is negative_text_ids
    assert torch.equal(calls[0]["timestep"], torch.full((2,), 0.5))
    assert torch.equal(calls[1]["joint_attention_kwargs"]["ip_adapter_image_embeds"][0], torch.full((2, 1, 2), -7.0))


def test_kontext_clean_prediction_matches_flow_scheduler_parameterization():
    sample = torch.randn(2, 5, 4, requires_grad=True)
    velocity = torch.randn_like(sample, requires_grad=True)
    sigma = torch.tensor(0.7)
    clean = predict_clean_latent(sample, velocity, sigma)

    torch.testing.assert_close(clean, sample - sigma * velocity)
    grad_sample, grad_velocity = torch.autograd.grad(clean.sum(), (sample, velocity))
    assert grad_sample.abs().sum() > 0
    assert grad_velocity.abs().sum() > 0


def test_kontext_explicit_euler_matches_official_scheduler_step():
    scheduler = FlowMatchEulerDiscreteScheduler()
    scheduler.set_timesteps(sigmas=[0.8, 0.5], device="cpu")
    sample = torch.randn(2, 5, 4, dtype=torch.bfloat16)
    velocity = torch.randn_like(sample)
    official = scheduler.step(velocity, scheduler.timesteps[0], sample, return_dict=False)[0]
    actual = paper_euler_update(sample, velocity, scheduler.sigmas[0], scheduler.sigmas[1])

    torch.testing.assert_close(actual, official, rtol=0, atol=0)


def test_kontext_k1_zero_drift_full_loop_matches_official_pipeline():
    pipe = ToyKontextPipeline()
    initial = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4) / 10
    common = {
        "image": torch.zeros(1, 4, 1, 1),
        "prompt": "edit",
        "height": 16,
        "width": 16,
        "max_area": 16**2,
        "_auto_resize": False,
        "num_inference_steps": 3,
        "latents": initial,
        "output_type": "latent",
    }
    official = pipe(**common).images
    trajectory = pipe(
        **common,
        trajectory_config=StrengthTrajectoryConfig(
            enabled=True,
            strengths=(0.5,),
            lambda_strength_reward=0.0,
            use_shared_sde_noise=False,
        ),
    ).images

    torch.testing.assert_close(trajectory, official, rtol=0, atol=0)


def test_kontext_full_loop_k_greater_than_one_no_reward_is_identical():
    pipe = ToyKontextPipeline()
    initial = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4) / 10
    output = pipe(
        image=torch.zeros(1, 4, 1, 1),
        prompt="edit",
        height=16,
        width=16,
        max_area=16**2,
        _auto_resize=False,
        num_inference_steps=3,
        latents=initial,
        output_type="latent",
        trajectory_config=StrengthTrajectoryConfig(
            enabled=True,
            strengths=(0.2, 0.4, 0.6, 0.8),
            lambda_strength_reward=0.0,
            use_shared_sde_noise=False,
        ),
    )

    torch.testing.assert_close(output.images, output.images[:1].expand_as(output.images), rtol=0, atol=0)
    assert pipe.transformer.seen_batches[-3:] == [1, 1, 1]


def test_kontext_callback_uses_stable_materialized_branch_batch():
    pipe = ToyKontextPipeline()
    callback_batches = []

    def callback(pipeline, step, timestep, callback_kwargs):
        del pipeline, step, timestep
        callback_batches.append((callback_kwargs["latents"].shape[0], callback_kwargs["prompt_embeds"].shape[0]))
        return {}

    pipe(
        image=torch.zeros(1, 4, 1, 1),
        prompt="edit",
        height=16,
        width=16,
        max_area=16**2,
        _auto_resize=False,
        num_inference_steps=2,
        latents=torch.zeros(1, 3, 4),
        output_type="latent",
        callback_on_step_end=callback,
        callback_on_step_end_tensor_inputs=["latents", "prompt_embeds"],
        trajectory_config=StrengthTrajectoryConfig(
            enabled=True,
            strengths=(0.2, 0.5, 0.8),
            lambda_strength_reward=0.0,
            use_shared_sde_noise=False,
        ),
    )

    assert callback_batches == [(3, 3), (3, 3)]
    assert pipe.transformer.seen_batches == [3, 3]


def test_kontext_reward_decode_preserves_image_gradient():
    class VAE:
        config = SimpleNamespace(scaling_factor=2.0, shift_factor=0.25)

        def decode(self, latent, return_dict=False):
            return (latent * 0.5,)

    pipe = SimpleNamespace(
        vae=VAE(),
        vae_scale_factor=1,
        _unpack_latents=lambda latent, height, width, scale: latent,
    )
    clean = torch.rand(2, 3, 2, 2, requires_grad=True)
    image = FluxKontextStrengthTrajectoryPipeline._decode_kontext_clean_latent_for_reward(pipe, clean, 2, 2)
    gradient = torch.autograd.grad(image.sum(), clean)[0]

    assert image.shape == (2, 3, 2, 2)
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_kontext_reward_gradient_reaches_current_latent():
    updated, _, trace, _ = _run_toy_step(reward=True)

    assert torch.isfinite(updated).all()
    assert min(value for row in trace["branch_reward_grad_norms"] for value in row) > 0


def test_kontext_no_reward_branches_are_identical():
    updated, _, trace, _ = _run_toy_step(num_strengths=4)
    grouped = updated.reshape(2, 4, *updated.shape[1:])

    torch.testing.assert_close(grouped, grouped[:, :1].expand_as(grouped), rtol=0, atol=0)
    assert trace["max_shared_noise_difference"] == 0


@pytest.mark.parametrize("chunk_size", [1, 2, 4])
def test_kontext_chunking_preserves_output_and_rng(chunk_size):
    expected, expected_next, _, _ = _run_toy_step(chunk_size=None)
    actual, actual_next, _, _ = _run_toy_step(chunk_size=chunk_size)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_next, expected_next, rtol=0, atol=0)


@pytest.mark.parametrize("num_strengths", [1, 2, 5])
def test_kontext_different_k_preserves_base_stochastic_path(num_strengths):
    expected, expected_next, _, _ = _run_toy_step(num_strengths=1)
    actual, actual_next, _, _ = _run_toy_step(num_strengths=num_strengths)
    actual_base = actual.reshape(2, num_strengths, *actual.shape[1:])[:, 0]

    torch.testing.assert_close(actual_base, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_next, expected_next, rtol=0, atol=0)


def test_kontext_lazy_no_reward_prefix_runs_only_base_batch():
    updated, _, trace, effective_batch = _run_toy_step(materialized=False)
    expanded = expand_shared_initial_latents(updated, 3)

    assert effective_batch == 2
    assert trace["effective_model_batch"] == 2
    assert trace["branches_materialized"] is False
    assert expanded.shape[0] == 6


def test_kontext_freezes_both_text_encoders_image_encoder_and_reward():
    modules = [torch.nn.Linear(1, 1) for _ in range(6)]
    pipe = SimpleNamespace(
        transformer=modules[0],
        vae=modules[1],
        text_encoder=modules[2],
        text_encoder_2=modules[3],
        image_encoder=modules[4],
    )
    reward = modules[5]
    FluxKontextStrengthTrajectoryPipeline._freeze_kontext_trajectory_modules(pipe, reward)

    assert all(not parameter.requires_grad for module in modules for parameter in module.parameters())


def test_kontext_freezes_reward_before_no_grad_context_preparation():
    class PreparingReward:
        def __init__(self):
            self.model = torch.nn.Linear(1, 1)
            self.frozen_during_prepare = None
            self.grad_enabled_during_prepare = None

        def prepare_context(self, context, *, device, base_batch_size, num_strengths):
            self.frozen_during_prepare = all(not parameter.requires_grad for parameter in self.model.parameters())
            self.grad_enabled_during_prepare = torch.is_grad_enabled()

    reward = PreparingReward()
    pipe = SimpleNamespace(
        transformer=torch.nn.Linear(1, 1),
        vae=torch.nn.Linear(1, 1),
        text_encoder=torch.nn.Linear(1, 1),
        text_encoder_2=torch.nn.Linear(1, 1),
        image_encoder=torch.nn.Linear(1, 1),
    )
    FluxKontextStrengthTrajectoryPipeline._freeze_kontext_trajectory_modules(pipe, reward)
    guidance = FluxKontextStrengthTrajectoryPipeline._prepare_strength_reward_guidance(
        pipe,
        reward,
        StrengthRewardContext(source_endpoint=torch.ones(1, 3, 2, 2, requires_grad=True)),
        lambda_strength_reward=1.0,
        device=torch.device("cpu"),
        base_batch_size=1,
        num_strengths=2,
    )

    assert guidance is not None
    assert reward.frozen_during_prepare is True
    assert reward.grad_enabled_during_prepare is False


def test_real_kontext_k1_matches_official_when_model_is_available():
    model_path = os.getenv("FLUX_KONTEXT_MODEL_PATH")
    if not model_path:
        pytest.skip("FLUX_KONTEXT_MODEL_PATH is not set.")
    if not torch.cuda.is_available():
        pytest.skip("The real Kontext parity smoke test requires CUDA.")

    pipe = FluxKontextStrengthTrajectoryPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to("cuda")
    source = torch.full((1, 3, 64, 64), 0.5, device="cuda")
    common = {
        "image": source,
        "prompt": "Make the object blue.",
        "height": 64,
        "width": 64,
        "max_area": 64**2,
        "_auto_resize": False,
        "num_inference_steps": 2,
        "output_type": "latent",
    }
    official = pipe(**common, generator=torch.Generator(device="cuda").manual_seed(7)).images
    trajectory = pipe(
        **common,
        generator=torch.Generator(device="cuda").manual_seed(7),
        trajectory_config=StrengthTrajectoryConfig(
            enabled=True,
            strengths=(0.5,),
            lambda_strength_reward=0.0,
            use_shared_sde_noise=False,
        ),
    ).images
    difference = (official - trajectory).float().abs()

    assert difference.max() < 1e-5, (
        f"engineering parity failed: max={difference.max().item()}, mean={difference.mean().item()}"
    )
