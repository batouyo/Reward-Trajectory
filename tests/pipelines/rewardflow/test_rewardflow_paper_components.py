import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.rewardflow.paper_components import (
    PaperRewardFlowConfig,
    clean_latent_kl_energy,
    paper_euler_update,
    paper_gamma_schedule,
    predict_clean_latent,
    sample_langevin_noise,
)
from diffusers.pipelines.rewardflow.pipeline_rewardflow_flux import FluxRewardFlowPipeline
from diffusers.pipelines.rewardflow.rewards import (
    Qwen25VQAReward,
    SigLIPReward,
    StaticRewardGuidance,
    qwen_vqa_token_reward,
)
from diffusers.pipelines.rewardflow.semantic_parser import (
    SemanticParseResult,
    build_semantic_parser_prompt,
    load_cached_parse,
    parse_semantic_parser_json,
    save_cached_parse,
)


def test_predict_clean_latent_matches_flow_scheduler_formula():
    sample = torch.randn(2, 4, 3, dtype=torch.float64, requires_grad=True)
    model_output = torch.randn_like(sample, requires_grad=True)
    sigma = torch.tensor([0.75, 0.25], dtype=sample.dtype)

    actual = predict_clean_latent(sample, model_output, sigma)
    expected = sample - sigma[:, None, None] * model_output

    assert actual.dtype == sample.dtype
    assert actual.device == sample.device
    assert torch.equal(actual, expected)
    actual.sum().backward()
    assert sample.grad is not None
    assert model_output.grad is not None


def test_paper_reverse_drift_matches_scheduler_euler():
    sample = torch.randn(2, 5, 4)
    model_output = torch.randn_like(sample)
    sigma = torch.tensor(0.8)
    sigma_next = torch.tensor(0.55)

    expected = sample + (sigma_next - sigma) * model_output
    actual = paper_euler_update(sample, model_output, sigma, sigma_next)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_paper_reverse_drift_matches_real_scheduler_step(dtype):
    scheduler = FlowMatchEulerDiscreteScheduler()
    scheduler.set_timesteps(2)
    sample = torch.randn(2, 5, 4, dtype=dtype)
    model_output = torch.randn_like(sample)
    sigma = scheduler.sigmas[0]
    sigma_next = scheduler.sigmas[1]

    expected = scheduler.step(model_output, scheduler.timesteps[0], sample, return_dict=False)[0]
    actual = paper_euler_update(sample, model_output, sigma, sigma_next)

    assert torch.equal(actual, expected)


def test_gamma_schedule_is_monotonically_decreasing():
    sigmas = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0])
    gamma = paper_gamma_schedule(sigmas, sigmas[0], gamma_min=0.01, gamma_max=0.2, rho=2.0)

    assert torch.all(gamma[:-1] >= gamma[1:])
    assert gamma[0] == pytest.approx(0.2)
    assert gamma[-1] == pytest.approx(0.01)


def test_langevin_noise_reproducible_with_generator():
    sample = torch.zeros(2, 3)
    first = sample_langevin_noise(sample, 0.2, 0.1, generator=torch.Generator().manual_seed(7))
    second = sample_langevin_noise(sample, 0.2, 0.1, generator=torch.Generator().manual_seed(7))
    third = sample_langevin_noise(sample, 0.2, 0.1, generator=torch.Generator().manual_seed(8))

    assert torch.equal(first, second)
    assert not torch.equal(first, third)


def test_kl_gradient_points_toward_source():
    current = torch.tensor([[2.0, -1.0]], requires_grad=True)
    source = torch.tensor([[0.5, 0.25]])
    clean_pred = current
    energy = clean_latent_kl_energy(clean_pred, source)
    grad = torch.autograd.grad(energy, current)[0]

    before = torch.linalg.vector_norm(clean_pred.detach() - source)
    after = torch.linalg.vector_norm((current.detach() - 0.1 * grad) - source)

    assert after < before


def test_reward_gradient_flows_through_clean_predictor():
    latent = torch.tensor([[0.4, -0.2]], requires_grad=True)
    sigma = torch.tensor(0.6)
    velocity = 2.0 * latent
    clean_pred = predict_clean_latent(latent, velocity, sigma)
    decoded = clean_pred.square()
    reward = decoded.sum()

    grad = torch.autograd.grad(reward, latent)[0]

    assert grad is not None
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0


def test_paper_pipeline_step_matches_deterministic_euler_without_guidance():
    latents = torch.randn(1, 3, 4)
    velocity = torch.randn_like(latents)
    pipe = SimpleNamespace(
        scheduler=SimpleNamespace(sigmas=torch.tensor([0.8, 0.5])),
        _paper_reward_guidance=None,
        _predict_velocity=lambda *args, **kwargs: velocity,
    )
    config = PaperRewardFlowConfig(enabled=True, use_kl=False, use_sde_noise=False)

    actual = FluxRewardFlowPipeline._paper_langevin_step(
        pipe,
        latents=latents,
        step_index=0,
        timestep=torch.tensor([800.0]),
        latent_ids=torch.empty(1, 3, 4),
        prompt_embeds=torch.empty(1, 1, 1),
        text_ids=torch.empty(1, 1, 4),
        image_latents=None,
        image_latent_ids=None,
        negative_prompt_embeds=None,
        negative_text_ids=None,
        guidance_scale=1.0,
        source_clean_latent=None,
        paper_config=config,
        generator=None,
    )
    expected = latents + (0.5 - 0.8) * velocity

    torch.testing.assert_close(actual, expected)


def test_legacy_rewardflow_path_unchanged():
    class RecordingScheduler:
        def __init__(self):
            self.calls = []

        def step(self, model_output, timestep, sample, return_dict):
            self.calls.append((model_output, timestep, sample, return_dict))
            return (sample + model_output,)

    scheduler = RecordingScheduler()
    pipe = SimpleNamespace(scheduler=scheduler)
    sample = torch.tensor([1.0])
    model_output = torch.tensor([2.0])
    timestep = torch.tensor(500.0)

    result = FluxRewardFlowPipeline._legacy_scheduler_step(pipe, model_output, timestep, sample)

    assert torch.equal(result, torch.tensor([3.0]))
    assert scheduler.calls == [(model_output, timestep, sample, False)]
    assert PaperRewardFlowConfig().enabled is False


def test_paper_config_rejects_unknown_hyperparameters():
    with pytest.raises(ValueError, match="lambda_reward"):
        PaperRewardFlowConfig(enabled=True, static_reward_weights={"siglip": 1.0}).validate(
            reward_enabled=True, has_source_image=True
        )
    with pytest.raises(ValueError, match="gamma_min"):
        PaperRewardFlowConfig(enabled=True, use_kl=False, use_sde_noise=True).validate(
            reward_enabled=False, has_source_image=False
        )


def test_static_reward_guidance_uses_exact_named_weights():
    image = torch.tensor([[[[0.25]]]], requires_grad=True)
    guidance = StaticRewardGuidance(
        rewards={
            "linear": lambda image, prompt: image.sum(),
            "quadratic": lambda image, prompt: image.square().sum(),
        },
        weights={"linear": 2.0, "quadratic": -0.5},
    )

    total, values = guidance.compute(image, "unused")
    grad = torch.autograd.grad(total, image)[0]

    assert set(values) == {"linear", "quadratic"}
    assert total.item() == pytest.approx(2.0 * 0.25 - 0.5 * 0.25**2)
    assert grad.item() == pytest.approx(2.0 - 0.25)


def test_paper_reward_step_backpropagates_through_denoiser_without_normalization():
    latents = torch.tensor([[[0.4, -0.2]]])
    sigma = 0.25
    sigma_next = 0.1
    lambda_reward = 2.0
    velocity_scale = 0.5
    reward_guidance = StaticRewardGuidance(
        rewards={"toy": lambda image, prompt: image.sum()},
        weights={"toy": 1.0},
    )
    pipe = SimpleNamespace(
        scheduler=SimpleNamespace(sigmas=torch.tensor([sigma, sigma_next])),
        _paper_reward_guidance=reward_guidance,
        _paper_reward_prompt="toy",
        _predict_velocity=lambda latent, *args, **kwargs: velocity_scale * latent,
        _decode_clean_latent_for_reward=lambda clean, latent_ids: clean,
    )
    config = PaperRewardFlowConfig(
        enabled=True,
        lambda_reward=lambda_reward,
        use_kl=False,
        static_reward_weights={"toy": 1.0},
    )

    actual = FluxRewardFlowPipeline._paper_langevin_step(
        pipe,
        latents=latents,
        step_index=0,
        timestep=torch.tensor([250.0]),
        latent_ids=torch.empty(1, 1, 4),
        prompt_embeds=torch.empty(1, 1, 1),
        text_ids=torch.empty(1, 1, 4),
        image_latents=None,
        image_latent_ids=None,
        negative_prompt_embeds=None,
        negative_text_ids=None,
        guidance_scale=1.0,
        source_clean_latent=None,
        paper_config=config,
        generator=None,
    )

    eta = sigma - sigma_next
    velocity = velocity_scale * latents
    # d(sum(z - sigma * velocity_scale * z))/dz = 1 - sigma * velocity_scale.
    raw_reward_grad = torch.full_like(latents, 1 - sigma * velocity_scale)
    expected = latents + eta * (-velocity + lambda_reward * raw_reward_grad)
    torch.testing.assert_close(actual, expected)


def test_semantic_parser_prompt_and_json_validation():
    instruction = "Add black sunglasses."
    prompt = build_semantic_parser_prompt(instruction)
    result = parse_semantic_parser_json(
        """{
          "short_prompts": [
            "Add black sunglasses",
            "Align glasses to face",
            "Preserve subject identity",
            "Preserve original lighting",
            "Maintain realistic shadows"
          ],
          "qna": {"question": "What is on the face?", "answer": "Black sunglasses."}
        }"""
    )

    assert instruction in prompt
    assert "JSON only" in prompt
    assert result.question == "What is on the face?"
    assert len(result.short_prompts) == 5


@pytest.mark.parametrize(
    "payload, match",
    [
        ('{"short_prompts": [], "qna": {"question": "q", "answer": "a"}}', "5 to 12"),
        (
            '{"short_prompts": ["one two three four five six seven", "b", "c", "d", "e"], '
            '"qna": {"question": "q", "answer": "a"}}',
            "six-word",
        ),
        (
            '{"short_prompts": ["a", "b", "c", "d", "e"], "qna": {"question": "q", "answer": "a"}, "extra": true}',
            "exactly",
        ),
    ],
)
def test_semantic_parser_rejects_invalid_schema(payload, match):
    with pytest.raises(ValueError, match=match):
        parse_semantic_parser_json(payload)


def test_semantic_parser_cache_round_trip(tmp_path):
    path = tmp_path / "nested" / "semantic-cache.json"
    instruction = "Turn the car blue."
    result = SemanticParseResult(
        short_prompts=[
            "Make car blue",
            "Preserve car shape",
            "Keep original viewpoint",
            "Match scene lighting",
            "Retain reflections",
        ],
        question="What color is the car?",
        answer="Blue.",
    )

    assert load_cached_parse(path, instruction) is None
    save_cached_parse(path, instruction, result)

    assert load_cached_parse(path, instruction) == result
    assert load_cached_parse(path, "Different instruction") is None


def test_qwen_vqa_token_objective_matches_negative_ce_plus_margin():
    logits = torch.tensor([[1.0, 3.0, 2.0], [4.0, 0.0, 1.0]], requires_grad=True)
    targets = torch.tensor([1, 2])
    margin = 0.75
    lambda_margin = 0.4

    reward = qwen_vqa_token_reward(
        logits,
        targets,
        margin=margin,
        lambda_margin=lambda_margin,
    )
    correct = torch.tensor([3.0, 1.0])
    max_other = torch.tensor([2.0, 4.0])
    expected = (
        torch.log_softmax(logits, dim=-1)[torch.arange(2), targets]
        - lambda_margin * torch.relu(torch.tensor(margin) - correct + max_other)
    ).mean()
    grad = torch.autograd.grad(reward, logits)[0]

    torch.testing.assert_close(reward, expected)
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0


def test_siglip_reward_gradient_through_paper_step_when_local_model_is_available():
    model_path = os.getenv("REWARDFLOW_SIGLIP_MODEL")
    if not model_path or not Path(model_path).exists():
        pytest.skip("Set REWARDFLOW_SIGLIP_MODEL to run the SigLIP paper-step integration test.")
    if not torch.cuda.is_available():
        pytest.skip("SigLIP integration test requires CUDA.")

    device = torch.device("cuda")
    reward = SigLIPReward(
        model_path,
        device=device,
        dtype=torch.float32,
        local_files_only=True,
    )
    for parameter in reward.model.parameters():
        parameter.requires_grad_(False)
    reward.prepare("a red square", device=device)
    guidance = StaticRewardGuidance({"siglip": reward}, {"siglip": 1.0})
    latents = torch.randn(1, 3, 32, 32, device=device)
    pipe = SimpleNamespace(
        scheduler=SimpleNamespace(sigmas=torch.tensor([0.25, 0.1], device=device)),
        _paper_reward_guidance=guidance,
        _paper_reward_prompt="a red square",
        _predict_velocity=lambda latent, *args, **kwargs: 0.1 * latent,
        _decode_clean_latent_for_reward=lambda clean, latent_ids: clean.sigmoid(),
        last_paper_trace=[],
    )
    config = PaperRewardFlowConfig(
        enabled=True,
        lambda_reward=0.01,  # Arbitrary test-only coefficient; not a paper default.
        use_kl=False,
        collect_trace=True,
        static_reward_weights={"siglip": 1.0},
    )

    updated = FluxRewardFlowPipeline._paper_langevin_step(
        pipe,
        latents=latents,
        step_index=0,
        timestep=torch.tensor([250.0], device=device),
        latent_ids=torch.empty(1, 32 * 32, 4, device=device),
        prompt_embeds=torch.empty(1, 1, 1, device=device),
        text_ids=torch.empty(1, 1, 4, device=device),
        image_latents=None,
        image_latent_ids=None,
        negative_prompt_embeds=None,
        negative_text_ids=None,
        guidance_scale=1.0,
        source_clean_latent=None,
        paper_config=config,
        generator=None,
    )

    assert updated.shape == latents.shape
    assert torch.isfinite(updated).all()
    assert pipe.last_paper_trace[0]["reward_grad_norm"] > 0


def test_qwen_vqa_image_gradient_when_local_model_is_available():
    model_path = os.getenv("REWARDFLOW_QWEN_VQA_MODEL")
    if not model_path or not Path(model_path).exists():
        pytest.skip("Set REWARDFLOW_QWEN_VQA_MODEL to run the Qwen2.5-VL image-gradient integration test.")
    if not torch.cuda.is_available():
        pytest.skip("Qwen2.5-VL integration test requires CUDA.")

    reward_model = Qwen25VQAReward(
        question="What color is the square?",
        answer="Red.",
        model_id=model_path,
        margin=0.5,
        lambda_margin=0.25,
        device="cuda",
        dtype=torch.bfloat16,
        local_files_only=True,
    )
    image = torch.rand(1, 3, 56, 56, device="cuda", requires_grad=True)
    native_pixels, native_grid = reward_model._differentiable_image_inputs(image)
    reference = reward_model.processor.image_processor(
        images=image.detach().cpu(),
        do_rescale=False,
        return_tensors="pt",
    )
    torch.testing.assert_close(native_grid.cpu(), reference["image_grid_thw"])
    torch.testing.assert_close(native_pixels.detach().cpu(), reference["pixel_values"], rtol=1e-4, atol=1e-4)

    reward = reward_model(image, "unused")
    grad = torch.autograd.grad(reward, image)[0]

    assert grad is not None
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0


def test_paper_trace_contains_auditable_step_values():
    latents = torch.randn(1, 2, 3)
    pipe = SimpleNamespace(
        scheduler=SimpleNamespace(sigmas=torch.tensor([0.7, 0.2])),
        _paper_reward_guidance=None,
        _predict_velocity=lambda latent, *args, **kwargs: 0.25 * latent,
        last_paper_trace=[],
    )
    config = PaperRewardFlowConfig(enabled=True, use_kl=False, collect_trace=True)

    FluxRewardFlowPipeline._paper_langevin_step(
        pipe,
        latents=latents,
        step_index=0,
        timestep=torch.tensor([700.0]),
        latent_ids=torch.empty(1, 2, 4),
        prompt_embeds=torch.empty(1, 1, 1),
        text_ids=torch.empty(1, 1, 4),
        image_latents=None,
        image_latent_ids=None,
        negative_prompt_embeds=None,
        negative_text_ids=None,
        guidance_scale=1.0,
        source_clean_latent=None,
        paper_config=config,
        generator=None,
    )

    assert set(pipe.last_paper_trace[0]) == {
        "step",
        "timestep",
        "sigma",
        "sigma_next",
        "eta",
        "gamma",
        "total_reward",
        "each_reward_value",
        "reward_grad_norm",
        "kl_energy",
        "kl_grad_norm",
        "backbone_drift_norm",
        "langevin_noise_norm",
        "clean_pred_norm",
        "latent_norm",
    }
