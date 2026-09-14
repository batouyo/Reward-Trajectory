"""Diagnose reward timing and native-drift dominance on real FLUX.1-Kontext.

This opt-in H20 script changes no production pipeline API. Its subclass only
adds measurement, a reward-step mask, and a temporary first-three-step native
drift scale. The color rewards are differentiable diagnostic probes, not
semantic or endpoint-relative edit-strength rewards.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from diffusers import FluxKontextStrengthTrajectoryPipeline
from diffusers.pipelines.rewardflow import StrengthRewardContext, StrengthTrajectoryConfig
from diffusers.pipelines.rewardflow.paper_components import flow_step_size, paper_euler_update, predict_clean_latent


STRENGTHS = (0.2, 0.5, 0.8)


class BlueDirectionStrengthReward:
    """Toy probe that increases B - 0.5 * (R + G)."""

    def __call__(self, *, image, target_strength, context):
        del context
        means = image.float().mean(dim=(2, 3))
        score = means[:, 2] - 0.5 * (means[:, 0] + means[:, 1])
        return target_strength.float() * score


class RedDirectionStrengthReward:
    """Conflicting toy probe that increases R - 0.5 * (G + B)."""

    def __call__(self, *, image, target_strength, context):
        del context
        means = image.float().mean(dim=(2, 3))
        score = means[:, 0] - 0.5 * (means[:, 1] + means[:, 2])
        return target_strength.float() * score


class DiagnosticKontextTrajectoryPipeline(FluxKontextStrengthTrajectoryPipeline):
    """Script-local instrumentation and native-drift scaling only."""

    def configure_diagnostic(
        self,
        *,
        reward_schedule: str,
        early_kontext_scale: float,
        total_steps: int,
        strengths: tuple[float, ...],
    ) -> None:
        self._diagnostic_reward_schedule = reward_schedule
        self._diagnostic_early_kontext_scale = early_kontext_scale
        self._diagnostic_total_steps = total_steps
        self._diagnostic_strengths = strengths
        self._diagnostic_step_index = -1
        self.diagnostic_update_trace = []

    def _diagnostic_reward_active(self, step_index: int) -> bool:
        schedule = self._diagnostic_reward_schedule
        if schedule == "none":
            return False
        if schedule == "all":
            return True
        if schedule == "first":
            return step_index == 0
        if schedule == "first3":
            return step_index < 3
        if schedule == "last3":
            return step_index >= self._diagnostic_total_steps - 3
        raise ValueError(f"Unknown diagnostic reward schedule: {schedule}.")

    def _kontext_strength_trajectory_step(self, **kwargs):
        step_index = kwargs["step_index"]
        self._diagnostic_step_index = step_index
        has_reward = kwargs["strength_reward_guidance"] is not None
        kwargs["reward_active"] = has_reward and self._diagnostic_reward_active(step_index)
        return super()._kontext_strength_trajectory_step(**kwargs)

    def _kontext_strength_trajectory_chunk(self, **kwargs):
        latents = kwargs["latents"]
        sigma = kwargs["sigma"]
        sigma_next = kwargs["sigma_next"]
        reward_active = kwargs["reward_active"]
        trajectory_config = kwargs["trajectory_config"]
        strength_reward_guidance = kwargs["strength_reward_guidance"]
        target_strengths = kwargs["target_strengths"]
        branch_layout = kwargs["branch_layout"]
        strength_reward_context = kwargs["strength_reward_context"]
        forward_kwargs = {
            "image_latents": kwargs["image_latents"],
            "image_ids": kwargs["image_ids"],
            "prompt_embeds": kwargs["prompt_embeds"],
            "pooled_prompt_embeds": kwargs["pooled_prompt_embeds"],
            "text_ids": kwargs["text_ids"],
            "guidance": kwargs["guidance"],
            "do_true_cfg": kwargs["do_true_cfg"],
            "true_cfg_scale": kwargs["true_cfg_scale"],
            "negative_prompt_embeds": kwargs["negative_prompt_embeds"],
            "negative_pooled_prompt_embeds": kwargs["negative_pooled_prompt_embeds"],
            "negative_text_ids": kwargs["negative_text_ids"],
            "image_embeds": kwargs["image_embeds"],
            "negative_image_embeds": kwargs["negative_image_embeds"],
        }

        branch_rewards = None
        reward_grad = torch.zeros_like(latents)
        if reward_active:
            if strength_reward_guidance is None or target_strengths is None or branch_layout is None:
                raise RuntimeError("An active diagnostic reward step requires guidance, targets, and layout.")
            latent_var = latents.detach().requires_grad_(True)
            with torch.enable_grad():
                velocity = self._predict_kontext_velocity(latent_var, kwargs["timestep"], **forward_kwargs)
                clean_pred = predict_clean_latent(latent_var, velocity, sigma)
                clean_image = self._decode_kontext_clean_latent_for_reward(
                    clean_pred,
                    kwargs["height"],
                    kwargs["width"],
                )
                branch_rewards = strength_reward_guidance.compute(
                    image=clean_image,
                    target_strength=target_strengths,
                    context=strength_reward_context,
                    layout=branch_layout,
                )
                reward_grad = torch.autograd.grad(branch_rewards.sum(), latent_var, allow_unused=True)[0]
                if reward_grad is None:
                    raise RuntimeError("Diagnostic reward gradient did not reach the current latent.")
                if not torch.isfinite(reward_grad).all():
                    raise RuntimeError("Diagnostic reward gradient contains non-finite values.")
            base_latents = latent_var.detach()
        else:
            with torch.no_grad():
                velocity = self._predict_kontext_velocity(latents, kwargs["timestep"], **forward_kwargs)
            base_latents = latents

        step_index = self._diagnostic_step_index
        native_scale = self._diagnostic_early_kontext_scale if step_index < 3 else 1.0
        scaled_velocity = velocity.detach() * native_scale
        reward_drift = trajectory_config.lambda_strength_reward * reward_grad.detach() if reward_active else None

        # Counterfactual native-only state and full state use the exact production
        # update helper. Their difference is the BF16-representable marginal
        # reward update actually written to the latent state.
        native_only = paper_euler_update(base_latents, scaled_velocity, sigma, sigma_next)
        updated = paper_euler_update(
            base_latents,
            scaled_velocity,
            sigma,
            sigma_next,
            reward_drift=reward_drift,
            langevin_noise=kwargs["langevin_noise"],
        )
        native_update = native_only.float() - base_latents.float()
        reward_update = updated.float() - native_only.float()

        eta = flow_step_size(sigma, sigma_next, base_latents.float())
        native_precast = -eta * scaled_velocity.float()
        reward_precast = torch.zeros_like(native_precast)
        if reward_drift is not None:
            reward_precast = eta * reward_drift.float()

        native_norms = torch.linalg.vector_norm(native_update.flatten(start_dim=1), dim=1)
        reward_norms = torch.linalg.vector_norm(reward_update.flatten(start_dim=1), dim=1)
        native_precast_norms = torch.linalg.vector_norm(native_precast.flatten(start_dim=1), dim=1)
        reward_precast_norms = torch.linalg.vector_norm(reward_precast.flatten(start_dim=1), dim=1)
        raw_gradient_norms = torch.linalg.vector_norm(reward_grad.float().flatten(start_dim=1), dim=1)
        cosine = F.cosine_similarity(native_update.flatten(start_dim=1), reward_update.flatten(start_dim=1), dim=1)
        precast_cosine = F.cosine_similarity(
            native_precast.flatten(start_dim=1), reward_precast.flatten(start_dim=1), dim=1
        )

        reward_values = (
            branch_rewards.detach().float().tolist() if branch_rewards is not None else [None] * len(latents)
        )
        for index, strength in enumerate(self._diagnostic_strengths):
            native_norm = native_precast_norms[index].item()
            reward_norm = reward_precast_norms[index].item()
            bf16_native_norm = native_norms[index].item()
            bf16_reward_norm = reward_norms[index].item()
            has_direction = native_norm > 0 and reward_norm > 0
            has_bf16_direction = bf16_native_norm > 0 and bf16_reward_norm > 0
            self.diagnostic_update_trace.append(
                {
                    "step": step_index,
                    "sigma": float(torch.as_tensor(sigma).detach().cpu()),
                    "sigma_next": float(torch.as_tensor(sigma_next).detach().cpu()),
                    "eta": float((torch.as_tensor(sigma) - torch.as_tensor(sigma_next)).detach().cpu()),
                    "strength": strength,
                    "native_scale": native_scale,
                    "reward_active": reward_active,
                    "native_update_norm": native_norm,
                    "reward_update_norm": reward_norm,
                    "reward_to_native_ratio": reward_norm / native_norm if native_norm > 0 else None,
                    "native_reward_cosine": precast_cosine[index].item() if has_direction else None,
                    "bf16_effective_native_update_norm": bf16_native_norm,
                    "bf16_effective_reward_marginal_norm": bf16_reward_norm,
                    "bf16_effective_reward_to_native_ratio": (
                        bf16_reward_norm / bf16_native_norm if bf16_native_norm > 0 else None
                    ),
                    "bf16_effective_native_reward_cosine": (cosine[index].item() if has_bf16_direction else None),
                    "raw_reward_gradient_norm": raw_gradient_norms[index].item(),
                    "reward_value": reward_values[index],
                }
            )

        return updated, None if branch_rewards is None else branch_rewards.detach(), raw_gradient_norms


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.getenv("FLUX_KONTEXT_MODEL_PATH"))
    parser.add_argument("--source", required=True)
    parser.add_argument(
        "--prompt",
        default="Make the weighted training ball blue while preserving its shape, texture, lighting, and background.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--lambda-strength-reward", type=float, default=0.1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--measurement-only", action="store_true")
    args = parser.parse_args()
    if not args.model:
        parser.error("Set FLUX_KONTEXT_MODEL_PATH or pass --model.")
    if args.steps < 3:
        parser.error("--steps must be at least 3 for early/late diagnostics.")
    if args.lambda_strength_reward <= 0 or not math.isfinite(args.lambda_strength_reward):
        parser.error("--lambda-strength-reward must be finite and positive.")
    return args


def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    array = image.detach().clamp(0, 1).mul(255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array, mode="RGB")


def _scores(images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    means = images.float().mean(dim=(2, 3))
    blue = means[:, 2] - 0.5 * (means[:, 0] + means[:, 1])
    red = means[:, 0] - 0.5 * (means[:, 1] + means[:, 2])
    return blue, red


def _float_tag(value: float) -> str:
    return format(value, "g").replace(".", "p")


def _make_grid(
    source: Image.Image,
    baseline: Image.Image,
    branch_images: list[Image.Image],
    title: str,
) -> Image.Image:
    labels = ["Source", "Baseline", *(f"strength={strength:g}" for strength in STRENGTHS)]
    images = [source, baseline, *branch_images]
    width, height = source.size
    header = 44
    grid = Image.new("RGB", (len(images) * width, height + header), "white")
    draw = ImageDraw.Draw(grid)
    draw.text((6, 4), title, fill="black")
    for index, (label, image) in enumerate(zip(labels, images)):
        left = index * width
        draw.text((left + 6, 23), label, fill="black")
        grid.paste(image.resize((width, height), Image.Resampling.LANCZOS), (left, header))
    return grid


def _run(
    pipe,
    args,
    source,
    *,
    reward_type: str,
    reward_schedule: str,
    early_kontext_scale: float,
    reward_lambda: float,
):
    pipe.configure_diagnostic(
        reward_schedule=reward_schedule,
        early_kontext_scale=early_kontext_scale,
        total_steps=args.steps,
        strengths=STRENGTHS,
    )
    final_latents = {}
    branch_divergence = []

    def callback(_pipe, step, _timestep, callback_kwargs):
        latents = callback_kwargs["latents"].reshape(1, len(STRENGTHS), *callback_kwargs["latents"].shape[1:])
        reference = latents[:, :1].float()
        branch_divergence.append(
            {
                "step": step,
                "max_abs_from_strength_0.2": [
                    (latents[:, index : index + 1].float() - reference).abs().max().item()
                    for index in range(len(STRENGTHS))
                ],
            }
        )
        if step == args.steps - 1:
            final_latents["value"] = latents.reshape(-1, *latents.shape[2:]).detach().float().cpu()
        return callback_kwargs

    reward = None
    if reward_lambda > 0:
        reward = BlueDirectionStrengthReward() if reward_type == "blue" else RedDirectionStrengthReward()
    config = StrengthTrajectoryConfig(
        enabled=True,
        strengths=STRENGTHS,
        lambda_strength_reward=reward_lambda,
        reward_start_step=0,
        reward_every_n_steps=1,
        use_shared_sde_noise=False,
        lazy_branch_materialization=False,
        branch_chunk_size=None,
        collect_trace=True,
    )
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    torch.cuda.synchronize(args.device)
    started = time.perf_counter()
    output = pipe(
        image=source,
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        max_area=args.height * args.width,
        _auto_resize=False,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.steps,
        output_type="pt",
        generator=generator,
        trajectory_config=config,
        strength_reward=reward,
        strength_reward_context=StrengthRewardContext(prompt=args.prompt),
        callback_on_step_end=callback,
        callback_on_step_end_tensor_inputs=["latents"],
    )
    torch.cuda.synchronize(args.device)
    if "value" not in final_latents:
        raise RuntimeError("Final latent callback was not reached.")
    return {
        "latents": final_latents["value"],
        "images": output.images.detach().float().cpu(),
        "update_trace": list(pipe.diagnostic_update_trace),
        "pipeline_trace": list(pipe.last_strength_trajectory_trace),
        "branch_divergence": branch_divergence,
        "runtime_seconds": time.perf_counter() - started,
    }


def _summarize_run(run, baseline) -> list[dict]:
    blue_scores, red_scores = _scores(run["images"])
    rows = []
    for index, strength in enumerate(STRENGTHS):
        latent_difference = run["latents"][index] - baseline["latents"][index]
        image_difference = run["images"][index] - baseline["images"][index]
        rows.append(
            {
                "strength": strength,
                "final_latent_difference": torch.linalg.vector_norm(latent_difference).item(),
                "final_latent_mean_abs_difference": latent_difference.abs().mean().item(),
                "final_latent_max_abs_difference": latent_difference.abs().max().item(),
                "final_image_difference": image_difference.abs().mean().item(),
                "blue_score": blue_scores[index].item(),
                "red_score": red_scores[index].item(),
            }
        )
    return rows


def _experiment_specs(measurement_only: bool) -> list[dict]:
    specs = [
        {
            "experiment": "baseline",
            "reward_type": "blue",
            "reward_schedule": "none",
            "early_kontext_scale": 1.0,
            "lambda": 0.0,
        },
        {
            "experiment": "measurement_all_steps",
            "reward_type": "blue",
            "reward_schedule": "all",
            "early_kontext_scale": 1.0,
        },
    ]
    if measurement_only:
        return specs
    specs.extend(
        [
            {
                "experiment": "timing_first_step",
                "reward_type": "blue",
                "reward_schedule": "first",
                "early_kontext_scale": 1.0,
            },
            {
                "experiment": "timing_first_3",
                "reward_type": "blue",
                "reward_schedule": "first3",
                "early_kontext_scale": 1.0,
            },
            {
                "experiment": "timing_last_3",
                "reward_type": "blue",
                "reward_schedule": "last3",
                "early_kontext_scale": 1.0,
            },
            *(
                {
                    "experiment": f"control_scale_{_float_tag(scale)}",
                    "reward_type": "blue",
                    "reward_schedule": "none",
                    "early_kontext_scale": scale,
                    "lambda": 0.0,
                }
                for scale in (0.5, 0.25, 0.0)
            ),
            *(
                {
                    "experiment": f"dominance_blue_scale_{_float_tag(scale)}",
                    "reward_type": "blue",
                    "reward_schedule": "first3",
                    "early_kontext_scale": scale,
                }
                for scale in (1.0, 0.5, 0.25, 0.0)
            ),
            *(
                {
                    "experiment": f"dominance_red_scale_{_float_tag(scale)}",
                    "reward_type": "red",
                    "reward_schedule": "first3",
                    "early_kontext_scale": scale,
                }
                for scale in (1.0, 0.5, 0.25)
            ),
        ]
    )
    return specs


def _save_images(output_dir: Path, source: Image.Image, baseline, name: str, run, title: str) -> None:
    run_dir = output_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    baseline_image = _tensor_to_pil(baseline["images"][0])
    baseline_image.save(run_dir / "baseline.png")
    branch_images = []
    for index, strength in enumerate(STRENGTHS):
        image = _tensor_to_pil(run["images"][index])
        image.save(run_dir / f"strength_{_float_tag(strength)}.png")
        branch_images.append(image)
    _make_grid(source, baseline_image, branch_images, title).save(output_dir / f"grid_{name}.png")


def _flatten_csv_rows(experiment: dict, final_rows: list[dict], update_trace: list[dict]) -> list[dict]:
    final_by_strength = {row["strength"]: row for row in final_rows}
    rows = []
    for update in update_trace:
        rows.append(
            {
                "experiment": experiment["experiment"],
                "reward_type": experiment["reward_type"],
                "reward_schedule": experiment["reward_schedule"],
                "early_kontext_scale": experiment["early_kontext_scale"],
                "lambda": experiment["lambda"],
                "comparison_baseline": experiment["comparison_baseline"],
                **update,
                **final_by_strength[update["strength"]],
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Real Kontext diagnostics require CUDA.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source = Image.open(args.source).convert("RGB").resize((args.width, args.height), Image.Resampling.LANCZOS)
    source.save(output_dir / "source.png")

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    torch.cuda.set_device(device)
    print(f"Loading {args.model} on {device} as {dtype}", flush=True)
    pipe = DiagnosticKontextTrajectoryPipeline.from_pretrained(
        args.model,
        torch_dtype=dtype,
        local_files_only=True,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    specs = _experiment_specs(args.measurement_only)
    runs = {}
    normalized_specs = []
    for spec in specs:
        normalized = dict(spec)
        normalized.setdefault("lambda", args.lambda_strength_reward)
        normalized_specs.append(normalized)

    for spec in normalized_specs:
        print(
            f"Running {spec['experiment']}: reward={spec['reward_type']} schedule={spec['reward_schedule']} "
            f"early_scale={spec['early_kontext_scale']:g} lambda={spec['lambda']:g}",
            flush=True,
        )
        runs[spec["experiment"]] = _run(
            pipe,
            args,
            source,
            reward_type=spec["reward_type"],
            reward_schedule=spec["reward_schedule"],
            early_kontext_scale=spec["early_kontext_scale"],
            reward_lambda=spec["lambda"],
        )

    baseline = runs["baseline"]
    all_csv_rows = []
    experiment_reports = []
    for spec in normalized_specs:
        run = runs[spec["experiment"]]
        scale = spec["early_kontext_scale"]
        matched_control_name = "baseline" if scale == 1.0 else f"control_scale_{_float_tag(scale)}"
        matched_control = runs.get(matched_control_name, baseline)
        final_rows = _summarize_run(run, matched_control)
        report_spec = {**spec, "comparison_baseline": matched_control_name}
        all_csv_rows.extend(_flatten_csv_rows(report_spec, final_rows, run["update_trace"]))
        report = {
            **report_spec,
            "runtime_seconds": run["runtime_seconds"],
            "final_metrics": final_rows,
            "global_baseline_final_metrics": _summarize_run(run, baseline),
            "update_trace": run["update_trace"],
            "branch_divergence": run["branch_divergence"],
        }
        experiment_reports.append(report)
        _save_images(
            output_dir,
            source,
            matched_control,
            spec["experiment"],
            run,
            f"{spec['experiment']} | reward={spec['reward_type']} | schedule={spec['reward_schedule']} | "
            f"early scale={spec['early_kontext_scale']:g} | lambda={spec['lambda']:g}",
        )
        print(json.dumps({"experiment": spec["experiment"], "final_metrics": final_rows}, indent=2), flush=True)

    baseline_pairwise = [
        (baseline["latents"][index] - baseline["latents"][0]).abs().max().item() for index in range(len(STRENGTHS))
    ]
    report = {
        "scope": "timing/native-dominance diagnosis with toy color rewards; no production algorithm change",
        "model_path": os.path.realpath(args.model),
        "source_path": os.path.realpath(args.source),
        "prompt": args.prompt,
        "gpu_name": torch.cuda.get_device_name(device),
        "device": str(device),
        "dtype": str(dtype),
        "height": args.height,
        "width": args.width,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "strengths": list(STRENGTHS),
        "baseline_exact_latent_invariant": all(value == 0 for value in baseline_pairwise),
        "baseline_pairwise_max_abs": baseline_pairwise,
        "experiments": experiment_reports,
    }
    _write_csv(output_dir / "summary.csv", all_csv_rows)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print("FINAL_REPORT=" + json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
