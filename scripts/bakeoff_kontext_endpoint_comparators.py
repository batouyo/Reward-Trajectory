"""Run the v4 endpoint-comparator bake-off without running the controller.

Oracle images are held-out FLUX controller outputs. Pixel blends are secondary
diagnostics and are never used to define, calibrate, or select prompts.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from diffusers.pipelines.rewardflow.endpoint_comparator_metrics import (
    average_ranks,
    gradient_direction_gate,
    ordering_diagnostics,
    spearman_correlation,
)
from diffusers.pipelines.rewardflow.endpoint_feature_distance import FeatureEndpointDistanceReward
from diffusers.pipelines.rewardflow.pairwise_endpoint_semantic import PairwiseEndpointSemanticReward
from diffusers.pipelines.rewardflow.relative_endpoint_parser import parse_relative_endpoint_semantic_json
from diffusers.pipelines.rewardflow.relative_endpoint_semantic import (
    RELATIVE_ENDPOINT_CHOICES,
    build_relative_endpoint_comparison_prompt,
)
from diffusers.pipelines.rewardflow.rewards import Qwen25VQATeacherForcedScorer


DEFAULT_MODEL = "/data15/hyp/weight/FLUX.1-Kontext-dev"
DEFAULT_QWEN = "/data15/hyp/weight/reward_models/Qwen2.5-VL-3B-Instruct"
DEFAULT_SOURCE = "/data15/hyp/dataset/kontinuous_kontext/raw/source_images/source_000000.png"
DEFAULT_SPEC = "examples/rewardflow/relative_endpoint_ball_spec.json"
DEFAULT_FULL = "experiments/relative_endpoint_semantic_v3/native_full_MODEL_GENERATED.png"
DEFAULT_ORACLE = "/tmp/st_formal_shared_t4_mask"
DEFAULT_OUTPUT = "experiments/endpoint_comparator_bakeoff_v4"
DEFAULT_PROMPT = "Make the weighted training ball blue while preserving its shape, texture, lighting, and background."
FORMAL_STATEMENTS = {"ball_color": "The two images show a similar visible color state for the weighted training ball."}
ORACLE_STRENGTHS = (0.2, 0.5, 0.8)
PIXEL_ALPHAS = (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0)
GRADIENT_STEPS = (1e-5, 3e-5, 1e-4, 3e-4)


def _args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.getenv("FLUX_KONTEXT_MODEL_PATH", DEFAULT_MODEL))
    parser.add_argument("--qwen-model", default=os.getenv("QWEN25_VL_MODEL_PATH", DEFAULT_QWEN))
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--relative-spec", default=DEFAULT_SPEC)
    parser.add_argument("--full", default=DEFAULT_FULL)
    parser.add_argument("--oracle-output-dir", default=DEFAULT_ORACLE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--qwen-dtype",
        choices=("bf16", "fp32", "both"),
        default="both",
        help="Explicit checkpoint dtype. Formal bake-off requires both.",
    )
    args = parser.parse_args()
    for name in ("model", "qwen_model", "source", "relative_spec", "full", "oracle_output_dir"):
        value = Path(getattr(args, name))
        if not value.exists():
            parser.error(f"`--{name.replace('_', '-')}` must point to an existing local path: {value}")
    if args.prompt != DEFAULT_PROMPT:
        parser.error("The formal v4 ball bake-off uses the preregistered edit prompt.")
    return args


def _serialize(value):
    if torch.is_tensor(value):
        value = value.detach().float().cpu()
        return value.item() if value.numel() == 1 else value.tolist()
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_serialize(item) for item in value]
    return value


def _json(path: Path, value) -> None:
    path.write_text(json.dumps(_serialize(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(_serialize(rows))


def _tensor(path: str | Path, device: torch.device) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((256, 256), Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype=np.float32).copy() / 255
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def _oracle_path(directory: str | Path, strength: float) -> Path:
    return Path(directory) / f"final_{str(strength).replace('.', 'p')}.png"


def _load_spec(path: str | Path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return parse_relative_endpoint_semantic_json(json.dumps(payload.get("spec", payload)))


def _load_images(args, device):
    source = _tensor(args.source, device)
    full = _tensor(args.full, device)
    images = {
        "source": source,
        **{
            f"oracle_{strength:g}": _tensor(_oracle_path(args.oracle_output_dir, strength), device)
            for strength in ORACLE_STRENGTHS
        },
        "native_full": full,
    }
    images.update({f"pixel_{alpha:g}": (1 - alpha) * source + alpha * full for alpha in PIXEL_ALPHAS})
    return source, full, images


def _model_dtype_audit(scorer, requested: str):
    parameter_dtypes = {}
    for parameter in scorer.model.parameters():
        key = str(parameter.dtype).replace("torch.", "")
        parameter_dtypes[key] = parameter_dtypes.get(key, 0) + parameter.numel()
    expected = "bfloat16" if requested == "bf16" else "float32"
    floating = {key for key in parameter_dtypes if key in {"float16", "bfloat16", "float32", "float64"}}
    return {
        "requested": requested,
        "parameter_elements_by_dtype": parameter_dtypes,
        "floating_parameter_dtypes": sorted(floating),
        "true_checkpoint_dtype": floating == {expected},
    }


def _measure(device, function):
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    result = function()
    torch.cuda.synchronize(device)
    return result, {
        "runtime_seconds": time.perf_counter() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
    }


def _score_summary(scores: dict[str, float], raw_endpoint_range: float | None = None):
    oracle = [scores[f"oracle_{strength:g}"] for strength in ORACLE_STRENGTHS]
    ordering = ordering_diagnostics(oracle)
    endpoint_range = scores["native_full"] - scores["source"] if raw_endpoint_range is None else raw_endpoint_range
    return {
        "endpoint_source": scores["source"],
        "endpoint_full": scores["native_full"],
        "endpoint_range": endpoint_range,
        "endpoint_valid": math.isfinite(endpoint_range) and endpoint_range > 0,
        "oracle_scores": dict(zip(("0.2", "0.5", "0.8"), oracle)),
        "oracle_ordering": ordering,
        "oracle_spearman": spearman_correlation(oracle, ORACLE_STRENGTHS),
        "minimum_adjacent_oracle_gap": min(ordering["adjacent_gaps"]),
    }


def _gradient_audit(score_fn, candidate: torch.Tensor, scorer, score_name: str):
    image = candidate.detach().requires_grad_(True)
    score = score_fn(image)
    gradient = torch.autograd.grad(score, image)[0]
    rms = gradient.float().square().mean().sqrt()
    audit = {
        "candidate_provenance": "held_out_model_generated_oracle_0p5",
        "optimized_score": score_name,
        "initial_score": score,
        "gradient_l2_norm": torch.linalg.vector_norm(gradient.float()),
        "gradient_rms": rms,
        "gradient_finite": bool(torch.isfinite(gradient).all().item()),
        "gradient_nonzero": bool((gradient.abs().sum() > 0).item()),
        "directions": {},
    }
    for direction, sign in (("toward_full", 1.0), ("toward_source", -1.0)):
        trials = []
        for step_rms in GRADIENT_STEPS:
            stepped_image = (image.detach() + sign * step_rms * gradient.detach() / rms.clamp_min(1e-12)).clamp(0, 1)
            with torch.no_grad():
                stepped = score_fn(stepped_image)
            correct = stepped > score if direction == "toward_full" else stepped < score
            trials.append({"step_rms": step_rms, "new_score": stepped, "direction_correct": bool(correct.item())})
        gate = gradient_direction_gate(trials)
        audit["directions"][direction] = {"trials": trials, **gate}
    audit.update(
        {
            "qwen_parameters_require_grad": any(parameter.requires_grad for parameter in scorer.model.parameters()),
            "qwen_parameters_have_grad": any(parameter.grad is not None for parameter in scorer.model.parameters()),
        }
    )
    finite_nonzero = audit["gradient_finite"] and audit["gradient_nonzero"]
    audit["toward_full_pass"] = finite_nonzero and audit["directions"]["toward_full"]["passed"]
    audit["toward_source_pass"] = finite_nonzero and audit["directions"]["toward_source"]["passed"]
    return audit


def _swapped_prompt(prompt: str) -> str:
    return prompt.replace("A. Image 1\nB. Image 3", "A. Image 3\nB. Image 1")


def _ab_forward(scorer, source, full, candidate, spec, primitive, *, swapped=False):
    prompt_1 = build_relative_endpoint_comparison_prompt(spec, primitive, first_reference="source")
    prompt_2 = build_relative_endpoint_comparison_prompt(spec, primitive, first_reference="full")
    if swapped:
        prompt_1, prompt_2 = _swapped_prompt(prompt_1), _swapped_prompt(prompt_2)
    logits_1, logprobs_1 = scorer.score_multi_image_single_token_choices_with_logits(
        (source, candidate, full), prompt_1, RELATIVE_ENDPOINT_CHOICES
    )
    logits_2, logprobs_2 = scorer.score_multi_image_single_token_choices_with_logits(
        (full, candidate, source), prompt_2, RELATIVE_ENDPOINT_CHOICES
    )
    if swapped:
        logit_margin_1 = logits_1[0] - logits_1[1]
        logprob_margin_1 = logprobs_1[0] - logprobs_1[1]
        logit_margin_2 = logits_2[1] - logits_2[0]
        logprob_margin_2 = logprobs_2[1] - logprobs_2[0]
    else:
        logit_margin_1 = logits_1[1] - logits_1[0]
        logprob_margin_1 = logprobs_1[1] - logprobs_1[0]
        logit_margin_2 = logits_2[0] - logits_2[1]
        logprob_margin_2 = logprobs_2[0] - logprobs_2[1]
    return {
        "logits_1": logits_1,
        "logprobs_1": logprobs_1,
        "logits_2": logits_2,
        "logprobs_2": logprobs_2,
        "logit_margin_1": logit_margin_1,
        "logprob_margin_1": logprob_margin_1,
        "logit_margin_2": logit_margin_2,
        "logprob_margin_2": logprob_margin_2,
        "symmetrized_logit_margin": 0.5 * (logit_margin_1 + logit_margin_2),
        "symmetrized_margin": 0.5 * (logprob_margin_1 + logprob_margin_2),
    }


def _ab_bakeoff(scorer, spec, source, full, images, dtype_name):
    primitive = spec.primitives[0]
    rows = []
    precision = []
    values = {}
    directional = {"forward_1": {}, "forward_2": {}, "label_swapped": {}}
    with torch.no_grad():
        for image_id, image in images.items():
            result = _ab_forward(scorer, source, full, image, spec, primitive)
            score = float(result["symmetrized_margin"])
            values[image_id] = score
            directional["forward_1"][image_id] = float(result["logprob_margin_1"])
            directional["forward_2"][image_id] = float(result["logprob_margin_2"])
            for order in (1, 2):
                logits = result[f"logits_{order}"]
                logprobs = result[f"logprobs_{order}"]
                precision.append(
                    {
                        "image_id": image_id,
                        "dtype": dtype_name,
                        "forward_order": order,
                        "logit_A": float(logits[0]),
                        "logit_B": float(logits[1]),
                        "logit_difference_B_minus_A": float(logits[1] - logits[0]),
                        "logprob_A": float(logprobs[0]),
                        "logprob_B": float(logprobs[1]),
                        "logprob_difference_B_minus_A": float(logprobs[1] - logprobs[0]),
                        "remapped_logit_margin": float(result[f"logit_margin_{order}"]),
                        "margin": float(result[f"logprob_margin_{order}"]),
                        "symmetrized_margin": score,
                    }
                )
            rows.append(
                {
                    "image_id": image_id,
                    "provenance": "PIXEL_BLEND_DIAGNOSTIC_NOT_MODEL_GENERATED_OUTPUT"
                    if image_id.startswith("pixel_")
                    else "MODEL_GENERATED_OR_ENDPOINT_INPUT",
                    "forward_1_margin": float(result["logprob_margin_1"]),
                    "forward_2_margin": float(result["logprob_margin_2"]),
                    "symmetrized_margin": score,
                    "order_bias": float(0.5 * (result["logprob_margin_1"] - result["logprob_margin_2"])),
                }
            )
        for image_id in ("source", "oracle_0.2", "oracle_0.5", "oracle_0.8", "native_full"):
            swapped = _ab_forward(scorer, source, full, images[image_id], spec, primitive, swapped=True)
            directional["label_swapped"][image_id] = float(swapped["symmetrized_margin"])
    summary = _score_summary(values)
    primary_ids = ("source", "oracle_0.2", "oracle_0.5", "oracle_0.8", "native_full")
    forward_summaries = {
        name: _score_summary({key: mapping[key] for key in primary_ids}) for name, mapping in directional.items()
    }
    max_order_change = max(abs(directional["forward_1"][key] - directional["forward_2"][key]) for key in primary_ids)
    label_changes = [abs(directional["label_swapped"][key] - values[key]) for key in primary_ids]
    order = {
        "max_forward_order_score_change": max_order_change,
        "max_label_swap_score_change": max(label_changes),
        "forward_summaries": forward_summaries,
        "order_robust": all(
            item["endpoint_valid"] and item["oracle_ordering"]["strict_order_pass"]
            for item in forward_summaries.values()
        ),
    }
    gradient = _gradient_audit(
        lambda image: _ab_forward(scorer, source, full, image, spec, primitive)["symmetrized_margin"],
        images["oracle_0.5"],
        scorer,
        "symmetrized_three_image_ab_logprob_margin",
    )
    return rows, precision, summary, order, gradient


def _pairwise_bakeoff(scorer, spec, source, full, images):
    objective = PairwiseEndpointSemanticReward(
        scorer,
        spec,
        source,
        full,
        statements=FORMAL_STATEMENTS,
        fail_on_endpoint_validation=False,
    )
    primitive = spec.primitives[0]
    rows = []
    scores = {}
    candidate_first = {}
    reference_first = {}
    with torch.no_grad():
        for image_id, image in images.items():
            values = objective.primitive_affinities(image, primitive)
            anchor = objective._anchors[primitive.id]
            coordinate = (values["raw_margin"] - anchor["source_margin"]) / anchor["dynamic_range"]
            scores[image_id] = float(coordinate)
            candidate_first[image_id] = float(values["candidate_first_margin"])
            reference_first[image_id] = float(values["reference_first_margin"])
            rows.append(
                {
                    "image_id": image_id,
                    "provenance": "PIXEL_BLEND_DIAGNOSTIC_NOT_MODEL_GENERATED_OUTPUT"
                    if image_id.startswith("pixel_")
                    else "MODEL_GENERATED_OR_ENDPOINT_INPUT",
                    **{key: float(value) for key, value in values.items()},
                    "relative_coordinate": float(coordinate),
                }
            )
    raw_range = float(objective._anchors[primitive.id]["dynamic_range"])
    summary = _score_summary(scores, raw_endpoint_range=raw_range)
    primary_ids = ("source", "oracle_0.2", "oracle_0.5", "oracle_0.8", "native_full")
    order_summaries = {
        "candidate_first": _score_summary({key: candidate_first[key] for key in primary_ids}),
        "reference_first": _score_summary({key: reference_first[key] for key in primary_ids}),
    }
    order = {
        "max_order_score_change": max(abs(candidate_first[key] - reference_first[key]) for key in primary_ids),
        "directional_summaries": order_summaries,
        "order_robust": all(
            item["endpoint_valid"] and item["oracle_ordering"]["strict_order_pass"]
            for item in order_summaries.values()
        ),
        "criterion": "both unsymmetrized image orders preserve positive endpoint range and strict oracle ordering",
    }
    gradient = _gradient_audit(
        lambda image: objective.primitive_affinities(image, primitive)["raw_margin"],
        images["oracle_0.5"],
        scorer,
        "symmetric_pairwise_endpoint_affinity_margin",
    )
    return objective, rows, summary, order, gradient


def _feature_bakeoff(scorer, spec, source, full, images):
    objective = FeatureEndpointDistanceReward(scorer, spec, source, full)
    primitive = spec.primitives[0]
    rows = []
    scores = {}
    with torch.no_grad():
        for image_id, image in images.items():
            values = objective.primitive_distances(image, primitive)
            scores[image_id] = float(values["cosine_distance_ratio"])
            rows.append(
                {
                    "image_id": image_id,
                    "provenance": "PIXEL_BLEND_DIAGNOSTIC_NOT_MODEL_GENERATED_OUTPUT"
                    if image_id.startswith("pixel_")
                    else "MODEL_GENERATED_OR_ENDPOINT_INPUT",
                    **{key: float(value) for key, value in values.items()},
                }
            )
    summary = _score_summary(scores)
    order = {
        "max_order_score_change": 0.0,
        "order_robust": True,
        "criterion": "not applicable: each image is independently encoded with the same focus prompt",
    }
    gradient = _gradient_audit(
        lambda image: objective.primitive_distances(image, primitive)["cosine_distance_ratio"],
        images["oracle_0.5"],
        scorer,
        "focus_conditioned_cosine_distance_ratio",
    )
    return objective, rows, summary, order, gradient


def _method_gates(summary, order, gradient):
    return {
        "endpoint_valid": summary["endpoint_valid"],
        "oracle_strict_order": summary["oracle_ordering"]["strict_order_pass"],
        "oracle_no_ties": summary["oracle_ordering"]["tie_count"] == 0,
        "toward_source_small_steps": gradient["toward_source_pass"],
        "toward_full_small_steps": gradient["toward_full_pass"],
        "order_robust": order["order_robust"],
    }


def _select_method(methods):
    passing = [name for name, result in methods.items() if all(result["gates"].values())]
    if not passing:
        return None, "No comparator passed every preregistered gate."
    pairwise = "pairwise sentence FP32"
    feature = "feature distance FP32"
    if pairwise in passing and feature in passing:
        pair_gap = methods[pairwise]["summary"]["minimum_adjacent_oracle_gap"]
        feature_gap = methods[feature]["summary"]["minimum_adjacent_oracle_gap"]
        # ASSUMPTION: "close" is not numerically defined by the research brief.
        # We predeclare pairwise as close when its minimum gap is at least 90%
        # of the feature baseline's; the threshold is reported, not hidden.
        if pair_gap >= 0.9 * feature_gap:
            return pairwise, "Both passed; pairwise retained >=90% of the feature minimum gap and is preferred."
    complexity = {
        "three-image A/B BF16": 2,
        "three-image A/B FP32": 2,
        pairwise: 0,
        feature: 1,
    }
    passing.sort(
        key=lambda name: (
            -methods[name]["summary"]["minimum_adjacent_oracle_gap"],
            methods[name]["order"].get("max_order_score_change", math.inf),
            complexity[name],
        )
    )
    return passing[0], "Selected by minimum adjacent oracle gap, then order bias, then implementation simplicity."


def main():
    args = _args()
    if not torch.cuda.is_available():
        raise RuntimeError("The real comparator bake-off requires CUDA.")
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = _load_spec(args.relative_spec)
    if [primitive.id for primitive in spec.primitives] != ["ball_color"] or spec.unresolved_instruction_items:
        raise ValueError("Formal v4 requires the preregistered one-primitive human-audited ball spec.")
    source, full, images = _load_images(args, device)
    methods = {}
    precision_rows = []
    runtime = {}
    dtype_audit = {}

    requested_dtypes = (
        (("bf16", torch.bfloat16), ("fp32", torch.float32))
        if args.qwen_dtype == "both"
        else ((args.qwen_dtype, torch.bfloat16 if args.qwen_dtype == "bf16" else torch.float32),)
    )
    for dtype_name, torch_dtype in requested_dtypes:
        scorer = Qwen25VQATeacherForcedScorer(args.qwen_model, device=device, dtype=torch_dtype, local_files_only=True)
        dtype_audit[dtype_name] = _model_dtype_audit(scorer, dtype_name)
        ab_result, timing = _measure(device, partial(_ab_bakeoff, scorer, spec, source, full, images, dtype_name))
        ab_rows, precision, summary, order, gradient = ab_result
        label = f"three-image A/B {dtype_name.upper()}"
        methods[label] = {
            "summary": summary,
            "order": order,
            "gradient": gradient,
            "gates": _method_gates(summary, order, gradient),
        }
        runtime[label] = timing
        precision_rows.extend(precision)
        _csv(output_dir / f"three_image_ab_{dtype_name}.csv", ab_rows)
        _csv(output_dir / "precision_audit.csv", precision_rows)
        if dtype_name == "fp32":
            fidelity, fidelity_timing = _measure(
                device,
                partial(
                    scorer.compare_multi_image_with_official_processor,
                    (images["oracle_0.5"], source),
                    "Pairwise preprocessing fidelity diagnostic.",
                ),
            )
            fidelity["engineering_gate_pixel_cosine_at_least_0p999"] = bool(
                fidelity["grid_equal"] and fidelity["pixel_cosine"] >= 0.999
            )
            fidelity["threshold_is_engineering_not_paper"] = True
            fidelity["runtime"] = fidelity_timing
            _json(output_dir / "processor_fidelity_pairwise.json", fidelity)

            pair_result, timing = _measure(device, partial(_pairwise_bakeoff, scorer, spec, source, full, images))
            pair_objective, rows, summary, order, gradient = pair_result
            label = "pairwise sentence FP32"
            methods[label] = {
                "summary": summary,
                "order": order,
                "gradient": gradient,
                "gates": _method_gates(summary, order, gradient),
            }
            runtime[label] = timing
            _csv(output_dir / "pairwise_sentence_fp32.csv", rows)

            feature_result, timing = _measure(device, partial(_feature_bakeoff, scorer, spec, source, full, images))
            feature_objective, rows, summary, order, gradient = feature_result
            label = "feature distance FP32"
            methods[label] = {
                "summary": summary,
                "order": order,
                "gradient": gradient,
                "gates": _method_gates(summary, order, gradient),
            }
            runtime[label] = timing
            _csv(output_dir / "feature_distance_fp32.csv", rows)
            del pair_objective, feature_objective
        del scorer
        gc.collect()
        torch.cuda.empty_cache()

    recommended, selection_reason = _select_method(methods)
    endpoint_report = {
        name: {"summary": result["summary"], "gates": result["gates"]} for name, result in methods.items()
    }
    ordering_report = {
        name: {
            "oracle": result["summary"]["oracle_ordering"],
            "spearman": result["summary"]["oracle_spearman"],
        }
        for name, result in methods.items()
    }
    gradient_report = {name: result["gradient"] for name, result in methods.items()}
    order_report = {name: result["order"] for name, result in methods.items()}
    tie_audit = {
        "input": [0, 0, 1, 1, 2],
        "average_ranks": average_ranks([0, 0, 1, 1, 2]),
        "spearman_against_increasing_index": spearman_correlation([0, 0, 1, 1, 2]),
    }
    report = {
        "scope": "Endpoint comparator bake-off v4; comparator only; controller frozen and not run",
        "old_head_expected": "152705b58e45ed053f3b850bf6ef9ad34ff90973",
        "config": vars(args),
        "preregistered_pairwise_statement": FORMAL_STATEMENTS,
        "feature_representation": {
            "layer": "last",
            "token_position": "final prompt token",
            "normalization": "L2",
            "primary_distance": "cosine distance ratio",
        },
        "selection_close_rule_assumption": "pairwise minimum adjacent gap >= 90% of feature gap",
        "oracle_provenance": "held-out MODEL-GENERATED FLUX CONTROLLER PROBES; not used to design comparator",
        "pixel_probe_provenance": "PIXEL BLEND DIAGNOSTIC; NOT MODEL-GENERATED OUTPUT",
        "dtype_audit": dtype_audit,
        "runtime_and_vram": runtime,
        "methods": methods,
        "recommended_comparator": recommended,
        "selection_reason": selection_reason,
        "controller_run": False,
        "controller_remains_frozen": True,
    }
    _json(output_dir / "endpoint_gate_report.json", endpoint_report)
    _json(output_dir / "oracle_ordering_report.json", ordering_report)
    _json(output_dir / "gradient_direction_report.json", gradient_report)
    _json(output_dir / "order_bias_report.json", order_report)
    _json(output_dir / "metric_tie_audit.json", tie_audit)
    _json(output_dir / "comparator_bakeoff_report.json", report)
    print(json.dumps(_serialize({"recommended_comparator": recommended, "methods": methods}), indent=2))


if __name__ == "__main__":
    main()
