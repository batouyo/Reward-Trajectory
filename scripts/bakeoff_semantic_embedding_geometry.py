"""Compare CLIP, SigLIP, and Qwen-hidden with one endpoint-axis geometry."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from diffusers.pipelines.rewardflow.endpoint_comparator_metrics import (
    SMALL_GRADIENT_STEPS,
    gradient_direction_gate,
    ordering_diagnostics,
)
from diffusers.pipelines.rewardflow.endpoint_embedding_geometry import (
    CachedEndpointEmbeddingGeometry,
)
from diffusers.pipelines.rewardflow.rewards import Qwen25VQATeacherForcedScorer
from diffusers.pipelines.rewardflow.semantic_feature_scorers import (
    CLIPImageFeatureScorer,
    QwenHiddenFeatureScorer,
    SigLIPImageFeatureScorer,
)


DEFAULT_OUTPUT = "experiments/semantic_embedding_geometry_v6"
DEFAULT_CLIP = "/data15/hyp/weight/reward_models/clip-vit-large-patch14"
DEFAULT_SIGLIP = "/data15/hyp/weight/reward_models/siglip-so400m-patch14-384"
DEFAULT_QWEN = "/data15/hyp/weight/reward_models/Qwen2.5-VL-3B-Instruct"
GRADIENT_STEPS = (*SMALL_GRADIENT_STEPS, 3e-4)
IMAGE_ORDER = ("source", "oracle_0.2", "oracle_0.5", "oracle_0.8", "full")
PROBE_IDS = ("oracle_0.2", "oracle_0.5", "oracle_0.8")


def _args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--clip-model", default=os.getenv("REWARDFLOW_CLIP_MODEL_PATH", DEFAULT_CLIP))
    parser.add_argument("--siglip-model", default=os.getenv("REWARDFLOW_SIGLIP_MODEL_PATH", DEFAULT_SIGLIP))
    parser.add_argument("--qwen-model", default=os.getenv("REWARDFLOW_QWEN_MODEL_PATH", DEFAULT_QWEN))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--endpoint-audit-only", action="store_true")
    args = parser.parse_args()
    if args.manifest is None:
        args.manifest = str(Path(args.output_dir) / "suite_manifest.json")
    return args


def _json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _tensor(path: str, device: torch.device) -> torch.Tensor:
    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32).copy() / 255
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def _images(case: dict[str, object], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "source": _tensor(case["source_path"], device),
        "oracle_0.2": _tensor(case["probe_paths"]["0.2"], device),
        "oracle_0.5": _tensor(case["probe_paths"]["0.5"], device),
        "oracle_0.8": _tensor(case["probe_paths"]["0.8"], device),
        "full": _tensor(case["full_path"], device),
    }


def _qwen_metadata(scorer, model_id: str) -> dict[str, object]:
    config = scorer.model.config
    return {
        "model_id": model_id,
        "local_path": str(Path(model_id).resolve()) if Path(model_id).exists() else None,
        "revision": getattr(config, "_commit_hash", None) or "unavailable_local_snapshot",
        "parameter_count": sum(parameter.numel() for parameter in scorer.model.parameters()),
        "dtype": str(next(scorer.model.parameters()).dtype).removeprefix("torch."),
        "native_image_resolution": "dynamic Qwen smart-resize from checkpoint processor config",
        "parameters_frozen": all(not parameter.requires_grad for parameter in scorer.model.parameters()),
    }


def _semantic_audit(scorer, cases, device):
    rows = []
    validity = {}
    with torch.no_grad():
        for case in cases:
            case_images = _images(case, device)
            case_valid = True
            for image_id in IMAGE_ORDER:
                image = case_images[image_id]
                source_score = float(
                    scorer.score_answer(image, case["endpoint_question"], case["source_answer"]).item()
                )
                target_score = float(
                    scorer.score_answer(image, case["endpoint_question"], case["target_answer"]).item()
                )
                preferred = "source" if source_score > target_score else "target"
                rows.append(
                    {
                        "case_id": case["case_id"],
                        "image_id": image_id,
                        "source_answer_score": source_score,
                        "target_answer_score": target_score,
                        "preferred_endpoint_state": preferred,
                        "used_for_continuous_geometry": False,
                        "used_for_endpoint_semantic_validation": image_id in {"source", "full"},
                        "diagnostic_only": image_id not in {"source", "full"},
                    }
                )
            source_row = rows[-5]
            full_row = rows[-1]
            case_valid = (
                source_row["preferred_endpoint_state"] == "source" and full_row["preferred_endpoint_state"] == "target"
            )
            validity[case["case_id"]] = case_valid
    return rows, validity


def _output_dict(output) -> dict[str, float]:
    return {
        "progress": float(output.progress.detach()),
        "off_axis": float(output.off_axis.detach()),
        "source_distance": float(output.source_distance.detach()),
        "full_distance": float(output.full_distance.detach()),
        "axis_norm": float(output.axis_norm.detach()),
        "cosine_to_source": float(output.cosine_to_source.detach()),
        "cosine_to_full": float(output.cosine_to_full.detach()),
        "cosine_distance_source": float(output.cosine_distance_source.detach()),
        "cosine_distance_full": float(output.cosine_distance_full.detach()),
    }


def _gradient_audit(geometry, candidate: torch.Tensor) -> dict[str, object]:
    base = candidate.detach().clone().requires_grad_(True)
    baseline = geometry(base)
    directions = {}
    base_gradient = torch.autograd.grad(baseline.progress, base)[0]
    gradient_rms = base_gradient.float().square().mean().sqrt()
    finite = bool(torch.isfinite(base_gradient).all().item())
    nonzero = bool((gradient_rms > 0).item())
    for direction, loss_sign in (("toward_full", -1.0), ("toward_source", 1.0)):
        image = candidate.detach().clone().requires_grad_(True)
        output = geometry(image)
        loss = loss_sign * output.progress
        loss_gradient = torch.autograd.grad(loss, image)[0]
        loss_gradient_rms = loss_gradient.float().square().mean().sqrt()
        trials = []
        for step in GRADIENT_STEPS:
            updated = (image - step * loss_gradient / loss_gradient_rms.clamp_min(1e-12)).detach().clamp(0, 1)
            with torch.no_grad():
                shifted = geometry(updated)
            new_progress = float(shifted.progress)
            old_progress = float(output.progress.detach())
            correct = new_progress > old_progress if direction == "toward_full" else new_progress < old_progress
            trials.append(
                {
                    "step_rms": step,
                    "old_progress": old_progress,
                    "new_progress": new_progress,
                    "progress_change": new_progress - old_progress,
                    "old_off_axis": float(output.off_axis.detach()),
                    "new_off_axis": float(shifted.off_axis),
                    "off_axis_change": float(shifted.off_axis - output.off_axis.detach()),
                    "direction_correct": bool(correct),
                }
            )
        directions[direction] = {"trials": trials, **gradient_direction_gate(trials)}
    return {
        "gradient_l2": float(torch.linalg.vector_norm(base_gradient.float())),
        "gradient_rms": float(gradient_rms),
        "gradient_finite": finite,
        "gradient_nonzero": nonzero,
        "directions": directions,
        "toward_source_pass": finite and nonzero and directions["toward_source"]["passed"],
        "toward_full_pass": finite and nonzero and directions["toward_full"]["passed"],
    }


def _case_metrics(rows, gradient, endpoint_valid):
    probes = [next(row for row in rows if row["image_id"] == image_id)["progress"] for image_id in PROBE_IDS]
    ordering = ordering_diagnostics(probes)
    minimum_gap = min(ordering["adjacent_gaps"])
    span = probes[2] - probes[0]
    axis_norm = rows[0]["axis_norm"]
    return {
        "endpoint_semantic_valid": endpoint_valid,
        "axis_nondegenerate": math.isfinite(axis_norm) and axis_norm > 1e-4,
        "p_0.2": probes[0],
        "p_0.5": probes[1],
        "p_0.8": probes[2],
        **ordering,
        "minimum_adjacent_gap": minimum_gap,
        "probe_span": span,
        "low_probe_distance_from_full": 1 - probes[0],
        "mean_probe_progress": statistics.mean(probes),
        "all_probes_in_wide_range": all(-0.25 <= value <= 1.25 for value in probes),
        "mean_probe_off_axis": statistics.mean(
            next(row for row in rows if row["image_id"] == image_id)["off_axis"] for image_id in PROBE_IDS
        ),
        "toward_source_gradient_pass": gradient["toward_source_pass"],
        "toward_full_gradient_pass": gradient["toward_full_pass"],
    }


def _formal_gate(metrics):
    gates = {
        "endpoint_semantic_valid": metrics["endpoint_semantic_valid"],
        "axis_norm_finite_above_1e-4": metrics["axis_nondegenerate"],
        "strict_order": metrics["strict_order_pass"],
        "no_ties": metrics["tie_count"] == 0,
        "minimum_adjacent_gap_above_0p02": metrics["minimum_adjacent_gap"] > 0.02,
        "probe_span_at_least_0p10": metrics["probe_span"] >= 0.10,
        "p_0p2_at_most_0p90": metrics["p_0.2"] <= 0.90,
        "all_probes_in_minus_0p25_to_1p25": metrics["all_probes_in_wide_range"],
        "gradient_toward_source": metrics["toward_source_gradient_pass"],
        "gradient_toward_full": metrics["toward_full_gradient_pass"],
    }
    return {**gates, "passed": all(gates.values())}


def _evaluate_encoder(name, scorer_for_case, cases, validity, device):
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    rows = []
    metrics = {}
    gradients = {}
    for case in cases:
        case_images = _images(case, device)
        scorer = scorer_for_case(case)
        geometry = CachedEndpointEmbeddingGeometry(scorer, case_images["source"], case_images["full"])
        case_rows = []
        with torch.no_grad():
            for image_id in IMAGE_ORDER:
                values = _output_dict(geometry(case_images[image_id]))
                cosine_sum = values["cosine_distance_source"] + values["cosine_distance_full"]
                row = {
                    "encoder": name,
                    "case_id": case["case_id"],
                    "image_id": image_id,
                    "provenance": (
                        case["source_provenance"]
                        if image_id == "source"
                        else case["full_provenance"]
                        if image_id == "full"
                        else case["probe_provenance"]
                    ),
                    **values,
                    "legacy_ratio_if_available": (
                        values["cosine_distance_source"] / (cosine_sum + 1e-8) if name == "qwen_hidden" else ""
                    ),
                }
                rows.append(row)
                case_rows.append(row)
        gradient = _gradient_audit(geometry, case_images["oracle_0.5"])
        gradients[case["case_id"]] = gradient
        metrics[case["case_id"]] = _case_metrics(case_rows, gradient, validity[case["case_id"]])
    torch.cuda.synchronize(device)
    return {
        "rows": rows,
        "case_metrics": metrics,
        "gradients": gradients,
        "runtime_seconds": time.perf_counter() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
    }


def _aggregate(result):
    cases = list(result["case_metrics"].values())
    strict_count = sum(case["strict_order_pass"] for case in cases)
    gradient_count = sum(case["toward_source_gradient_pass"] and case["toward_full_gradient_pass"] for case in cases)
    return {
        "case_count": len(cases),
        "endpoint_valid_case_count": sum(case["endpoint_semantic_valid"] for case in cases),
        "strict_order_case_count": strict_count,
        "strict_order_rate": strict_count / len(cases),
        "gradient_both_pass_count": gradient_count,
        "gradient_both_pass_rate": gradient_count / len(cases),
        "median_min_adjacent_gap": statistics.median(case["minimum_adjacent_gap"] for case in cases),
        "median_probe_span": statistics.median(case["probe_span"] for case in cases),
        "median_p_0.2": statistics.median(case["p_0.2"] for case in cases),
        "median_off_axis": statistics.median(case["mean_probe_off_axis"] for case in cases),
        "extrapolation_case_count": sum(not case["all_probes_in_wide_range"] for case in cases),
        "all_endpoint_axes_finite_nondegenerate": all(case["axis_nondegenerate"] for case in cases),
        "runtime_seconds": result["runtime_seconds"],
        "peak_allocated_gib": result["peak_allocated_gib"],
        "peak_reserved_gib": result["peak_reserved_gib"],
    }


def _candidate_gate(formal_gate, aggregate):
    gates = {
        "formal_ball_all_gates": formal_gate["passed"],
        "all_cases_endpoint_semantic_valid": aggregate["endpoint_valid_case_count"] == aggregate["case_count"],
        "strict_order_rate_at_least_0p80": aggregate["strict_order_rate"] >= 0.8,
        "gradient_both_pass_rate_at_least_0p80": aggregate["gradient_both_pass_rate"] >= 0.8,
        "median_probe_span_at_least_0p10": aggregate["median_probe_span"] >= 0.10,
        "median_p_0p2_at_most_0p90": aggregate["median_p_0.2"] <= 0.90,
        "all_endpoint_axes_finite_nondegenerate": aggregate["all_endpoint_axes_finite_nondegenerate"],
    }
    return {**gates, "status": "SEMANTIC_GEOMETRY_CANDIDATE_PASS" if all(gates.values()) else "FAIL"}


def _plots(output_dir, formal_rows, all_rows, aggregates):
    import matplotlib.pyplot as plt

    labels = ["Source", ".2", ".5", ".8", "Full"]
    for field, filename, ylabel in (
        ("progress", "formal_ball_projection_plot.png", "Source→Full axis progress"),
        ("off_axis", "formal_ball_off_axis_plot.png", "Normalized off-axis deviation"),
    ):
        fig, axis = plt.subplots(figsize=(8, 5))
        for encoder in ("clip", "siglip", "qwen_hidden"):
            rows = [row for row in formal_rows if row["encoder"] == encoder]
            axis.plot(labels, [row[field] for row in rows], marker="o", label=encoder)
        axis.set_title("MODEL-GENERATED DIAGNOSTIC PROBES — Formal ball")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.3)
        axis.legend()
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    encoders = ("clip", "siglip", "qwen_hidden")
    axes[0].bar(encoders, [aggregates[name]["strict_order_rate"] for name in encoders])
    axes[0].set_ylabel("Strict-order case rate")
    axes[0].set_ylim(0, 1)
    axes[1].bar(encoders, [aggregates[name]["median_probe_span"] for name in encoders])
    axes[1].set_ylabel("Median probe span")
    fig.suptitle("MODEL-GENERATED DIAGNOSTIC PROBES — Suite summary")
    fig.tight_layout()
    fig.savefig(output_dir / "suite_summary_plot.png", dpi=160)
    plt.close(fig)


def main():
    args = _args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if manifest.get("controller_run") is not False:
        raise RuntimeError("v6 requires controller_run=false.")
    cases = manifest["cases"]
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.init()

    qwen = Qwen25VQATeacherForcedScorer(args.qwen_model, device=device, dtype=torch.float32, local_files_only=True)
    semantic_rows, validity = _semantic_audit(qwen, cases, device)
    _csv(output_dir / "qwen_semantic_state_audit.csv", semantic_rows)
    _json(output_dir / "endpoint_semantic_validity.json", validity)
    if args.endpoint_audit_only:
        print(json.dumps(validity, indent=2))
        return
    invalid = [case_id for case_id, passed in validity.items() if not passed]
    if invalid:
        raise RuntimeError(f"Endpoint semantic audit failed before encoder geometry: {invalid}")

    results = {}
    model_metadata = {"qwen_hidden": _qwen_metadata(qwen, args.qwen_model)}
    results["qwen_hidden"] = _evaluate_encoder(
        "qwen_hidden",
        lambda case, qwen_scorer=qwen: QwenHiddenFeatureScorer(qwen_scorer, case["comparison_focus"]),
        cases,
        validity,
        device,
    )
    del qwen
    torch.cuda.empty_cache()

    formal_source = _images(cases[0], device)["source"]
    processor_fidelity = {}
    for name, scorer_class, model_id in (
        ("clip", CLIPImageFeatureScorer, args.clip_model),
        ("siglip", SigLIPImageFeatureScorer, args.siglip_model),
    ):
        scorer = scorer_class(model_id, device=device, dtype=torch.float32, local_files_only=True)
        fidelity = scorer.processor_fidelity(formal_source)
        processor_fidelity[name] = fidelity
        if not fidelity["engineering_gate_embedding_cosine_at_least_0p999"]:
            _json(output_dir / "processor_fidelity.json", processor_fidelity)
            raise RuntimeError(f"{name} processor fidelity failed before geometry evaluation: {fidelity}")
        model_metadata[name] = scorer.metadata()
        results[name] = _evaluate_encoder(name, lambda case, scorer=scorer: scorer, cases, validity, device)
        del scorer
        torch.cuda.empty_cache()

    all_rows = [row for name in ("clip", "siglip", "qwen_hidden") for row in results[name]["rows"]]
    formal_case_id = cases[0]["case_id"]
    formal_rows = [row for row in all_rows if row["case_id"] == formal_case_id]
    _csv(output_dir / "formal_ball_geometry.csv", formal_rows)
    _csv(output_dir / "suite_geometry_scores.csv", all_rows)
    gradients = {name: result["gradients"] for name, result in results.items()}
    case_metrics = {name: result["case_metrics"] for name, result in results.items()}
    _json(
        output_dir / "formal_ball_gradient_audit.json",
        {name: value[formal_case_id] for name, value in gradients.items()},
    )
    _json(output_dir / "suite_gradient_audit.json", gradients)
    _json(output_dir / "suite_case_metrics.json", case_metrics)
    aggregates = {name: _aggregate(result) for name, result in results.items()}
    formal_gates = {name: _formal_gate(result["case_metrics"][formal_case_id]) for name, result in results.items()}
    candidate_gates = {name: _candidate_gate(formal_gates[name], aggregates[name]) for name in results}
    _json(output_dir / "aggregate_geometry_metrics.json", aggregates)
    _json(output_dir / "model_metadata.json", model_metadata)
    _json(output_dir / "processor_fidelity.json", processor_fidelity)

    passing = [name for name, gate in candidate_gates.items() if gate["status"] == "SEMANTIC_GEOMETRY_CANDIDATE_PASS"]
    passing.sort(
        key=lambda name: (
            -aggregates[name]["strict_order_rate"],
            -aggregates[name]["gradient_both_pass_rate"],
            -aggregates[name]["median_min_adjacent_gap"],
            -aggregates[name]["median_probe_span"],
            aggregates[name]["median_off_axis"],
            aggregates[name]["runtime_seconds"],
            aggregates[name]["peak_allocated_gib"],
        )
    )
    selection = {
        "selection_rule_frozen_before_results": True,
        "candidate_gates": candidate_gates,
        "winner": passing[0] if passing else None,
        "no_winner": not passing,
        "controller_run": False,
        "controller_integrated_or_optimized_in_v6": False,
        "perceptual_percentage_calibration_established": False,
    }
    _json(output_dir / "selection_report.json", selection)
    _plots(output_dir, formal_rows, all_rows, aggregates)
    report = {
        "scope": "CLIP/SigLIP/Qwen-hidden endpoint-axis geometry bake-off",
        "primary_coordinate": "unclamped normalized Source-to-Full axis projection",
        "controller_run": False,
        "probe_provenance": "MODEL-GENERATED PIXEL-ORACLE DIAGNOSTIC; not semantic ground truth or method output",
        "endpoint_semantic_validity": validity,
        "formal_ball_case_metrics": {name: result["case_metrics"][formal_case_id] for name, result in results.items()},
        "suite_case_metrics": case_metrics,
        "formal_gates": formal_gates,
        "aggregate_metrics": aggregates,
        "selection": selection,
    }
    _json(output_dir / "geometry_bakeoff_report.json", report)
    print(
        json.dumps(
            {"controller_run": False, "winner": selection["winner"], "candidate_gates": candidate_gates}, indent=2
        )
    )


if __name__ == "__main__":
    main()
