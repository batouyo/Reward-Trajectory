"""Run the frozen v7 text-conditioned semantic-geometry bake-off.

This script never generates probes and never invokes a controller. It reads the
five frozen v6 cases, parses Source/Native-Full once, and changes only the
semantic direction used by CLIP/SigLIP.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from diffusers.pipelines.rewardflow.endpoint_comparator_metrics import (
    SMALL_GRADIENT_STEPS,
    gradient_direction_gate,
    ordering_diagnostics,
)
from diffusers.pipelines.rewardflow.relative_endpoint_parser import fingerprint_endpoint_image
from diffusers.pipelines.rewardflow.rewards import Qwen25VQATeacherForcedScorer
from diffusers.pipelines.rewardflow.semantic_feature_scorers import (
    CLIPImageFeatureScorer,
    SigLIPImageFeatureScorer,
)
from diffusers.pipelines.rewardflow.text_conditioned_semantic import (
    TextConditionedSemanticGeometry,
    TextSemanticEndpointDirectionError,
    TianyuAITextConditionedSemanticParser,
    load_cached_text_conditioned_parse,
    save_cached_text_conditioned_parse,
)


DEFAULT_OUTPUT = "experiments/text_conditioned_semantic_geometry_v7"
DEFAULT_MANIFEST = "experiments/semantic_embedding_geometry_v6/suite_manifest.json"
DEFAULT_V6_SCORES = "experiments/semantic_embedding_geometry_v6/suite_geometry_scores.csv"
DEFAULT_V6_AGGREGATE = "experiments/semantic_embedding_geometry_v6/aggregate_geometry_metrics.json"
DEFAULT_CLIP = "/data15/hyp/weight/reward_models/clip-vit-large-patch14"
DEFAULT_SIGLIP = "/data15/hyp/weight/reward_models/siglip-so400m-patch14-384"
DEFAULT_QWEN = "/data15/hyp/weight/reward_models/Qwen2.5-VL-3B-Instruct"
EXPECTED_CASES = (
    "ball_color_seed_20260914",
    "ball_color_seed_20260915",
    "ball_color_seed_20260916",
    "scene_reimagination_seed_20260914",
    "environment_night_seed_20260914",
)
IMAGE_ORDER = ("source", "oracle_0.2", "oracle_0.5", "oracle_0.8", "full")
PROBE_IDS = ("oracle_0.2", "oracle_0.5", "oracle_0.8")
GRADIENT_STEPS = (*SMALL_GRADIENT_STEPS, 3e-4)


def _args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--v6-scores", default=DEFAULT_V6_SCORES)
    parser.add_argument("--v6-aggregate", default=DEFAULT_V6_AGGREGATE)
    parser.add_argument("--clip-model", default=os.getenv("REWARDFLOW_CLIP_MODEL_PATH", DEFAULT_CLIP))
    parser.add_argument("--siglip-model", default=os.getenv("REWARDFLOW_SIGLIP_MODEL_PATH", DEFAULT_SIGLIP))
    parser.add_argument("--qwen-model", default=os.getenv("REWARDFLOW_QWEN_MODEL_PATH", DEFAULT_QWEN))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--parse-only", action="store_true")
    return parser.parse_args()


def _json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _csv(path: Path, rows: list[dict[str, object]], fieldnames=None) -> None:
    if fieldnames is None:
        if not rows:
            raise ValueError(f"Cannot infer columns for empty CSV `{path}`.")
        fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _tensor(path: str, device: torch.device) -> torch.Tensor:
    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32).copy() / 255
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def _images(case, device):
    return {
        "source": _tensor(case["source_path"], device),
        "oracle_0.2": _tensor(case["probe_paths"]["0.2"], device),
        "oracle_0.5": _tensor(case["probe_paths"]["0.5"], device),
        "oracle_0.8": _tensor(case["probe_paths"]["0.8"], device),
        "full": _tensor(case["full_path"], device),
    }


def _validate_frozen_manifest(manifest):
    case_ids = tuple(case["case_id"] for case in manifest["cases"])
    if case_ids != EXPECTED_CASES:
        raise RuntimeError(f"The v7 frozen cases changed: {case_ids!r}.")
    if manifest.get("controller_run") is not False:
        raise RuntimeError("v7 requires controller_run=false.")
    for case in manifest["cases"]:
        for path in (case["source_path"], case["full_path"], *case["probe_paths"].values()):
            if not Path(path).is_file():
                raise FileNotFoundError(path)
        actual = {
            "source": fingerprint_endpoint_image(case["source_path"]),
            "full": fingerprint_endpoint_image(case["full_path"]),
            **{strength: fingerprint_endpoint_image(path) for strength, path in case["probe_paths"].items()},
        }
        expected = {
            "source": case["source_fingerprint_sha256"],
            "full": case["full_fingerprint_sha256"],
            **case["probe_fingerprints_sha256"],
        }
        if actual != expected:
            raise RuntimeError(f"Frozen image fingerprint mismatch for {case['case_id']}.")


def _parse_cases(cases, output_dir):
    cache_path = output_dir / "semantic_parse_cache.json"
    parser = None
    records = {}
    calls = 0
    cache_hits = 0
    for case in cases:
        record = load_cached_text_conditioned_parse(
            cache_path,
            case["source_fingerprint_sha256"],
            case["full_fingerprint_sha256"],
            case["instruction"],
        )
        if record is None:
            parser = parser or TianyuAITextConditionedSemanticParser()
            record = parser.parse(case["source_path"], case["full_path"], case["instruction"])
            save_cached_text_conditioned_parse(cache_path, record)
            calls += 1
        else:
            cache_hits += 1
        records[case["case_id"]] = record
    serializable = {
        case_id: {"spec": asdict(record.spec), "provenance": record.provenance} for case_id, record in records.items()
    }
    _json(output_dir / "semantic_parse_records.json", serializable)
    text_rows = []
    for case_id, record in records.items():
        for primitive in record.spec.primitives:
            text_rows.append(
                {
                    "case_id": case_id,
                    "primitive_id": primitive.id,
                    "object": primitive.object,
                    "attribute": primitive.attribute,
                    "source_state": primitive.source_state,
                    "target_state": primitive.target_state,
                    "source_semantic_text": primitive.source_semantic_text,
                    "target_semantic_text": primitive.target_semantic_text,
                    "parser_provider": record.provenance["provider"],
                    "parser_model": record.provenance["model"],
                }
            )
    _csv(output_dir / "semantic_text_pairs.csv", text_rows)
    return records, {"online_parse_calls": calls, "cache_hits": cache_hits}


def _endpoint_audit(qwen, cases, records, device):
    rows = []
    validity = {}
    with torch.no_grad():
        for case in cases:
            images = _images(case, device)
            for primitive in records[case["case_id"]].spec.primitives:
                endpoint_rows = []
                for image_id in ("source", "full"):
                    source_score = float(
                        qwen.score_answer(images[image_id], primitive.endpoint_question, primitive.source_answer)
                    )
                    target_score = float(
                        qwen.score_answer(images[image_id], primitive.endpoint_question, primitive.target_answer)
                    )
                    preferred = "source" if source_score > target_score else "target"
                    row = {
                        "case_id": case["case_id"],
                        "primitive_id": primitive.id,
                        "image_id": image_id,
                        "question": primitive.endpoint_question,
                        "source_answer": primitive.source_answer,
                        "target_answer": primitive.target_answer,
                        "source_answer_score": source_score,
                        "target_answer_score": target_score,
                        "preferred_endpoint_state": preferred,
                        "used_for_continuous_progress": False,
                    }
                    rows.append(row)
                    endpoint_rows.append(row)
                validity[f"{case['case_id']}::{primitive.id}"] = (
                    endpoint_rows[0]["preferred_endpoint_state"] == "source"
                    and endpoint_rows[1]["preferred_endpoint_state"] == "target"
                )
    return {"rows": rows, "primitive_validity": validity}


def _output_values(output):
    return {
        "source_similarity": float(output.source_similarity.detach()),
        "target_similarity": float(output.target_similarity.detach()),
        "semantic_margin": float(output.semantic_margin.detach()),
        "progress": float(output.progress.detach()),
        "endpoint_dynamic_range": float(output.endpoint_dynamic_range.detach()),
        "text_axis_norm": float(output.text_axis_norm.detach()),
        "image_delta_text_alignment": float(output.image_delta_text_alignment.detach()),
        "orthogonal_ratio": float(output.orthogonal_ratio.detach()),
    }


def _gradient_audit(geometry, candidate):
    base = candidate.detach().clone().requires_grad_(True)
    output = geometry(base)
    gradient = torch.autograd.grad(output.progress, base)[0]
    rms = gradient.float().square().mean().sqrt()
    result = {
        "gradient_l2": float(torch.linalg.vector_norm(gradient.float())),
        "gradient_rms": float(rms),
        "gradient_finite": bool(torch.isfinite(gradient).all()),
        "gradient_nonzero": bool(rms > 0),
        "directions": {},
    }
    for direction, loss_sign in (("toward_full", -1.0), ("toward_source", 1.0)):
        image = candidate.detach().clone().requires_grad_(True)
        before = geometry(image)
        loss_gradient = torch.autograd.grad(loss_sign * before.progress, image)[0]
        loss_rms = loss_gradient.float().square().mean().sqrt()
        trials = []
        for step in GRADIENT_STEPS:
            shifted_image = (image - step * loss_gradient / loss_rms.clamp_min(1e-12)).detach().clamp(0, 1)
            with torch.no_grad():
                after = geometry(shifted_image)
            change = float(after.progress - before.progress.detach())
            correct = change > 0 if direction == "toward_full" else change < 0
            trials.append(
                {
                    "step_rms": step,
                    "old_progress": float(before.progress.detach()),
                    "new_progress": float(after.progress),
                    "progress_change": change,
                    "semantic_margin_change": float(after.semantic_margin - before.semantic_margin.detach()),
                    "alignment_change": float(
                        after.image_delta_text_alignment - before.image_delta_text_alignment.detach()
                    ),
                    "direction_correct": bool(correct),
                    "diagnostic_only": step == 3e-4,
                }
            )
        result["directions"][direction] = {"trials": trials, **gradient_direction_gate(trials)}
    result["toward_source_pass"] = (
        result["gradient_finite"] and result["gradient_nonzero"] and result["directions"]["toward_source"]["passed"]
    )
    result["toward_full_pass"] = (
        result["gradient_finite"] and result["gradient_nonzero"] and result["directions"]["toward_full"]["passed"]
    )
    return result


def _primitive_metrics(rows, gradient, endpoint_audit_valid):
    probes = [next(row for row in rows if row["image_id"] == key)["progress"] for key in PROBE_IDS]
    ordering = ordering_diagnostics(probes)
    alignments = [
        next(row for row in rows if row["image_id"] == key)["image_delta_text_alignment"] for key in PROBE_IDS
    ]
    orthogonal = [next(row for row in rows if row["image_id"] == key)["orthogonal_ratio"] for key in PROBE_IDS]
    return {
        "endpoint_audit_valid": endpoint_audit_valid,
        "text_endpoint_direction_valid": True,
        "p_0.2": probes[0],
        "p_0.5": probes[1],
        "p_0.8": probes[2],
        **ordering,
        "minimum_adjacent_gap": min(ordering["adjacent_gaps"]),
        "probe_span": probes[2] - probes[0],
        "all_probes_in_wide_range": all(-0.25 <= value <= 1.25 for value in probes),
        "median_alignment": statistics.median(alignments),
        "median_orthogonal_ratio": statistics.median(orthogonal),
        "toward_source_gradient_pass": gradient["toward_source_pass"],
        "toward_full_gradient_pass": gradient["toward_full_pass"],
    }


def _evaluate_encoder(name, scorer, cases, records, endpoint_validity, device):
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    rows = []
    gradients = {}
    primitive_metrics = {}
    token_metadata = {}
    failures = {}
    for case in cases:
        images = _images(case, device)
        for primitive in records[case["case_id"]].spec.primitives:
            key = f"{case['case_id']}::{primitive.id}"
            try:
                geometry = TextConditionedSemanticGeometry(
                    scorer,
                    images["source"],
                    images["full"],
                    primitive.source_semantic_text,
                    primitive.target_semantic_text,
                )
            except TextSemanticEndpointDirectionError as error:
                failures[key] = {"status": "TEXT_SEMANTIC_ENDPOINT_DIRECTION_FAIL", **error.diagnostics}
                continue
            token_metadata[key] = {
                "source": geometry.source_text_metadata,
                "target": geometry.target_text_metadata,
            }
            primitive_rows = []
            with torch.no_grad():
                for image_id in IMAGE_ORDER:
                    row = {
                        "encoder": name,
                        "case_id": case["case_id"],
                        "primitive_id": primitive.id,
                        "image_id": image_id,
                        "provenance": (
                            case["source_provenance"]
                            if image_id == "source"
                            else case["full_provenance"]
                            if image_id == "full"
                            else case["probe_provenance"]
                        ),
                        **_output_values(geometry(images[image_id])),
                    }
                    rows.append(row)
                    primitive_rows.append(row)
            gradient = _gradient_audit(geometry, images["oracle_0.5"])
            gradients[key] = gradient
            primitive_metrics[key] = _primitive_metrics(primitive_rows, gradient, endpoint_validity.get(key, False))
    torch.cuda.synchronize(device)
    case_metrics = {}
    for case in cases:
        prefix = f"{case['case_id']}::"
        keys = [key for key in endpoint_validity if key.startswith(prefix)]
        values = [primitive_metrics.get(key) for key in keys]
        complete = all(value is not None for value in values)
        valid_values = [value for value in values if value is not None]
        weights = {primitive.id: primitive.weight for primitive in records[case["case_id"]].spec.primitives}
        weighted = {}
        for field in ("p_0.2", "p_0.5", "p_0.8"):
            available = [(key, primitive_metrics[key][field]) for key in keys if key in primitive_metrics]
            denominator = sum(weights[key.split("::", 1)[1]] for key, _ in available)
            weighted[field] = (
                sum(weights[key.split("::", 1)[1]] * value for key, value in available) / denominator
                if denominator
                else math.nan
            )
        case_metrics[case["case_id"]] = {
            "primitive_count": len(keys),
            "all_primitives_evaluated": complete,
            "endpoint_semantic_valid": complete and all(value["endpoint_audit_valid"] for value in valid_values),
            "text_endpoint_direction_valid": complete,
            "strict_order_pass": complete and all(value["strict_order_pass"] for value in valid_values),
            "tie_count": sum(value["tie_count"] for value in valid_values) if complete else math.nan,
            "toward_source_gradient_pass": complete
            and all(value["toward_source_gradient_pass"] for value in valid_values),
            "toward_full_gradient_pass": complete
            and all(value["toward_full_gradient_pass"] for value in valid_values),
            "minimum_adjacent_gap": min((value["minimum_adjacent_gap"] for value in valid_values), default=math.nan),
            "probe_span": min((value["probe_span"] for value in valid_values), default=math.nan),
            "all_probes_in_wide_range": complete and all(value["all_probes_in_wide_range"] for value in valid_values),
            "median_alignment": statistics.median([value["median_alignment"] for value in valid_values])
            if valid_values
            else math.nan,
            "median_orthogonal_ratio": statistics.median([value["median_orthogonal_ratio"] for value in valid_values])
            if valid_values
            else math.nan,
            **weighted,
        }
    return {
        "rows": rows,
        "gradients": gradients,
        "primitive_metrics": primitive_metrics,
        "case_metrics": case_metrics,
        "failures": failures,
        "token_metadata": token_metadata,
        "runtime_seconds": time.perf_counter() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
    }


def _formal_gate(metrics):
    gates = {
        "endpoint_semantic_direction_valid": metrics["endpoint_semantic_valid"],
        "text_endpoint_direction_valid": metrics["text_endpoint_direction_valid"],
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


def _aggregate(result, formal_case):
    cases = list(result["case_metrics"].values())
    strict = sum(value["strict_order_pass"] for value in cases)
    source_grad = sum(value["toward_source_gradient_pass"] for value in cases)
    full_grad = sum(value["toward_full_gradient_pass"] for value in cases)
    endpoint = sum(value["endpoint_semantic_valid"] and value["text_endpoint_direction_valid"] for value in cases)
    finite_spans = [value["probe_span"] for value in cases if math.isfinite(value["probe_span"])]
    finite_gaps = [value["minimum_adjacent_gap"] for value in cases if math.isfinite(value["minimum_adjacent_gap"])]
    orthogonal = [
        value["median_orthogonal_ratio"] for value in cases if math.isfinite(value["median_orthogonal_ratio"])
    ]
    formal = _formal_gate(result["case_metrics"][formal_case])
    aggregate = {
        "case_count": len(cases),
        "endpoint_valid_case_count": endpoint,
        "endpoint_direction_valid_rate": endpoint / len(cases),
        "strict_order_case_count": strict,
        "strict_order_rate": strict / len(cases),
        "toward_source_gradient_pass_count": source_grad,
        "toward_full_gradient_pass_count": full_grad,
        "gradient_both_pass_count": sum(
            value["toward_source_gradient_pass"] and value["toward_full_gradient_pass"] for value in cases
        ),
        "gradient_both_pass_rate": sum(
            value["toward_source_gradient_pass"] and value["toward_full_gradient_pass"] for value in cases
        )
        / len(cases),
        "median_probe_span": statistics.median(finite_spans) if finite_spans else math.nan,
        "median_p_0.2": statistics.median(value["p_0.2"] for value in cases),
        "worst_minimum_adjacent_gap": min(finite_gaps) if finite_gaps else math.nan,
        "median_orthogonal_ratio": statistics.median(orthogonal) if orthogonal else math.nan,
        "extrapolation_case_count": sum(not value["all_probes_in_wide_range"] for value in cases),
        "runtime_seconds": result["runtime_seconds"],
        "peak_allocated_gib": result["peak_allocated_gib"],
        "peak_reserved_gib": result["peak_reserved_gib"],
        "formal_ball_gates": formal,
    }
    passed = (
        endpoint == 5
        and strict == 5
        and source_grad == 5
        and full_grad == 5
        and aggregate["median_probe_span"] >= 0.10
        and formal["passed"]
    )
    aggregate["status"] = (
        "TEXT_CONDITIONED_CANDIDATE_PASS"
        if passed
        else "PROMISING_BUT_NOT_STABLE"
        if endpoint == 5 and strict == 4
        else "FAIL"
    )
    return aggregate


def _v6_lookup(path):
    lookup = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["encoder"] in {"clip", "siglip"} and row["image_id"] in PROBE_IDS:
                lookup[(row["encoder"], row["case_id"], row["image_id"])] = float(row["progress"])
    return lookup


def _comparison_rows(cases, results, v6_scores):
    rows = []
    for encoder in ("clip", "siglip"):
        for case in cases:
            case_id = case["case_id"]
            v6 = [v6_scores[(encoder, case_id, image_id)] for image_id in PROBE_IDS]
            metrics = results[encoder]["case_metrics"][case_id]
            v7 = [metrics[f"p_0.{index}"] for index in (2, 5, 8)]
            rows.append(
                {
                    "encoder": encoder,
                    "case_id": case_id,
                    "v6_p_0.2": v6[0],
                    "v6_p_0.5": v6[1],
                    "v6_p_0.8": v6[2],
                    "v6_ordered": v6[0] < v6[1] < v6[2],
                    "v7_p_0.2": v7[0],
                    "v7_p_0.5": v7[1],
                    "v7_p_0.8": v7[2],
                    "v7_ordered": metrics["strict_order_pass"],
                    "known_v6_inversion": (encoder, case_id)
                    in {
                        ("siglip", "ball_color_seed_20260915"),
                        ("clip", "environment_night_seed_20260914"),
                        ("siglip", "environment_night_seed_20260914"),
                    },
                    "known_inversion_fixed": not (v6[0] < v6[1]) and v7[0] < v7[1],
                }
            )
    return rows


def _aggregate_comparison_table(v6_aggregate, aggregates, comparisons):
    rows = []
    for encoder in ("clip", "siglip"):
        v6 = v6_aggregate[encoder]
        v6_cases = [row for row in comparisons if row["encoder"] == encoder]
        v6_worst_gap = min(
            min(row["v6_p_0.5"] - row["v6_p_0.2"], row["v6_p_0.8"] - row["v6_p_0.5"]) for row in v6_cases
        )
        rows.append(
            {
                "method": f"{encoder.upper()} image-axis v6",
                "strict_order_rate": v6["strict_order_rate"],
                "gradient_pass_rate": v6["gradient_both_pass_rate"],
                "median_span": v6["median_probe_span"],
                "worst_minimum_adjacent_gap": v6_worst_gap,
                "endpoint_direction_valid_rate": v6["endpoint_valid_case_count"] / v6["case_count"],
            }
        )
        v7 = aggregates[encoder]
        rows.append(
            {
                "method": f"{encoder.upper()} text-conditioned v7",
                "strict_order_rate": v7["strict_order_rate"],
                "gradient_pass_rate": v7["gradient_both_pass_rate"],
                "median_span": v7["median_probe_span"],
                "worst_minimum_adjacent_gap": v7["worst_minimum_adjacent_gap"],
                "endpoint_direction_valid_rate": v7["endpoint_direction_valid_rate"],
            }
        )
    return rows


def main():
    args = _args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    _validate_frozen_manifest(manifest)
    cases = manifest["cases"]
    records, parse_runtime = _parse_cases(cases, output_dir)
    if args.parse_only:
        print(json.dumps({"controller_run": False, **parse_runtime}, indent=2))
        return

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.init()
    qwen = Qwen25VQATeacherForcedScorer(args.qwen_model, device=device, dtype=torch.float32, local_files_only=True)
    audit = _endpoint_audit(qwen, cases, records, device)
    _json(output_dir / "endpoint_audit.json", audit)
    del qwen
    torch.cuda.empty_cache()

    results = {}
    model_metadata = {}
    processor_fidelity = {}
    formal_source = _images(cases[0], device)["source"]
    for name, scorer_class, model_path in (
        ("clip", CLIPImageFeatureScorer, args.clip_model),
        ("siglip", SigLIPImageFeatureScorer, args.siglip_model),
    ):
        scorer = scorer_class(model_path, device=device, dtype=torch.float32, local_files_only=True)
        processor_fidelity[name] = scorer.processor_fidelity(formal_source)
        if not processor_fidelity[name]["engineering_gate_embedding_cosine_at_least_0p999"]:
            raise RuntimeError(f"{name} processor fidelity failed.")
        model_metadata[name] = scorer.metadata()
        results[name] = _evaluate_encoder(name, scorer, cases, records, audit["primitive_validity"], device)
        del scorer
        torch.cuda.empty_cache()

    all_rows = results["clip"]["rows"] + results["siglip"]["rows"]
    formal_rows = [row for row in all_rows if row["case_id"] == EXPECTED_CASES[0]]
    _csv(output_dir / "formal_ball_text_geometry.csv", formal_rows)
    _csv(output_dir / "suite_text_geometry.csv", all_rows)
    _json(output_dir / "gradient_audit.json", {name: value["gradients"] for name, value in results.items()})
    formal_case = EXPECTED_CASES[0]
    aggregates = {name: _aggregate(result, formal_case) for name, result in results.items()}
    v6_aggregate = json.loads(Path(args.v6_aggregate).read_text(encoding="utf-8"))
    aggregate_details = {
        "clip_image_axis_v6": v6_aggregate["clip"],
        "clip_text_conditioned_v7": aggregates["clip"],
        "siglip_image_axis_v6": v6_aggregate["siglip"],
        "siglip_text_conditioned_v7": aggregates["siglip"],
    }
    comparisons = _comparison_rows(cases, results, _v6_lookup(args.v6_scores))
    _csv(output_dir / "v6_vs_v7_comparison.csv", comparisons)
    aggregate_table = _aggregate_comparison_table(v6_aggregate, aggregates, comparisons)
    _json(output_dir / "aggregate_metrics.json", {"table": aggregate_table, "details": aggregate_details})

    passing = [name for name, value in aggregates.items() if value["status"] == "TEXT_CONDITIONED_CANDIDATE_PASS"]
    passing.sort(
        key=lambda name: (
            -aggregates[name]["strict_order_rate"],
            -aggregates[name]["gradient_both_pass_rate"],
            -aggregates[name]["worst_minimum_adjacent_gap"],
            -aggregates[name]["median_probe_span"],
            aggregates[name]["median_orthogonal_ratio"],
            aggregates[name]["runtime_seconds"],
        )
    )
    selection = {
        "selection_rule_frozen_before_results": True,
        "encoder_status": {name: value["status"] for name, value in aggregates.items()},
        "recommended_encoder_for_next_controller_test": passing[0] if passing else None,
        "controller_run": False,
        "perceptual_percentage_calibration_established": False,
    }
    _json(output_dir / "selection_report.json", selection)
    report = {
        "experiment_version": "v7 — Text-Conditioned Semantic Progress Geometry",
        "primary_formula": "p(I)=(m(I)-m(Source))/(m(Full)-m(Source)+eps), unclamped",
        "margin_formula": "m(I)=dot(f(I), t_target)-dot(f(I), t_source)",
        "controller_run": False,
        "probe_provenance": "MODEL-GENERATED PIXEL-ORACLE DIAGNOSTIC; not semantic ground truth or method output",
        "parser_runtime": parse_runtime,
        "endpoint_audit": audit,
        "case_metrics": {name: value["case_metrics"] for name, value in results.items()},
        "primitive_metrics": {name: value["primitive_metrics"] for name, value in results.items()},
        "direction_failures": {name: value["failures"] for name, value in results.items()},
        "token_metadata": {name: value["token_metadata"] for name, value in results.items()},
        "model_metadata": model_metadata,
        "processor_fidelity": processor_fidelity,
        "aggregate_metrics": {"table": aggregate_table, "details": aggregate_details},
        "known_inversion_comparison": [row for row in comparisons if row["known_v6_inversion"]],
        "selection": selection,
    }
    _json(output_dir / "text_conditioned_semantic_geometry_report.json", report)
    print(json.dumps({"controller_run": False, "aggregates": aggregates, "selection": selection}, indent=2))


if __name__ == "__main__":
    main()
