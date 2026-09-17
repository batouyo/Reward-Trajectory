"""Compare v6 whole-image CLIP/SigLIP geometry with one velocity-derived crop.

Diagnostic only: no probe generation, semantic reward, or controller optimization.
Run from the repository root with PYTHONPATH=src.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from bakeoff_semantic_embedding_geometry import (
    DEFAULT_CLIP,
    DEFAULT_SIGLIP,
    IMAGE_ORDER,
    _images,
    _output_dict,
)
from PIL import Image, ImageDraw, ImageOps

from diffusers.pipelines.rewardflow.endpoint_embedding_geometry import CachedEndpointEmbeddingGeometry
from diffusers.pipelines.rewardflow.pipeline_flux_kontext_terminal_control import FluxKontextTerminalControlPipeline
from diffusers.pipelines.rewardflow.region_semantic_geometry import (
    AggregatedVelocityRegion,
    aggregate_velocity_region,
    summarize_probe_geometry,
)
from diffusers.pipelines.rewardflow.semantic_feature_scorers import CLIPImageFeatureScorer, SigLIPImageFeatureScorer


DEFAULT_MANIFEST = "experiments/semantic_embedding_geometry_v6/suite_manifest.json"
DEFAULT_OUTPUT = "experiments/region_semantic_geometry_v8"
DEFAULT_KONTEXT = "/data15/hyp/weight/FLUX.1-Kontext-dev"
CASE_ID = "ball_color_seed_20260914"
CONTROL_STEPS = 4
TOPK_FRACTION = 0.25
PADDING_FRACTION = 0.10
PROVENANCE = {
    "source": "SOURCE INPUT — reused from v6 manifest",
    "oracle_0.2": "MODEL-GENERATED PIXEL-ORACLE DIAGNOSTIC — reused from v6",
    "oracle_0.5": "MODEL-GENERATED PIXEL-ORACLE DIAGNOSTIC — reused from v6",
    "oracle_0.8": "MODEL-GENERATED PIXEL-ORACLE DIAGNOSTIC — reused from v6",
    "full": "MODEL-GENERATED NATIVE FULL — reused from v6 manifest",
}


def _args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--kontext-model", default=os.getenv("FLUX_KONTEXT_MODEL_PATH", DEFAULT_KONTEXT))
    parser.add_argument("--clip-model", default=os.getenv("REWARDFLOW_CLIP_MODEL_PATH", DEFAULT_CLIP))
    parser.add_argument("--siglip-model", default=os.getenv("REWARDFLOW_SIGLIP_MODEL_PATH", DEFAULT_SIGLIP))
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _case_and_paths(manifest_path: Path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = manifest["probe_generator_config"]
    expected = {
        "steps": 12,
        "control_steps": CONTROL_STEPS,
        "control_mask_mode": "velocity-topk",
        "control_mask_topk_fraction": TOPK_FRACTION,
        "guidance_scale": 2.5,
        "dtype": "bfloat16",
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"v6 generation config mismatch for {key}: {config.get(key)!r} != {value!r}")
    if manifest.get("controller_run") is not False:
        raise ValueError("v6 source manifest must describe controller_run=false")
    matching = [case for case in manifest["cases"] if case["case_id"] == CASE_ID]
    if len(matching) != 1 or matching[0]["seed"] != 20260914:
        raise ValueError("Exactly one formal ball case with seed 20260914 is required")
    case = matching[0]
    paths = {
        "source": Path(case["source_path"]),
        "oracle_0.2": Path(case["probe_paths"]["0.2"]),
        "oracle_0.5": Path(case["probe_paths"]["0.5"]),
        "oracle_0.8": Path(case["probe_paths"]["0.8"]),
        "full": Path(case["full_path"]),
    }
    expected_hashes = {
        "source": case["source_fingerprint_sha256"],
        "oracle_0.2": case["probe_fingerprints_sha256"]["0.2"],
        "oracle_0.5": case["probe_fingerprints_sha256"]["0.5"],
        "oracle_0.8": case["probe_fingerprints_sha256"]["0.8"],
        "full": case["full_fingerprint_sha256"],
    }
    hashes = {}
    for image_id, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing v6 input {image_id}: {path}")
        hashes[image_id] = _sha256(path)
        if hashes[image_id] != expected_hashes[image_id]:
            raise ValueError(f"v6 input fingerprint changed for {image_id}: {path}")
    return case, config, paths, hashes


def _parity(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    delta = actual.detach().float() - expected.detach().float()
    return {"max_abs_diff": float(delta.abs().max()), "mean_abs_diff": float(delta.abs().mean())}


def _region_definition(
    region: AggregatedVelocityRegion,
    inputs,
    *,
    args,
    case,
    config,
    paths,
    hashes,
    parity,
) -> dict:
    x0, y0, x1, y1 = region.bbox_xyxy
    height, width = inputs.height, inputs.width
    return {
        "scope": "Fixed region for one v6 formal ball diagnostic; not a paper method component or segmentation model.",
        "case_id": case["case_id"],
        "git_head_before_experiment": os.popen("git rev-parse HEAD").read().strip(),
        "manifest": str(Path(args.manifest).resolve()),
        "image_paths": {key: str(path.resolve()) for key, path in paths.items()},
        "image_sha256": hashes,
        "image_provenance": PROVENANCE,
        "kontext_model": str(Path(args.kontext_model).resolve()),
        "generation_config": {
            "seed": case["seed"],
            "instruction": case["instruction"],
            "height": height,
            "width": width,
            "num_inference_steps": config["steps"],
            "guidance_scale": config["guidance_scale"],
            "dtype": config["dtype"],
        },
        "native_zero_control_parity": parity,
        "timestep_count": CONTROL_STEPS,
        "topk_fraction": TOPK_FRACTION,
        "aggregation_rule": (
            "DIAGNOSTIC ASSUMPTION: independently min-max normalize each of the first four existing velocity "
            "discrepancy token-score maps, average them, then use the existing stable velocity_topk_mask at 25%. "
            "This fixed-region aggregation is not a paper-defined method component."
        ),
        "per_step_score_min_max": [list(pair) for pair in region.per_step_score_ranges],
        "token_grid_shape_hw": [inputs.sampling_token_height, inputs.sampling_token_width],
        "image_resolution_hw": [height, width],
        "token_to_pixel_mapping": "nearest-neighbor upsampling of the fixed binary token mask",
        "raw_bbox_xyxy_exclusive": list(region.raw_bbox_xyxy),
        "bbox_xyxy_exclusive": list(region.bbox_xyxy),
        "padding_rule": "ceil(10% of raw bbox width/height) on each side, clamped to image bounds",
        "padding_fraction_each_side": PADDING_FRACTION,
        "requested_padding_xy_px": list(region.padding_xy),
        "active_token_fraction": region.active_token_fraction,
        "bbox_area_fraction_of_image": (x1 - x0) * (y1 - y0) / (height * width),
        "same_bbox_for_all_five_images": True,
        "crop_not_blackout": True,
    }


def _save_visuals(output_dir: Path, paths: dict[str, Path], region: AggregatedVelocityRegion) -> None:
    with Image.open(paths["source"]) as opened:
        source = opened.convert("RGB")
    width, height = source.size
    score = region.score_map.detach().cpu().numpy()
    heat = Image.fromarray(np.uint8(np.clip(score, 0, 1) * 255), mode="L").resize(
        (width, height), Image.Resampling.NEAREST
    )
    heat = ImageOps.colorize(heat, black="#08264a", white="#fff0a8")
    mask = Image.fromarray(region.pixel_mask.detach().cpu().numpy().astype(np.uint8) * 255, mode="L")
    overlay_color = Image.new("RGBA", source.size, (255, 30, 30, 0))
    overlay_color.putalpha(mask.point(lambda value: int(value * 0.45)))
    overlay = Image.alpha_composite(source.convert("RGBA"), overlay_color).convert("RGB")
    bbox_image = source.copy()
    x0, y0, x1, y1 = region.bbox_xyxy
    ImageDraw.Draw(bbox_image).rectangle((x0, y0, x1 - 1, y1 - 1), outline="red", width=3)
    top_images = (source, heat, mask.convert("RGB"), overlay, bbox_image)
    top_names = ("source", "velocity_heatmap", "binary_edit_mask", "mask_overlay_source", "bbox_overlay_source")
    for image, name in zip(top_images, top_names):
        image.save(output_dir / f"{name}.png")
    bottom_images = []
    for image_id in IMAGE_ORDER:
        with Image.open(paths[image_id]) as opened:
            crop = opened.convert("RGB").crop(region.bbox_xyxy)
        name = (
            "source"
            if image_id == "source"
            else "full"
            if image_id == "full"
            else image_id.replace("oracle_", "probe_")
        )
        crop.save(output_dir / f"{name}_crop.png")
        bottom_images.append(crop)
    tile_w, tile_h, label_h = 256, 256, 26
    grid = Image.new("RGB", (tile_w * 5, (tile_h + label_h) * 2), "white")
    draw = ImageDraw.Draw(grid)
    for row, (images, names) in enumerate(((top_images, top_names), (bottom_images, IMAGE_ORDER))):
        for column, (image, name) in enumerate(zip(images, names)):
            left, top = column * tile_w, row * (tile_h + label_h)
            draw.text((left + 5, top + 5), name, fill="black")
            grid.paste(image.resize((tile_w, tile_h), Image.Resampling.LANCZOS), (left, top + label_h))
    grid.save(output_dir / "overview_grid.png")


def _crop(image: torch.Tensor, bbox: tuple[int, int, int, int]) -> torch.Tensor:
    x0, y0, x1, y1 = bbox
    return image[:, :, y0:y1, x0:x1]


def _gradient_audit(geometry: CachedEndpointEmbeddingGeometry, candidate: torch.Tensor, bbox) -> dict:
    image = candidate.detach().clone().requires_grad_(True)
    progress = geometry(_crop(image, bbox)).progress
    gradient = torch.autograd.grad(progress, image)[0]
    x0, y0, x1, y1 = bbox
    inside = gradient[:, :, y0:y1, x0:x1]
    outside = gradient.clone()
    outside[:, :, y0:y1, x0:x1] = 0
    total_l1 = gradient.float().abs().sum()
    inside_l1 = inside.float().abs().sum()
    outside_l1 = outside.float().abs().sum()
    finite = bool(torch.isfinite(gradient).all())
    strict_subregion = x0 > 0 or y0 > 0 or x1 < image.shape[-1] or y1 < image.shape[-2]
    return {
        "candidate": "v6 stored probe_0.5 image",
        "crop_operation": "tensor slicing of original candidate image",
        "progress": float(progress.detach()),
        "gradient_finite": finite,
        "gradient_nonzero": bool((total_l1 > 0).item()),
        "gradient_rms_full_image": float(gradient.float().square().mean().sqrt()),
        "gradient_rms_inside_crop": float(inside.float().square().mean().sqrt()),
        "gradient_l1_inside_crop": float(inside_l1),
        "gradient_l1_outside_crop": float(outside_l1),
        "gradient_l1_fraction_inside_crop": float(inside_l1 / total_l1) if total_l1 > 0 else 0.0,
        "gradient_outside_crop_max_abs": float(outside.abs().max()),
        "gradient_spatial_localization_informative": strict_subregion,
        "gradient_concentrated_in_crop": (
            finite and bool((total_l1 > 0).item()) and bool((outside_l1 == 0).item()) if strict_subregion else None
        ),
        "interpretation": (
            "Outside-crop zeros follow from slicing; this is a differentiability audit, not reward validation."
            if strict_subregion
            else "The bbox is the full image: gradient localization cannot be assessed, only finiteness/nonzero."
        ),
    }


def _v6_whole_alignment(rows: list[dict], v6_path: Path, encoder: str) -> dict:
    with v6_path.open(newline="", encoding="utf-8") as handle:
        baseline = {
            row["image_id"]: row
            for row in csv.DictReader(handle)
            if row["encoder"] == encoder and row["case_id"] == CASE_ID
        }
    fields = (
        "progress",
        "off_axis",
        "source_distance",
        "full_distance",
        "axis_norm",
        "cosine_to_source",
        "cosine_to_full",
        "cosine_distance_source",
        "cosine_distance_full",
    )
    if set(baseline) != set(IMAGE_ORDER):
        raise ValueError(f"v6 whole-image baseline incomplete for {encoder}")
    differences = {
        image_id: {field: abs(float(row[field]) - float(baseline[image_id][field])) for field in fields}
        for row in rows
        if (image_id := row["image_id"]) in baseline
    }
    return {
        "baseline_csv": str(v6_path.resolve()),
        "max_abs_difference_across_v6_geometry_fields": max(
            difference for image in differences.values() for difference in image.values()
        ),
        "per_image_field_abs_differences": differences,
    }


def main() -> None:
    args = _args()
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite an existing nonempty diagnostic directory: {output_dir}")
    case, config, paths, hashes = _case_and_paths(Path(args.manifest))
    with Image.open(paths["source"]) as opened:
        source_image = opened.convert("RGB")
    width, height = source_image.size
    if (width, height) != (256, 256):
        raise ValueError("Formal v6 ball source must be 256x256")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The formal v8 FLUX-Kontext diagnostic requires an available CUDA device")
    torch.cuda.set_device(device)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipe = FluxKontextTerminalControlPipeline.from_pretrained(
        args.kontext_model, torch_dtype=torch.bfloat16, local_files_only=True
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    inputs = pipe.prepare_terminal_control_inputs(
        image=source_image,
        prompt=case["instruction"],
        height=height,
        width=width,
        num_inference_steps=config["steps"],
        guidance_scale=config["guidance_scale"],
        generator=torch.Generator(device=device).manual_seed(case["seed"]),
    )
    masks, zero_control = pipe.prepare_velocity_edit_masks(
        inputs, control_steps=CONTROL_STEPS, mode="velocity-topk", topk_fraction=TOPK_FRACTION
    )
    parity = _parity(inputs.native_final_latent, zero_control.final_latent)
    if parity["max_abs_diff"] != 0 or parity["mean_abs_diff"] != 0:
        raise RuntimeError(f"IMPLEMENTATION_PARITY_FAIL: native versus zero-control latent {parity}")
    if len(masks.scores) != CONTROL_STEPS:
        raise RuntimeError("Mask preparation did not return exactly four early-step score maps")
    region = aggregate_velocity_region(
        masks.scores,
        token_height=inputs.sampling_token_height,
        token_width=inputs.sampling_token_width,
        image_height=height,
        image_width=width,
        topk_fraction=TOPK_FRACTION,
        padding_fraction=PADDING_FRACTION,
    )
    definition = _region_definition(
        region, inputs, args=args, case=case, config=config, paths=paths, hashes=hashes, parity=parity
    )
    del pipe, inputs, masks, zero_control
    torch.cuda.empty_cache()
    _write_json(output_dir / "region_definition.json", definition)
    _save_visuals(output_dir, paths, region)

    images = _images(case, device)
    rows, summary, gradients, metadata = [], {}, {}, {}
    v6_path = Path(args.manifest).parent / "formal_ball_geometry.csv"
    for name, scorer_class, model_id in (
        ("clip", CLIPImageFeatureScorer, args.clip_model),
        ("siglip", SigLIPImageFeatureScorer, args.siglip_model),
    ):
        scorer = scorer_class(model_id, device=device, dtype=torch.float32, local_files_only=True)
        fidelity = scorer.processor_fidelity(images["source"])
        if not fidelity["engineering_gate_embedding_cosine_at_least_0p999"]:
            raise RuntimeError(f"v6 processor fidelity failed for {name}: {fidelity}")
        metadata[name] = {"model": scorer.metadata(), "processor_fidelity": fidelity}
        view_rows = {}
        geometries = {}
        for view in ("whole", "crop"):
            view_image = (lambda image: image) if view == "whole" else (lambda image: _crop(image, region.bbox_xyxy))
            geometry = CachedEndpointEmbeddingGeometry(
                scorer, view_image(images["source"]), view_image(images["full"])
            )
            geometries[view] = geometry
            current_rows = []
            with torch.no_grad():
                for image_id in IMAGE_ORDER:
                    result = {
                        "encoder": name,
                        "view": view,
                        "case_id": CASE_ID,
                        "image_id": image_id,
                        "provenance": PROVENANCE[image_id],
                        **_output_dict(geometry(view_image(images[image_id]))),
                    }
                    rows.append(result)
                    current_rows.append(result)
            view_rows[view] = current_rows
        whole = summarize_probe_geometry(view_rows["whole"])
        crop = summarize_probe_geometry(view_rows["crop"])
        summary[name] = {
            "whole": whole,
            "crop": crop,
            "crop_minus_whole": {
                key: crop[key] - whole[key]
                for key in (
                    "gap_0.2_to_0.5",
                    "gap_0.5_to_0.8",
                    "minimum_adjacent_gap",
                    "probe_span_0.2_to_0.8",
                    "mean_probe_off_axis",
                )
            },
            "v6_whole_alignment": _v6_whole_alignment(view_rows["whole"], v6_path, name),
        }
        gradients[name] = _gradient_audit(geometries["crop"], images["oracle_0.5"], region.bbox_xyxy)
        del geometries, scorer
        torch.cuda.empty_cache()

    with (output_dir / "geometry_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    _write_json(
        output_dir / "comparison_summary.json",
        {
            "case_id": CASE_ID,
            "hypothesis": "Whole-image unedited background may dilute local Source-to-Full feature change.",
            "metric": "Unchanged v6 CachedEndpointEmbeddingGeometry; progress is not a semantic percentage.",
            "probes_regenerated": False,
            "same_fixed_bbox_for_every_image_and_encoder": True,
            "region_crop_is_strict_subregion": definition["bbox_area_fraction_of_image"] < 1.0,
            "hypothesis_test_informative": definition["bbox_area_fraction_of_image"] < 1.0,
            "status": (
                "REGION_COMPARISON_AVAILABLE"
                if definition["bbox_area_fraction_of_image"] < 1.0
                else "NO_REGION_CONTRAST_FULL_IMAGE_BBOX"
            ),
            "region_definition": "region_definition.json",
            "encoder_metadata": metadata,
            "comparisons": summary,
        },
    )
    _write_json(output_dir / "gradient_audit.json", gradients)
    print(
        json.dumps(
            {
                "region_bbox_xyxy": list(region.bbox_xyxy),
                "bbox_area_fraction": definition["bbox_area_fraction_of_image"],
                "comparisons": {
                    name: {view: summary[name][view] for view in ("whole", "crop")} for name in ("clip", "siglip")
                },
                "gradient_finite_nonzero": {
                    name: gradients[name]["gradient_finite"] and gradients[name]["gradient_nonzero"]
                    for name in ("clip", "siglip")
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
