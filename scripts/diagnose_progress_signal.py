#!/usr/bin/env python3
"""Measure SigLIP/DINO progress signal on source, native target, and VeloEdit states."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rewardflow_calibration.optimization.progress_reward import ProgressEstimator
from rewardflow_calibration.rollout.veloedit import VeloEditCompatibleRollout, VeloEditRolloutConfig


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/data15/hyp/weight/FLUX.1-Kontext-dev")
    parser.add_argument("--source", default="/home/hyp/Code/VeloEdit/testdata/7.jpg")
    parser.add_argument("--prompt", default="make him old")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--cache-dir", default="/data15/hyp/weight")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/progress_signal"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--float32", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    dtype = torch.float32 if args.float32 else torch.bfloat16
    native_config = VeloEditRolloutConfig(
        steps=args.steps, seed=args.seed,
        first_step_align_steps=0, preserve_steps=0, edit_steps=0,
    )
    velo_config = VeloEditRolloutConfig(steps=args.steps, seed=args.seed)
    rollout = VeloEditCompatibleRollout(args.model, device=device, dtype=dtype)
    source_image = Image.open(args.source).convert("RGB")
    native_prepared = rollout.prepare(source_image, args.prompt, config=native_config, seed=args.seed)
    velo_prepared = rollout.prepare(source_image, args.prompt, config=velo_config, seed=args.seed)
    source_pixels = rollout.pipeline.image_processor.preprocess(
        native_prepared.working_image, native_prepared.height, native_prepared.width
    ).to(device=device, dtype=torch.float32)
    source = ((source_pixels + 1) / 2).clamp(0, 1)
    with torch.no_grad():
        native_full = rollout.rollout_native(native_prepared, config=native_config).detach()
        velo_images = rollout.rollout(velo_prepared, args.alphas, config=velo_config).detach()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    rows.append({
        "image": "source", "alpha": None, "siglip_progress": 0.0,
        "siglip_drift": 0.0, "dino_progress": 0.0, "dino_drift": 0.0,
    })
    for alpha, image in zip(args.alphas, velo_images):
        rows.append({"image": f"veloedit_alpha_{alpha:.2f}", "alpha": alpha, "tensor": image})
    rows.append({"image": "native_full", "alpha": None, "tensor": native_full})
    for backbone in ("siglip", "dino"):
        estimator = ProgressEstimator(backbone, device=device, cache_dir=args.cache_dir)
        anchors = estimator.set_anchors(source, native_full)
        for row in rows[1:]:
            path = args.output_dir / f"{row['image']}.png"
            value = row["tensor"]
            pixels = (value[0] if value.ndim == 4 else value).detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            from PIL import Image as PILImage
            import numpy as np
            PILImage.fromarray(np.rint(pixels * 255).astype("uint8"), mode="RGB").save(path)
            with torch.no_grad():
                score = estimator(value)
            row[f"{backbone}_progress"] = float(score.raw_progress.mean().cpu())
            row[f"{backbone}_drift"] = float(score.drift.mean().cpu())
            row[f"{backbone}_anchor_diagnostics"] = anchors
        del estimator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    numeric = rows[1:-1]
    for row in rows:
        row.pop("tensor", None)
    for backbone in ("siglip", "dino"):
        progress = [float(row[f"{backbone}_progress"]) for row in numeric]
        adjacent = [progress[i + 1] - progress[i] for i in range(len(progress) - 1)]
        rows[-1][f"{backbone}_analysis"] = {
            "dynamic_range": max(progress) - min(progress) if progress else 0.0,
            "near_tie_adjacent_pairs": sum(abs(delta) < 0.02 for delta in adjacent),
            "adjacent_order_reversals": sum(delta < -0.02 for delta in adjacent),
        }
    disagreement = [
        abs(float(row["siglip_progress"]) - float(row["dino_progress"]))
        for row in numeric
    ]
    summary = {
        "source": str(Path(args.source).resolve()),
        "prompt": args.prompt,
        "native_target": "Native FLUX-Kontext full-edit image",
        "veloedit_alpha_images_are_diagnostic_only": True,
        "dino_representation": "normalized mean of DINOv2 patch tokens; CLS excluded",
        "rows": rows,
        "siglip_dino_mean_absolute_progress_difference": (
            sum(disagreement) / len(disagreement) if disagreement else 0.0
        ),
        "siglip_dino_disagreement_threshold": 0.2,
        "siglip_dino_severe_disagreement": (
            (sum(disagreement) / len(disagreement) if disagreement else 0.0) > 0.2
        ),
    }
    (args.output_dir / "progress_signal.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
