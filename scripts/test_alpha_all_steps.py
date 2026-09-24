#!/usr/bin/env python3
"""Test if letting alpha affect all steps fixes the collapse issue."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rewardflow_calibration.rollout.veloedit import (
    VeloEditCompatibleRollout,
    VeloEditRolloutConfig,
)
from rewardflow_calibration.utils.images import image_tensor, save_tensor_image


def main():
    torch.manual_seed(42)
    device = torch.device("cuda:0")

    # Setup
    model_path = "/data15/hyp/weight/FLUX.1-Kontext-dev"
    source_path = "/home/hyp/Code/VeloEdit/testdata/7.jpg"
    prompt = "Make him old."
    output_dir = Path("outputs/test_alpha_all_steps_simple")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = VeloEditRolloutConfig(
        steps=15,
        seed=42,
        guidance_scale=2.5,
        first_step_align_steps=4,
        preserve_steps=4,
        edit_steps=4,
        similarity_threshold=0.8,
    )

    print("Loading pipeline...")
    runner = VeloEditCompatibleRollout(model_path, device=device)

    print("Preparing...")
    source_pil = Image.open(source_path).convert("RGB")
    prepared = runner.prepare(source_pil, prompt, config=config, seed=42)

    # Test critical alpha values
    test_alphas = [0.0, 0.5, 0.55, 0.6, 0.75, 0.9, 1.0]

    print(f"Generating images for alphas: {test_alphas}")
    for alpha in test_alphas:
        print(f"  Generating alpha={alpha:.2f}...")
        images = runner.rollout(prepared, [alpha], config=config)
        save_tensor_image(images, output_dir / f"alpha_{alpha:.2f}.png")
        del images
        torch.cuda.empty_cache()

    print(f"\nDone! Check results in: {output_dir}")
    print("\nKey comparisons to make:")
    print("  1. Compare alpha=0.50 vs alpha=0.55 - should be visibly different now")
    print("  2. Compare alpha=0.50 vs alpha=0.60 - should show smooth progression")
    print("  3. Compare alpha=0.75 vs alpha=0.90 - should also show smooth progression")


if __name__ == "__main__":
    main()
