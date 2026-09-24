#!/usr/bin/env python3
"""Calculate perceptual distances between adjacent alpha values."""

from pathlib import Path
import torch
from PIL import Image
import numpy as np

def load_image_tensor(path):
    img = Image.open(path).convert("RGB")
    array = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)

def simple_distance(img1, img2):
    """Simple MSE distance for quick comparison."""
    return ((img1 - img2) ** 2).mean().item()

def main():
    input_dir = Path("outputs/test_alpha_all_steps_simple")
    alphas = [0.0, 0.5, 0.55, 0.6, 0.75, 0.9, 1.0]

    images = {}
    for alpha in alphas:
        img_path = input_dir / f"alpha_{alpha:.2f}.png"
        images[alpha] = load_image_tensor(img_path)

    print("=" * 70)
    print("Perceptual Distance Analysis (α affecting ALL 15 steps)")
    print("=" * 70)
    print("\nAdjacent distances (MSE):")
    print("-" * 70)

    for i in range(len(alphas) - 1):
        alpha1, alpha2 = alphas[i], alphas[i + 1]
        dist = simple_distance(images[alpha1], images[alpha2])
        delta_alpha = alpha2 - alpha1
        print(f"  α={alpha1:.2f} → α={alpha2:.2f} (Δ={delta_alpha:.2f}):  distance={dist:.6f}")

    print("\n" + "=" * 70)
    print("Key observations to check:")
    print("=" * 70)
    print("\n1. Small α differences (0.50→0.55, 0.55→0.60) should show non-zero distances")
    print("2. Distances should generally increase with larger α jumps")
    print("3. Compare with original implementation where 0.50→0.55 collapsed to ~0")

if __name__ == "__main__":
    main()
