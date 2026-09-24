#!/usr/bin/env python3
"""Create comparison grid of alpha test results."""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def create_grid():
    input_dir = Path("outputs/test_alpha_all_steps_simple")
    output_path = Path("outputs/test_alpha_all_steps_simple/comparison_grid.png")

    alphas = [0.0, 0.5, 0.55, 0.6, 0.75, 0.9, 1.0]
    images = []

    for alpha in alphas:
        img_path = input_dir / f"alpha_{alpha:.2f}.png"
        img = Image.open(img_path).convert("RGB")
        # Resize to make grid more manageable
        img = img.resize((img.width // 2, img.height // 2), Image.LANCZOS)
        images.append((alpha, img))

    # Create 2 rows
    cols = 4
    rows = 2

    w, h = images[0][1].size
    label_height = 40

    grid_width = cols * w
    grid_height = rows * (h + label_height)

    grid = Image.new('RGB', (grid_width, grid_height), 'white')
    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        font = ImageFont.load_default()

    for idx, (alpha, img) in enumerate(images):
        row = idx // cols
        col = idx % cols

        x = col * w
        y = row * (h + label_height)

        # Draw label
        label = f"α={alpha:.2f}"
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_x = x + (w - text_w) // 2
        draw.text((text_x, y + 5), label, fill='black', font=font)

        # Paste image
        grid.paste(img, (x, y + label_height))

    grid.save(output_path)
    print(f"Saved comparison grid to: {output_path}")

if __name__ == "__main__":
    create_grid()
