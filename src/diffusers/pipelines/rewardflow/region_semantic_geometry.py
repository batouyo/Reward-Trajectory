"""Pure region aggregation and probe summaries for the v8 diagnostic only."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from .endpoint_comparator_metrics import ordering_diagnostics
from .terminal_control import velocity_topk_mask


@dataclass(frozen=True)
class AggregatedVelocityRegion:
    score_map: torch.Tensor
    token_mask: torch.Tensor
    pixel_mask: torch.Tensor
    raw_bbox_xyxy: tuple[int, int, int, int]
    bbox_xyxy: tuple[int, int, int, int]
    padding_xy: tuple[int, int]
    active_token_fraction: float
    per_step_score_ranges: tuple[tuple[float, float], ...]


def aggregate_velocity_region(
    scores: Sequence[torch.Tensor],
    *,
    token_height: int,
    token_width: int,
    image_height: int,
    image_width: int,
    topk_fraction: float = 0.25,
    padding_fraction: float = 0.10,
) -> AggregatedVelocityRegion:
    """Min-max each early step, average, then reuse native stable top-k selection.

    This fixed-image-region rule is a diagnostic assumption, not a paper method.
    """

    if not scores:
        raise ValueError("At least one early-step velocity score map is required")
    if min(token_height, token_width, image_height, image_width) <= 0:
        raise ValueError("Token-grid and image dimensions must be positive")
    if not math.isfinite(padding_fraction) or padding_fraction < 0:
        raise ValueError("padding_fraction must be finite and nonnegative")
    tokens = token_height * token_width
    normalized, ranges = [], []
    for step, score in enumerate(scores):
        if not torch.is_tensor(score) or score.shape != (1, tokens):
            raise ValueError(f"Velocity score at step {step} must have shape [1, {tokens}]")
        if not torch.isfinite(score).all():
            raise ValueError(f"Velocity score at step {step} contains non-finite values")
        value = score.detach().float()
        low, high = value.amin(), value.amax()
        if bool((high <= low).item()):
            raise ValueError(f"Velocity score at step {step} is constant; min-max normalization is undefined")
        normalized.append((value - low) / (high - low))
        ranges.append((float(low), float(high)))
    average = torch.stack(normalized).mean(dim=0)
    binary = velocity_topk_mask(average, topk_fraction)[0, :, 0].reshape(token_height, token_width)
    score_map = average.reshape(token_height, token_width).detach()
    pixel_mask = F.interpolate(binary[None, None], size=(image_height, image_width), mode="nearest")[0, 0].bool()
    ys, xs = torch.where(pixel_mask)
    if ys.numel() == 0:
        raise RuntimeError("Aggregated top-k mask unexpectedly contains no active pixels")
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    pad_x = math.ceil((x1 - x0) * padding_fraction)
    pad_y = math.ceil((y1 - y0) * padding_fraction)
    bbox = (max(0, x0 - pad_x), max(0, y0 - pad_y), min(image_width, x1 + pad_x), min(image_height, y1 + pad_y))
    return AggregatedVelocityRegion(
        score_map=score_map,
        token_mask=binary.detach(),
        pixel_mask=pixel_mask.detach(),
        raw_bbox_xyxy=(x0, y0, x1, y1),
        bbox_xyxy=bbox,
        padding_xy=(pad_x, pad_y),
        active_token_fraction=float(binary.mean()),
        per_step_score_ranges=tuple(ranges),
    )


def summarize_probe_geometry(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    """Reuse v6 ordering semantics for the three cached pixel-oracle probes."""

    by_id = {str(row["image_id"]): row for row in rows}
    ids = ("oracle_0.2", "oracle_0.5", "oracle_0.8")
    if len(by_id) != len(rows) or any(image_id not in by_id for image_id in ids):
        raise ValueError("Rows must have unique IDs and contain all three probes")
    progress = [float(by_id[image_id]["progress"]) for image_id in ids]
    off_axis = [float(by_id[image_id]["off_axis"]) for image_id in ids]
    if not all(math.isfinite(value) for value in progress + off_axis):
        raise ValueError("Probe geometry values must be finite")
    ordering = ordering_diagnostics(progress)
    gaps = (progress[1] - progress[0], progress[2] - progress[1])
    return {
        "p_0.2": progress[0],
        "p_0.5": progress[1],
        "p_0.8": progress[2],
        "strict_order": bool(ordering["strict_order_pass"]),
        "gap_0.2_to_0.5": gaps[0],
        "gap_0.5_to_0.8": gaps[1],
        "minimum_adjacent_gap": min(gaps),
        "probe_span_0.2_to_0.8": progress[2] - progress[0],
        "mean_probe_off_axis": statistics.mean(off_axis),
    }
