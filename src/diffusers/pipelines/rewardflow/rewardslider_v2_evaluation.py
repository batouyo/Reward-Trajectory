"""Fixed-density evaluation helpers for RewardSlider V2."""

from __future__ import annotations

import torch

from .rewardslider_v2_lpips import lpips_trajectory_stats


def interpolate_strengths(learned_alphas: torch.Tensor, requested: torch.Tensor) -> torch.Tensor:
    """Piecewise-linear alpha(s) over uniformly indexed learned nodes."""
    if learned_alphas.ndim != 1 or requested.ndim != 1:
        raise ValueError("Alpha grids must be one-dimensional.")
    if learned_alphas.numel() < 3 or requested.numel() < 1:
        raise ValueError("Need at least one interior learned node and two requests.")
    if learned_alphas[0] != 0 or learned_alphas[-1] != 1 or torch.any(learned_alphas[1:] <= learned_alphas[:-1]):
        raise ValueError("Learned alphas must be strictly ordered with endpoints 0 and 1.")
    if requested[0] < 0 or requested[-1] > 1 or torch.any(requested[1:] < requested[:-1]):
        raise ValueError("Requested strengths must be sorted in [0, 1].")
    positions = torch.linspace(0, 1, learned_alphas.numel(), device=learned_alphas.device, dtype=learned_alphas.dtype)
    indices = torch.searchsorted(positions, requested.to(positions), right=True).clamp(1, positions.numel() - 1)
    left = indices - 1
    fraction = (requested.to(positions) - positions[left]) / (positions[indices] - positions[left])
    return learned_alphas[left] + fraction * (learned_alphas[indices] - learned_alphas[left])


def fixed_grid_metrics(images: torch.Tensor, distance, *, requested: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    if requested is None:
        requested = torch.linspace(0, 1, images.shape[0], device=images.device, dtype=images.dtype)
    stats = lpips_trajectory_stats(distance(images[:-1], images[1:]).reshape(-1).float())
    normalized = stats.normalized_distances
    return {
        "requested_strengths": requested,
        "adjacent_lpips": stats.distances,
        "normalized_lpips": normalized,
        "kl": stats.kl_uniform,
        "max_gap": normalized.max(),
        "min_gap": normalized.min(),
        "max_min_ratio": normalized.max() / normalized.min().clamp_min(1e-8),
        "path_length": stats.path_length,
        "endpoint_distance": distance(images[:1], images[-1:]).reshape(-1).mean(),
    }
