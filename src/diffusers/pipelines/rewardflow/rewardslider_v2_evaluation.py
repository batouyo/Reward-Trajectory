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


def _piecewise_interpolate_nodes(node_values: torch.Tensor, requested: torch.Tensor) -> torch.Tensor:
    """Interpolate control tensors on the uniformly indexed learned-node axis."""
    if node_values.ndim < 1 or node_values.shape[0] < 2:
        raise ValueError("node_values must have at least two nodes on dimension 0.")
    requested = requested.to(device=node_values.device, dtype=node_values.dtype)
    if requested.ndim != 1 or requested.numel() == 0:
        raise ValueError("requested strengths must be a non-empty one-dimensional tensor.")
    if requested[0] < 0 or requested[-1] > 1 or torch.any(requested[1:] < requested[:-1]):
        raise ValueError("requested strengths must be sorted in [0, 1].")
    positions = torch.linspace(0, 1, node_values.shape[0], device=node_values.device, dtype=node_values.dtype)
    indices = torch.searchsorted(positions, requested, right=True).clamp(1, positions.numel() - 1)
    left = indices - 1
    fraction = (requested - positions[left]) / (positions[indices] - positions[left])
    fraction = fraction.reshape((fraction.shape[0],) + (1,) * (node_values.ndim - 1))
    return (1 - fraction) * node_values[left] + fraction * node_values[indices]


def interpolate_rewardslider_controls(
    learned_alphas: torch.Tensor,
    v_goals: list[torch.Tensor] | tuple[torch.Tensor, ...],
    requested_strengths: torch.Tensor,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """Interpolate alpha and endpoint-zero V_goal controls, never image tensors."""
    if learned_alphas.ndim != 1 or learned_alphas.numel() < 3:
        raise ValueError("learned_alphas must contain endpoints and an interior node.")
    if len(v_goals) == 0:
        raise ValueError("v_goals must contain controlled timestep tensors.")
    if any(goal.ndim < 2 for goal in v_goals):
        raise ValueError("Each V_goal tensor must have a branch dimension and payload dimensions.")
    if any(goal.shape[0] != learned_alphas.numel() - 2 for goal in v_goals):
        raise ValueError("Each V_goal tensor must contain one row per interior learned node.")
    full_goals = []
    for goal in v_goals:
        zeros = torch.zeros((1,) + tuple(goal.shape[1:]), device=goal.device, dtype=goal.dtype)
        full_goals.append(torch.cat((zeros, goal, zeros), dim=0))
    alpha = interpolate_strengths(learned_alphas, requested_strengths)
    interpolated = tuple(_piecewise_interpolate_nodes(goal, requested_strengths) for goal in full_goals)
    return alpha, interpolated


def generate_model_fixed_grid(
    learned_alphas: torch.Tensor,
    v_goals: list[torch.Tensor] | tuple[torch.Tensor, ...],
    requested_strengths: torch.Tensor,
    *,
    source_image: torch.Tensor,
    native_image: torch.Tensor,
    rematerialize_inputs,
    unroll_callback,
    decode_callback,
    control_steps: int,
    use_checkpointing: bool,
) -> dict[str, object]:
    """Generate a fixed grid by running the model on interpolated controls."""
    if requested_strengths.numel() < 3 or requested_strengths[0] != 0 or requested_strengths[-1] != 1:
        raise ValueError("fixed grid must include 0 and 1 endpoints.")
    alpha, all_goals = interpolate_rewardslider_controls(learned_alphas, v_goals, requested_strengths)
    interior_count = requested_strengths.numel() - 2
    fixed_inputs = rematerialize_inputs(num_branches=interior_count)
    fixed_unroll = unroll_callback(
        fixed_inputs,
        alpha[1:-1],
        [goal[1:-1] for goal in all_goals],
        control_steps=control_steps,
        use_checkpointing=use_checkpointing,
    )
    interior_images = decode_callback(fixed_unroll, fixed_inputs)
    images = torch.cat((source_image, interior_images, native_image), dim=0)
    if images.shape[0] != requested_strengths.numel():
        raise AssertionError("model-generated fixed grid has an unexpected node count.")
    return {
        "requested_strengths": requested_strengths,
        "alpha": alpha,
        "v_goals": all_goals,
        "inputs": fixed_inputs,
        "unroll": fixed_unroll,
        "interior_images": interior_images,
        "images": images,
        "provenance": "MODEL_GENERATED_REWARDSLIDER_OUTPUT",
        "pixel_blend_used": False,
        "interior_model_forward_count": interior_count,
    }


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
