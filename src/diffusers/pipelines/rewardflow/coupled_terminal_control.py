"""RewardSlider V1 coupled, independent terminal-trajectory controls.

This is a research path separate from RewardFlow's paper reproduction and
from the legacy/shared-direction terminal controller.  The K middle branches
own independent FP32 velocity tensors.  ``D`` is a detached soft prior and is
never used to parameterize a branch as a scalar multiple.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .paper_components import paper_euler_update
from .strength_trajectory import expand_shared_initial_latents
from .terminal_control import TerminalControlUnrollOutput


CoupledVelocityFn = Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]


@dataclass(frozen=True)
class CoupledControlPrior:
    """Detached native keep-edit directions and continuous token relevance."""

    directions: tuple[torch.Tensor, ...]
    raw_scores: tuple[torch.Tensor, ...]
    relevance: tuple[torch.Tensor, ...]


def initialize_independent_coupled_controls(
    directions: Sequence[torch.Tensor],
    *,
    num_branches: int,
    betas: Sequence[float] | None = None,
    noise_std: float = 0.0,
) -> nn.ParameterList:
    """Initialize one independent ``[K,tokens,channels]`` parameter per step.

    ``betas`` only set initial values; no shared scalar or shared storage is
    retained after this function returns.
    """

    if num_branches < 1:
        raise ValueError("`num_branches` must be positive.")
    if not directions:
        raise ValueError("At least one detached keep-edit direction is required.")
    if betas is None:
        betas = tuple(1 - (index + 1) / (num_branches + 1) for index in range(num_branches))
    if len(betas) != num_branches:
        raise ValueError("`betas` must provide exactly one initialization factor per branch.")
    if noise_std < 0:
        raise ValueError("`noise_std` must be non-negative.")
    parameters = []
    for direction in directions:
        if direction.ndim != 3 or direction.shape[0] != 1:
            raise ValueError("Each keep-edit direction must have shape [1, tokens, channels].")
        base = direction.detach().to(dtype=torch.float32)
        value = torch.stack([float(beta) * base[0] for beta in betas], dim=0).clone()
        if noise_std:
            value.add_(torch.randn_like(value) * noise_std)
        parameters.append(nn.Parameter(value))
    return nn.ParameterList(parameters)


def soft_relevance_from_velocity_scores(scores: Sequence[torch.Tensor], *, eps: float = 1e-8) -> tuple[torch.Tensor, ...]:
    """Map detached non-negative token scores to continuous maps in ``[0,1]``.

    Max normalization keeps the prior interpretable and deliberately does not
    threshold, select top-k, or multiply controls by a mask.
    """

    if eps <= 0:
        raise ValueError("`eps` must be positive.")
    output = []
    for score in scores:
        if score.ndim != 2 or score.shape[0] != 1 or not torch.isfinite(score).all():
            raise ValueError("Each score must be a finite [1, tokens] tensor.")
        maximum = score.detach().float().amax(dim=1, keepdim=True)
        normalized = torch.where(maximum > eps, score.detach().float() / maximum.clamp_min(eps), torch.zeros_like(score))
        output.append(normalized.clamp(0, 1).detach())
    return tuple(output)


def make_coupled_prior(
    directions: Sequence[torch.Tensor], scores: Sequence[torch.Tensor]
) -> CoupledControlPrior:
    if len(directions) != len(scores):
        raise ValueError("Every controlled step needs both a direction and a score.")
    frozen_directions = tuple(direction.detach().float().clone() for direction in directions)
    frozen_scores = tuple(score.detach().float().clone() for score in scores)
    return CoupledControlPrior(
        directions=frozen_directions,
        raw_scores=frozen_scores,
        relevance=soft_relevance_from_velocity_scores(frozen_scores),
    )


def unroll_coupled_velocity_controls(
    initial_latent: torch.Tensor,
    timesteps: Sequence[torch.Tensor],
    sigmas: torch.Tensor | Sequence[float],
    velocity_fn: CoupledVelocityFn,
    controls: Sequence[torch.Tensor],
    *,
    use_checkpointing: bool = True,
) -> TerminalControlUnrollOutput:
    """Differentiably unroll K independent branches batched in one forward.

    ``initial_latent`` remains a B=1 captured native state.  Each control has
    shape ``[K,tokens,channels]`` and creates the K branch dimension.  Later
    native dynamics intentionally retain their graph: only model parameters
    are frozen by the caller.
    """

    if initial_latent.ndim != 3 or initial_latent.shape[0] != 1:
        raise ValueError("Coupled terminal control requires one captured [1,tokens,channels] latent.")
    if not controls:
        raise ValueError("At least one controlled timestep is required.")
    if len(controls) > len(timesteps):
        raise ValueError("Control prefix cannot exceed available timesteps.")
    num_branches = controls[0].shape[0]
    expected_shape = (num_branches, *initial_latent.shape[1:])
    if num_branches < 1 or any(control.shape != expected_shape for control in controls):
        raise ValueError("Every coupled control must share shape [K,tokens,channels].")
    if any(control.dtype != torch.float32 for control in controls):
        raise ValueError("Coupled control master parameters must be FP32.")
    sigmas = torch.as_tensor(sigmas, device=initial_latent.device)
    if sigmas.ndim != 1 or sigmas.numel() != len(timesteps) + 1:
        raise ValueError("`sigmas` must contain one value more than `timesteps`.")

    latent = expand_shared_initial_latents(initial_latent, num_branches)
    states = [latent]
    native_control_velocities = []
    for step_index, timestep in enumerate(timesteps):
        timestep = torch.as_tensor(timestep, device=latent.device)

        def predict(current: torch.Tensor) -> torch.Tensor:
            return velocity_fn(current, timestep, step_index)

        native = checkpoint(predict, latent, use_reentrant=False) if use_checkpointing and latent.requires_grad else predict(latent)
        if native.shape != latent.shape:
            raise ValueError("Batched native velocity must have the current latent shape.")
        if step_index < len(controls):
            native_control_velocities.append(native.detach())
            velocity = native + controls[step_index].to(dtype=native.dtype)
        else:
            velocity = native
        latent = paper_euler_update(latent, velocity, sigmas[step_index], sigmas[step_index + 1])
        states.append(latent)
    return TerminalControlUnrollOutput(
        final_latent=latent,
        states=tuple(states),
        native_control_velocities=tuple(native_control_velocities),
        controlled_step_indices=tuple(range(len(controls))),
    )


def control_band_loss(
    controls: Sequence[torch.Tensor], directions: Sequence[torch.Tensor], relevance: Sequence[torch.Tensor], *, eps: float = 1e-8
) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
    """Softly prefer the keep-to-native control corridor without projection."""

    if not (len(controls) == len(directions) == len(relevance)):
        raise ValueError("Controls, directions, and relevance must have identical step counts.")
    losses, diagnostics = [], []
    for control, direction, map_ in zip(controls, directions, relevance):
        if direction.shape[0] != 1 or control.shape[1:] != direction.shape[1:] or map_.shape != direction.shape[:2]:
            raise ValueError("Invalid coupled control prior shapes.")
        direction = direction.to(control)
        denominator = direction.square().sum(dim=-1, keepdim=True).clamp_min(eps)
        coefficient = (control * direction).sum(dim=-1, keepdim=True) / denominator
        parallel = coefficient * direction
        orthogonal = control - parallel
        # Small-D locations get no unreliable direction projection signal;
        # relevance still makes spatial/energy penalties well-defined there.
        weight = map_.to(control).unsqueeze(-1)
        wrong = F.relu(-coefficient).square()
        beyond = F.relu(coefficient - 1).square()
        orth_ratio = orthogonal.square().sum(dim=-1, keepdim=True) / denominator
        losses.append(
            ((wrong + beyond + orth_ratio) * weight).sum()
            / (weight.sum() * control.shape[0]).clamp_min(eps)
        )
        flat_control, flat_direction = control.float().flatten(1), direction.float().expand_as(control).flatten(1)
        diagnostics.append(
            {
                "coefficient_mean": coefficient.detach().mean(),
                "coefficient_std": coefficient.detach().std(unbiased=False),
                "negative_fraction": (coefficient.detach() < 0).float().mean(),
                "above_one_fraction": (coefficient.detach() > 1).float().mean(),
                "orthogonal_energy_ratio": orth_ratio.detach().mean(),
                "cosine_to_direction": F.cosine_similarity(flat_control, flat_direction, dim=1).detach().mean(),
                "control_to_direction_norm": (flat_control.norm(dim=1) / flat_direction.norm(dim=1).clamp_min(eps)).detach().mean(),
            }
        )
    return torch.stack(losses).mean(), diagnostics


def spatial_prior_loss(controls: Sequence[torch.Tensor], relevance: Sequence[torch.Tensor], *, eps: float = 1e-8) -> torch.Tensor:
    """Penalize only the *fraction* of control energy outside relevance.

    This intentionally does not reward ``U -> 0``: scaling a non-zero control
    leaves the loss unchanged.  A zero control has a finite zero contribution
    and is handled by ``eps`` rather than producing a NaN.
    """

    values = []
    for control, map_ in zip(controls, relevance):
        weight = (1 - map_.to(control)).unsqueeze(-1)
        squared = control.float().square()
        low_energy = (squared * weight).sum(dim=(1, 2))
        total_energy = squared.sum(dim=(1, 2))
        values.append((low_energy / total_energy.clamp_min(eps)).mean())
    return torch.stack(values).mean()


def control_no_jump_loss(controls: Sequence[torch.Tensor], *, max_jump_ratio: float = 3.0, eps: float = 1e-8) -> torch.Tensor:
    """Optional control-only outlier-jump penalty, defaulted off by the caller.

    This deliberately uses only ``[U_weak, U_mid, U_strong, 0]``.  It neither
    includes ``D`` as an endpoint nor asks adjacent jumps to be equal.
    """

    if max_jump_ratio <= 1:
        raise ValueError("`max_jump_ratio` must be greater than one.")
    values = []
    for control in controls:
        if control.ndim != 3 or control.shape[0] < 1:
            raise ValueError("Each coupled control must have shape [K,tokens,channels].")
        nodes = torch.cat((control, torch.zeros_like(control[:1])), dim=0)
        jumps = (nodes[1:] - nodes[:-1]).float().square().mean(dim=(1, 2)).sqrt()
        values.append(F.relu(jumps.max() / jumps.mean().clamp_min(eps) - max_jump_ratio).square())
    return torch.stack(values).mean()


def control_energy_loss(controls: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.stack([control.float().square().mean() for control in controls]).mean()


def adjacent_ranking_loss(progress: torch.Tensor, *, margin: float = 0.01) -> torch.Tensor:
    if progress.ndim != 1 or progress.numel() < 2:
        raise ValueError("Progress must contain an ordered one-dimensional trajectory.")
    if margin < 0:
        raise ValueError("Ranking margin must be non-negative.")
    return F.relu(float(margin) - (progress[1:] - progress[:-1])).mean()


def relative_gap_loss(
    distances: torch.Tensor,
    *,
    min_gap_fraction: float = 0.05,
    max_gap_fraction: float = 0.55,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scale-free collapse/jump loss over each trajectory's path fractions."""

    if distances.ndim != 1 or distances.numel() < 1:
        raise ValueError("Adjacent distances must be one-dimensional.")
    if not 0 <= min_gap_fraction <= max_gap_fraction <= 1:
        raise ValueError("Gap fractions must satisfy 0 <= min <= max <= 1.")
    fractions = distances / distances.sum().clamp_min(eps)
    collapse = F.relu(min_gap_fraction - fractions).mean()
    jump = F.relu(fractions - max_gap_fraction).mean()
    return collapse + jump, collapse, jump, fractions


def coarse_anchor_indices(num_nodes: int, *, max_anchors: int = 5) -> tuple[int, ...]:
    """Select unique, ordered endpoint-inclusive anchors for a dense path."""

    if num_nodes < 2:
        raise ValueError("A trajectory needs at least source and native-full nodes.")
    if max_anchors < 2:
        raise ValueError("`max_anchors` must be at least two.")
    count = min(max_anchors, num_nodes)
    # Python round has banker's behavior. Tensor round gives the documented
    # linspace-round construction consistently across supported Python builds.
    indices = torch.linspace(0, num_nodes - 1, count).round().to(torch.int64).tolist()
    unique = tuple(dict.fromkeys(indices))
    if unique[0] != 0 or unique[-1] != num_nodes - 1 or len(unique) != count:
        raise RuntimeError("Anchor construction lost an endpoint or uniqueness.")
    return unique


def semantic_coverage_loss(
    progress: torch.Tensor,
    *,
    anchor_indices: Sequence[int] | None = None,
    min_fraction: float = 0.05,
    max_fraction: float = 0.55,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Coarse semantic coverage without imposing equal semantic spacing."""

    if progress.ndim != 1 or progress.numel() < 2:
        raise ValueError("Progress must be a one-dimensional trajectory.")
    if not 0 <= min_fraction <= max_fraction <= 1:
        raise ValueError("Semantic fractions must satisfy 0 <= min <= max <= 1.")
    indices = tuple(anchor_indices or coarse_anchor_indices(progress.numel()))
    if len(indices) < 2 or indices[0] != 0 or indices[-1] != progress.numel() - 1:
        raise ValueError("Anchors must be ordered and include both endpoints.")
    anchors = progress[list(indices)]
    gaps = anchors[1:] - anchors[:-1]
    # Endpoint coordinates are defined as zero and one by this objective, but
    # normalize defensively for toy diagnostics and numerical auditability.
    fractions = gaps / gaps.sum().clamp_min(eps)
    collapse = F.relu(min_fraction - fractions.min())
    jump = F.relu(fractions.max() - max_fraction)
    return collapse + jump, collapse, jump, gaps


def fine_jump_loss(
    distances: torch.Tensor, *, max_fraction: float = 0.55, eps: float = 1e-8
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Allow tiny dense adjacent gaps; penalize only the single worst jump."""

    if distances.ndim != 1 or distances.numel() < 1:
        raise ValueError("Adjacent distances must be one-dimensional.")
    if not 0 <= max_fraction <= 1:
        raise ValueError("`max_fraction` must lie in [0, 1].")
    fractions = distances / distances.sum().clamp_min(eps)
    worst = fractions.max()
    return F.relu(worst - max_fraction), fractions, worst


def triangle_deficit_loss(
    adjacent: torch.Tensor, skip: torch.Tensor, *, eps: float = 1e-8
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Local-normalized non-detour penalty for every consecutive triplet."""

    if adjacent.ndim != 1 or skip.ndim != 1 or adjacent.numel() != skip.numel() + 1:
        raise ValueError("Need N adjacent and N-1 skip distances.")
    raw_deficits = F.relu(adjacent[:-1] + adjacent[1:] - skip)
    normalized = raw_deficits / skip.clamp_min(eps)
    return normalized.mean(), raw_deficits, normalized


def scalar_direction_residual_diagnostics(
    controls: Sequence[torch.Tensor], directions: Sequence[torch.Tensor], *, eps: float = 1e-8
) -> list[dict[str, torch.Tensor]]:
    """Measure each branch's departure from its best scalar multiple of ``D``."""

    if len(controls) != len(directions):
        raise ValueError("Every control needs one detached keep-edit direction.")
    diagnostics = []
    for control, direction in zip(controls, directions):
        if control.ndim != 3 or direction.shape != (1, *control.shape[1:]):
            raise ValueError("Expected control [K,tokens,channels] and direction [1,tokens,channels].")
        direction = direction.to(control)
        denominator = direction.float().square().sum().clamp_min(eps)
        beta = (control.float() * direction.float()).sum(dim=(1, 2)) / denominator
        residual = control.float() - beta[:, None, None] * direction.float()
        residual_ratio = residual.flatten(1).norm(dim=1) / control.float().flatten(1).norm(dim=1).clamp_min(eps)
        diagnostics.append({"beta_star": beta.detach(), "residual_ratio": residual_ratio.detach()})
    return diagnostics


def weighted_source_preservation(images: torch.Tensor, source: torch.Tensor, relevance_image: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    if images.ndim != 4 or source.shape[0] != 1 or source.shape[1:] != images.shape[1:]:
        raise ValueError("Candidates must be [K,3,H,W] and source [1,3,H,W].")
    if relevance_image.shape != (1, 1, *images.shape[-2:]):
        raise ValueError("Image relevance must be [1,1,H,W].")
    weight = 1 - relevance_image.to(images)
    absolute = (images.float() - source.to(images).float()).abs()
    return (absolute * weight).sum() / (weight.sum() * images.shape[1] * images.shape[0]).clamp_min(eps)


def endpoint_progress_batch(source_feature: torch.Tensor, candidate_features: torch.Tensor, full_feature: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """Unclamped CLIP endpoint-axis coordinates for ranking only."""

    source, full = source_feature.float().reshape(-1), full_feature.float().reshape(-1)
    candidates = candidate_features.float()
    if candidates.ndim != 2 or candidates.shape[1] != source.numel() or full.shape != source.shape:
        raise ValueError("Endpoint feature dimensions do not match.")
    axis = full - source
    return ((candidates - source) * axis).sum(dim=1) / axis.square().sum().clamp_min(eps)
