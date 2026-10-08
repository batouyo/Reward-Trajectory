"""Pure tensor helpers for diagnosing backward velocity directions.

These functions do not implement an optimizer or use the trajectory acceptance
gate.  They are intentionally independent of the production controller so the
diagnostic can measure directions that the controller would reject.
"""

from __future__ import annotations

import math
from typing import Callable, Mapping

import torch


def _validate_direction(direction: torch.Tensor, name: str = "direction") -> None:
    if direction.ndim != 3:
        raise ValueError(f"{name} must have shape [steps, tokens, channels]")
    if direction.numel() == 0 or not torch.isfinite(direction).all():
        raise ValueError(f"{name} must be non-empty and finite")


def mask_direction(direction: torch.Tensor, hard_mask: torch.Tensor, *, sign: float = 1.0) -> torch.Tensor:
    """Apply a frozen boolean support mask and sign without normalization."""
    _validate_direction(direction)
    mask = torch.as_tensor(hard_mask, device=direction.device, dtype=torch.bool)
    if mask.shape != direction.shape:
        raise ValueError("hard_mask must have the same shape as direction")
    if not math.isfinite(sign):
        raise ValueError("sign must be finite")
    return direction.detach().float() * mask.float() * float(sign)


def veloedit_backward_direction(edit_direction: torch.Tensor) -> torch.Tensor:
    """Return the direction from native velocity toward the VeloEdit reference."""
    _validate_direction(edit_direction, "edit_direction")
    return -edit_direction.detach().float()


def scale_global_direction(
    direction: torch.Tensor,
    native_velocity: torch.Tensor,
    requested_ratio: float,
    *,
    active_mask: torch.Tensor | None = None,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Scale all steps with one scalar to a global residual/native RMS ratio."""
    _validate_direction(direction)
    native = torch.as_tensor(native_velocity, device=direction.device).detach().float()
    if native.shape != direction.shape or not torch.isfinite(native).all():
        raise ValueError("native_velocity must be finite and match direction shape")
    if not math.isfinite(requested_ratio) or requested_ratio <= 0:
        raise ValueError("requested_ratio must be finite and positive")
    if eps <= 0:
        raise ValueError("eps must be positive")
    direction = direction.detach().float()
    native_global = native.square().mean().sqrt()
    direction_rms = direction.square().mean().sqrt()
    if float(native_global) <= eps or float(direction_rms) <= eps:
        raise FloatingPointError("native velocity and direction must have non-zero RMS")
    scale = float(requested_ratio) * native_global / direction_rms
    residual = direction * scale
    native_per_step = native.square().mean(dim=(1, 2)).sqrt()
    residual_per_step = residual.square().mean(dim=(1, 2)).sqrt()
    active_ratio = None
    if active_mask is not None:
        mask = torch.as_tensor(active_mask, device=direction.device, dtype=torch.bool)
        if mask.shape != direction.shape:
            raise ValueError("active_mask must match direction shape")
        if bool(mask.any()):
            active_native = native[mask]
            active_residual = residual[mask]
            denom = active_native.square().mean().sqrt()
            if float(denom) > eps:
                active_ratio = float((active_residual.square().mean().sqrt() / denom).cpu())
    actual_global = float((residual.square().mean().sqrt() / native_global).cpu())
    return residual, {
        "requested_global_ratio": float(requested_ratio),
        "actual_global_ratio": actual_global,
        "active_mask_ratio": active_ratio,
        "per_step_ratio": (residual_per_step / native_per_step.clamp_min(eps)).cpu().tolist(),
        "scalar": float(scale.cpu()),
    }


def scale_isolated_timestep(
    direction: torch.Tensor,
    native_velocity: torch.Tensor,
    step: int,
    requested_ratio: float = 0.02,
    *,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Scale one step to its own native RMS ratio and leave other steps zero."""
    _validate_direction(direction)
    native = torch.as_tensor(native_velocity, device=direction.device).detach().float()
    if native.shape != direction.shape:
        raise ValueError("native_velocity must match direction shape")
    if not 0 <= step < direction.shape[0]:
        raise IndexError("step is out of range")
    if not math.isfinite(requested_ratio) or requested_ratio <= 0:
        raise ValueError("requested_ratio must be finite and positive")
    step_dir = direction.detach().float()[step]
    step_native = native[step]
    native_rms = step_native.square().mean().sqrt()
    direction_rms = step_dir.square().mean().sqrt()
    if float(native_rms) <= eps or float(direction_rms) <= eps:
        raise FloatingPointError("isolated timestep direction/native RMS is zero")
    residual = torch.zeros_like(direction, dtype=torch.float32)
    residual[step] = step_dir * (float(requested_ratio) * native_rms / direction_rms)
    return residual, {
        "step": int(step),
        "requested_step_ratio": float(requested_ratio),
        "actual_step_ratio": float((residual[step].square().mean().sqrt() / native_rms).cpu()),
        "per_step_nonzero": [bool(torch.count_nonzero(residual[i])) for i in range(direction.shape[0])],
    }


def cosine_similarity(first: torch.Tensor, second: torch.Tensor, *, eps: float = 1e-12) -> float | None:
    """Cosine similarity, returning None when either vector has zero energy."""
    if first.shape != second.shape:
        raise ValueError("cosine inputs must have matching shapes")
    a, b = first.detach().float().flatten(), second.detach().float().flatten()
    denom = a.norm() * b.norm()
    if float(denom) <= eps:
        return None
    return float(((a @ b) / denom).cpu())


def compare_directions(first: torch.Tensor, second: torch.Tensor) -> dict[str, object]:
    """Return global and timestep-wise cosine similarities."""
    _validate_direction(first, "first direction")
    _validate_direction(second, "second direction")
    if first.shape != second.shape:
        raise ValueError("direction tensors must match")
    return {
        "global_cosine": cosine_similarity(first, second),
        "per_step_cosine": [
            cosine_similarity(first[i], second[i]) for i in range(first.shape[0])
        ],
    }


def parallel_orthogonal_energy(direction: torch.Tensor, reference: torch.Tensor) -> dict[str, object]:
    """Measure energy parallel and orthogonal to a reference direction."""
    _validate_direction(direction)
    _validate_direction(reference, "reference")
    if direction.shape != reference.shape:
        raise ValueError("direction and reference must match")

    def fractions(a: torch.Tensor, b: torch.Tensor) -> tuple[float | None, float | None]:
        a, b = a.detach().float().flatten(), b.detach().float().flatten()
        a_energy, b_energy = a.square().sum(), b.square().sum()
        if float(a_energy) <= 1e-24 or float(b_energy) <= 1e-24:
            return None, None
        parallel = ((a @ b).square() / (a_energy * b_energy)).clamp(0, 1)
        return float(parallel.cpu()), float((1 - parallel).cpu())

    parallel, orthogonal = fractions(direction, reference)
    per_step = [fractions(direction[i], reference[i]) for i in range(direction.shape[0])]
    return {
        "parallel_energy_fraction": parallel,
        "orthogonal_energy_fraction": orthogonal,
        "per_step_parallel_energy_fraction": [item[0] for item in per_step],
        "per_step_orthogonal_energy_fraction": [item[1] for item in per_step],
    }


def temporal_energy(direction: torch.Tensor) -> dict[str, object]:
    """Return per-step RMS and energy fractions that sum to one when nonzero."""
    _validate_direction(direction)
    value = direction.detach().float()
    step_rms = value.square().mean(dim=(1, 2)).sqrt()
    energy = step_rms.square()
    total = energy.sum()
    fractions = torch.zeros_like(energy) if float(total) <= 1e-24 else energy / total
    return {
        "rms_per_step": step_rms.cpu().tolist(),
        "energy_fraction_per_step": fractions.cpu().tolist(),
        "energy_fraction_sum": float(fractions.sum().cpu()),
    }


def component_gradients(
    losses: Mapping[str, torch.Tensor], residual: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Differentiate each scalar reward component independently w.r.t. residual."""
    names = list(losses)
    gradients: dict[str, torch.Tensor] = {}
    for index, name in enumerate(names):
        loss = losses[name]
        if loss.numel() != 1:
            raise ValueError(f"loss component {name!r} must be scalar")
        if not loss.requires_grad:
            gradients[name] = torch.zeros_like(residual)
            continue
        gradient = torch.autograd.grad(
            loss,
            residual,
            retain_graph=index < len(names) - 1,
            allow_unused=True,
        )[0]
        gradients[name] = torch.zeros_like(residual) if gradient is None else gradient.detach()
    return gradients


def evaluate_proxy_final_pair(
    rollout,
    prepared,
    config,
    residual: torch.Tensor,
    goal_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run proxy and full rollouts with the exact same residual tensor, no gate."""
    with torch.no_grad():
        proxy = rollout.rollout_native(
            prepared,
            config=config,
            goal_residual=residual,
            early_stop_steps=goal_steps,
        )
        final = rollout.rollout_native(
            prepared,
            config=config,
            goal_residual=residual,
            early_stop_steps=None,
        )
    return proxy, final


def capture_final_gradient(compute: Callable[[], Mapping[str, torch.Tensor]]) -> dict[str, object]:
    """Capture final-gradient OOM without discarding already computed diagnostics."""
    try:
        return {"final_gradient_status": "ok", "gradients": dict(compute()), "error": None}
    except torch.cuda.OutOfMemoryError as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"final_gradient_status": "oom", "gradients": None, "error": str(exc)}
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"final_gradient_status": "oom", "gradients": None, "error": str(exc)}
