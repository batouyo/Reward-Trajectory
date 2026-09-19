"""Small optimization-state helpers used by the auditable V2 runner."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch


@dataclass
class BestTrajectoryState:
    """Snapshot the best calibration point without retaining an autograd graph."""

    best_kl: float | None = None
    best_alpha_logits: torch.Tensor | None = None
    best_alphas: torch.Tensor | None = None
    best_iteration: int | None = None

    def update(self, parameterization, *, kl: float | torch.Tensor, iteration: int) -> bool:
        value = float(kl.detach().item() if torch.is_tensor(kl) else kl)
        if not torch.isfinite(torch.tensor(value)):
            raise ValueError("Trajectory KL must be finite.")
        if self.best_kl is not None and value >= self.best_kl:
            return False
        self.best_kl = value
        self.best_alpha_logits = parameterization.interval_logits.detach().clone()
        self.best_alphas = parameterization.alphas.detach().clone()
        self.best_iteration = int(iteration)
        return True

    def restore(self, parameterization) -> bool:
        if self.best_alpha_logits is None:
            return False
        if parameterization.interval_logits.shape != self.best_alpha_logits.shape:
            raise ValueError("Best trajectory snapshot has incompatible topology.")
        with torch.no_grad():
            parameterization.interval_logits.copy_(
                self.best_alpha_logits.to(device=parameterization.interval_logits.device)
            )
        return True


@dataclass(frozen=True)
class CoordinateSearchResult:
    alphas: torch.Tensor
    kl: torch.Tensor
    accepted_steps: int
    rejected_steps: int
    sweeps: int


def coordinate_search_alphas(
    initial_alphas: torch.Tensor,
    evaluate,
    *,
    initial_delta: float = 0.02,
    min_delta: float = 0.001,
    margin: float = 1e-4,
) -> CoordinateSearchResult:
    """Deterministic coordinate line search using the real trajectory evaluator."""
    if initial_alphas.ndim != 1 or initial_alphas.numel() < 3:
        raise ValueError("At least one interior alpha is required.")
    if initial_delta <= 0 or min_delta <= 0 or min_delta > initial_delta or margin < 0:
        raise ValueError("Invalid coordinate-search step or margin.")
    current = initial_alphas.detach().clone().float()
    if current[0] != 0 or current[-1] != 1 or torch.any(current[1:] <= current[:-1]):
        raise ValueError("Initial alphas must be strictly ordered with endpoints 0 and 1.")
    current_kl = torch.as_tensor(evaluate(current)).detach().float().reshape(())
    accepted = rejected = sweeps = 0
    delta = float(initial_delta)
    while delta >= min_delta:
        improved = False
        sweeps += 1
        for index in range(1, current.numel() - 1):
            best_candidate = current
            best_value = current_kl
            for direction in (-1.0, 1.0):
                candidate = current.clone()
                candidate[index] += direction * delta
                lower = candidate[index - 1] + margin
                upper = candidate[index + 1] - margin
                if candidate[index] < lower or candidate[index] > upper:
                    rejected += 1
                    continue
                value = torch.as_tensor(evaluate(candidate)).detach().float().reshape(())
                if value < best_value:
                    best_candidate, best_value = candidate, value
                else:
                    rejected += 1
            if best_value < current_kl:
                current, current_kl = best_candidate, best_value
                accepted += 1
                improved = True
        if not improved:
            delta *= 0.5
    return CoordinateSearchResult(current, current_kl, accepted, rejected, sweeps)


@dataclass(frozen=True)
class LocalInsertResult:
    alphas: torch.Tensor
    alpha: torch.Tensor
    d_left: torch.Tensor
    d_right: torch.Tensor
    kl: torch.Tensor
    balance_ratio: torch.Tensor


def local_insert_line_search(
    alphas: torch.Tensor,
    *,
    interval: int,
    evaluate,
    fractions=(0.25, 0.5, 0.75),
) -> LocalInsertResult:
    """Choose an inserted alpha by a real local LPIPS balance evaluator."""
    if not 0 <= interval < alphas.numel() - 1:
        raise ValueError("Insertion interval is out of range.")
    candidates = []
    for fraction in fractions:
        candidate = alphas.detach().clone()
        candidate[interval + 1] = alphas[interval] + fraction * (alphas[interval + 1] - alphas[interval])
        d_left, d_right, kl = evaluate(candidate)
        total = torch.as_tensor(d_left) + torch.as_tensor(d_right)
        balance = (torch.as_tensor(d_left) - torch.as_tensor(d_right)).abs() / total.clamp_min(1e-8)
        candidates.append((balance, candidate, torch.as_tensor(d_left), torch.as_tensor(d_right), torch.as_tensor(kl)))
    balance, candidate, d_left, d_right, kl = min(candidates, key=lambda item: float(item[0]))
    return LocalInsertResult(candidate, candidate[interval + 1], d_left, d_right, kl, balance)


@dataclass(frozen=True)
class HybridAcceptanceResult:
    accepted: bool
    current_kl: torch.Tensor
    rejected_steps: int


def hybrid_acceptance_step(parameter, optimizer, *, current_kl, step, evaluate, tolerance=0.0):
    """Run one optimizer step and restore both params and Adam state on rejection."""
    parameters = (parameter,) if isinstance(parameter, torch.Tensor) else tuple(parameter)
    snapshots = [item.detach().clone() for item in parameters]
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    step()
    new_kl = torch.as_tensor(evaluate()).detach().float().reshape(())
    old_kl = torch.as_tensor(current_kl).detach().float().reshape(())
    if new_kl <= old_kl + tolerance:
        return HybridAcceptanceResult(True, new_kl, 0)
    with torch.no_grad():
        for item, snapshot in zip(parameters, snapshots):
            item.copy_(snapshot)
    optimizer.load_state_dict(optimizer_state)
    return HybridAcceptanceResult(False, old_kl, 1)
