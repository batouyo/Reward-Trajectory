"""Small optimization-state helpers used by the auditable V2 runner."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass
class BestTrajectoryState:
    """Snapshot the best calibration point without retaining an autograd graph."""

    best_kl: float | None = None
    best_alpha_logits: torch.Tensor | None = None
    best_alphas: torch.Tensor | None = None
    best_v_goals: tuple[torch.Tensor, ...] | None = None
    best_topology_signature: tuple | None = None
    best_iteration: int | None = None

    def update(
        self,
        parameterization,
        *,
        kl: float | torch.Tensor,
        iteration: int,
        v_goals: Sequence[torch.Tensor] | None = None,
    ) -> bool:
        value = float(kl.detach().item() if torch.is_tensor(kl) else kl)
        if not torch.isfinite(torch.tensor(value)):
            raise ValueError("Trajectory KL must be finite.")
        if self.best_kl is not None and value >= self.best_kl:
            return False
        self.best_kl = value
        self.best_alpha_logits = parameterization.interval_logits.detach().clone()
        self.best_alphas = parameterization.alphas.detach().clone()
        self.best_v_goals = None if v_goals is None else tuple(goal.detach().clone() for goal in v_goals)
        self.best_topology_signature = (
            tuple(self.best_alpha_logits.shape),
            None if self.best_v_goals is None else tuple(tuple(goal.shape) for goal in self.best_v_goals),
        )
        self.best_iteration = int(iteration)
        return True

    def restore(self, parameterization, v_goals: Sequence[torch.Tensor] | None = None) -> bool:
        if self.best_alpha_logits is None:
            return False
        if parameterization.interval_logits.shape != self.best_alpha_logits.shape:
            raise ValueError("Best trajectory snapshot has incompatible topology.")
        if (self.best_v_goals is None) != (v_goals is None):
            raise ValueError("Best trajectory snapshot and live V_goal topology differ.")
        if v_goals is not None:
            if len(v_goals) != len(self.best_v_goals):
                raise ValueError("Best trajectory snapshot has incompatible V_goal topology.")
            if any(goal.shape != saved.shape for goal, saved in zip(v_goals, self.best_v_goals)):
                raise ValueError("Best trajectory snapshot has incompatible V_goal shapes.")
        with torch.no_grad():
            parameterization.interval_logits.copy_(
                self.best_alpha_logits.to(device=parameterization.interval_logits.device)
            )
            if v_goals is not None:
                for goal, saved in zip(v_goals, self.best_v_goals):
                    goal.copy_(saved.to(device=goal.device, dtype=goal.dtype))
        return True



@dataclass
class BestQualityState:
    """Best image-repair checkpoint under a trajectory-KL constraint."""

    best_deficit: float | None = None
    best_kl: float | None = None
    best_preservation: float | None = None
    best_quality: float | None = None
    best_alpha_logits: torch.Tensor | None = None
    best_v_goals: tuple[torch.Tensor, ...] | None = None
    best_iteration: int | None = None
    preservation_only: bool = True

    def update(self, parameterization, *, kl, deficit, preservation, quality, iteration, v_goals):
        values = [float(value.detach().item() if torch.is_tensor(value) else value) for value in (kl, deficit, preservation)]
        quality_value = None if quality is None else float(quality.detach().item() if torch.is_tensor(quality) else quality)
        if not all(torch.isfinite(torch.tensor(value)) for value in values):
            raise ValueError("Quality checkpoint values must be finite.")
        if quality_value is not None and not torch.isfinite(torch.tensor(quality_value)):
            raise ValueError("Quality checkpoint quality value must be finite.")
        if self.best_deficit is not None and values[1] >= self.best_deficit:
            return False
        self.best_kl, self.best_deficit, self.best_preservation = values
        self.best_quality = quality_value
        self.best_alpha_logits = parameterization.interval_logits.detach().clone()
        self.best_v_goals = tuple(goal.detach().clone() for goal in v_goals)
        self.best_iteration = int(iteration)
        self.preservation_only = quality is None
        return True

    def restore(self, parameterization, v_goals) -> bool:
        if self.best_alpha_logits is None or parameterization.interval_logits.shape != self.best_alpha_logits.shape:
            return False
        if len(v_goals) != len(self.best_v_goals) or any(goal.shape != saved.shape for goal, saved in zip(v_goals, self.best_v_goals)):
            return False
        with torch.no_grad():
            parameterization.interval_logits.copy_(self.best_alpha_logits.to(parameterization.interval_logits))
            for goal, saved in zip(v_goals, self.best_v_goals):
                goal.copy_(saved.to(device=goal.device, dtype=goal.dtype))
        return True
def v_goal_off_axis_diagnostics(v_goals: Sequence[torch.Tensor], native_directions: Sequence[torch.Tensor], *, eps: float = 1e-8) -> dict[str, object]:
    """Report V_goal geometry relative to detached native residual directions."""
    if len(v_goals) != len(native_directions):
        raise ValueError("v_goals and native_directions must have the same timestep count.")
    per_timestep = []
    nonzero = []
    for goal, direction in zip(v_goals, native_directions):
        direction = direction.to(device=goal.device, dtype=goal.dtype)
        beta = (goal * direction).flatten(1).sum(dim=1) / (direction.square().flatten(1).sum(dim=1) + eps)
        residual = goal - beta.reshape((-1,) + (1,) * (goal.ndim - 1)) * direction
        goal_norm = goal.float().flatten(1).norm(dim=1)
        residual_norm = residual.float().flatten(1).norm(dim=1)
        zero = goal_norm <= eps
        rows = []
        for index in range(goal.shape[0]):
            ratio = None if bool(zero[index]) else float((residual_norm[index] / (goal_norm[index] + eps)).detach())
            if ratio is not None:
                nonzero.append(ratio)
            rows.append({"branch": index, "goal_norm": float(goal_norm[index].detach()), "beta": float(beta[index].detach()), "off_axis_ratio": ratio, "zero_vgoal": bool(zero[index])})
        per_timestep.append(rows)
    return {"per_timestep": per_timestep, "mean_nonzero_off_axis_ratio": None if not nonzero else sum(nonzero) / len(nonzero)}


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
    candidates: tuple[dict, ...]
    search_left: torch.Tensor
    search_right: torch.Tensor
    old_alpha_width: torch.Tensor
    old_visual_gap: torch.Tensor
    split_max_ratio: torch.Tensor

    @property
    def old_gap(self) -> torch.Tensor:
        """Backward-compatible alias for the alpha-space width."""
        return self.old_alpha_width


def local_insert_line_search(
    inserted_alphas: torch.Tensor,
    *,
    inserted_index: int | None = None,
    search_left: torch.Tensor | float | None = None,
    search_right: torch.Tensor | float | None = None,
    pre_insert_alphas: torch.Tensor | None = None,
    interval: int | None = None,
    evaluate,
    old_visual_gap: torch.Tensor | float | None = None,
    fractions=(0.25, 0.5, 0.75),
) -> LocalInsertResult:
    """Choose an inserted alpha over the complete pre-insertion interval."""
    if inserted_index is None:
        if interval is None:
            raise ValueError("`inserted_index` or legacy `interval` is required.")
        inserted_index = interval + 1
    if not 0 < inserted_index < inserted_alphas.numel() - 1:
        raise ValueError("Inserted node index is out of range.")
    if pre_insert_alphas is not None:
        if interval is None or not 0 <= interval < pre_insert_alphas.numel() - 1:
            raise ValueError("A valid pre-insertion interval is required.")
        search_left = pre_insert_alphas[interval]
        search_right = pre_insert_alphas[interval + 1]
    if search_left is None or search_right is None:
        raise ValueError("The complete original interval must be supplied.")
    search_left = torch.as_tensor(search_left, device=inserted_alphas.device, dtype=inserted_alphas.dtype)
    search_right = torch.as_tensor(search_right, device=inserted_alphas.device, dtype=inserted_alphas.dtype)
    if not bool(search_right > search_left):
        raise ValueError("Search interval must be strictly increasing.")
    old_alpha_width = search_right - search_left
    if old_visual_gap is None:
        # Legacy tensor-only callers have no image metric; the real runner
        # always supplies the fresh LPIPS visual gap explicitly.
        old_visual_gap = old_alpha_width
    old_visual_gap = torch.as_tensor(old_visual_gap, device=inserted_alphas.device, dtype=inserted_alphas.dtype)
    if not bool(torch.isfinite(old_visual_gap) and old_visual_gap > 0):
        raise ValueError("`old_visual_gap` must be finite and positive.")
    candidates = []
    for fraction in fractions:
        candidate = inserted_alphas.detach().clone()
        candidate[inserted_index] = search_left + fraction * old_alpha_width
        d_left, d_right, kl = evaluate(candidate)
        total = torch.as_tensor(d_left) + torch.as_tensor(d_right)
        balance = (torch.as_tensor(d_left) - torch.as_tensor(d_right)).abs() / total.clamp_min(1e-8)
        candidates.append({
            "fraction": float(fraction),
            "alpha": float(candidate[inserted_index]),
            "d_left": float(torch.as_tensor(d_left).detach()),
            "d_right": float(torch.as_tensor(d_right).detach()),
            "balance_ratio": float(balance.detach()),
            "kl": float(torch.as_tensor(kl).detach()),
            "alphas": candidate.detach().cpu().tolist(),
        })
    selected = min(candidates, key=lambda item: item["balance_ratio"])
    candidate = inserted_alphas.detach().clone()
    candidate[inserted_index] = selected["alpha"]
    d_left, d_right, kl = evaluate(candidate)
    d_left = torch.as_tensor(d_left)
    d_right = torch.as_tensor(d_right)
    kl = torch.as_tensor(kl)
    balance = (d_left - d_right).abs() / (d_left + d_right).clamp_min(1e-8)
    split_max_ratio = torch.maximum(d_left, d_right) / old_visual_gap.clamp_min(1e-8)
    return LocalInsertResult(
        candidate, candidate[inserted_index], d_left, d_right, kl, balance,
        tuple(candidates), search_left, search_right, old_alpha_width, old_visual_gap, split_max_ratio,
    )


@dataclass(frozen=True)
class HybridAcceptanceResult:
    accepted: bool
    current_kl: torch.Tensor
    rejected_steps: int


@dataclass
class OptimizationTransaction:
    """Atomic snapshot for a joint alpha/V_goal proposal."""

    alpha_parameters: tuple[torch.Tensor, ...]
    v_goal_parameters: tuple[torch.Tensor, ...]
    alpha_values: tuple[torch.Tensor, ...]
    v_goal_values: tuple[torch.Tensor, ...]
    alpha_optimizer_state: dict
    vgoal_optimizer_state: dict
    scheduler: object | None = None
    scheduler_state: dict | None = None

    @classmethod
    def capture(cls, *, alpha_parameters, v_goal_parameters, alpha_optimizer, vgoal_optimizer, scheduler=None):
        alpha_parameters = tuple(alpha_parameters)
        v_goal_parameters = tuple(v_goal_parameters)
        scheduler_state = None
        if scheduler is not None:
            scheduler_state = {
                "phase": scheduler.phase,
                "phase_iterations": scheduler.phase_iterations,
                "phase_reference_kl": scheduler.phase_reference_kl,
                "healthy_streak": scheduler._healthy_streak,
                "bad_streak": scheduler._bad_streak,
                "topology_events": copy.deepcopy(scheduler.topology_events),
            }
        return cls(
            alpha_parameters, v_goal_parameters,
            tuple(parameter.detach().clone() for parameter in alpha_parameters),
            tuple(parameter.detach().clone() for parameter in v_goal_parameters),
            copy.deepcopy(alpha_optimizer.state_dict()),
            copy.deepcopy(vgoal_optimizer.state_dict()),
            scheduler, scheduler_state,
        )

    def restore(self, alpha_optimizer, vgoal_optimizer) -> None:
        with torch.no_grad():
            for parameter, value in zip(self.alpha_parameters, self.alpha_values):
                parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
            for parameter, value in zip(self.v_goal_parameters, self.v_goal_values):
                parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
        alpha_optimizer.load_state_dict(copy.deepcopy(self.alpha_optimizer_state))
        vgoal_optimizer.load_state_dict(copy.deepcopy(self.vgoal_optimizer_state))
        if self.scheduler is not None and self.scheduler_state is not None:
            self.scheduler.phase = self.scheduler_state["phase"]
            self.scheduler.phase_iterations = self.scheduler_state["phase_iterations"]
            self.scheduler.phase_reference_kl = self.scheduler_state["phase_reference_kl"]
            self.scheduler._healthy_streak = self.scheduler_state["healthy_streak"]
            self.scheduler._bad_streak = self.scheduler_state["bad_streak"]
            self.scheduler.topology_events = copy.deepcopy(self.scheduler_state["topology_events"])
            self.scheduler._set_parameter_permissions()


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
