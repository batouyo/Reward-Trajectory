"""Minimal auditable RewardSlider V2 runner facade and CLI."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import torch
from torch import nn

from .rewardslider_v2_scheduler import RewardSliderV2Scheduler


def build_rewardslider_v2_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the RewardSlider V2 optimization loop.")
    parser.add_argument("--initial-nodes", type=int, default=5)
    parser.add_argument("--max-nodes", type=int, default=10)
    parser.add_argument("--control-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha-lr", type=float, default=1e-3)
    parser.add_argument("--vgoal-lr", type=float, default=1e-3)
    parser.add_argument("--trajectory-kl-threshold", type=float, default=0.15)
    parser.add_argument("--trajectory-patience", type=int, default=3)
    parser.add_argument("--local-refine-iters", type=int, default=10)
    parser.add_argument("--joint-refine-iters", type=int, default=10)
    parser.add_argument("--enable-prune", action="store_true")
    parser.add_argument("--enable-insert", action="store_true")
    parser.add_argument("--quality-reward", default=None)
    parser.add_argument("--use-checkpointing", action="store_true")
    parser.add_argument("--output-jsonl", type=Path, default=Path("rewardslider_v2.jsonl"))
    return parser


def _norm(value: torch.Tensor | None) -> float | None:
    return None if value is None else float(value.detach().float().norm().item())


class RewardSliderV2Runner:
    """Log one deterministic optimization step; FLUX execution is injected by caller."""

    def __init__(self, alpha_parameter: nn.Parameter, v_goals: Sequence[nn.Parameter], output_jsonl: str | Path):
        self.alpha_parameter = alpha_parameter
        self.v_goals = tuple(v_goals)
        self.output_jsonl = Path(output_jsonl)
        self.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        self.scheduler = RewardSliderV2Scheduler(
            nn.ParameterList([alpha_parameter]), self.v_goals, min_repair_iterations=1
        )
        self.iteration = 0

    def step(
        self,
        trajectory_loss: torch.Tensor,
        quality_loss: torch.Tensor,
        *,
        trajectory_kl: float,
        reward_values: Sequence[float] = (),
        normalized_rewards: Sequence[float] = (),
        dynamic_weights: Sequence[float] = (),
        topology_event: dict | None = None,
    ) -> dict:
        started = time.perf_counter()
        self.scheduler.backward(trajectory_loss=trajectory_loss, quality_loss=quality_loss)
        self.scheduler.advance(trajectory_kl)
        record = {
            "iteration": self.iteration,
            "trajectory": {
                "current_number_of_nodes": self.alpha_parameter.numel() + 1,
                "trajectory_kl": float(trajectory_kl),
            },
            "reward": {
                "raw_values": list(reward_values),
                "normalized_values": list(normalized_rewards),
                "dynamic_weights": list(dynamic_weights),
            },
            "gradient": {
                "alpha_gradient_norm": _norm(self.alpha_parameter.grad),
                "v_goal_gradient_norms": [_norm(goal.grad) for goal in self.v_goals],
                "finite": all(parameter.grad is None or torch.isfinite(parameter.grad).all().item() for parameter in (self.alpha_parameter, *self.v_goals)),
            },
            "control": {"v_goal_norms": [_norm(goal) for goal in self.v_goals]},
            "topology": topology_event or {"operation": None},
            "system": {"phase": self.scheduler.phase, "elapsed_seconds": time.perf_counter() - started, "nan_inf": False},
        }
        with self.output_jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        self.iteration += 1
        return record


def main() -> None:
    args = build_rewardslider_v2_parser().parse_args()
    torch.manual_seed(args.seed)
    raise RuntimeError("Instantiate RewardSliderV2Runner from a prepared FLUX-Kontext pipeline; no implicit weight loading is performed.")


if __name__ == "__main__":
    main()
