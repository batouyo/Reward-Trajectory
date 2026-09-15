"""Endpoint-comparative semantic reward for controlled image-edit trajectories.

This is an independently gated research reward, not a RewardFlow paper
component. Its score is a VLM endpoint-preference margin, not human semantic
strength and not an image, latent, or velocity interpolation coordinate.
"""

from __future__ import annotations

import math
from typing import Any, Protocol, Sequence

import torch

from .relative_endpoint_parser import RelativeEndpointPrimitiveSpec, RelativeEndpointSemanticSpec
from .terminal_control import TerminalObjectiveOutput


RELATIVE_ENDPOINT_CHOICES = ("A", "B")


class MultiImageChoiceScorer(Protocol):
    def score_multi_image_single_token_choices(
        self,
        images: Sequence[torch.Tensor],
        question: str,
        choices: tuple[str, ...] = RELATIVE_ENDPOINT_CHOICES,
    ) -> torch.Tensor: ...


class RelativeEndpointValidationError(ValueError):
    def __init__(self, diagnostics: dict[str, Any]):
        self.diagnostics = diagnostics
        super().__init__("Relative endpoint reward validation failed; inspect endpoint diagnostics.")


def build_relative_endpoint_comparison_prompt(
    spec: RelativeEndpointSemanticSpec,
    primitive: RelativeEndpointPrimitiveSpec,
    *,
    first_reference: str,
) -> str:
    """Render a comparison prompt for [reference, candidate, other reference]."""

    if first_reference not in {"source", "full"}:
        raise ValueError("`first_reference` must be `source` or `full`.")
    image_1 = "SOURCE" if first_reference == "source" else "NATIVE FULL-EDIT"
    image_3 = "NATIVE FULL-EDIT" if first_reference == "source" else "SOURCE"
    constraints = "; ".join(spec.preserve_constraints) if spec.preserve_constraints else "none specified"
    return f"""You are comparing an image-editing trajectory.

Editing instruction:
{spec.edit_instruction}

Focus ONLY on:
{primitive.comparison_focus}

Ignore unrelated visual differences such as:
{constraints}

Image 1 is the {image_1} reference.
Image 2 is the CURRENT CANDIDATE.
Image 3 is the {image_3} reference.

Considering only the requested semantic change, which reference has an edited semantic state more similar to Image 2?

A. Image 1
B. Image 3

Answer with exactly one letter: A or B."""


def _number(value: torch.Tensor) -> float:
    return float(value.detach().float().cpu())


class RelativeEndpointSemanticReward:
    """Symmetrized Source/Candidate/Full preference-margin objective."""

    def __init__(
        self,
        scorer: MultiImageChoiceScorer,
        spec: RelativeEndpointSemanticSpec,
        source_image: torch.Tensor,
        full_image: torch.Tensor,
        *,
        denominator_epsilon: float = 1e-6,
        fail_on_endpoint_validation: bool = True,
    ):
        if not isinstance(spec, RelativeEndpointSemanticSpec) or not spec.primitives:
            raise ValueError("Relative endpoint reward requires a validated semantic spec.")
        if source_image.shape != full_image.shape or source_image.ndim != 4 or source_image.shape[1] != 3:
            raise ValueError("Source and full images must share shape [B, 3, H, W].")
        if source_image.shape[0] != 1:
            raise ValueError("Relative endpoint reward currently supports endpoint batch size one.")
        if not math.isfinite(denominator_epsilon) or denominator_epsilon <= 0:
            raise ValueError("`denominator_epsilon` must be finite and positive.")
        self.scorer = scorer
        self.spec = spec
        self.source_image = source_image.detach()
        self.full_image = full_image.detach()
        self.denominator_epsilon = float(denominator_epsilon)
        self._anchors: dict[str, dict[str, torch.Tensor]] = {}
        diagnostics: dict[str, Any] = {}
        failed = False
        with torch.no_grad():
            for primitive in spec.primitives:
                source = self.primitive_margin(self.source_image, primitive)
                full = self.primitive_margin(self.full_image, primitive)
                dynamic_range = (full["raw_margin"] - source["raw_margin"]).detach()
                finite = all(torch.isfinite(value).all().item() for value in (*source.values(), *full.values()))
                source_prefers_source_both_orders = bool(
                    (source["forward_1_margin"] < 0).item() and (source["forward_2_margin"] < 0).item()
                )
                full_prefers_full_both_orders = bool(
                    (full["forward_1_margin"] > 0).item() and (full["forward_2_margin"] > 0).item()
                )
                range_valid = bool((dynamic_range > self.denominator_epsilon).item())
                valid = finite and source_prefers_source_both_orders and full_prefers_full_both_orders and range_valid
                failed = failed or not valid
                self._anchors[primitive.id] = {
                    "source_margin": source["raw_margin"].detach(),
                    "full_margin": full["raw_margin"].detach(),
                    "dynamic_range": dynamic_range,
                }
                diagnostics[primitive.id] = {
                    "source": {name: _number(value) for name, value in source.items()},
                    "full": {name: _number(value) for name, value in full.items()},
                    "dynamic_range": _number(dynamic_range),
                    "finite": finite,
                    "source_prefers_source_both_orders": source_prefers_source_both_orders,
                    "full_prefers_full_both_orders": full_prefers_full_both_orders,
                    "dynamic_range_valid": range_valid,
                    "valid": valid,
                }
        self.endpoint_diagnostics = diagnostics
        self.endpoint_validation_failed = failed
        if failed and fail_on_endpoint_validation:
            raise RelativeEndpointValidationError(diagnostics)

    @property
    def model(self):
        return getattr(self.scorer, "model", None)

    def primitive_margin(
        self,
        candidate_image: torch.Tensor,
        primitive: RelativeEndpointPrimitiveSpec,
    ) -> dict[str, torch.Tensor]:
        if candidate_image.ndim != 4 or candidate_image.shape[0] != 1 or candidate_image.shape[1] != 3:
            raise ValueError("Candidate image must have shape [1, 3, H, W].")
        prompt_1 = build_relative_endpoint_comparison_prompt(self.spec, primitive, first_reference="source")
        score_1 = self.scorer.score_multi_image_single_token_choices(
            (self.source_image, candidate_image, self.full_image), prompt_1, RELATIVE_ENDPOINT_CHOICES
        )
        prompt_2 = build_relative_endpoint_comparison_prompt(self.spec, primitive, first_reference="full")
        score_2 = self.scorer.score_multi_image_single_token_choices(
            (self.full_image, candidate_image, self.source_image), prompt_2, RELATIVE_ENDPOINT_CHOICES
        )
        if score_1.shape != (2,) or score_2.shape != (2,):
            raise ValueError("Relative endpoint scorer must return exactly two choice log-probabilities.")
        forward_1 = score_1[1] - score_1[0]
        forward_2 = score_2[0] - score_2[1]
        return {
            "forward_1_logprob_a": score_1[0],
            "forward_1_logprob_b": score_1[1],
            "forward_1_margin": forward_1,
            "forward_2_logprob_a": score_2[0],
            "forward_2_logprob_b": score_2[1],
            "forward_2_margin": forward_2,
            "raw_margin": 0.5 * (forward_1 + forward_2),
            "order_bias": 0.5 * (forward_1 - forward_2),
        }

    def margin_vector(self, image: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        margins = []
        diagnostics = {}
        for primitive in self.spec.primitives:
            values = self.primitive_margin(image, primitive)
            anchor = self._anchors[primitive.id]
            coordinate = (values["raw_margin"] - anchor["source_margin"].to(image.device)) / anchor[
                "dynamic_range"
            ].to(image.device)
            margins.append(values["raw_margin"])
            diagnostics[primitive.id] = {
                **values,
                "source_margin": anchor["source_margin"].to(image.device),
                "full_margin": anchor["full_margin"].to(image.device),
                "dynamic_range": anchor["dynamic_range"].to(image.device),
                "relative_coordinate": coordinate,
                "coordinate_clamped_for_logging_only": coordinate.clamp(0, 1),
            }
        return torch.stack(margins), diagnostics

    def _target(self, target_strengths, reference: torch.Tensor) -> torch.Tensor:
        ids = [primitive.id for primitive in self.spec.primitives]
        if isinstance(target_strengths, dict):
            if set(target_strengths) != set(ids):
                raise ValueError("Target mapping must contain exactly every primitive ID.")
            values = [target_strengths[primitive_id] for primitive_id in ids]
        elif torch.is_tensor(target_strengths) and target_strengths.numel() > 1:
            values = target_strengths.flatten()
        elif isinstance(target_strengths, Sequence) and not isinstance(target_strengths, (str, bytes)):
            values = target_strengths
        else:
            values = [target_strengths] * len(ids)
        target = torch.as_tensor(values, device=reference.device, dtype=reference.dtype).flatten()
        if target.numel() != len(ids) or not torch.isfinite(target).all():
            raise ValueError("Provide one finite target per primitive.")
        if bool(((target < 0) | (target > 1)).any().item()):
            raise ValueError("Target strengths must lie in [0, 1].")
        return target

    def __call__(self, image: torch.Tensor, target_strengths) -> TerminalObjectiveOutput:
        margins, diagnostics = self.margin_vector(image)
        target = self._target(target_strengths, margins)
        source = torch.stack([self._anchors[p.id]["source_margin"].to(image.device) for p in self.spec.primitives])
        full = torch.stack([self._anchors[p.id]["full_margin"].to(image.device) for p in self.spec.primitives])
        dynamic_range = full - source
        target_margin = source + target * dynamic_range
        normalized_residual = (margins - target_margin) / dynamic_range
        weights = margins.new_tensor([primitive.weight for primitive in self.spec.primitives])
        weights = weights / weights.sum()
        coordinates = (margins - source) / dynamic_range
        for index, primitive in enumerate(self.spec.primitives):
            diagnostics[primitive.id].update(
                {
                    "target_margin": target_margin[index],
                    "normalized_residual": normalized_residual[index],
                    "semantic_loss": normalized_residual[index].square(),
                }
            )
        return TerminalObjectiveOutput(
            objective_name="relative_endpoint_semantic_margin",
            loss=(weights * normalized_residual.square()).sum(),
            objective_error=(weights * normalized_residual.abs()).sum(),
            source_score=(weights * source).sum(),
            full_score=(weights * full).sum(),
            target_score=(weights * target_margin).sum(),
            achieved_score=(weights * coordinates).sum(),
            diagnostics=diagnostics,
        )


def audit_endpoint_answers(scorer, spec: RelativeEndpointSemanticSpec, source_image, full_image) -> dict[str, Any]:
    """Teacher-forced endpoint sanity check; never used as the continuous reward."""

    report = {}
    with torch.no_grad():
        for primitive in spec.primitives:
            source_source = scorer.score_answer(source_image, primitive.endpoint_question, primitive.source_answer)
            source_target = scorer.score_answer(source_image, primitive.endpoint_question, primitive.target_answer)
            full_source = scorer.score_answer(full_image, primitive.endpoint_question, primitive.source_answer)
            full_target = scorer.score_answer(full_image, primitive.endpoint_question, primitive.target_answer)
            source_valid = bool((source_source > source_target).item())
            full_valid = bool((full_target > full_source).item())
            report[primitive.id] = {
                "source_source_answer_score": _number(source_source),
                "source_target_answer_score": _number(source_target),
                "full_source_answer_score": _number(full_source),
                "full_target_answer_score": _number(full_target),
                "source_prefers_source_answer": source_valid,
                "full_prefers_target_answer": full_valid,
                "valid": source_valid and full_valid,
                "used_for_continuous_reward": False,
            }
    return report
