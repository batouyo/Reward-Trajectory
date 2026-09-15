import json

import pytest
import torch

from diffusers.pipelines.rewardflow.endpoint_feature_distance import FeatureEndpointDistanceReward
from diffusers.pipelines.rewardflow.feature_controller_evaluation import (
    BLIND_IDENTITIES,
    DENSE_STRENGTHS,
    EVALUATION_MASK_PROVENANCE,
    FEATURE_CONTROL_PROVENANCE,
    NATIVE_FULL_PROVENANCE,
    SOURCE_INPUT_PROVENANCE,
    TRAINING_STRENGTHS,
    build_blind_permutations,
    dense_feature_curve_diagnostics,
    endpoint_difference_evaluation_mask,
    endpoint_pixel_diagnostics,
    parse_visual_judge_json,
    remap_visual_judgment,
    summarize_visual_judgments,
    visual_judge_json_schema,
)
from diffusers.pipelines.rewardflow.relative_endpoint_parser import parse_relative_endpoint_semantic_json
from diffusers.pipelines.rewardflow.terminal_control import (
    initialize_velocity_controls,
    masked_effective_controls,
    update_best_control_checkpoint,
)


def _spec():
    return parse_relative_endpoint_semantic_json(
        json.dumps(
            {
                "edit_instruction": "change the object",
                "primitives": [
                    {
                        "id": "object_color",
                        "object": "object",
                        "attribute": "color",
                        "edit_description": "change color",
                        "comparison_focus": "visible object color",
                        "source_state": "dark",
                        "target_state": "bright",
                        "endpoint_question": "What color is the object?",
                        "source_answer": "Dark.",
                        "target_answer": "Bright.",
                        "weight": 1.0,
                    }
                ],
                "preserve_constraints": [],
                "unresolved_instruction_items": [],
            }
        )
    )


class FeatureScorer:
    def __init__(self):
        self.model = torch.nn.Linear(1, 1, bias=False)
        self.model.requires_grad_(False)

    def focus_conditioned_representation(self, image, prompt):
        del prompt
        value = image.mean()
        return torch.stack((1 - value, value))


def _reward():
    source = torch.zeros(1, 3, 4, 4, requires_grad=True)
    full = torch.ones(1, 3, 4, 4, requires_grad=True)
    return FeatureEndpointDistanceReward(FeatureScorer(), _spec(), source, full), source, full


def test_zero_controls_share_native_path_and_strength_one_is_exactly_zero():
    reference = torch.zeros(1, 4, 2)
    directions = initialize_velocity_controls(reference, 2)
    masks = (torch.ones(1, 4, 1), torch.ones(1, 4, 1))
    outputs = [masked_effective_controls(directions, masks, strength=value) for value in TRAINING_STRENGTHS]
    assert all(torch.equal(left, right) for families in zip(*outputs) for left in families for right in families)
    strength_one = masked_effective_controls(directions, masks, strength=1.0)
    assert all(torch.equal(control, torch.zeros_like(control)) for control in strength_one)


def test_feature_reward_has_nonzero_shared_control_gradient_and_frozen_context():
    reward, source, full = _reward()
    shared_control = torch.tensor(0.2, requires_grad=True)
    image = torch.sigmoid(shared_control).expand_as(source)
    loss = sum(reward(image, strength).loss for strength in TRAINING_STRENGTHS)
    gradient = torch.autograd.grad(loss, shared_control)[0]
    assert torch.isfinite(gradient) and gradient.abs() > 0
    assert all(
        not feature.requires_grad for endpoints in reward._endpoint_features.values() for feature in endpoints.values()
    )
    assert not reward.source_image.requires_grad and not reward.full_image.requires_grad
    assert source.grad is None and full.grad is None
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in reward.model.parameters())


def test_best_checkpoint_selection_uses_only_supplied_feature_error():
    control = torch.nn.Parameter(torch.tensor([1.0]))
    best = update_best_control_checkpoint(None, iteration=0, objective_error=torch.tensor(0.4), controls=(control,))
    control.data.fill_(9.0)
    unchanged = update_best_control_checkpoint(
        best, iteration=1, objective_error=torch.tensor(0.5), controls=(control,)
    )
    updated = update_best_control_checkpoint(
        unchanged, iteration=2, objective_error=torch.tensor(0.1), controls=(control,)
    )
    assert unchanged is best and torch.equal(best.controls[0], torch.tensor([1.0]))
    assert updated.iteration == 2 and torch.equal(updated.controls[0], torch.tensor([9.0]))


def test_dense_nodes_are_held_out_from_optimizer_except_registered_training_nodes():
    assert TRAINING_STRENGTHS == (0.2, 0.5, 0.8)
    assert DENSE_STRENGTHS == tuple(index / 10 for index in range(11))
    assert set(DENSE_STRENGTHS) - set(TRAINING_STRENGTHS) == {0.0, 0.1, 0.3, 0.4, 0.6, 0.7, 0.9, 1.0}


def test_blind_permutations_are_unique_reproducible_and_remap_correctly():
    first = build_blind_permutations()
    second = build_blind_permutations()
    assert first == second and len(first) == 6
    assert len({tuple(mapping.values()) for mapping in first}) == 6
    payload = _valid_judgment()
    remapped = remap_visual_judgment(payload, first[0])
    assert remapped["ordered_identities"] == [first[0][label] for label in payload["ordered_labels"]]


def _valid_judgment():
    return {
        "ordered_labels": ["A", "B", "C", "D", "E"],
        "per_image": {
            label: {
                "endpoint_relation": relation,
                "semantic_description": f"visible state {label}",
                "preservation_violations": [],
            }
            for label, relation in zip(
                ("A", "B", "C", "D", "E"),
                ("closer_to_source", "closer_to_source", "intermediate", "closer_to_full", "closer_to_full"),
            )
        },
        "indistinguishable_pairs": [],
    }


def test_visual_judge_json_parser_is_strict():
    payload = _valid_judgment()
    assert parse_visual_judge_json(json.dumps(payload)) == payload
    invalid = dict(payload, percentage=50)
    with pytest.raises(ValueError, match="unexpected"):
        parse_visual_judge_json(json.dumps(invalid))
    invalid = json.loads(json.dumps(payload))
    invalid["ordered_labels"][-1] = "A"
    with pytest.raises(ValueError, match="permutation"):
        parse_visual_judge_json(json.dumps(invalid))


def test_visual_gate_requires_five_exact_orders_and_majority_relations():
    trials = []
    for _ in range(6):
        trials.append(
            {
                "ordered_identities": list(BLIND_IDENTITIES),
                "per_identity": {
                    "source": {"endpoint_relation": "closer_to_source", "preservation_violations": []},
                    "best_0.2": {"endpoint_relation": "closer_to_source", "preservation_violations": []},
                    "best_0.5": {"endpoint_relation": "intermediate", "preservation_violations": []},
                    "best_0.8": {"endpoint_relation": "closer_to_full", "preservation_violations": []},
                    "native_full": {"endpoint_relation": "closer_to_full", "preservation_violations": []},
                },
                "indistinguishable_identity_pairs": [],
            }
        )
    summary = summarize_visual_judgments(trials)
    assert summary["exact_order_count"] == 6
    assert summary["primary_visual_ordering_gate"] and summary["visual_weak_mid_strong"] == "PASS"


def test_evaluation_mask_depends_only_on_endpoints_and_pixel_metric_is_not_feature_loss():
    source = torch.zeros(1, 3, 4, 4)
    full = source.clone()
    full[:, :, :2] = 1
    candidate_a = torch.rand_like(source)
    candidate_b = torch.rand_like(source)
    mask_a = endpoint_difference_evaluation_mask(source, full)
    mask_b = endpoint_difference_evaluation_mask(source, full)
    assert torch.equal(mask_a, mask_b) and mask_a.sum() == 4
    diagnostics_a = endpoint_pixel_diagnostics(candidate_a, source, full, mask_a)
    diagnostics_b = endpoint_pixel_diagnostics(candidate_b, source, full, mask_a)
    assert set(diagnostics_a) == set(diagnostics_b)
    reward = FeatureEndpointDistanceReward(FeatureScorer(), _spec(), source, full)
    feature_loss = reward(candidate_a, 0.5).loss.detach().clone()
    endpoint_pixel_diagnostics(candidate_a, source, full, mask_a)
    assert torch.equal(feature_loss, reward(candidate_a, 0.5).loss.detach())


def test_dense_curve_reports_ties_reversals_and_jumps_separately():
    diagnostics = dense_feature_curve_diagnostics((0, 0.1, 0.2, 0.3), (0, 0.1, 0.1, -0.2))
    assert not diagnostics["strict_order_pass"]
    assert diagnostics["tie_count"] == 1
    assert diagnostics["descending_inversion_count"] > 0
    assert diagnostics["sudden_jump_indices"] == [2]


def test_provenance_categories_are_explicit_and_disjoint():
    values = {
        SOURCE_INPUT_PROVENANCE,
        NATIVE_FULL_PROVENANCE,
        FEATURE_CONTROL_PROVENANCE,
        EVALUATION_MASK_PROVENANCE,
    }
    assert len(values) == 4
    assert "NOT USED BY REWARD OR CONTROLLER" in EVALUATION_MASK_PROVENANCE


def test_visual_judge_schema_uses_tianyuai_supported_subset_but_parser_keeps_uniqueness_checks():
    import json
    import pytest

    assert "uniqueItems" not in json.dumps(visual_judge_json_schema())

    invalid = _valid_judgment()
    invalid["indistinguishable_pairs"] = [["A", "A"]]
    with pytest.raises(ValueError, match="two distinct labels"):
        parse_visual_judge_json(json.dumps(invalid))

    invalid = _valid_judgment()
    invalid["per_image"]["A"]["preservation_violations"] = ["identity", "identity"]
    with pytest.raises(ValueError, match="preservation_violations"):
        parse_visual_judge_json(json.dumps(invalid))
