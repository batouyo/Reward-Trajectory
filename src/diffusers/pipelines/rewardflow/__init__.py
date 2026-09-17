from typing import TYPE_CHECKING

from ...utils import (
    DIFFUSERS_SLOW_IMPORT,
    OptionalDependencyNotAvailable,
    _LazyModule,
    get_objects_from_module,
    is_torch_available,
    is_transformers_available,
)


_dummy_objects = {}
_additional_imports = {}
_import_structure = {
    "pipeline_output": ["FluxRewardFlowPipelineOutput", "StrengthTrajectoryPipelineOutput"],
    "semantic_parser": [
        "SEMANTIC_CACHE_SCHEMA_VERSION",
        "SEMANTIC_PARSER_VERSION",
        "SemanticParseResult",
        "build_semantic_parser_prompt",
        "fingerprint_image",
        "load_cached_parse",
        "make_semantic_cache_key",
        "parse_semantic_parser_json",
        "save_cached_parse",
    ],
    "relative_endpoint_parser": [
        "DEFAULT_RELATIVE_ENDPOINT_PARSER_MODEL",
        "DEFAULT_RELATIVE_ENDPOINT_PARSER_PROVIDER",
        "DEFAULT_RELATIVE_ENDPOINT_PARSER_BASE_URL",
        "RELATIVE_ENDPOINT_CACHE_SCHEMA_VERSION",
        "RELATIVE_ENDPOINT_PARSER_VERSION",
        "OpenAIRelativeEndpointParser",
        "TianyuAIRelativeEndpointParser",
        "RelativeEndpointParseRecord",
        "RelativeEndpointPrimitiveSpec",
        "RelativeEndpointSemanticSpec",
        "build_relative_endpoint_semantic_parser_prompt",
        "fingerprint_endpoint_image",
        "load_cached_relative_endpoint_parse",
        "make_human_relative_endpoint_parse_record",
        "make_relative_endpoint_cache_key",
        "parse_relative_endpoint_semantic_json",
        "relative_endpoint_json_schema",
        "save_cached_relative_endpoint_parse",
    ],
}

try:
    if not (is_transformers_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils import dummy_torch_and_transformers_objects  # noqa F403

    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_objects))
else:
    _import_structure["endpoint_embedding_geometry"] = [
        "CachedEndpointEmbeddingGeometry",
        "EndpointGeometryOutput",
        "endpoint_axis_geometry",
    ]
    _import_structure["endpoint_comparator_metrics"] = [
        "SMALL_GRADIENT_STEPS",
        "average_ranks",
        "gradient_direction_gate",
        "ordering_diagnostics",
        "spearman_correlation",
    ]
    _import_structure["endpoint_feature_distance"] = [
        "FeatureEndpointDistanceReward",
        "build_focus_conditioned_feature_prompt",
    ]
    _import_structure["feature_controller_evaluation"] = [
        "BLIND_IDENTITIES",
        "BLIND_LABELS",
        "DENSE_STRENGTHS",
        "EVALUATION_MASK_PROVENANCE",
        "FEATURE_CONTROL_PROVENANCE",
        "NATIVE_FULL_PROVENANCE",
        "PRESERVATION_CATEGORIES",
        "SOURCE_INPUT_PROVENANCE",
        "TRAINING_STRENGTHS",
        "VISUAL_JUDGE_PERMUTATION_SEED",
        "VISUAL_JUDGE_PROMPT_VERSION",
        "build_blind_permutations",
        "dense_feature_curve_diagnostics",
        "endpoint_difference_evaluation_mask",
        "endpoint_pixel_diagnostics",
        "format_strength_tag",
        "parse_visual_judge_json",
        "remap_visual_judgment",
        "summarize_visual_judgments",
        "visual_judge_json_schema",
    ]
    _import_structure["paper_components"] = [
        "PaperRewardFlowConfig",
        "clean_latent_kl_energy",
        "flow_step_size",
        "freeze_module_parameters",
        "paper_euler_update",
        "paper_gamma_schedule",
        "predict_clean_latent",
        "reverse_flow_drift",
        "sample_langevin_noise",
    ]
    _import_structure["pairwise_endpoint_semantic"] = [
        "PairwiseEndpointSemanticReward",
        "PairwiseEndpointValidationError",
        "build_pairwise_endpoint_affinity_prompt",
    ]
    _import_structure["ordinal_semantic_progress"] = [
        "ORDINAL_CHOICE_LABELS",
        "ORDINAL_SEMANTIC_PROGRESS_CACHE_SCHEMA_VERSION",
        "ORDINAL_SEMANTIC_PROGRESS_PARSER_VERSION",
        "ORDINAL_STAGE_NODES",
        "EndpointOrdinalValidationError",
        "EndpointRelativeOrdinalSemanticProgressReward",
        "OrdinalQuestionSpec",
        "OrdinalSemanticPrimitiveSpec",
        "OrdinalSemanticProgressSpec",
        "build_ordinal_choice_prompt",
        "build_ordinal_semantic_progress_parser_prompt",
        "load_cached_ordinal_semantic_progress_spec",
        "make_ordinal_semantic_progress_cache_key",
        "ordinal_choice_distribution",
        "parse_ordinal_semantic_progress_json",
        "save_cached_ordinal_semantic_progress_spec",
    ]
    _import_structure["pipeline_rewardflow_flux"] = ["FluxRewardFlowPipeline"]
    _import_structure["pipeline_flux_kontext_strength_trajectory"] = ["FluxKontextStrengthTrajectoryPipeline"]
    _import_structure["pipeline_flux_kontext_terminal_control"] = [
        "FluxKontextTerminalControlPipeline",
        "KontextTerminalControlInputs",
    ]
    _import_structure["pipeline_flux_kontext_coupled_control"] = [
        "CoupledKontextControlInputs",
        "FluxKontextCoupledControlPipeline",
    ]
    _import_structure["coupled_terminal_control"] = [
        "CoupledControlPrior",
        "adjacent_ranking_loss",
        "control_band_loss",
        "control_energy_loss",
        "control_no_jump_loss",
        "initialize_independent_coupled_controls",
        "make_coupled_prior",
        "relative_gap_loss",
        "scalar_direction_residual_diagnostics",
        "soft_relevance_from_velocity_scores",
        "spatial_prior_loss",
        "triangle_deficit_loss",
        "unroll_coupled_velocity_controls",
        "weighted_source_preservation",
    ]
    _import_structure["trajectory_objectives"] = [
        "CoupledTrajectoryObjective",
        "RewardSliderV1LossWeights",
        "RewardSliderV1ObjectiveOutput",
    ]
    _import_structure["strength_trajectory"] = [
        "StrengthBranchLayout",
        "StrengthRewardBatchContext",
        "StrengthRewardContext",
        "StrengthRewardFn",
        "StrengthRewardGuidance",
        "StrengthTrajectoryConfig",
        "align_strength_reward_context",
        "assert_branch_local_reward",
        "expand_for_strengths",
        "expand_shared_initial_latents",
        "flatten_strength_branches",
        "group_trajectory_images",
        "is_strength_reward_step",
        "make_flat_strength_tensor",
        "make_strength_branch_layout",
        "max_shared_noise_difference",
        "sample_base_langevin_noise",
        "sample_shared_langevin_noise",
        "unflatten_strength_branches",
        "validate_strength_reward_context",
        "validate_trajectory_mode",
    ]
    _import_structure["rewards"] = [
        "Qwen25VQAReward",
        "Qwen25VQATeacherForcedScorer",
        "ResearchStaticRewardGuidance",
        "StaticRewardGuidance",
        "qwen_vqa_token_reward",
    ]
    _import_structure["relative_endpoint_semantic"] = [
        "RELATIVE_ENDPOINT_CHOICES",
        "RelativeEndpointSemanticReward",
        "RelativeEndpointValidationError",
        "audit_endpoint_answers",
        "build_relative_endpoint_comparison_prompt",
    ]
    _import_structure["semantic_progress"] = [
        "SEMANTIC_PROGRESS_CACHE_SCHEMA_VERSION",
        "SEMANTIC_PROGRESS_PARSER_VERSION",
        "EndpointRelativeSemanticProgressReward",
        "EndpointSemanticValidationError",
        "SemanticPrimitiveSpec",
        "SemanticProgressSpec",
        "build_semantic_progress_parser_prompt",
        "load_cached_semantic_progress_spec",
        "make_semantic_progress_cache_key",
        "parse_semantic_progress_json",
        "save_cached_semantic_progress_spec",
    ]
    _import_structure["semantic_feature_scorers"] = [
        "CLIPImageFeatureScorer",
        "ImageFeatureScorer",
        "ImageTextFeatureScorer",
        "QwenHiddenFeatureScorer",
        "SigLIPImageFeatureScorer",
    ]
    _import_structure["text_conditioned_semantic"] = [
        "TEXT_CONDITIONED_SEMANTIC_CACHE_SCHEMA_VERSION",
        "TEXT_CONDITIONED_SEMANTIC_PARSER_VERSION",
        "TextConditionedParseRecord",
        "TextConditionedPrimitiveSpec",
        "TextConditionedSemanticGeometry",
        "TextConditionedSemanticSpec",
        "TextSemanticEndpointDirectionError",
        "TextSemanticGeometryOutput",
        "TianyuAITextConditionedSemanticParser",
        "build_text_conditioned_semantic_parser_prompt",
        "load_cached_text_conditioned_parse",
        "make_text_conditioned_semantic_cache_key",
        "parse_text_conditioned_semantic_json",
        "save_cached_text_conditioned_parse",
        "text_conditioned_semantic_json_schema",
    ]
    _import_structure["terminal_control"] = [
        "BestTerminalControlCheckpoint",
        "BlueEndpointTargetLoss",
        "EndpointPixelTargetLoss",
        "MonotonicStrengthCalibration",
        "TerminalControlUnrollOutput",
        "TerminalObjectiveOutput",
        "VelocityEditMasks",
        "amplitude_scaled_effective_controls",
        "blue_direction_score",
        "build_velocity_edit_masks",
        "endpoint_soft_mask",
        "freeze_terminal_control_modules",
        "initialize_velocity_controls",
        "masked_effective_controls",
        "normalized_control_energy",
        "normalized_effective_control_energy",
        "source_restoring_velocity",
        "unroll_terminal_velocity_controls",
        "update_best_control_checkpoint",
    ]
if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        if not (is_transformers_available() and is_torch_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ...utils.dummy_torch_and_transformers_objects import *  # noqa F403
    else:
        from .endpoint_comparator_metrics import (
            SMALL_GRADIENT_STEPS,
            average_ranks,
            gradient_direction_gate,
            ordering_diagnostics,
            spearman_correlation,
        )
        from .endpoint_embedding_geometry import (
            CachedEndpointEmbeddingGeometry,
            EndpointGeometryOutput,
            endpoint_axis_geometry,
        )
        from .endpoint_feature_distance import (
            FeatureEndpointDistanceReward,
            build_focus_conditioned_feature_prompt,
        )
        from .feature_controller_evaluation import (
            BLIND_IDENTITIES,
            BLIND_LABELS,
            DENSE_STRENGTHS,
            EVALUATION_MASK_PROVENANCE,
            FEATURE_CONTROL_PROVENANCE,
            NATIVE_FULL_PROVENANCE,
            PRESERVATION_CATEGORIES,
            SOURCE_INPUT_PROVENANCE,
            TRAINING_STRENGTHS,
            VISUAL_JUDGE_PERMUTATION_SEED,
            VISUAL_JUDGE_PROMPT_VERSION,
            build_blind_permutations,
            dense_feature_curve_diagnostics,
            endpoint_difference_evaluation_mask,
            endpoint_pixel_diagnostics,
            format_strength_tag,
            parse_visual_judge_json,
            remap_visual_judgment,
            summarize_visual_judgments,
            visual_judge_json_schema,
        )
        from .ordinal_semantic_progress import (
            ORDINAL_CHOICE_LABELS,
            ORDINAL_SEMANTIC_PROGRESS_CACHE_SCHEMA_VERSION,
            ORDINAL_SEMANTIC_PROGRESS_PARSER_VERSION,
            ORDINAL_STAGE_NODES,
            EndpointOrdinalValidationError,
            EndpointRelativeOrdinalSemanticProgressReward,
            OrdinalQuestionSpec,
            OrdinalSemanticPrimitiveSpec,
            OrdinalSemanticProgressSpec,
            build_ordinal_choice_prompt,
            build_ordinal_semantic_progress_parser_prompt,
            load_cached_ordinal_semantic_progress_spec,
            make_ordinal_semantic_progress_cache_key,
            ordinal_choice_distribution,
            parse_ordinal_semantic_progress_json,
            save_cached_ordinal_semantic_progress_spec,
        )
        from .pairwise_endpoint_semantic import (
            PairwiseEndpointSemanticReward,
            PairwiseEndpointValidationError,
            build_pairwise_endpoint_affinity_prompt,
        )
        from .paper_components import (
            PaperRewardFlowConfig,
            clean_latent_kl_energy,
            flow_step_size,
            freeze_module_parameters,
            paper_euler_update,
            paper_gamma_schedule,
            predict_clean_latent,
            reverse_flow_drift,
            sample_langevin_noise,
        )
        from .pipeline_flux_kontext_strength_trajectory import FluxKontextStrengthTrajectoryPipeline
        from .pipeline_flux_kontext_terminal_control import (
            FluxKontextTerminalControlPipeline,
            KontextTerminalControlInputs,
        )
        from .pipeline_flux_kontext_coupled_control import (
            CoupledKontextControlInputs,
            FluxKontextCoupledControlPipeline,
        )
        from .coupled_terminal_control import (
            CoupledControlPrior,
            adjacent_ranking_loss,
            control_band_loss,
            control_energy_loss,
            control_no_jump_loss,
            initialize_independent_coupled_controls,
            make_coupled_prior,
            relative_gap_loss,
            scalar_direction_residual_diagnostics,
            soft_relevance_from_velocity_scores,
            spatial_prior_loss,
            triangle_deficit_loss,
            unroll_coupled_velocity_controls,
            weighted_source_preservation,
        )
        from .trajectory_objectives import (
            CoupledTrajectoryObjective,
            RewardSliderV1LossWeights,
            RewardSliderV1ObjectiveOutput,
        )
        from .pipeline_rewardflow_flux import FluxRewardFlowPipeline
        from .relative_endpoint_semantic import (
            RELATIVE_ENDPOINT_CHOICES,
            RelativeEndpointSemanticReward,
            RelativeEndpointValidationError,
            audit_endpoint_answers,
            build_relative_endpoint_comparison_prompt,
        )
        from .rewards import (
            Qwen25VQAReward,
            Qwen25VQATeacherForcedScorer,
            ResearchStaticRewardGuidance,
            StaticRewardGuidance,
            qwen_vqa_token_reward,
        )
        from .semantic_feature_scorers import (
            CLIPImageFeatureScorer,
            ImageFeatureScorer,
            ImageTextFeatureScorer,
            QwenHiddenFeatureScorer,
            SigLIPImageFeatureScorer,
        )
        from .semantic_progress import (
            SEMANTIC_PROGRESS_CACHE_SCHEMA_VERSION,
            SEMANTIC_PROGRESS_PARSER_VERSION,
            EndpointRelativeSemanticProgressReward,
            EndpointSemanticValidationError,
            SemanticPrimitiveSpec,
            SemanticProgressSpec,
            build_semantic_progress_parser_prompt,
            load_cached_semantic_progress_spec,
            make_semantic_progress_cache_key,
            parse_semantic_progress_json,
            save_cached_semantic_progress_spec,
        )
        from .strength_trajectory import (
            StrengthBranchLayout,
            StrengthRewardBatchContext,
            StrengthRewardContext,
            StrengthRewardFn,
            StrengthRewardGuidance,
            StrengthTrajectoryConfig,
            align_strength_reward_context,
            assert_branch_local_reward,
            expand_for_strengths,
            expand_shared_initial_latents,
            flatten_strength_branches,
            group_trajectory_images,
            is_strength_reward_step,
            make_flat_strength_tensor,
            make_strength_branch_layout,
            max_shared_noise_difference,
            sample_base_langevin_noise,
            sample_shared_langevin_noise,
            unflatten_strength_branches,
            validate_strength_reward_context,
            validate_trajectory_mode,
        )
        from .terminal_control import (
            BestTerminalControlCheckpoint,
            BlueEndpointTargetLoss,
            EndpointPixelTargetLoss,
            MonotonicStrengthCalibration,
            TerminalControlUnrollOutput,
            TerminalObjectiveOutput,
            VelocityEditMasks,
            amplitude_scaled_effective_controls,
            blue_direction_score,
            build_velocity_edit_masks,
            endpoint_soft_mask,
            freeze_terminal_control_modules,
            initialize_velocity_controls,
            masked_effective_controls,
            normalized_control_energy,
            normalized_effective_control_energy,
            source_restoring_velocity,
            unroll_terminal_velocity_controls,
            update_best_control_checkpoint,
        )
        from .text_conditioned_semantic import (
            TEXT_CONDITIONED_SEMANTIC_CACHE_SCHEMA_VERSION,
            TEXT_CONDITIONED_SEMANTIC_PARSER_VERSION,
            TextConditionedParseRecord,
            TextConditionedPrimitiveSpec,
            TextConditionedSemanticGeometry,
            TextConditionedSemanticSpec,
            TextSemanticEndpointDirectionError,
            TextSemanticGeometryOutput,
            TianyuAITextConditionedSemanticParser,
            build_text_conditioned_semantic_parser_prompt,
            load_cached_text_conditioned_parse,
            make_text_conditioned_semantic_cache_key,
            parse_text_conditioned_semantic_json,
            save_cached_text_conditioned_parse,
            text_conditioned_semantic_json_schema,
        )

    from .relative_endpoint_parser import (
        DEFAULT_RELATIVE_ENDPOINT_PARSER_BASE_URL,
        DEFAULT_RELATIVE_ENDPOINT_PARSER_MODEL,
        DEFAULT_RELATIVE_ENDPOINT_PARSER_PROVIDER,
        RELATIVE_ENDPOINT_CACHE_SCHEMA_VERSION,
        RELATIVE_ENDPOINT_PARSER_VERSION,
        OpenAIRelativeEndpointParser,
        RelativeEndpointParseRecord,
        RelativeEndpointPrimitiveSpec,
        RelativeEndpointSemanticSpec,
        TianyuAIRelativeEndpointParser,
        build_relative_endpoint_semantic_parser_prompt,
        fingerprint_endpoint_image,
        load_cached_relative_endpoint_parse,
        make_human_relative_endpoint_parse_record,
        make_relative_endpoint_cache_key,
        parse_relative_endpoint_semantic_json,
        relative_endpoint_json_schema,
        save_cached_relative_endpoint_parse,
    )
    from .semantic_parser import (
        SEMANTIC_CACHE_SCHEMA_VERSION,
        SEMANTIC_PARSER_VERSION,
        SemanticParseResult,
        build_semantic_parser_prompt,
        fingerprint_image,
        load_cached_parse,
        make_semantic_cache_key,
        parse_semantic_parser_json,
        save_cached_parse,
    )

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )

    for name, value in _dummy_objects.items():
        setattr(sys.modules[__name__], name, value)
    for name, value in _additional_imports.items():
        setattr(sys.modules[__name__], name, value)
