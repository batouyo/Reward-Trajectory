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
    "pipeline_output": ["FluxRewardFlowPipelineOutput"],
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
}

try:
    if not (is_transformers_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils import dummy_torch_and_transformers_objects  # noqa F403

    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_objects))
else:
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
    _import_structure["pipeline_rewardflow_flux"] = ["FluxRewardFlowPipeline"]
    _import_structure["rewards"] = [
        "Qwen25VQAReward",
        "ResearchStaticRewardGuidance",
        "StaticRewardGuidance",
        "qwen_vqa_token_reward",
    ]
if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        if not (is_transformers_available() and is_torch_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ...utils.dummy_torch_and_transformers_objects import *  # noqa F403
    else:
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
        from .pipeline_rewardflow_flux import FluxRewardFlowPipeline
        from .rewards import (
            Qwen25VQAReward,
            ResearchStaticRewardGuidance,
            StaticRewardGuidance,
            qwen_vqa_token_reward,
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
