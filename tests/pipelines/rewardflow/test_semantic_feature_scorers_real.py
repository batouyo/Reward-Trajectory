import os

import pytest
import torch

from diffusers.pipelines.rewardflow.semantic_feature_scorers import (
    CLIPImageFeatureScorer,
    SigLIPImageFeatureScorer,
)


pytestmark = pytest.mark.skipif(
    not os.getenv("RUN_REWARDFLOW_REAL_SEMANTIC_ENCODERS"),
    reason="Set RUN_REWARDFLOW_REAL_SEMANTIC_ENCODERS=1 for local CLIP/SigLIP integration.",
)


@pytest.mark.parametrize(
    ("scorer_class", "environment_name"),
    [
        (CLIPImageFeatureScorer, "REWARDFLOW_CLIP_MODEL_PATH"),
        (SigLIPImageFeatureScorer, "REWARDFLOW_SIGLIP_MODEL_PATH"),
    ],
)
def test_real_projected_embedding_processor_fidelity_and_image_gradient(scorer_class, environment_name):
    model_path = os.getenv(environment_name)
    if not model_path:
        pytest.skip(f"Set {environment_name} to a local checkpoint.")
    scorer = scorer_class(model_path, device="cuda:0", dtype=torch.float32, local_files_only=True)
    image = torch.linspace(0, 1, 3 * 256 * 320, device="cuda:0").reshape(1, 3, 256, 320)
    image = image.requires_grad_(True)
    fidelity = scorer.processor_fidelity(image)
    print(environment_name, fidelity)
    assert fidelity["official_pixel_shape"] == fidelity["differentiable_pixel_shape"]
    assert fidelity["embedding_cosine"] >= 0.999
    gradient = torch.autograd.grad(scorer.encode_image(image)[0], image)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    text_feature = scorer.encode_text("a vivid blue weighted training ball")
    assert text_feature.ndim == 1 and torch.isfinite(text_feature).all()
    torch.testing.assert_close(torch.linalg.vector_norm(text_feature), torch.tensor(1.0, device="cuda:0"))
    assert text_feature.grad_fn is None and not text_feature.requires_grad
    assert scorer.last_text_metadata["truncated"] is False
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in scorer.model.parameters())
