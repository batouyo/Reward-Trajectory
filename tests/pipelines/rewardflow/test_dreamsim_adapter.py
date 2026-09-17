from __future__ import annotations

import os

import pytest
import torch

from diffusers.pipelines.rewardflow.dreamsim_adapter import DreamSimAdapter


_CACHE = os.getenv("DREAMSIM_CACHE_DIR")
_DEVICE = os.getenv("DREAMSIM_TEST_DEVICE", "cpu")


@pytest.mark.skipif(not _CACHE, reason="Set DREAMSIM_CACHE_DIR to run the official DreamSim integration test.")
def test_official_dreamsim_tensor_path_preserves_candidate_gradient_and_freezes_model():
    adapter = DreamSimAdapter(model_path=_CACHE, device=_DEVICE)
    candidate = torch.rand(1, 3, 48, 64, device=_DEVICE, requires_grad=True)
    reference = torch.rand(1, 3, 48, 64, device=_DEVICE)
    distance = adapter.distance(candidate, reference).mean()
    distance.backward()
    assert candidate.grad is not None
    assert torch.isfinite(candidate.grad).all()
    assert candidate.grad.abs().sum() > 0
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in adapter.model.parameters())


@pytest.mark.skipif(not _CACHE, reason="Set DREAMSIM_CACHE_DIR to run the official DreamSim integration test.")
def test_dreamsim_fidelity_audit_compares_official_fixed_image_path():
    adapter = DreamSimAdapter(model_path=_CACHE, device=_DEVICE)
    first = torch.rand(1, 3, 48, 64, device=_DEVICE)
    second = torch.rand(1, 3, 48, 64, device=_DEVICE)
    audit = adapter.fidelity_audit(first, second)
    assert set(audit) == {"official_distance", "differentiable_distance", "absolute_difference"}
    assert all(torch.isfinite(torch.tensor(value)) for value in audit.values())
