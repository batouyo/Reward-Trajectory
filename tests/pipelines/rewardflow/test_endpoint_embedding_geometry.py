import pytest
import torch

from diffusers.pipelines.rewardflow.endpoint_embedding_geometry import (
    CachedEndpointEmbeddingGeometry,
    endpoint_axis_geometry,
)


def test_source_progress_is_zero_and_full_progress_is_one():
    source = torch.tensor([1.0, 0.0])
    full = torch.tensor([0.0, 1.0])
    assert endpoint_axis_geometry(source, source, full).progress.abs() < 1e-7
    torch.testing.assert_close(endpoint_axis_geometry(source, full, full).progress, torch.tensor(1.0))


def test_linear_feature_midpoint_has_half_progress():
    source = torch.tensor([1.0, 0.0])
    full = torch.tensor([0.0, 1.0])
    midpoint = (source + full) / 2
    torch.testing.assert_close(endpoint_axis_geometry(source, midpoint, full).progress, torch.tensor(0.5))


def test_off_axis_point_preserves_projection_but_has_positive_deviation():
    source = torch.tensor([1.0, 0.0, 0.0])
    full = torch.tensor([0.0, 1.0, 0.0])
    on_axis = (source + full) / 2
    off_axis = on_axis + torch.tensor([0.0, 0.0, 1.0])
    baseline = endpoint_axis_geometry(source, on_axis, full)
    shifted = endpoint_axis_geometry(source, off_axis, full)
    torch.testing.assert_close(shifted.progress, baseline.progress)
    assert baseline.off_axis < 1e-6 and shifted.off_axis > 0


def test_progress_is_not_clamped():
    source = torch.tensor([1.0, 0.0])
    full = torch.tensor([0.0, 1.0])
    beyond = torch.nn.functional.normalize(torch.tensor([-1.0, 2.0]), dim=0)
    assert endpoint_axis_geometry(source, beyond, full).progress > 1


def test_degenerate_endpoint_axis_fails_fast():
    source = torch.tensor([1.0, 0.0])
    with pytest.raises(ValueError, match="degenerate"):
        endpoint_axis_geometry(source, source, source)


class _MeanEncoder:
    def encode_image(self, image):
        value = image.mean()
        return torch.stack((1 - value, value, value.square()))


def test_cached_endpoints_are_detached_and_candidate_gradient_flows():
    source = torch.zeros(1, 3, 4, 4, requires_grad=True)
    full = torch.ones_like(source, requires_grad=True)
    geometry = CachedEndpointEmbeddingGeometry(_MeanEncoder(), source, full)
    candidate = torch.full_like(source, 0.4, requires_grad=True)
    gradient = torch.autograd.grad(geometry(candidate).progress, candidate)[0]
    assert not geometry.source_feature.requires_grad and not geometry.full_feature.requires_grad
    assert source.grad is None and full.grad is None
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
