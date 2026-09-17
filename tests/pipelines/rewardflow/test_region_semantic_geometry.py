import pytest
import torch

from diffusers.pipelines.rewardflow.region_semantic_geometry import (
    aggregate_velocity_region,
    summarize_probe_geometry,
)


def test_velocity_aggregation_normalizes_each_step_and_reuses_stable_topk():
    base = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    scores = (base, base * 3 + 4, base * 0.5 - 2, base * 10)
    region = aggregate_velocity_region(
        scores,
        token_height=2,
        token_width=2,
        image_height=8,
        image_width=8,
        topk_fraction=0.25,
        padding_fraction=0.1,
    )
    torch.testing.assert_close(region.score_map.flatten(), torch.tensor([0, 1 / 3, 2 / 3, 1]))
    assert region.token_mask.flatten().tolist() == [0, 0, 0, 1]
    assert region.active_token_fraction == 0.25
    assert region.raw_bbox_xyxy == (4, 4, 8, 8)
    assert region.bbox_xyxy == (3, 3, 8, 8)
    assert region.pixel_mask.shape == (8, 8)
    assert int(region.pixel_mask.sum()) == 16


def test_velocity_aggregation_rejects_degenerate_or_mismatched_scores():
    kwargs = {"token_height": 2, "token_width": 2, "image_height": 8, "image_width": 8}
    with pytest.raises(ValueError, match="constant"):
        aggregate_velocity_region([torch.ones(1, 4)], **kwargs)
    with pytest.raises(ValueError, match="shape"):
        aggregate_velocity_region([torch.arange(3.0)[None]], **kwargs)
    with pytest.raises(ValueError, match="non-finite"):
        aggregate_velocity_region([torch.tensor([[0.0, 1.0, float("nan"), 3.0]])], **kwargs)


def test_probe_summary_uses_unclamped_v6_progress_and_exact_gaps():
    rows = [
        {"image_id": "oracle_0.2", "progress": -0.2, "off_axis": 0.3},
        {"image_id": "oracle_0.5", "progress": 0.1, "off_axis": 0.6},
        {"image_id": "oracle_0.8", "progress": 1.2, "off_axis": 0.9},
    ]
    result = summarize_probe_geometry(rows)
    assert result["strict_order"] is True
    assert result["p_0.2"] == -0.2
    assert result["p_0.8"] == 1.2
    assert result["gap_0.2_to_0.5"] == pytest.approx(0.3)
    assert result["gap_0.5_to_0.8"] == pytest.approx(1.1)
    assert result["probe_span_0.2_to_0.8"] == pytest.approx(1.4)
    assert result["mean_probe_off_axis"] == pytest.approx(0.6)


def test_probe_summary_exposes_inversion():
    rows = [
        {"image_id": "oracle_0.2", "progress": 0.4, "off_axis": 0.1},
        {"image_id": "oracle_0.5", "progress": 0.3, "off_axis": 0.1},
        {"image_id": "oracle_0.8", "progress": 0.6, "off_axis": 0.1},
    ]
    result = summarize_probe_geometry(rows)
    assert result["strict_order"] is False
    assert result["minimum_adjacent_gap"] == pytest.approx(-0.1)


def test_tensor_crop_gradient_is_zero_outside_bbox():
    image = torch.rand(1, 3, 8, 8, requires_grad=True)
    cropped = image[:, :, 3:8, 3:8]
    gradient = torch.autograd.grad(cropped.square().sum(), image)[0]
    assert gradient.abs().sum() > 0
    assert torch.count_nonzero(gradient[:, :, :3]) == 0
    assert torch.count_nonzero(gradient[:, :, :, :3]) == 0
