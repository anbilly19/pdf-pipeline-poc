"""Tests for the bbox overlay renderer and IoU utility."""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from src.ui.overlay import bbox_iou, render_page_with_bboxes


# ---------------------------------------------------------------------------
# bbox_iou
# ---------------------------------------------------------------------------

def test_iou_identical_boxes() -> None:
    assert bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)


def test_iou_no_overlap() -> None:
    assert bbox_iou([0, 0, 5, 5], [10, 10, 20, 20]) == pytest.approx(0.0)


def test_iou_partial_overlap() -> None:
    score = bbox_iou([0, 0, 10, 10], [5, 5, 15, 15])
    assert 0.0 < score < 1.0


def test_iou_contained() -> None:
    # inner box fully inside outer box
    score = bbox_iou([0, 0, 20, 20], [5, 5, 10, 10])
    assert score > 0.0


def test_iou_meets_threshold() -> None:
    # CLAUDE.md requires IoU >= 0.7 for acceptable bbox accuracy
    score = bbox_iou([0, 0, 100, 20], [2, 1, 102, 21])
    assert score >= 0.7


# ---------------------------------------------------------------------------
# render_page_with_bboxes
# ---------------------------------------------------------------------------

def test_render_missing_file_returns_none() -> None:
    result = render_page_with_bboxes("/nonexistent/page.png", [[0, 0, 10, 10]])
    assert result is None


def test_render_draws_highlights(tmp_path: Path) -> None:
    # create a blank white page PNG
    img = Image.new("RGB", (595, 842), color=(255, 255, 255))
    img_path = tmp_path / "page_0001.png"
    img.save(str(img_path))

    result = render_page_with_bboxes(
        str(img_path),
        bboxes=[[72.0, 100.0, 400.0, 130.0]],
        dpi=72,  # 1:1 scale for test simplicity
    )
    assert result is not None
    assert isinstance(result, Image.Image)


def test_render_invalid_bbox_skipped(tmp_path: Path) -> None:
    img = Image.new("RGB", (595, 842), color=(255, 255, 255))
    img_path = tmp_path / "page_0001.png"
    img.save(str(img_path))

    # invalid bbox (only 2 values) should not crash
    result = render_page_with_bboxes(str(img_path), bboxes=[[0, 0]])
    assert result is not None  # renders but skips bad bbox


def test_render_output_size_matches_input(tmp_path: Path) -> None:
    img = Image.new("RGB", (800, 1000), color=(200, 200, 200))
    img_path = tmp_path / "page.png"
    img.save(str(img_path))

    result = render_page_with_bboxes(str(img_path), bboxes=[[10, 10, 100, 50]], dpi=72)
    assert result is not None
    assert result.size == (800, 1000)
