"""Page image rendering with bbox highlight overlays.

Draws semi-transparent yellow highlight rectangles over the rendered
page PNG at the coordinates returned by the retrieval pipeline.
"""
from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

_HIGHLIGHT_FILL = (255, 255, 0, 80)   # semi-transparent yellow
_HIGHLIGHT_OUTLINE = (255, 140, 0, 200)  # orange border
_OUTLINE_WIDTH = 2


def render_page_with_bboxes(
    image_path: str,
    bboxes: list[list[float]],
    dpi: int = 150,
    pdf_dpi: int = 72,
) -> Image.Image | None:
    """Load a rendered page PNG and draw bbox highlights on it.

    Coordinates are in PDF points (72 dpi). The image was rendered at
    `dpi` resolution, so bboxes are scaled accordingly.

    Args:
        image_path: Path to the rendered page PNG.
        bboxes: List of [x0, y0, x1, y1] in PDF points.
        dpi: Render DPI used when the PNG was created.
        pdf_dpi: Native PDF coordinate DPI (always 72).

    Returns:
        PIL Image with highlights drawn, or None if the file is missing.
    """
    path = Path(image_path)
    if not path.exists():
        logger.warning("Page image not found: %s", image_path)
        return None

    img = Image.open(path).convert("RGBA")
    scale = dpi / pdf_dpi  # e.g. 150/72 ≈ 2.08

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for bbox in bboxes:
        if len(bbox) != 4:
            logger.warning("Skipping invalid bbox: %s", bbox)
            continue
        x0, y0, x1, y1 = [coord * scale for coord in bbox]
        draw.rectangle(
            [x0, y0, x1, y1],
            fill=_HIGHLIGHT_FILL,
            outline=_HIGHLIGHT_OUTLINE,
            width=_OUTLINE_WIDTH,
        )

    composite = Image.alpha_composite(img, overlay)
    return composite.convert("RGB")


def bbox_iou(bbox_a: list[float], bbox_b: list[float]) -> float:
    """Compute Intersection over Union between two bboxes.

    Used for evaluation (CLAUDE.md section 8 requires IoU >= 0.7).

    Args:
        bbox_a: [x0, y0, x1, y1]
        bbox_b: [x0, y0, x1, y1]

    Returns:
        IoU score in [0.0, 1.0].
    """
    ax0, ay0, ax1, ay1 = bbox_a
    bx0, by0, bx1, by1 = bbox_b

    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)

    intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if intersection == 0.0:
        return 0.0

    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0
