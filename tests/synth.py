"""Synthetic plot images with an analytically known ground truth (test helper)."""
from __future__ import annotations

import cv2
import numpy as np

_SS = 8   # super-sampling factor of the reference renderer


def draw_curve(img: np.ndarray, xs, ys, bgr, thickness: float = 3.0) -> None:
    """Draw a polyline with unbiased anti-aliasing (continuous pixel coordinates).

    OpenCV's own anti-aliased thick lines are slightly asymmetric (~0.2 px), which
    would blur the accuracy claims of the tests.  Here the stroke is rasterised at
    ``_SS`` times the resolution, box-filtered down and alpha-composited.
    """
    h, w = img.shape[:2]
    big = np.zeros((h * _SS, w * _SS), np.uint8)
    pts = np.round(np.column_stack([np.asarray(xs) * _SS - 0.5, np.asarray(ys) * _SS - 0.5])).astype(np.int32)
    cv2.polylines(big, [pts.reshape(-1, 1, 2)], False, 255, max(1, int(round(thickness * _SS))), cv2.LINE_8)
    alpha = cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)[..., None] / 255.0
    color = np.asarray(bgr, np.float32)[None, None, :]
    img[:] = np.round(img.astype(np.float32) * (1.0 - alpha) + color * alpha).astype(np.uint8)


def dashed_line(img: np.ndarray, p0, p1, bgr, dash: int = 6, gap: int = 4, thickness: int = 1) -> None:
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    length = float(np.hypot(*(p1 - p0)))
    direction = (p1 - p0) / max(length, 1e-9)
    pos = 0.0
    while pos < length:
        a = p0 + direction * pos
        b = p0 + direction * min(pos + dash, length)
        cv2.line(img, tuple(int(round(v)) for v in a), tuple(int(round(v)) for v in b), bgr, thickness)
        pos += dash + gap


def make_plot(width: int = 640, height: int = 420, margin: int = 50, grid: bool = True, seed: int = 0):
    """Blank plot: white page, black frame, dashed grey grid.  Returns ``(img, plot_rect_xyxy)``."""
    img = np.full((height, width, 3), 255, np.uint8)
    x0, y0, x1, y1 = margin, margin, width - margin, height - margin
    if grid:
        for i in range(1, 5):
            gx = x0 + (x1 - x0) * i / 5
            gy = y0 + (y1 - y0) * i / 5
            dashed_line(img, (gx, y0), (gx, y1), (150, 150, 150))
            dashed_line(img, (x0, gy), (x1, gy), (150, 150, 150))
    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 0), 2)
    return img, (x0, y0, x1, y1)
