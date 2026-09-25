"""Automatic suggestion of the plot area: the rectangle drawn by the axes' frame.

The frame of most charts is made of long, dark, straight lines.  Rows that contain a black/grey
run spanning a large part of the image width (and columns, likewise for the height) are frame
candidates; the outermost horizontal ones are the top and bottom edges.  Their end points give
the horizontal extent, so a frame whose right edge is a coloured secondary axis (as in the
"Charge" chart of the sample datasheet) is still found.  The result is only a *suggestion*; the
user can always redraw the area.
"""
from __future__ import annotations

import cv2
import numpy as np

from .models import Rect


def _longest_runs(dark: np.ndarray, axis: int) -> np.ndarray:
    """Length of the longest True run along ``axis`` for every row (axis=1) or column (axis=0)."""
    m = dark if axis == 1 else dark.T
    h, w = m.shape
    padded = np.zeros((h, w + 2), np.int8)
    padded[:, 1:-1] = m
    d = np.diff(padded, axis=1)
    best = np.zeros(h, np.int32)
    for i in range(h):
        starts = np.flatnonzero(d[i] == 1)
        if starts.size:
            ends = np.flatnonzero(d[i] == -1)
            best[i] = int((ends - starts).max())
    return best


def _longest_run_extent(line: np.ndarray) -> tuple[int, int]:
    """``(start, end_exclusive)`` of the longest True run of a 1-D boolean array."""
    padded = np.concatenate([[0], line.astype(np.int8), [0]])
    d = np.diff(padded)
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    k = int(np.argmax(ends - starts))
    return int(starts[k]), int(ends[k])


def _bands(flags: np.ndarray, max_gap: int = 1) -> list[tuple[int, int]]:
    """Runs of True (allowing tiny holes) as ``(first, last)`` index pairs."""
    idx = np.flatnonzero(flags)
    if idx.size == 0:
        return []
    out, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > max_gap + 1:
            out.append((int(start), int(prev)))
            start = i
        prev = i
    out.append((int(start), int(prev)))
    return out


def detect_plot_area(image_bgr: np.ndarray, min_frac: float = 0.40, dark_level: int = 110) -> Rect | None:
    """Rectangle spanned by the frame of the chart, or ``None`` when no frame is recognised.

    Returned in continuous image coordinates along the *centre lines* of the frame strokes.
    """
    h, w = image_bgr.shape[:2]
    if h < 40 or w < 40:
        return None
    img = np.ascontiguousarray(image_bgr)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    chroma = img.max(axis=2).astype(np.int16) - img.min(axis=2)
    dark_any = gray < dark_level
    dark_neutral = dark_any & (chroma < 60)         # black/grey only: a blue or red axis is no frame edge

    hbands = _bands(_longest_runs(dark_neutral, 1) >= min_frac * w)
    if len(hbands) < 2:
        return None
    extents = []
    for band in hbands:
        extents.append(_longest_run_extent(dark_any[(band[0] + band[1]) // 2]))

    # Outermost top/bottom pair whose lines match in extent and thickness (a stray line at the very
    # edge of a crop - the underline of a neighbouring heading - does not pair with the real frame).
    pair = None
    for i in range(len(hbands)):
        for j in range(len(hbands) - 1, i, -1):
            (a0, a1), (b0, b1) = extents[i], extents[j]
            ti, tj = hbands[i][1] - hbands[i][0] + 1, hbands[j][1] - hbands[j][0] + 1
            y_span = 0.5 * (hbands[j][0] + hbands[j][1]) - 0.5 * (hbands[i][0] + hbands[i][1])
            if (y_span >= 0.3 * h and abs(a0 - b0) <= 0.03 * w and abs(a1 - b1) <= 0.03 * w
                    and (b1 - a0) >= 0.3 * w and max(ti, tj) <= 2 * min(ti, tj) + 1):
                pair = (i, j)
                break
        if pair:
            break
    if pair is None:
        return None
    top, bottom = hbands[pair[0]], hbands[pair[1]]
    y0, y1 = 0.5 * (top[0] + top[1] + 1), 0.5 * (bottom[0] + bottom[1] + 1)
    thick = float(top[1] - top[0] + 1)
    xa = min(extents[pair[0]][0], extents[pair[1]][0])       # coloured pixels may end the lines: use the outer ends
    xb = max(extents[pair[0]][1], extents[pair[1]][1])

    # Prefer the centres of real vertical frame lines when they sit at the ends of the horizontal ones.
    vbands = _bands(_longest_runs(dark_neutral, 0) >= min_frac * h)
    x0, x1 = xa + 0.5 * thick, xb - 0.5 * thick
    tol = 0.02 * w + thick
    for band in vbands:
        c = 0.5 * (band[0] + band[1] + 1)
        if abs(band[0] - xa) <= tol:
            x0 = c
        if abs(band[1] + 1 - xb) <= tol:
            x1 = c
    if x1 - x0 < 0.3 * w:
        return None
    return Rect(float(x0), float(y0), float(x1), float(y1))
