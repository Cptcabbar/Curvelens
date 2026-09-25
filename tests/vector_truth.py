"""Test helper: exact curve geometry read from the PDF's vector layer.

The datasheet draws every curve as one stroked polyline in a pure colour.  Those
polylines are an independent ground truth for the raster extraction (they never
enter the application itself).
"""
from __future__ import annotations

import ctypes

import numpy as np
import pypdfium2 as pdfium
import pypdfium2.raw as raw

from core.models import Rect


def curve_polylines(pdf_path, page_index: int, frame_pt: Rect, min_segments: int = 20):
    """Return ``{(r, g, b): (x_pt, y_pt)}`` for stroked polylines inside ``frame_pt``.

    Coordinates are page points with y measured from the page top (y-down).
    """
    pdf = pdfium.PdfDocument(str(pdf_path))
    page = pdf[page_index]
    ph = page.get_size()[1]
    out: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]] = {}
    for obj in page.get_objects():
        if obj.type != raw.FPDF_PAGEOBJ_PATH:
            continue
        l, b, r, t = obj.get_bounds()
        y0, y1 = ph - t, ph - b
        if not (l >= frame_pt.x0 - 2 and r <= frame_pt.x1 + 2 and y0 >= frame_pt.y0 - 2 and y1 <= frame_pt.y1 + 2):
            continue
        n = raw.FPDFPath_CountSegments(obj.raw)
        if n < min_segments:
            continue
        cr, cg, cb, ca = (ctypes.c_uint() for _ in range(4))
        raw.FPDFPageObj_GetStrokeColor(obj.raw, cr, cg, cb, ca)
        color = (cr.value, cg.value, cb.value)
        pts = []
        for i in range(n):
            seg = raw.FPDFPath_GetPathSegment(obj.raw, i)
            x, y = ctypes.c_float(), ctypes.c_float()
            raw.FPDFPathSegment_GetPoint(seg, x, y)
            pts.append((x.value, y.value))
        a, b_, c, d, e, f = obj.get_matrix().get()
        arr = np.asarray(pts)
        xs = a * arr[:, 0] + c * arr[:, 1] + e
        ys_up = b_ * arr[:, 0] + d * arr[:, 1] + f
        if color in out:                       # same colour twice (e.g. legend swatch) -> keep the longest
            if len(xs) <= len(out[color][0]):
                continue
        out[color] = (xs, ph - ys_up)
    return out


def densify(x: np.ndarray, y: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray]:
    """Resample a polyline so consecutive vertices are at most ``step`` apart."""
    xs, ys = [x[0]], [y[0]]
    for i in range(1, len(x)):
        seg = float(np.hypot(x[i] - x[i - 1], y[i] - y[i - 1]))
        k = max(1, int(np.ceil(seg / step)))
        for t in range(1, k + 1):
            xs.append(x[i - 1] + (x[i] - x[i - 1]) * t / k)
            ys.append(y[i - 1] + (y[i] - y[i - 1]) * t / k)
    return np.asarray(xs), np.asarray(ys)


def distance_to_polyline(px: np.ndarray, py: np.ndarray, x: np.ndarray, y: np.ndarray, step: float = 0.25) -> np.ndarray:
    """Distance of each point ``(px, py)`` to the (densified) polyline."""
    from scipy.spatial import cKDTree
    dx, dy = densify(x, y, step)
    d, _ = cKDTree(np.column_stack([dx, dy])).query(np.column_stack([px, py]))
    return d
