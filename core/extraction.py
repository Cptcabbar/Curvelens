"""Colour-based curve extraction with sub-pixel column centroids.

Pipeline (``extract_curve``):

1. Convert the plot area to CIE Lab and compute the distance (Delta E, CIE76)
   to the reference colour the user picked -> colour mask.
2. Clean the mask: strip dashed/solid grid lines (needed for black curves whose
   colour equals the grid colour), drop specks, optionally open the mask, and
   keep only components that are long enough (text, legend swatches and grid
   fragments are short).
3. For every pixel column find the vertical runs of mask pixels and their
   weighted centroid (sub-pixel y).
4. Follow the curve from its most reliable column in both directions; when a
   column offers several runs, pick the one closest to the y extrapolated from
   the previous points (continuity tracking).

Everything works in continuous pixel coordinates (see ``core.models``) and is
independent of any GUI code.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Sequence

import cv2
import numpy as np

from .models import ExtractionParams, Rect


# --------------------------------------------------------------------------- #
# Colour handling
# --------------------------------------------------------------------------- #
def bgr_to_lab(img_bgr: np.ndarray) -> np.ndarray:
    """uint8 BGR image -> float32 Lab (L in 0..100, a/b roughly -128..127)."""
    return cv2.cvtColor(np.ascontiguousarray(img_bgr).astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)


def rgb_to_lab(rgb: Sequence[int]) -> np.ndarray:
    r, g, b = (int(v) for v in rgb)
    return bgr_to_lab(np.array([[[b, g, r]]], dtype=np.uint8))[0, 0]


def delta_e_map(img_bgr: np.ndarray, ref_rgb: Sequence[int]) -> np.ndarray:
    """Per-pixel CIE76 colour distance to ``ref_rgb`` (float32, same H x W)."""
    diff = bgr_to_lab(img_bgr) - rgb_to_lab(ref_rgb)
    return np.sqrt(np.einsum("...c,...c->...", diff, diff)).astype(np.float32)


def pick_reference_color(img_bgr: np.ndarray, x: float, y: float) -> tuple[int, int, int]:
    """RGB colour of the pixel under the continuous image position ``(x, y)``."""
    h, w = img_bgr.shape[:2]
    j = int(np.clip(np.floor(x), 0, w - 1))
    i = int(np.clip(np.floor(y), 0, h - 1))
    b, g, r = (int(v) for v in img_bgr[i, j, :3])
    return r, g, b


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class ExtractionResult:
    x_px: np.ndarray                     # continuous image x of each point
    y_px: np.ndarray                     # continuous image y (sub-pixel)
    thickness_px: np.ndarray             # height of the mask run behind each point
    mask: np.ndarray | None = None       # final mask in *full image* size (debug/overlay)
    stats: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.x_px)


# --------------------------------------------------------------------------- #
# Mask clean-up
# --------------------------------------------------------------------------- #
_MIN_DASHES = 6   # a line made of at least this many separate runs counts as dashed


def _flag_bands(flags: np.ndarray, max_width: int) -> np.ndarray:
    """Keep only runs of True no longer than ``max_width`` (grid lines are thin)."""
    out = np.zeros_like(flags)
    n = len(flags)
    i = 0
    while i < n:
        if flags[i]:
            j = i
            while j < n and flags[j]:
                j += 1
            if j - i <= max_width:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def _grid_reach(shape: tuple[int, int], params: ExtractionParams) -> int:
    """Widest band (px) still treated as a grid line."""
    return params.grid_max_width or max(3, int(round(0.012 * min(shape))))


def _count_runs(m: np.ndarray, axis: int) -> np.ndarray:
    """Number of separate True runs along ``axis`` of a 0/1 array."""
    if axis == 1:
        return (m[:, 1:] > m[:, :-1]).sum(axis=1) + m[:, 0]
    return (m[1:, :] > m[:-1, :]).sum(axis=0) + m[0, :]


def _grid_lines(mask: np.ndarray, params: ExtractionParams) -> tuple[np.ndarray, np.ndarray]:
    """Boolean vectors flagging the rows / columns that hold a grid line.

    A grid line is a thin band covered along (almost) its whole length, in every
    quarter of it.  Only two kinds are removed:

    * *dashed* lines (many separate runs) anywhere, and
    * solid lines at the very border of the area (the plot frame).

    A solid line in the interior may be a genuine flat curve (constant current,
    a plateau) and is left for the tracker, which ignores it unless it is the
    curve itself.
    """
    h, w = mask.shape
    if h < 8 or w < 8:
        return np.zeros(h, bool), np.zeros(w, bool)
    reach = _grid_reach(mask.shape, params)
    f = params.grid_fill
    m = mask.astype(np.uint8)

    row_q = np.stack([q.mean(axis=1) for q in np.array_split(m, 4, axis=1)])
    row_flag = (m.mean(axis=1) > f) & (row_q.min(axis=0) > 0.5 * f)
    dashed_r = _count_runs(m, 1) >= _MIN_DASHES
    border_r = np.zeros(h, bool)
    border_r[:reach] = border_r[h - reach:] = True
    rows = _flag_bands(row_flag & (dashed_r | border_r), reach)

    col_q = np.stack([q.mean(axis=0) for q in np.array_split(m, 4, axis=0)])
    col_flag = (m.mean(axis=0) > f) & (col_q.min(axis=0) > 0.5 * f)
    dashed_c = _count_runs(m, 0) >= _MIN_DASHES
    border_c = np.zeros(w, bool)
    border_c[:reach] = border_c[w - reach:] = True
    cols = _flag_bands(col_flag & (dashed_c | border_c), reach)
    return rows, cols


def _remove_grid(mask: np.ndarray, params: ExtractionParams) -> np.ndarray:
    """Delete grid lines, then bridge the small holes they cut into real curves."""
    rows, cols = _grid_lines(mask, params)
    if not rows.any() and not cols.any():
        return mask
    reach = _grid_reach(mask.shape, params)
    cleaned = mask.copy()
    cleaned[rows, :] = False
    cleaned[:, cols] = False
    c8 = cleaned.astype(np.uint8)
    # A curve crossing a horizontal grid line is cut vertically -> close vertically
    # inside the removed rows; likewise horizontally for removed columns.
    closed_v = cv2.morphologyEx(c8, cv2.MORPH_CLOSE, np.ones((2 * reach + 1, 1), np.uint8)).astype(bool)
    closed_h = cv2.morphologyEx(c8, cv2.MORPH_CLOSE, np.ones((1, 2 * reach + 1), np.uint8)).astype(bool)
    repaired = cleaned
    repaired[rows, :] |= closed_v[rows, :]
    repaired[:, cols] |= closed_h[:, cols]
    return repaired


def _stroke_thickness(mask: np.ndarray) -> float:
    """Typical thickness of the thickest strokes (ridge of the distance transform)."""
    if not mask.any():
        return 0.0
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    ridge = (dist >= cv2.dilate(dist, np.ones((3, 3), np.uint8))) & (dist > 0)
    vals = dist[ridge]
    return float(2.0 * np.percentile(vals, 90)) if vals.size else 0.0


def clean_mask(mask: np.ndarray, params: ExtractionParams) -> np.ndarray:
    """Grid removal, morphological opening, speck removal and component filtering."""
    if params.remove_grid:
        mask = _remove_grid(mask, params)

    open_size = params.open_size
    if params.dashed:
        open_size = 1                       # dashes are short: opening / length filtering would erase them
    elif open_size == 0:
        # Opening removes hair-thin residue but would erase 1-2 px lines, so it
        # only runs when the strokes are clearly thicker than that.
        open_size = 3 if _stroke_thickness(mask) >= 5.0 else 1
    if open_size > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, k).astype(bool)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros_like(mask)
    area = stats[1:, cv2.CC_STAT_AREA]
    width = stats[1:, cv2.CC_STAT_WIDTH]
    keep = area >= params.min_area
    if keep.any() and not params.dashed:
        longest = width[keep].max()
        keep &= width >= params.min_component_frac * longest
    lut = np.zeros(n, dtype=bool)
    lut[1:] = keep
    return lut[labels]


# --------------------------------------------------------------------------- #
# Column analysis and tracking
# --------------------------------------------------------------------------- #
def column_candidates(mask: np.ndarray, weights: np.ndarray, merge_gap: int) -> list[np.ndarray]:
    """Per column: array of ``[y_centroid, run_height, weight_sum]`` rows.

    Runs separated by at most ``merge_gap`` background pixels count as one run
    (a grid line or antialiasing hole must not split a curve in two).  Centroids
    are weighted by ``weights`` and returned in continuous coordinates (+0.5).
    """
    h, w = mask.shape
    out: list[np.ndarray] = []
    for j in range(w):
        rows = np.flatnonzero(mask[:, j])
        if rows.size == 0:
            out.append(np.empty((0, 3)))
            continue
        breaks = np.flatnonzero(np.diff(rows) > merge_gap + 1) + 1
        cands = []
        for seg in np.split(rows, breaks):
            wt = weights[seg, j]
            tot = float(wt.sum())
            yc = float((wt * (seg + 0.5)).sum() / tot) if tot > 0 else float(seg.mean() + 0.5)
            cands.append((yc, float(seg[-1] - seg[0] + 1), tot))
        out.append(np.asarray(cands))
    return out


def _predict(recent: deque, col: int) -> tuple[float, float]:
    """Extrapolated y at ``col`` and the current slope (px/column) from recent points."""
    if len(recent) < 2:
        return recent[-1][1], 0.0
    cols = np.array([c for c, _ in recent], dtype=float)
    ys = np.array([y for _, y in recent], dtype=float)
    slope, icpt = np.polyfit(cols, ys, 1)
    # Anchor the line at the newest point so a noisy fit cannot offset the prediction.
    last_c, last_y = recent[-1]
    return float(last_y + slope * (col - last_c)), float(slope)


def _walk(cands: list[np.ndarray], start: int, y0: float, step: int, params: ExtractionParams,
          base_tol: float, max_gap: int) -> list[tuple[int, float, float]]:
    """Follow the curve from ``start`` (exclusive) in direction ``step``."""
    n = len(cands)
    recent: deque = deque([(start, y0)], maxlen=max(2, params.slope_points))
    found: list[tuple[int, float, float]] = []
    gap = 0
    col = start + step
    while 0 <= col < n:
        c = cands[col]
        chosen = None
        if len(c):
            pred, slope = _predict(recent, col)
            k = int(np.argmin(np.abs(c[:, 0] - pred)))
            tol = base_tol + 0.5 * c[k, 1] + abs(slope) * (gap + 1)
            if abs(c[k, 0] - pred) <= tol:
                chosen = c[k]
        if chosen is None:
            gap += 1
            if gap > max_gap:
                break
        else:
            gap = 0
            recent.append((col, float(chosen[0])))
            found.append((col, float(chosen[0]), float(chosen[1])))
        col += step
    return found


def _pick_seed(cands: list[np.ndarray]) -> int | None:
    """Middle column of the longest stretch of columns with exactly one candidate."""
    single = np.array([len(c) == 1 for c in cands])
    best_len, best_mid, i, n = 0, None, 0, len(single)
    while i < n:
        if single[i]:
            j = i
            while j < n and single[j]:
                j += 1
            if j - i > best_len:
                best_len, best_mid = j - i, (i + j - 1) // 2
            i = j
        else:
            i += 1
    if best_mid is not None:
        return best_mid
    weights = [c[:, 2].max() if len(c) else -1.0 for c in cands]
    return int(np.argmax(weights)) if max(weights) > 0 else None


def _seed_from_click(cands: list[np.ndarray], col: int, y: float, roi_h: int) -> tuple[int, float, float] | None:
    """Candidate nearest to the clicked position ``(col, y)`` (searching outwards over a few columns)."""
    n = len(cands)
    tol = max(8.0, 0.04 * roi_h)
    for d in range(0, max(6, int(0.02 * n)) + 1):
        best = None
        for c in {col - d, col + d}:
            if 0 <= c < n and len(cands[c]):
                k = int(np.argmin(np.abs(cands[c][:, 0] - y)))
                dist = abs(cands[c][k, 0] - y)
                if dist <= tol + 0.5 * cands[c][k, 1] and (best is None or dist < best[0]):
                    best = (dist, c, float(cands[c][k, 0]), float(cands[c][k, 1]))
        if best is not None:
            return best[1], best[2], best[3]
    return None


def track_curve(cands: list[np.ndarray], params: ExtractionParams, roi_h: int,
                click: tuple[int, float] | None = None) -> list[tuple[int, float, float]]:
    """Continuity tracking over the per-column candidates -> ``[(col, y, run_height), ...]``.

    ``click = (column, y)`` starts the tracking on the curve the user pointed at (needed when
    several curves share one colour); without it, or when nothing is found there, the most
    reliable column is used.
    """
    n = len(cands)
    start = _seed_from_click(cands, click[0], click[1], roi_h) if click is not None else None
    if start is not None:
        seed, y0, h0 = start
    else:
        seed = _pick_seed(cands)
        if seed is None:
            return []
        c = cands[seed]
        k = int(np.argmax(c[:, 2]))
        y0, h0 = float(c[k, 0]), float(c[k, 1])
    base_tol = params.jump_tol or max(4.0, 0.02 * roi_h)
    max_gap = params.max_gap or max(8, int(0.06 * n))
    right = _walk(cands, seed, y0, +1, params, base_tol, max_gap)
    left = _walk(cands, seed, y0, -1, params, base_tol, max_gap)
    return list(reversed(left)) + [(seed, y0, h0)] + right


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def extract_curve(image_bgr: np.ndarray, ref_rgb: Sequence[int], plot_area: Rect | None = None,
                  exclusions: Sequence[Rect] = (), params: ExtractionParams | None = None,
                  keep_mask: bool = False, seed: tuple[float, float] | None = None) -> ExtractionResult:
    """Extract one curve of colour ``ref_rgb`` from ``image_bgr``.

    ``plot_area`` limits the search (everything outside is ignored);
    ``exclusions`` are rectangles inside it that are ignored too (legend boxes,
    text blocks).  ``seed`` is an image position on the wanted curve (where the user clicked); it
    selects which of several same-coloured curves is followed.  All coordinates are continuous
    image coordinates.
    """
    params = params or ExtractionParams()
    H, W = image_bgr.shape[:2]
    area = plot_area or Rect(0, 0, W, H)
    rs, cs = area.pixel_slices((H, W))
    crop = image_bgr[rs, cs]
    empty = ExtractionResult(np.empty(0), np.empty(0), np.empty(0), None, {"reason": "empty"})
    if crop.size == 0:
        return empty

    de = delta_e_map(crop, ref_rgb)
    mask = de <= params.delta_e
    for ex in exclusions:
        er, ec = ex.pixel_slices((H, W))
        r0, r1 = max(er.start - rs.start, 0), min(er.stop - rs.start, crop.shape[0])
        c0, c1 = max(ec.start - cs.start, 0), min(ec.stop - cs.start, crop.shape[1])
        if r1 > r0 and c1 > c0:
            mask[r0:r1, c0:c1] = False
    if not mask.any():
        return ExtractionResult(np.empty(0), np.empty(0), np.empty(0), None, {"reason": "no_match"})

    mask = clean_mask(mask, params)
    weights = np.clip(1.0 - de / max(params.delta_e, 1e-6), 0.05, 1.0)
    cands = column_candidates(mask, weights, params.merge_gap)
    click = None if seed is None else (int(np.floor(seed[0] - cs.start)), float(seed[1] - rs.start))
    pts = track_curve(cands, params, crop.shape[0], click)
    if not pts:
        return ExtractionResult(np.empty(0), np.empty(0), np.empty(0), None, {"reason": "no_curve"})

    cols = np.array([p[0] for p in pts], dtype=float)
    result = ExtractionResult(
        x_px=cols + cs.start + 0.5,
        y_px=np.array([p[1] for p in pts]) + rs.start,
        thickness_px=np.array([p[2] for p in pts]),
        stats={"columns": int(cols[-1] - cols[0] + 1), "points": len(pts),
               "coverage": len(pts) / max(1.0, cols[-1] - cols[0] + 1)},
    )
    if keep_mask:
        full = np.zeros((H, W), dtype=bool)
        full[rs, cs] = mask
        result.mask = full
    return result
