"""Look a value up on a curve by typing it: X -> Y and Y -> X.

Values between the curve's points are linearly interpolated (for a dashed curve that includes the gaps
between the dashes); nothing is invented outside the curve's own X range.  Uncertainties combine the
uncertainty of the curve's points with the local slope, so a steep piece of curve gives a wider +/- band.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .postprocess import CurveData

GAP_FACTOR = 8.0          # an interval this many times wider than the typical point spacing counts as a gap


@dataclass
class Answer:
    value: float
    unc: float                 # +/- ; inf when the level is (nearly) flat and the other coordinate is undetermined
    note: str = ""


def _unique(d: CurveData):
    ux, inv = np.unique(d.x, return_inverse=True)
    n = np.bincount(inv)
    avg = lambda v: np.bincount(inv, weights=v) / n
    return ux, avg(d.y), avg(d.x_unc), avg(d.y_unc)


def _slope(x: np.ndarray, y: np.ndarray, i: int, k: int = 3) -> float:
    """Least-squares slope of the curve around segment ``i -> i+1`` (uses up to ``2k`` neighbours)."""
    lo, hi = max(0, i - k + 1), min(len(x), i + k + 1)
    xs, ys = x[lo:hi], y[lo:hi]
    if len(xs) < 2 or np.ptp(xs) == 0:
        return 0.0
    return float(np.polyfit(xs, ys, 1)[0])


def value_at(d: CurveData, x: float) -> Answer | None:
    """Y of the curve at ``x``; ``None`` outside the curve's X range."""
    if len(d) == 0 or not math.isfinite(x):
        return None
    ux, uy, uxu, uyu = _unique(d)
    if x < ux[0] - 1e-12 or x > ux[-1] + 1e-12:
        return None
    if len(ux) == 1:
        return Answer(float(uy[0]), float(uyu[0]))
    x = min(max(x, ux[0]), ux[-1])
    y = float(np.interp(x, ux, uy))
    j = int(np.clip(np.searchsorted(ux, x, side="right") - 1, 0, len(ux) - 2))
    slope = _slope(ux, uy, j)
    yu = float(np.interp(x, ux, uyu))
    xu = float(np.interp(x, ux, uxu))
    unc = math.hypot(yu, slope * xu)
    note = ""
    spacing = float(np.median(np.diff(ux)))
    if spacing > 0 and (ux[j + 1] - ux[j]) > GAP_FACTOR * spacing:
        note = "iki nokta arasındaki boşluk doğrusal doldurulur"
    return Answer(y, unc, note)


def solve_x(d: CurveData, y: float) -> list[Answer]:
    """Every X where the curve equals ``y`` (a curve can cross a level more than once); empty if it never does."""
    if len(d) < 2 or not math.isfinite(y):
        return []
    x, yy = d.x, d.y
    n = len(x)
    spacing = float(np.median(np.diff(x))) if n > 1 else 0.0
    # Points within the curve's own noise of the level count as "at the level"; a dead band like this keeps a
    # noisy curve from producing dozens of crossings, and turns a plateau into one answer with its X range.
    noise = float(np.nanmedian(d.y_unc)) if np.any(np.isfinite(d.y_unc)) else 0.0
    if not math.isfinite(noise) or noise < 0:
        noise = 0.0
    floor = 1e-9 * max(float(np.ptp(yy)), abs(y), 1.0)
    exact = noise <= floor                               # noise-free data: any run of equal points is a plateau
    noise = max(noise, floor)
    dev = yy - y
    inband = np.abs(dev) <= noise

    hits: list[tuple[float, float, str, int]] = []       # (x, extra +/- from a plateau, note, index for the slope)
    last_out = -1                                        # last point outside the band
    i = 0
    while i < n:
        if not inband[i]:
            if last_out >= 0 and dev[last_out] * dev[i] < 0 and i - last_out == 1:
                t = (y - yy[last_out]) / (yy[i] - yy[last_out])
                hits.append((float(x[last_out] + t * (x[i] - x[last_out])), 0.0, "", last_out))
            last_out = i
            i += 1
            continue
        j = i
        while j + 1 < n and inband[j + 1]:
            j += 1
        ext = float(x[j] - x[i])
        flat = j > i and ext > 0 and (exact or ext >= 4.0 * spacing)
        mid = float(np.mean(x[i:j + 1])) if j > i else float(x[i])
        note = f"düz bölge: X = {x[i]:.6g} … {x[j]:.6g}" if flat else ""
        hits.append((mid, 0.5 * ext if flat else 0.0, note, min(i, n - 2)))
        last_out = -1                                    # a run of in-band points ends any pending direct crossing
        i = j + 1
        # the next out-of-band point may still sit on the other side of the level: it is covered by the run itself
    # Crossings that a noisy curve makes right next to each other are one answer.
    tol = max(3.0 * spacing, 0.004 * float(x[-1] - x[0]))
    merged: list[tuple[float, float, str, int]] = []
    group: list[tuple[float, float, str, int]] = []

    def flush() -> None:
        if not group:
            return
        if len(group) == 1:
            merged.append(group[0])
        else:
            xs_ = [g[0] for g in group]
            merged.append((float(np.mean(xs_)), 0.5 * (max(xs_) - min(xs_)),
                           "eğri bu seviyede birkaç kez kesişiyor (gürültü), ortalaması alındı", group[len(group) // 2][3]))
        group.clear()

    for h in hits:
        if group and (h[1] > 0 or group[-1][1] > 0 or h[0] - group[-1][0] > tol):
            flush()
        group.append(h)
    flush()
    hits = merged

    out: list[Answer] = []
    for hx, extra, note, seg in hits:
        xu = float(np.interp(hx, x, d.x_unc))
        yu = float(np.interp(hx, x, d.y_unc))
        slope = _slope(x, yy, seg)
        base = math.inf if abs(slope) < 1e-12 else math.hypot(xu, yu / abs(slope))
        unc = max(base, extra) if not extra else math.hypot(extra, xu)
        out.append(Answer(hx, unc, note))
    return out
