"""Smoothing, resampling and uncertainty of extracted curves (pixel -> data)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter

from .calibration import Calibration


@dataclass
class CurveData:
    """A curve in data units, sorted by ascending x."""

    name: str
    x: np.ndarray
    y: np.ndarray
    x_unc: np.ndarray          # +/- uncertainty of x at every point
    y_unc: np.ndarray

    def __len__(self) -> int:
        return len(self.x)


def smooth_savgol(y_px: np.ndarray, x_px: np.ndarray, window: int = 11, polyorder: int = 3) -> np.ndarray:
    """Savitzky-Golay smoothing of ``y_px`` along ``x_px`` (pixel space).

    Extracted points sit on (nearly) uniform pixel columns, which is what the
    filter assumes; small holes are bridged by linear interpolation first so the
    window always spans a uniform grid.  ``window`` is forced odd and clamped to
    the data length; ``polyorder`` is clamped below the window.
    """
    x_px = np.asarray(x_px, dtype=float)
    y_px = np.asarray(y_px, dtype=float)
    n = len(x_px)
    if n < 5:
        return y_px.copy()
    window = int(window) | 1                           # odd
    window = min(window, n if n % 2 else n - 1)
    polyorder = min(int(polyorder), window - 1)
    if window < 3 or polyorder < 1:
        return y_px.copy()

    step = float(np.median(np.diff(x_px)))
    if step <= 0:
        return y_px.copy()
    grid = np.arange(x_px[0], x_px[-1] + 0.5 * step, step)
    y_grid = np.interp(grid, x_px, y_px)
    smoothed = savgol_filter(y_grid, window, polyorder, mode="interp")
    return np.interp(x_px, grid, smoothed)


def to_curve_data(name: str, x_px, y_px, calibration: Calibration) -> CurveData:
    """Apply a calibration; result is sorted by x and carries per-point +/-0.5 px uncertainty."""
    x_px = np.asarray(x_px, dtype=float)
    y_px = np.asarray(y_px, dtype=float)
    x, y = calibration.pixel_to_data(x_px, y_px)
    ux, uy = calibration.uncertainty(x_px, y_px)
    order = np.argsort(x, kind="stable")
    return CurveData(name, np.asarray(x)[order], np.asarray(y)[order],
                     np.asarray(ux)[order], np.asarray(uy)[order])


def resample(curve: CurveData, step: float, max_gap: float | None = None,
             x_start: float | None = None, x_stop: float | None = None) -> CurveData:
    """Interpolate onto the grid ``k * step`` (multiples of ``step``) inside the curve's x range.

    No extrapolation.  With ``max_gap`` (in x units) grid points that fall into a
    hole of the source data wider than that become NaN instead of a made-up
    straight line.
    """
    if step <= 0:
        raise ValueError("Adım pozitif olmalı.")
    if len(curve) == 0:
        return CurveData(curve.name, *(np.empty(0) for _ in range(4)))
    lo = curve.x[0] if x_start is None else max(x_start, curve.x[0])
    hi = curve.x[-1] if x_stop is None else min(x_stop, curve.x[-1])
    k0 = int(np.ceil(lo / step - 1e-9))
    k1 = int(np.floor(hi / step + 1e-9))
    if k1 < k0:
        return CurveData(curve.name, *(np.empty(0) for _ in range(4)))
    grid = np.arange(k0, k1 + 1) * step

    # ``interp`` needs strictly increasing x; merge duplicate x by averaging.
    ux, inv = np.unique(curve.x, return_inverse=True)
    counts = np.bincount(inv)
    def avg(v):
        return np.bincount(inv, weights=v) / counts
    y = np.interp(grid, ux, avg(curve.y))
    xu = np.interp(grid, ux, avg(curve.x_unc))
    yu = np.interp(grid, ux, avg(curve.y_unc))
    if max_gap is not None:
        idx = np.searchsorted(ux, grid, side="right")
        left = ux[np.clip(idx - 1, 0, len(ux) - 1)]
        right = ux[np.clip(idx, 0, len(ux) - 1)]
        # A grid point sitting exactly on a data point is known, even at a hole's edge.
        exact = np.isclose(grid, left) | np.isclose(grid, right)
        inside_hole = ((right - left) > max_gap) & ~exact
        y = np.where(inside_hole, np.nan, y)
        xu = np.where(inside_hole, np.nan, xu)
        yu = np.where(inside_hole, np.nan, yu)
    return CurveData(curve.name, grid, y, xu, yu)


def uncertainty_summary(curve: CurveData) -> tuple[float, float, float, float]:
    """``(x_min, x_max, y_min, y_max)`` of the per-point uncertainties (equal for linear axes)."""
    if len(curve) == 0:
        return (float("nan"),) * 4
    return (float(np.nanmin(curve.x_unc)), float(np.nanmax(curve.x_unc)),
            float(np.nanmin(curve.y_unc)), float(np.nanmax(curve.y_unc)))
