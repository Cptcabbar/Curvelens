"""Hand corrections of an extracted curve: add, delete and move points, with undo / redo.

Works on :class:`core.postprocess.CurveData` in *data units*, so it does not care whether the curve came from
a PDF's vector layer or from an image, and it needs no GUI.  The editor keeps the untouched original, so a
curve can always be put back the way it was read.
"""
from __future__ import annotations

import numpy as np

from .postprocess import CurveData

_FIELDS = ("x", "y", "x_unc", "y_unc")
MAX_HISTORY = 200


def _copy(d: CurveData) -> CurveData:
    return CurveData(d.name, *(np.array(getattr(d, f), dtype=float) for f in _FIELDS))


class CurveEditor:
    """Editable copy of a curve.  Every change keeps the points sorted by ascending x."""

    def __init__(self, data: CurveData):
        self.original: CurveData = _copy(data)
        self._data: CurveData = _copy(data)
        self._undo: list[CurveData] = []
        self._redo: list[CurveData] = []

    # ------------------------------------------------------------------ state
    @property
    def data(self) -> CurveData:
        return self._data

    @property
    def n_changes(self) -> int:
        """Number of edits applied (undone edits do not count)."""
        return len(self._undo)

    @property
    def is_edited(self) -> bool:
        """True while the curve differs from the way it was read."""
        a, b = self._data, self.original
        return not all(np.array_equal(getattr(a, f), getattr(b, f)) for f in _FIELDS)

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def __len__(self) -> int:
        return len(self._data)

    # ------------------------------------------------------------------ helpers
    def _default_unc(self) -> tuple[float, float]:
        d = self._data
        if len(d) == 0:
            return 0.0, 0.0
        return float(np.nanmedian(d.x_unc)), float(np.nanmedian(d.y_unc))

    def _push(self) -> None:
        self._undo.append(_copy(self._data))
        if len(self._undo) > MAX_HISTORY:
            del self._undo[0]
        self._redo.clear()

    def _commit(self, x, y, xu, yu) -> None:
        order = np.argsort(x, kind="stable")
        self._data = CurveData(self._data.name, np.asarray(x, float)[order], np.asarray(y, float)[order],
                               np.asarray(xu, float)[order], np.asarray(yu, float)[order])

    # ------------------------------------------------------------------ edits
    def add_point(self, x: float, y: float, x_unc: float | None = None, y_unc: float | None = None) -> int:
        """Insert a point; returns its index.  Missing uncertainties default to the curve's median ones."""
        if not (np.isfinite(x) and np.isfinite(y)):
            raise ValueError("Nokta koordinatları sayı olmalı.")
        ux, uy = self._default_unc()
        d = self._data
        self._push()
        self._commit(np.append(d.x, x), np.append(d.y, y),
                     np.append(d.x_unc, ux if x_unc is None else x_unc),
                     np.append(d.y_unc, uy if y_unc is None else y_unc))
        # index of the new point (last among equal x, because the sort is stable)
        return int(np.searchsorted(self._data.x, x, side="right") - 1)

    def delete(self, indices) -> int:
        """Remove the points at ``indices``; returns how many were removed."""
        idx = np.unique(np.asarray(list(indices), dtype=int))
        d = self._data
        idx = idx[(idx >= 0) & (idx < len(d))]
        if idx.size == 0:
            return 0
        keep = np.ones(len(d), bool)
        keep[idx] = False
        self._push()
        self._commit(d.x[keep], d.y[keep], d.x_unc[keep], d.y_unc[keep])
        return int(idx.size)

    def indices_in_box(self, x0: float, x1: float, y0: float, y1: float) -> np.ndarray:
        d = self._data
        xa, xb = sorted((x0, x1))
        ya, yb = sorted((y0, y1))
        return np.nonzero((d.x >= xa) & (d.x <= xb) & (d.y >= ya) & (d.y <= yb))[0]

    def delete_in_box(self, x0: float, x1: float, y0: float, y1: float) -> int:
        """Remove every point inside the box (data units, borders included)."""
        return self.delete(self.indices_in_box(x0, x1, y0, y1))

    def move_point(self, index: int, x: float, y: float) -> int:
        """Put point ``index`` at ``(x, y)``; returns its new index (it may move in the x order)."""
        d = self._data
        if not (0 <= index < len(d)):
            raise IndexError("Böyle bir nokta yok.")
        if not (np.isfinite(x) and np.isfinite(y)):
            raise ValueError("Nokta koordinatları sayı olmalı.")
        nx, ny = d.x.copy(), d.y.copy()
        nx[index], ny[index] = x, y
        self._push()
        marker = np.zeros(len(d))
        marker[index] = 1.0
        order = np.argsort(nx, kind="stable")
        self._commit(nx, ny, d.x_unc, d.y_unc)
        return int(np.nonzero(marker[order])[0][0])

    def nearest(self, x: float, y: float, x_scale: float = 1.0, y_scale: float = 1.0) -> tuple[int, float] | None:
        """``(index, distance)`` of the point closest to ``(x, y)``; the axes are scaled by ``*_scale`` first
        (pass data-units-per-pixel to measure the distance in screen pixels)."""
        d = self._data
        if len(d) == 0:
            return None
        dist = np.hypot((d.x - x) / x_scale, (d.y - y) / y_scale)
        i = int(np.argmin(dist))
        return i, float(dist[i])

    # ------------------------------------------------------------------ history
    def undo(self) -> bool:
        if not self._undo:
            return False
        self._redo.append(self._data)
        self._data = self._undo.pop()
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        self._undo.append(self._data)
        self._data = self._redo.pop()
        return True

    def reset(self) -> bool:
        """Back to the curve as it was read (undoable like any other edit)."""
        if not self.is_edited:
            return False
        self._push()
        self._data = _copy(self.original)
        return True
