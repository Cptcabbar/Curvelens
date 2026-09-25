"""GUI-independent session state: calibration inputs, areas, curves, edits, export.

The window (``ui/main_window.py``) is a thin shell over :class:`Project`, so every
behaviour of the application can be tested without a display.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from .calibration import AxisCalibration, Calibration
from .export import AxisInfo, export_combined, export_separate
from .extraction import ExtractionResult, extract_curve, pick_reference_color
from .models import Curve, ExtractionParams, Rect
from .postprocess import CurveData, resample, smooth_savgol, to_curve_data


@dataclass
class AxisSetup:
    """Two clicked reference points of one axis plus its label and scale."""

    pixel: list[float | None] = field(default_factory=lambda: [None, None])
    value: list[float | None] = field(default_factory=lambda: [None, None])
    log: bool = False
    label: str = ""
    unit: str = ""

    def calibration(self) -> AxisCalibration | None:
        """The axis calibration, ``None`` while incomplete; ``ValueError`` when inconsistent."""
        if any(v is None for v in self.pixel + self.value):
            return None
        return AxisCalibration(self.pixel[0], self.value[0], self.pixel[1], self.value[1], self.log)

    def info(self) -> AxisInfo:
        return AxisInfo(self.label, self.unit, self.log)


@dataclass
class PostOptions:
    smooth: bool = False
    window: int = 15                 # Savitzky-Golay window (points, odd)
    polyorder: int = 3
    resample: bool = False
    step: float = 0.0                # x step of the output grid (data units)
    max_gap_factor: float = 5.0      # holes wider than this many steps become NaN


def nice_step(span: float, target_points: int = 200) -> float:
    """A 1-2-5 step giving roughly ``target_points`` samples over ``span``."""
    if not np.isfinite(span) or span <= 0:
        return 1.0
    raw = span / target_points
    mag = 10.0 ** np.floor(np.log10(raw))
    for m in (1, 2, 5, 10):
        if raw <= m * mag:
            return float(m * mag)
    return float(10 * mag)


class Project:
    """Everything the user builds up while digitising one chart."""

    UNDO_LIMIT = 100

    def __init__(self) -> None:
        self.image: np.ndarray | None = None
        self.source: str = ""
        self.x_axis = AxisSetup()
        self.y_axis = AxisSetup()
        self.plot_area: Rect | None = None
        self.exclusions: list[Rect] = []
        self.curves: list[Curve] = []
        self.post = PostOptions()
        self._undo: list[tuple[int, str, np.ndarray, np.ndarray]] = []
        self._redo: list[tuple[int, str, np.ndarray, np.ndarray]] = []

    # ------------------------------------------------------------------ image
    def set_image(self, image_bgr: np.ndarray, source: str = "") -> None:
        """Load a new image; everything tied to the previous image is dropped."""
        self.image = np.ascontiguousarray(image_bgr)
        self.source = source
        self.x_axis, self.y_axis = AxisSetup(), AxisSetup()
        self.plot_area = None
        self.exclusions = []
        self.curves = []
        self._undo.clear()
        self._redo.clear()

    # ------------------------------------------------------------------ calibration
    def calibration(self) -> Calibration | None:
        """Full calibration, ``None`` while an axis is incomplete (raises ``ValueError`` if invalid)."""
        ax, ay = self.x_axis.calibration(), self.y_axis.calibration()
        return Calibration(ax, ay) if ax and ay else None

    def calibration_error(self) -> str | None:
        """Human-readable reason why the calibration is unusable, ``None`` when fine or incomplete."""
        try:
            self.calibration()
        except ValueError as exc:
            return str(exc)
        return None

    def set_axis_point(self, axis: str, index: int, pixel: float | None = None, value: float | None = None) -> None:
        setup = self.x_axis if axis == "x" else self.y_axis
        if pixel is not None:
            setup.pixel[index] = float(pixel)
        if value is not None:
            setup.value[index] = float(value)

    # ------------------------------------------------------------------ areas
    def set_plot_area(self, rect: Rect | None) -> None:
        self.plot_area = rect

    def add_exclusion(self, rect: Rect) -> None:
        self.exclusions.append(rect)

    def remove_exclusion(self, index: int) -> None:
        del self.exclusions[index]

    def clear_exclusions(self) -> None:
        self.exclusions.clear()

    # ------------------------------------------------------------------ curves
    def add_curve(self, x: float, y: float, name: str | None = None) -> Curve:
        """New curve whose reference colour is the pixel under ``(x, y)``."""
        if self.image is None:
            raise ValueError("Önce bir görsel açın.")
        rgb = pick_reference_color(self.image, x, y)
        curve = self.add_curve_with_color(rgb, name)
        curve.seed = (float(x), float(y))
        return curve

    def add_curve_with_color(self, rgb: Sequence[int], name: str | None = None) -> Curve:
        curve = Curve(name or f"Eğri {len(self.curves) + 1}", tuple(int(v) for v in rgb))
        self.curves.append(curve)
        return curve

    def remove_curve(self, index: int) -> None:
        del self.curves[index]
        self._undo = [(i - (i > index), *rest) for i, *rest in self._undo if i != index]
        self._redo.clear()

    def extract(self, index: int, keep_mask: bool = False) -> ExtractionResult:
        """(Re-)run the automatic extraction of one curve with its own parameters."""
        if self.image is None:
            raise ValueError("Önce bir görsel açın.")
        curve = self.curves[index]
        res = extract_curve(self.image, curve.color_rgb, self.plot_area, self.exclusions, curve.params, keep_mask,
                            seed=curve.seed)
        self._push_undo(index, "extract")
        curve.set_points(res.x_px, res.y_px)
        curve.edited = False
        return res

    # ------------------------------------------------------------------ manual edits + undo
    def _push_undo(self, index: int, what: str) -> None:
        c = self.curves[index]
        self._undo.append((index, what, c.x_px.copy(), c.y_px.copy()))
        del self._undo[:-self.UNDO_LIMIT]
        self._redo.clear()

    def erase(self, index: int, rect: Rect) -> int:
        """Delete the points of one curve inside ``rect`` (returns how many)."""
        c = self.curves[index]
        inside = rect.contains(c.x_px, c.y_px) if len(c) else np.zeros(0, bool)
        n = int(inside.sum())
        if n:
            self._push_undo(index, "erase")
            c.x_px, c.y_px = c.x_px[~inside], c.y_px[~inside]
            c.edited = True
        return n

    def add_point(self, index: int, x: float, y: float) -> None:
        self._push_undo(index, "add")
        self.curves[index].add_point(x, y)
        self.curves[index].edited = True

    def reextract_unedited(self) -> list[int]:
        """Re-run the extraction of every curve the user has not touched by hand."""
        done = []
        for i, c in enumerate(self.curves):
            if not c.edited:
                self.extract(i)
                done.append(i)
        return done

    def can_undo(self) -> bool:
        return bool(self._undo)

    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo(self) -> int | None:
        """Revert the last edit; returns the affected curve index (or ``None``)."""
        if not self._undo:
            return None
        index, what, xs, ys = self._undo.pop()
        c = self.curves[index]
        self._redo.append((index, what, c.x_px.copy(), c.y_px.copy()))
        c.x_px, c.y_px = xs, ys
        return index

    def redo(self) -> int | None:
        if not self._redo:
            return None
        index, what, xs, ys = self._redo.pop()
        c = self.curves[index]
        self._undo.append((index, what, c.x_px.copy(), c.y_px.copy()))
        c.x_px, c.y_px = xs, ys
        return index

    # ------------------------------------------------------------------ results
    def curve_data(self, index: int, post: bool = True) -> CurveData | None:
        """Curve in data units (calibrated); with ``post`` smoothing/resampling are applied.

        ``None`` while the calibration is missing or the curve has no points.
        """
        cal = self.calibration()
        c = self.curves[index]
        if cal is None or len(c) == 0:
            return None
        y_px = c.y_px
        if post and self.post.smooth:
            y_px = smooth_savgol(c.y_px, c.x_px, self.post.window, self.post.polyorder)
        data = to_curve_data(c.name, c.x_px, y_px, cal)
        if post and self.post.resample and self.post.step > 0:
            gap = self.post.max_gap_factor * self.post.step
            if len(data) > 1:
                gap = max(gap, 5.0 * float(np.median(np.diff(data.x))))
            data = resample(data, self.post.step, max_gap=gap)
        return data

    def curve_quality(self, index: int) -> tuple[float, float] | None:
        """``(extent, density)`` of a curve: its x extent as a fraction of the plot-area width and the
        fraction of columns inside that extent that hold a point.  ``None`` for empty curves.

        A short extent or many holes usually mean the curve is partly hidden under another one,
        or that the colour threshold is too strict.
        """
        c = self.curves[index]
        if len(c) < 2 or self.image is None:
            return None
        width = self.plot_area.width if self.plot_area is not None else float(self.image.shape[1])
        span = float(c.x_px[-1] - c.x_px[0])
        return span / max(width, 1.0), min(1.0, len(c) / (span + 1.0))

    def suggested_step(self) -> float | None:
        """A sensible resampling step from the calibrated x extent of all curves."""
        spans = []
        for i in range(len(self.curves)):
            d = self.curve_data(i, post=False)
            if d is not None and len(d):
                spans.append(float(d.x[-1] - d.x[0]))
        return nice_step(max(spans)) if spans else None

    def smoothing_note(self) -> str | None:
        p = self.post
        return f"Savitzky-Golay (pencere={p.window | 1}, derece={p.polyorder})" if p.smooth else None

    # ------------------------------------------------------------------ export
    def _export_curves(self) -> list[CurveData]:
        out = [d for i in range(len(self.curves)) if (d := self.curve_data(i)) is not None and len(d)]
        if not out:
            raise ValueError("Dışa aktarılacak veri yok: kalibrasyonu tamamlayın ve en az bir eğri çıkarın.")
        return out

    def export(self, target: str | Path, combined: bool, per_point_uncertainty: bool = False) -> list[Path]:
        """Write CSV files.  ``target`` is a file (combined) or a directory (one file per curve)."""
        if self.calibration() is None:
            raise ValueError("Dışa aktarmak için iki eksenin de kalibrasyonu tamamlanmalı.")
        if combined and not (self.post.resample and self.post.step > 0):
            raise ValueError("Ortak X ızgarası için yeniden örnekleme etkin olmalı ve adım > 0 olmalı.")
        curves = self._export_curves()
        step = self.post.step if self.post.resample else None
        common = dict(source=self.source, step=step, smoothing=self.smoothing_note())
        xa, ya = self.x_axis.info(), self.y_axis.info()
        if combined:
            return [export_combined(curves, target, xa, ya, per_point_uncertainty=per_point_uncertainty, **common)]
        return export_separate(curves, target, xa, ya, per_point_uncertainty=per_point_uncertainty, **common)
