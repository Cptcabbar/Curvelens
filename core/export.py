"""CSV export: one file per curve, or one file with a shared X grid."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .postprocess import CurveData, uncertainty_summary


@dataclass
class AxisInfo:
    """Label and scale of one axis, written into the CSV header."""

    label: str = ""
    unit: str = ""
    log: bool = False

    @property
    def full_label(self) -> str:
        if self.label and self.unit and f"({self.unit})" not in self.label:
            return f"{self.label} ({self.unit})"
        return self.label or (f"({self.unit})" if self.unit else "")


def _fmt(v: float) -> str:
    return "nan" if v is None or not np.isfinite(v) else f"{v:.6g}"


def _unc_text(lo: float, hi: float, unit: str) -> str:
    suffix = f" {unit}" if unit else ""
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "n/a"
    if np.isclose(lo, hi, rtol=1e-3):
        return f"±{_fmt(hi)}{suffix}"
    return f"±{_fmt(lo)}..±{_fmt(hi)}{suffix} (log axis: varies with position)"


def _header(curves: Sequence[CurveData], x_axis: AxisInfo, y_axis: AxisInfo, source: str,
            step: float | None, smoothing: str | None, pixel_error: float) -> list[str]:
    lines = ["Plot Digitizer export"]
    if source:
        lines.append(f"source: {source}")
    lines.append(f"x axis: {x_axis.full_label or 'x'} [{'log10' if x_axis.log else 'linear'}]")
    lines.append(f"y axis: {y_axis.full_label or 'y'} [{'log10' if y_axis.log else 'linear'}]")
    if step:
        lines.append(f"resampled to x step: {_fmt(step)}{' ' + x_axis.unit if x_axis.unit else ''}")
    if smoothing:
        lines.append(f"smoothing: {smoothing}")
    lines.append(f"uncertainty: +/-{_fmt(pixel_error)} px position error converted with the calibration resolution")
    for c in curves:
        xl, xh, yl, yh = uncertainty_summary(c)
        lines.append(f"curve \"{c.name}\": x {_unc_text(xl, xh, x_axis.unit)}, y {_unc_text(yl, yh, y_axis.unit)}")
    return lines


def _write(path: Path, header: list[str], df: pd.DataFrame, sep: str) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        for line in header:
            fh.write(f"# {line}\n")
        df.to_csv(fh, index=False, sep=sep, float_format="%.6g", lineterminator="\n")


def safe_filename(name: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .")
    return cleaned or "curve"


def export_separate(curves: Sequence[CurveData], directory: str | Path, x_axis: AxisInfo, y_axis: AxisInfo,
                    source: str = "", step: float | None = None, smoothing: str | None = None,
                    pixel_error: float = 0.5, per_point_uncertainty: bool = True, sep: str = ",") -> list[Path]:
    """Write ``<directory>/<curve name>.csv`` for every curve; returns the paths."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    xl, yl = x_axis.full_label or "x", y_axis.full_label or "y"
    written: list[Path] = []
    used: set[str] = set()
    for c in curves:
        stem = safe_filename(c.name)
        base, k = stem, 2
        while stem.lower() in used:                       # duplicate names must not overwrite each other
            stem, k = f"{base}_{k}", k + 1
        used.add(stem.lower())
        data = {xl: c.x, yl: c.y}
        if per_point_uncertainty:
            data[f"±x"] = c.x_unc
            data[f"±y"] = c.y_unc
        path = directory / f"{stem}.csv"
        _write(path, _header([c], x_axis, y_axis, source, step, smoothing, pixel_error), pd.DataFrame(data), sep)
        written.append(path)
    return written


def export_combined(curves: Sequence[CurveData], path: str | Path, x_axis: AxisInfo, y_axis: AxisInfo,
                    source: str = "", step: float | None = None, smoothing: str | None = None,
                    pixel_error: float = 0.5, per_point_uncertainty: bool = False, sep: str = ",") -> Path:
    """One file: shared X column, one Y column per curve (NaN where a curve has no data).

    Curves are expected to share a grid (resample them with the same ``step``);
    otherwise the union of all x values is used and missing entries stay NaN.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    xl = x_axis.full_label or "x"
    yl = y_axis.full_label or "y"
    frames = []
    for i, c in enumerate(curves):
        cols = {"x": c.x, f"y{i}": c.y}
        if per_point_uncertainty:
            cols[f"uy{i}"] = c.y_unc
        frames.append(pd.DataFrame(cols).drop_duplicates("x").set_index("x"))
    df = pd.concat(frames, axis=1).sort_index() if frames else pd.DataFrame()
    df.index.name = xl
    rename = {}
    for i, c in enumerate(curves):
        rename[f"y{i}"] = f"{c.name} [{yl}]"
        rename[f"uy{i}"] = f"{c.name} ±y"
    df = df.rename(columns=rename).reset_index()
    _write(path, _header(curves, x_axis, y_axis, source, step, smoothing, pixel_error), df, sep)
    return path
