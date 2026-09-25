"""Remembers the axis calibration of picture charts, so the same chart of the same PDF is never calibrated twice.

The calibration is what the user clicked and typed (two marks per axis with their values and names), stored per
PDF *content* (a hash of the file, so a copy or a renamed file is still recognised) and per chart (page and the
picture's place on the page).  Everything else - the curves - is read again from the picture when the chart is
opened, which takes a moment and cannot go stale.

The store is one small JSON file in the user's application-data folder (``%APPDATA%\\PlotDigitizer`` on Windows,
overridable with the ``PLOT_DIGITIZER_DATA`` environment variable).  A damaged file is set aside and never stops the
application.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

from .raster_charts import AxisPoints

VERSION = 1
FILE_NAME = "calibrations.json"
MAX_PDFS = 300                   # oldest entries are dropped beyond this


def data_dir() -> Path:
    override = os.environ.get("PLOT_DIGITIZER_DATA")
    if override:
        return Path(override)
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    else:
        base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "PlotDigitizer"


def pdf_fingerprint(path: str | Path) -> str:
    """SHA-1 of the file's bytes (a few ms for a datasheet)."""
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def chart_key(page_index: int, bbox, image_shape) -> str:
    """Identifies a picture chart inside its PDF: page, place on the page (points) and pixel size."""
    h, w = image_shape[:2]
    return f"p{page_index}:{bbox.x0:.0f},{bbox.y0:.0f},{bbox.x1:.0f},{bbox.y1:.0f}:{w}x{h}"


def _axis_to_json(a: AxisPoints | None):
    if a is None:
        return None
    return {"px": [float(a.px[0]), float(a.px[1])], "value": [float(a.value[0]), float(a.value[1])],
            "name": a.name, "log": bool(a.log)}


def _axis_from_json(d) -> AxisPoints | None:
    if not d:
        return None
    px, value = d["px"], d["value"]
    if len(px) != 2 or len(value) != 2:
        raise ValueError("bad axis")
    return AxisPoints((float(px[0]), float(px[1])), (float(value[0]), float(value[1])), str(d.get("name", "")),
                      bool(d.get("log", False)))


class Saved(NamedTuple):
    x: AxisPoints
    y: AxisPoints
    y2: AxisPoints | None
    saved: str                                    # when, "YYYY-MM-DD HH:MM"
    plot: tuple[float, float, float, float] | None = None      # plot area the user drew (picture pixels), if any


class CalibrationStore:
    """``get`` / ``save`` / ``forget`` of the calibration of one chart of one PDF."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else data_dir() / FILE_NAME

    # ------------------------------------------------------------------ file
    def _load(self) -> dict:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data.get("version") == VERSION and isinstance(data.get("pdfs"), dict):
                return data
        except FileNotFoundError:
            pass
        except (OSError, ValueError):
            self._set_aside()
        return {"version": VERSION, "pdfs": {}}

    def _set_aside(self) -> None:
        """Keep an unreadable file as ``*.bad`` (once) instead of overwriting the user's data blindly."""
        try:
            bad = self.path.with_suffix(".bad")
            if not bad.exists():
                self.path.replace(bad)
        except OSError:
            pass

    def _write(self, data: dict) -> bool:
        pdfs = data["pdfs"]
        if len(pdfs) > MAX_PDFS:
            for k in sorted(pdfs, key=lambda k: pdfs[k].get("used", ""))[:len(pdfs) - MAX_PDFS]:
                del pdfs[k]
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
            tmp.replace(self.path)
            return True
        except OSError:
            return False                                   # read-only profile etc.: calibration just is not remembered

    # ------------------------------------------------------------------ API
    def get(self, pdf_hash: str, key: str) -> Saved | None:
        """What is remembered for the chart, or ``None`` when nothing usable is stored."""
        entry = self._load()["pdfs"].get(pdf_hash, {}).get("charts", {}).get(key)
        if not entry:
            return None
        try:
            x, y = _axis_from_json(entry["x"]), _axis_from_json(entry["y"])
            y2 = _axis_from_json(entry.get("y2"))
            plot = entry.get("plot")
            plot = tuple(float(v) for v in plot) if plot else None
            if plot is not None and len(plot) != 4:
                raise ValueError("bad plot")
        except (KeyError, TypeError, ValueError):
            return None
        return Saved(x, y, y2, str(entry.get("saved", "")), plot)

    def save(self, pdf_hash: str, pdf_name: str, key: str, x: AxisPoints, y: AxisPoints, y2: AxisPoints | None,
             plot: tuple[float, float, float, float] | None = None) -> bool:
        data = self._load()
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        rec = data["pdfs"].setdefault(pdf_hash, {"name": pdf_name, "charts": {}})
        rec["name"], rec["used"] = pdf_name, now
        rec.setdefault("charts", {})[key] = {"x": _axis_to_json(x), "y": _axis_to_json(y), "y2": _axis_to_json(y2),
                                             "plot": [float(v) for v in plot] if plot else None, "saved": now}
        return self._write(data)

    def forget(self, pdf_hash: str, key: str) -> bool:
        data = self._load()
        charts = data["pdfs"].get(pdf_hash, {}).get("charts", {})
        if key not in charts:
            return False
        del charts[key]
        if not charts:
            del data["pdfs"][pdf_hash]
        return self._write(data)

    def has(self, pdf_hash: str, key: str) -> bool:
        return self.get(pdf_hash, key) is not None
