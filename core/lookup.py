"""Lookup tables of a curve: X -> Y on a regular X grid, ready to show, print or export."""
from __future__ import annotations

import html
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from .postprocess import CurveData, resample
from .vector_charts import ChartData, CurveResult


def auto_step(x_min: float, x_max: float, target_rows: int = 60) -> float:
    """A 1-2-5 step that gives about ``target_rows`` rows between ``x_min`` and ``x_max``."""
    span = float(x_max - x_min)
    if not math.isfinite(span) or span <= 0:
        return 1.0
    raw = span / target_rows
    mag = 10.0 ** math.floor(math.log10(raw))
    for m in (1, 2, 5, 10):
        if raw <= m * mag * (1 + 1e-9):
            return float(m * mag)
    return 10.0 * mag


def decimals_for(uncertainty: float, cap: int = 8) -> int:
    """Decimal places worth printing for a value known to +/- ``uncertainty``."""
    if not math.isfinite(uncertainty) or uncertainty <= 0:
        return 3
    return int(min(cap, max(0, math.ceil(-math.log10(uncertainty)))))


def step_decimals(step: float) -> int:
    """Decimals needed to write multiples of ``step`` exactly (0.25 -> 2, 50 -> 0)."""
    for d in range(0, 9):
        if abs(round(step, d) - step) < 1e-9 * max(1.0, abs(step)):
            return d
    return 8


@dataclass
class LookupTable:
    chart_title: str
    curve_label: str
    style_label: str
    x_header: str
    y_header: str
    x: np.ndarray
    y: np.ndarray                      # NaN where the curve has no value
    step: float | None                 # None = the curve's own points
    x_decimals: int
    y_decimals: int
    x_unc: float
    y_unc: float
    source: str = ""
    page: int = 0
    notes: list[str] = field(default_factory=list)
    kind: str = "vector"               # "vector" or "raster": how the curve was read
    n_changes: int = 0                 # hand corrections applied to the curve (0 = as read)

    def __len__(self) -> int:
        return len(self.x)

    @property
    def method_text(self) -> str:
        base = ("PDF'in vektör verisinden okundu (piksel ölçümü değil)" if self.kind == "vector"
                else "PDF içindeki grafik görselinden (piksel) okundu")
        return base + (f"; eğri elle düzeltildi ({self.n_changes} değişiklik)" if self.n_changes else "")

    def fmt_x(self, v: float) -> str:
        return f"{v:.{self.x_decimals}f}"

    def fmt_y(self, v: float) -> str:
        return "—" if not math.isfinite(v) else f"{v:.{self.y_decimals}f}"

    def rows(self) -> list[tuple[str, str]]:
        return [(self.fmt_x(a), self.fmt_y(b)) for a, b in zip(self.x, self.y)]


def build_lookup(chart: ChartData, curve: CurveResult, step: float | None = "auto", source: str = "") -> LookupTable:
    """Lookup table of ``curve``.

    ``step="auto"`` picks a round X step (about 60 rows); ``step=None`` lists the curve's own points;
    a number gives the X step explicitly.  Values between the curve's points are linearly
    interpolated (for dashed curves that includes the gaps between the dashes); outside the
    curve's X range nothing is invented.
    """
    data: CurveData = curve.data
    yfit = chart.y_fit_of(curve)
    xfit = chart.x_fit
    x_unc = float(np.nanmax(data.x_unc)) if len(data) else 0.0
    y_unc = float(np.nanmax(data.y_unc)) if len(data) else 0.0
    if step == "auto":
        step = auto_step(float(data.x[0]), float(data.x[-1]))
    if step is None:
        ux, inv = np.unique(data.x, return_inverse=True)           # duplicate x -> average
        counts = np.bincount(inv)
        x, y = ux, np.bincount(inv, weights=data.y) / counts
        xdec = max(decimals_for(x_unc), 1) if len(x) else 0
    else:
        r = resample(data, float(step))
        x, y = r.x, r.y
        xdec = step_decimals(float(step))
    ydec = decimals_for(y_unc)
    xh = (f"{xfit.title}" if xfit and xfit.title else "X")
    yh = (f"{yfit.title}" if yfit and yfit.title else "Y")
    return LookupTable(chart.title, curve.label, curve.style_label, xh, yh, x, y, step, xdec, ydec, x_unc, y_unc,
                       source, chart.chart.page_index + 1, list(chart.notes), chart.kind, curve.n_changes)


def _unit_of(header: str) -> str:
    return header[header.rfind("(") + 1:header.rfind(")")] if "(" in header and header.endswith(")") else ""


def unit_of(header: str) -> str:
    """'Voltage (V)' -> 'V'; '' when the header has no unit in brackets."""
    return _unit_of(header.strip())


def name_of(header: str) -> str:
    """'Voltage (V)' -> 'Voltage'."""
    h = header.strip()
    return h[:h.rfind("(")].strip() if _unit_of(h) else h


def uncertainty_text(t: LookupTable) -> str:
    ux = f"±{t.x_unc:.2g} {_unit_of(t.x_header)}".strip()
    uy = f"±{t.y_unc:.2g} {_unit_of(t.y_header)}".strip()
    return f"X {ux}, Y {uy}"


def report_columns(n_rows: int) -> int:
    """How many side-by-side column blocks make a tidy A4 page for ``n_rows`` rows."""
    return 1 if n_rows <= 38 else 2 if n_rows <= 150 else 3


def to_html(t: LookupTable, columns: int = 1) -> str:
    """Printable report: title, what the curve is, and the table (optionally split into several column blocks)."""
    e = html.escape
    rows = t.rows()
    if columns > 1 and rows:
        per = math.ceil(len(rows) / columns)
        blocks = [rows[i:i + per] for i in range(0, len(rows), per)]
        for b in blocks:
            b.extend([("", "")] * (per - len(b)))
        body_rows = list(zip(*blocks))
        head = "".join(f"<th>{e(t.x_header)}</th><th>{e(t.y_header)}</th>" for _ in blocks)
        body = "".join("<tr>" + "".join(f"<td align='right'>{e(a)}</td><td align='right'>{e(b)}</td>" for a, b in r)
                       + "</tr>" for r in body_rows)
    else:
        head = f"<th>{e(t.x_header)}</th><th>{e(t.y_header)}</th>"
        body = "".join(f"<tr><td align='right'>{e(a)}</td><td align='right'>{e(b)}</td></tr>" for a, b in rows)
    step_txt = "eğrinin kendi noktaları" if t.step is None else f"{t.step:g} {_unit_of(t.x_header)} aralıkla".strip()
    notes = " · ".join(html.escape(n) for n in t.notes)
    meta = [
        ("Grafik", t.chart_title),
        ("Eğri", f"{t.curve_label} ({t.style_label}) — Y: {t.y_header}, X: {t.x_header}"),
        ("Satır", f"{len(rows)} satır, {step_txt}. {t.method_text}; olası belirsizlik {uncertainty_text(t)}"),
        ("Kaynak", (f"{t.source} · sayfa {t.page}" if t.source else f"sayfa {t.page}")
                   + " · " + datetime.now().strftime("%d.%m.%Y %H:%M")),
    ]
    if notes:
        meta.insert(3, ("Notlar", t.notes and " · ".join(t.notes)))
    meta_html = "".join(f"<tr><td><b>{e(k)}</b></td><td>{e(v)}</td></tr>" for k, v in meta)
    return (
        "<html><head><meta charset='utf-8'></head><body style='font-family:Segoe UI,Arial,sans-serif; font-size:9pt'>"
        f"<h3 style='margin:0 0 4px 0'>Lookup tablosu — {e(t.curve_label)}</h3>"
        f"<table cellspacing='0' cellpadding='1'>{meta_html}</table>"
        "<p style='margin:2px'></p>"
        f"<table border='1' cellspacing='0' cellpadding='1' width='100%' style='border-collapse:collapse; font-size:8.5pt'>"
        f"<thead><tr bgcolor='#dfe6f0'>{head}</tr></thead><tbody>{body}</tbody></table>"
        "</body></html>"
    )


def write_csv(t: LookupTable, path: str | Path) -> Path:
    """CSV with ``#`` comment lines (source, axes, uncertainty) followed by ``X,Y`` rows."""
    path = Path(path)
    lines = [f"# Lookup tablosu: {t.chart_title} / {t.curve_label} ({t.style_label})",
             f"# Kaynak: {t.source} sayfa {t.page} ({t.method_text})",
             f"# Olası belirsizlik: {uncertainty_text(t)}",
             "# Aralıklarda doğrusal interpolasyon; eğrinin X aralığı dışında değer yoktur"]
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        fh.write("\n".join(lines) + "\n")
        fh.write(f"{t.x_header},{t.y_header}\n")
        for a, b in zip(t.x, t.y):
            fh.write(f"{t.fmt_x(a)},{'' if not math.isfinite(b) else t.fmt_y(b)}\n")
    return path
