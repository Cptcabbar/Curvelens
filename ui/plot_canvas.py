"""A plain chart drawn from numbers: used when an imported table is not matched with an existing chart.

The picture is a real raster (so the normal image view can show it and put markers on it) and comes with the
exact mapping between data values and picture pixels.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPolygonF

PALETTE = [(31, 119, 180), (214, 39, 40), (44, 160, 44), (255, 127, 14), (148, 103, 189), (140, 86, 75), (227, 119, 194),
           (23, 190, 207)]


def nice_ticks(lo: float, hi: float, target: int = 7) -> list[float]:
    """Round tick values (1-2-5 steps) covering ``[lo, hi]``."""
    span = hi - lo
    if not math.isfinite(span) or span <= 0:
        return [lo]
    raw = span / max(target, 2)
    mag = 10.0 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if raw <= m * mag * (1 + 1e-9))
    first = math.ceil(lo / step - 1e-9) * step
    n = int(math.floor((hi - first) / step + 1e-9)) + 1
    return [first + i * step for i in range(max(n, 1))]


def _fmt(v: float, step: float) -> str:
    dec = 0 if step >= 1 else min(6, int(math.ceil(-math.log10(step))))
    s = f"{v:.{dec}f}"
    return "0" if float(s) == 0 else s


@dataclass
class GraphImage:
    image: np.ndarray                      # BGR uint8
    x0: float                              # data range shown
    x1: float
    y0: float
    y1: float
    left: float                            # plot rectangle in picture pixels
    top: float
    right: float
    bottom: float

    def to_px(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        px = self.left + (x - self.x0) / (self.x1 - self.x0) * (self.right - self.left)
        py = self.bottom - (y - self.y0) / (self.y1 - self.y0) * (self.bottom - self.top)
        return px, py

    def to_data(self, px, py):
        px, py = np.asarray(px, float), np.asarray(py, float)
        x = self.x0 + (px - self.left) / (self.right - self.left) * (self.x1 - self.x0)
        y = self.y0 + (self.bottom - py) / (self.bottom - self.top) * (self.y1 - self.y0)
        return x, y


def _range(values: list[np.ndarray]) -> tuple[float, float]:
    v = np.concatenate([a[np.isfinite(a)] for a in values]) if values else np.array([0.0, 1.0])
    if v.size == 0:
        return 0.0, 1.0
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < 1e-12 * max(1.0, abs(hi)):
        pad = abs(hi) * 0.1 or 1.0
        return lo - pad, hi + pad
    pad = 0.05 * (hi - lo)
    ticks_lo, ticks_hi = lo - pad, hi + pad
    return ticks_lo, ticks_hi


def render_graph(series: list[tuple[str, np.ndarray, np.ndarray]], x_title: str, y_title: str, title: str = "",
                 size: tuple[int, int] = (1200, 760)) -> GraphImage:
    """Line + points chart of ``series`` (name, x, y) with axes, grid, tick labels and a legend."""
    W, H = size
    left, right, top, bottom = 110.0, W - 40.0, (60.0 if title else 34.0), H - 100.0
    xs = [s[1] for s in series]
    ys = [s[2] for s in series]
    x0, x1 = _range(xs)
    y0, y1 = _range(ys)
    g = GraphImage(np.zeros((H, W, 3), np.uint8), x0, x1, y0, y1, left, top, right, bottom)

    img = QImage(W, H, QImage.Format.Format_RGB888)
    img.fill(QColor(255, 255, 255))
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    small, big = QFont("Segoe UI", 13), QFont("Segoe UI", 15, QFont.Weight.DemiBold)
    grid_pen = QPen(QColor(215, 215, 215), 1.0, Qt.PenStyle.DashLine)
    text_pen = QPen(QColor(40, 40, 40))

    def X(v): return float(g.to_px(v, y0)[0])
    def Y(v): return float(g.to_px(x0, v)[1])

    xt, yt = nice_ticks(x0, x1), nice_ticks(y0, y1)
    xstep = (xt[1] - xt[0]) if len(xt) > 1 else 1.0
    ystep = (yt[1] - yt[0]) if len(yt) > 1 else 1.0
    p.setFont(small)
    fm = p.fontMetrics()
    for v in xt:
        p.setPen(grid_pen)
        p.drawLine(QPointF(X(v), top), QPointF(X(v), bottom))
        p.setPen(text_pen)
        label = _fmt(v, xstep)
        p.drawText(QPointF(X(v) - fm.horizontalAdvance(label) / 2, bottom + 26), label)
    for v in yt:
        p.setPen(grid_pen)
        p.drawLine(QPointF(left, Y(v)), QPointF(right, Y(v)))
        p.setPen(text_pen)
        label = _fmt(v, ystep)
        p.drawText(QPointF(left - 12 - fm.horizontalAdvance(label), Y(v) + 5), label)
    p.setPen(QPen(QColor(0, 0, 0), 2.0))
    p.drawRect(int(left), int(top), int(right - left), int(bottom - top))

    for i, (name, x, y) in enumerate(series):
        col = QColor(*PALETTE[i % len(PALETTE)])
        ok = np.isfinite(x) & np.isfinite(y)
        pts = QPolygonF([QPointF(*(float(a) for a in g.to_px(xx, yy))) for xx, yy in zip(x[ok], y[ok])])
        p.setPen(QPen(col, 2.2))
        p.drawPolyline(pts)

    p.setFont(big)
    p.setPen(text_pen)
    fb = p.fontMetrics()
    p.drawText(QPointF((left + right) / 2 - fb.horizontalAdvance(x_title) / 2, bottom + 62), x_title)
    p.save()
    p.translate(30, (top + bottom) / 2 + fb.horizontalAdvance(y_title) / 2)
    p.rotate(-90)
    p.drawText(QPointF(0, 0), y_title)
    p.restore()
    if title:
        p.drawText(QPointF(left, 34), title)
    if len(series) > 1:                                              # legend inside the plot, top right
        p.setFont(small)
        for i, (name, _x, _y) in enumerate(series):
            yy = top + 22 + 22 * i
            p.setPen(QPen(QColor(*PALETTE[i % len(PALETTE)]), 3.0))
            p.drawLine(QPointF(right - 190, yy - 5), QPointF(right - 160, yy - 5))
            p.setPen(text_pen)
            p.drawText(QPointF(right - 152, yy), name[:22])
    p.end()

    ptr = img.constBits()
    arr = np.frombuffer(ptr, np.uint8, count=img.bytesPerLine() * H).reshape(H, img.bytesPerLine())[:, :W * 3]
    g.image = np.ascontiguousarray(arr.reshape(H, W, 3)[..., ::-1])          # RGB -> BGR
    return g
