"""Application icon drawn in code (no binary assets to ship)."""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap


def make_pixmap(size: int = 256) -> QPixmap:
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    s = float(size)
    p.setBrush(QColor(28, 45, 74))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawRoundedRect(QRectF(0.02 * s, 0.02 * s, 0.96 * s, 0.96 * s), 0.18 * s, 0.18 * s)

    axis = QPen(QColor(235, 238, 245), 0.035 * s, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
    p.setPen(axis)
    p.drawLine(QPointF(0.20 * s, 0.17 * s), QPointF(0.20 * s, 0.80 * s))
    p.drawLine(QPointF(0.20 * s, 0.80 * s), QPointF(0.84 * s, 0.80 * s))

    for colour, pts in ((QColor(255, 96, 96), [(0.24, 0.30), (0.42, 0.34), (0.60, 0.40), (0.72, 0.50), (0.80, 0.68)]),
                        (QColor(90, 230, 120), [(0.24, 0.40), (0.42, 0.45), (0.58, 0.51), (0.68, 0.60), (0.74, 0.72)])):
        path = QPainterPath(QPointF(pts[0][0] * s, pts[0][1] * s))
        for x, y in pts[1:]:
            path.lineTo(x * s, y * s)
        p.setPen(QPen(colour, 0.05 * s, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)
        p.setBrush(QColor(255, 255, 255))
        p.setPen(Qt.PenStyle.NoPen)
        for x, y in pts[1:-1]:
            p.drawEllipse(QPointF(x * s, y * s), 0.022 * s, 0.022 * s)
    p.end()
    return pm


def make_icon() -> QIcon:
    icon = QIcon()
    for size in (16, 24, 32, 48, 64, 128, 256):
        icon.addPixmap(make_pixmap(size))
    return icon
