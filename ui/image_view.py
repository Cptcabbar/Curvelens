"""Zoom/pan image view with an overlay for calibration, areas and extracted points."""
from __future__ import annotations

import math
from enum import Enum, auto
from typing import Sequence

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QCursor, QImage, QPainter, QPainterPath, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (QGraphicsItem, QGraphicsLineItem, QGraphicsPixmapItem, QGraphicsRectItem,
                               QGraphicsScene, QGraphicsSimpleTextItem, QGraphicsView)

from core.models import Rect


class Tool(Enum):
    PAN = auto()
    CALIB = auto()          # click one calibration point (slot chosen by the window)
    PICK_COLOR = auto()     # click a curve to take its colour
    PLOT_AREA = auto()      # drag the plot-area rectangle
    EXCLUDE = auto()        # drag an exclusion rectangle
    ERASE = auto()          # drag a box; points inside are deleted
    ADD_POINT = auto()      # click to add a point
    MOVE_POINT = auto()     # drag a point of the selected curve to a new place
    DELETE_POINT = auto()   # click a point of the selected curve to delete it
    PICK_CURVE = auto()     # click a curve of an image chart to read it (colour + seed)

POINT_TOOLS = {Tool.CALIB, Tool.PICK_COLOR, Tool.ADD_POINT, Tool.PICK_CURVE}
RECT_TOOLS = {Tool.PLOT_AREA, Tool.EXCLUDE, Tool.ERASE}
HIT_RADIUS = 12.0       # screen pixels within which a click grabs a point

MIN_ZOOM, MAX_ZOOM = 0.02, 64.0


def ndarray_to_pixmap(img_bgr: np.ndarray) -> QPixmap:
    img = np.ascontiguousarray(img_bgr)
    h, w = img.shape[:2]
    qimg = QImage(img.data, w, h, int(img.strides[0]), QImage.Format.Format_BGR888)
    return QPixmap.fromImage(qimg)        # fromImage copies, the numpy buffer may go away


class PointsItem(QGraphicsItem):
    """Extracted points of one curve, drawn as dots of constant on-screen size."""

    def __init__(self, image_rect: QRectF):
        super().__init__()
        self._rect = QRectF(image_rect)
        self._poly = QPolygonF()
        self._color = QColor(255, 0, 0)
        self._halo = QColor(255, 255, 255)
        self._size = 4.0
        self.setZValue(10)

    def set_size(self, size: float) -> None:
        self._size = size
        self.update()

    def set_points(self, xs: np.ndarray, ys: np.ndarray, color: QColor, size: float) -> None:
        self.prepareGeometryChange()
        self._poly = QPolygonF([QPointF(float(x), float(y)) for x, y in zip(xs, ys)])
        self._color = QColor(color)
        luminance = 0.299 * color.red() + 0.587 * color.green() + 0.114 * color.blue()
        self._halo = QColor(255, 255, 255) if luminance < 140 else QColor(0, 0, 0)
        self._size = size
        self.update()

    def boundingRect(self) -> QRectF:
        return self._rect

    def paint(self, painter: QPainter, option, widget=None) -> None:
        if self._poly.isEmpty():
            return
        for colour, width in ((self._halo, self._size + 2.0), (self._color, self._size)):
            pen = QPen(colour, width)
            pen.setCosmetic(True)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawPoints(self._poly)


class ImageView(QGraphicsView):
    """Shows the image; wheel = zoom at the cursor, drag = pan (or draw, depending on the tool)."""

    clicked = Signal(float, float)                 # scene (= image pixel) coordinates
    rectSelected = Signal(object)                  # core.models.Rect
    cursorMoved = Signal(float, float)             # nan, nan when the cursor leaves the image
    cancelled = Signal()
    pointMoved = Signal(int, float, float)         # index into the edit points, new scene x, y
    pointDeleteRequested = Signal(int)             # index into the edit points

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setBackgroundBrush(QBrush(QColor(45, 45, 48)))
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        self._pix_item: QGraphicsPixmapItem | None = None
        self._image_rect = QRectF()
        self._tool = Tool.PAN
        self._panning = False
        self._space = False
        self._last_pos = QPointF()
        self._press_pos = QPointF()
        self._drag_start: QPointF | None = None
        self._rubber: QGraphicsRectItem | None = None
        self._calib_items: list = []
        self._area_item: QGraphicsRectItem | None = None
        self._excl_items: list[QGraphicsRectItem] = []
        self._curve_items: list[PointsItem] = []
        self._highlight_item = None
        self._points_visible = True
        self._auto_fit = True          # keep the image fitted until the user zooms/pans by hand
        self._edit_xs = np.empty(0)    # points of the selected curve that can be grabbed / deleted (image pixels)
        self._edit_ys = np.empty(0)
        self._move_index: int | None = None
        self._move_ring = None
        self._marker_items: list = []
        self._edit_mode = False
        self.set_tool(Tool.PAN)

    # ------------------------------------------------------------------ image
    def set_image(self, img_bgr: np.ndarray | None) -> None:
        self._scene.clear()
        self._pix_item = None
        self._calib_items, self._excl_items, self._curve_items = [], [], []
        self._area_item = None
        self._rubber = None
        self._highlight_item = None
        self._move_ring, self._move_index, self._marker_items = None, None, []
        self._edit_xs, self._edit_ys = np.empty(0), np.empty(0)
        if img_bgr is None:
            self._image_rect = QRectF()
            return
        pm = ndarray_to_pixmap(img_bgr)
        self._pix_item = self._scene.addPixmap(pm)
        self._pix_item.setZValue(0)
        self._image_rect = QRectF(0, 0, pm.width(), pm.height())
        self._scene.setSceneRect(self._image_rect.adjusted(-2000, -2000, 2000, 2000))
        self.resetTransform()
        self.fit_in_view()

    def has_image(self) -> bool:
        return self._pix_item is not None

    # ------------------------------------------------------------------ zoom
    @property
    def zoom(self) -> float:
        return self.transform().m11()

    def _apply_zoom_quality(self) -> None:
        if self._pix_item is not None:
            # Crisp pixels when magnified (needed to pick tick marks), smooth when shrunk.
            self._pix_item.setTransformationMode(
                Qt.TransformationMode.FastTransformation if self.zoom >= 1.0 else Qt.TransformationMode.SmoothTransformation)

    def zoom_by(self, factor: float, anchor_view_pos: QPointF | None = None) -> None:
        if not self.has_image():
            return
        self._auto_fit = False
        target = min(max(self.zoom * factor, MIN_ZOOM), MAX_ZOOM)
        factor = target / self.zoom
        pos = anchor_view_pos if anchor_view_pos is not None else QPointF(self.viewport().rect().center())
        before = self.mapToScene(pos.toPoint())
        self.scale(factor, factor)
        after = self.mapToScene(pos.toPoint())
        delta = after - before
        self.translate(delta.x(), delta.y())
        self._apply_zoom_quality()

    def fit_in_view(self) -> None:
        if self.has_image():
            self._auto_fit = True
            self.fitInView(self._image_rect, Qt.AspectRatioMode.KeepAspectRatio)
            self._apply_zoom_quality()

    def wheelEvent(self, event) -> None:
        if not self.has_image():
            return
        steps = event.angleDelta().y() / 120.0
        if steps:
            self.zoom_by(1.25 ** steps, event.position())
        event.accept()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._auto_fit and self.has_image():
            self.fitInView(self._image_rect, Qt.AspectRatioMode.KeepAspectRatio)
            self._apply_zoom_quality()

    # ------------------------------------------------------------------ tools
    def set_tool(self, tool: Tool) -> None:
        self._tool = tool
        self._cancel_drag()
        self._update_cursor()

    @property
    def tool(self) -> Tool:
        return self._tool

    def _update_cursor(self) -> None:
        if self._panning:
            shape = Qt.CursorShape.ClosedHandCursor
        elif self._tool == Tool.PAN or self._space:
            shape = Qt.CursorShape.OpenHandCursor
        else:
            shape = Qt.CursorShape.CrossCursor
        self.viewport().setCursor(QCursor(shape))

    def _cancel_drag(self) -> None:
        self._drag_start = None
        if self._rubber is not None:
            self._scene.removeItem(self._rubber)
            self._rubber = None
        self._move_index = None
        self._clear_move_ring()

    # ------------------------------------------------------------------ editable points
    def set_edit_points(self, xs: np.ndarray | None, ys: np.ndarray | None) -> None:
        """Points (image pixels, in the curve's own index order) that MOVE_POINT / DELETE_POINT can grab."""
        self._edit_xs = np.empty(0) if xs is None else np.asarray(xs, float)
        self._edit_ys = np.empty(0) if ys is None else np.asarray(ys, float)

    def set_edit_mode(self, on: bool) -> None:
        """Editing makes the selected curve's dots bigger so single points are easy to hit."""
        self._edit_mode = on
        for it in self._curve_items:
            if it.zValue() > 10:
                it.set_size(7.0 if on else 5.0)

    def _hit_point(self, view_pos: QPointF) -> int | None:
        if self._edit_xs.size == 0:
            return None
        t = self.viewportTransform()
        vx = t.m11() * self._edit_xs + t.m21() * self._edit_ys + t.dx()
        vy = t.m12() * self._edit_xs + t.m22() * self._edit_ys + t.dy()
        d = np.hypot(vx - view_pos.x(), vy - view_pos.y())
        i = int(np.argmin(d))
        return i if d[i] <= HIT_RADIUS else None

    def _show_move_ring(self, scene_pos: QPointF) -> None:
        if self._move_ring is None:
            ring = self._scene.addEllipse(-9, -9, 18, 18)
            ring.setPen(self._cosmetic_pen(QColor(255, 255, 0), 2.0))
            ring.setBrush(QBrush(QColor(255, 255, 0, 60)))
            ring.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
            ring.setZValue(30)
            self._move_ring = ring
        self._move_ring.setPos(scene_pos)

    def _clear_move_ring(self) -> None:
        if self._move_ring is not None:
            self._scene.removeItem(self._move_ring)
            self._move_ring = None

    def set_markers(self, points: Sequence[tuple[float, float]]) -> None:
        """Crosshairs at ``(x_px, y_px)`` (e.g. the answer of a value query); an empty list clears them."""
        for it in self._marker_items:
            self._scene.removeItem(it)
        self._marker_items = []
        if not self.has_image():
            return
        for x, y in points:
            for colour, width in ((QColor(255, 255, 255), 4.0), (QColor(220, 0, 90), 2.0)):
                path = QPainterPath()
                path.moveTo(-12, 0)
                path.lineTo(12, 0)
                path.moveTo(0, -12)
                path.lineTo(0, 12)
                path.addEllipse(-6, -6, 12, 12)
                item = self._scene.addPath(path)
                item.setPen(self._cosmetic_pen(colour, width))
                item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
                item.setPos(float(x), float(y))
                item.setZValue(25)
                self._marker_items.append(item)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space = True
            self._update_cursor()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Escape:
            self._cancel_drag()
            self.cancelled.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space = False
            self._update_cursor()
            event.accept()
            return
        super().keyReleaseEvent(event)

    def mousePressEvent(self, event) -> None:
        if not self.has_image():
            return
        self.setFocus()
        pos = event.position()
        self._press_pos = pos
        left = event.button() == Qt.MouseButton.LeftButton
        if event.button() == Qt.MouseButton.MiddleButton or (left and (self._tool == Tool.PAN or self._space)):
            self._panning = True
            self._last_pos = pos
            self._update_cursor()
            event.accept()
            return
        if left and self._tool in RECT_TOOLS:
            self._drag_start = self.mapToScene(pos.toPoint())
            event.accept()
            return
        if left and self._tool == Tool.MOVE_POINT:
            idx = self._hit_point(pos)
            if idx is not None:
                self._move_index = idx
                self._show_move_ring(self.mapToScene(pos.toPoint()))
            else:                                   # nothing to grab here: drag pans, as usual
                self._panning = True
                self._last_pos = pos
                self._update_cursor()
            event.accept()
            return
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        pos = event.position()
        scene = self.mapToScene(pos.toPoint())
        if self._image_rect.contains(scene):
            self.cursorMoved.emit(scene.x(), scene.y())
        else:
            self.cursorMoved.emit(math.nan, math.nan)
        if self._move_index is not None:
            self._show_move_ring(scene)
            event.accept()
            return
        if self._panning:
            self._auto_fit = False
            d = pos - self._last_pos
            self._last_pos = pos
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(d.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(d.y()))
            event.accept()
            return
        if self._drag_start is not None:
            rect = QRectF(self._drag_start, scene).normalized()
            if self._rubber is None:
                self._rubber = self._scene.addRect(rect)
                pen = QPen(QColor(255, 255, 0), 1.5, Qt.PenStyle.DashLine)
                pen.setCosmetic(True)
                self._rubber.setPen(pen)
                self._rubber.setBrush(QBrush(QColor(255, 255, 0, 40)))
                self._rubber.setZValue(20)
            self._rubber.setRect(rect)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        pos = event.position()
        moved = (pos - self._press_pos).manhattanLength()
        if self._move_index is not None:
            idx, self._move_index = self._move_index, None
            self._clear_move_ring()
            if event.button() == Qt.MouseButton.LeftButton and moved >= 3:
                p = self.mapToScene(pos.toPoint())
                x = min(max(p.x(), 0.0), self._image_rect.width())
                y = min(max(p.y(), 0.0), self._image_rect.height())
                self.pointMoved.emit(idx, x, y)
            event.accept()
            return
        if self._panning:
            self._panning = False
            self._update_cursor()
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and self._tool == Tool.DELETE_POINT and moved < 4:
            idx = self._hit_point(pos)
            if idx is not None:
                self.pointDeleteRequested.emit(idx)
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            if self._drag_start is not None:
                start, end = self._drag_start, self.mapToScene(pos.toPoint())
                self._cancel_drag()
                if moved >= 4:
                    r = Rect.from_points(start.x(), start.y(), end.x(), end.y())
                    r = Rect(max(r.x0, 0.0), max(r.y0, 0.0), min(r.x1, self._image_rect.width()),
                             min(r.y1, self._image_rect.height()))
                    if r.width > 0 and r.height > 0:
                        self.rectSelected.emit(r)
                event.accept()
                return
            if self._tool in POINT_TOOLS and moved < 4:
                p = self.mapToScene(pos.toPoint())
                if self._image_rect.contains(p):
                    self.clicked.emit(p.x(), p.y())
                event.accept()
                return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event) -> None:
        self.cursorMoved.emit(math.nan, math.nan)
        super().leaveEvent(event)

    # ------------------------------------------------------------------ overlays
    def _cosmetic_pen(self, color: QColor, width: float = 1.0, style=Qt.PenStyle.SolidLine) -> QPen:
        pen = QPen(color, width, style)
        pen.setCosmetic(True)
        return pen

    def set_calibration_markers(self, markers: Sequence[tuple[str, float, str]]) -> None:
        """``(axis, pixel, label)``: a vertical guide for ``axis == 'x'``, horizontal for ``'y'``."""
        for it in self._calib_items:
            self._scene.removeItem(it)
        self._calib_items = []
        if not self.has_image():
            return
        w, h = self._image_rect.width(), self._image_rect.height()
        for axis, pixel, label in markers:
            colour = QColor(0, 120, 255) if axis == "x" else QColor(255, 0, 200)
            line = (QGraphicsLineItem(pixel, 0, pixel, h) if axis == "x" else QGraphicsLineItem(0, pixel, w, pixel))
            line.setPen(self._cosmetic_pen(colour, 1.0, Qt.PenStyle.DashLine))
            line.setZValue(15)
            self._scene.addItem(line)
            text = QGraphicsSimpleTextItem(label)
            text.setBrush(QBrush(colour))
            text.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
            text.setPos(pixel + 3, 3) if axis == "x" else text.setPos(3, pixel + 3)
            text.setZValue(16)
            self._scene.addItem(text)
            self._calib_items += [line, text]

    def set_plot_area(self, rect: Rect | None) -> None:
        if self._area_item is not None:
            self._scene.removeItem(self._area_item)
            self._area_item = None
        if rect is not None and self.has_image():
            item = self._scene.addRect(QRectF(rect.x0, rect.y0, rect.width, rect.height))
            item.setPen(self._cosmetic_pen(QColor(0, 200, 90), 1.5, Qt.PenStyle.DashLine))
            item.setZValue(12)
            self._area_item = item

    def set_exclusions(self, rects: Sequence[Rect]) -> None:
        for it in self._excl_items:
            self._scene.removeItem(it)
        self._excl_items = []
        if not self.has_image():
            return
        for r in rects:
            item = self._scene.addRect(QRectF(r.x0, r.y0, r.width, r.height))
            item.setPen(self._cosmetic_pen(QColor(230, 40, 40), 1.5))
            item.setBrush(QBrush(QColor(230, 40, 40, 55)))
            item.setZValue(11)
            self._excl_items.append(item)

    def set_curves(self, curves: Sequence[tuple[np.ndarray, np.ndarray, QColor, bool]]) -> None:
        """``(x_px, y_px, colour, is_selected)`` for every curve."""
        for it in self._curve_items:
            self._scene.removeItem(it)
        self._curve_items = []
        if not self.has_image():
            return
        for xs, ys, colour, selected in curves:
            item = PointsItem(self._image_rect)
            item.set_points(xs, ys, colour, (7.0 if self._edit_mode else 5.0) if selected else 3.5)
            item.setVisible(self._points_visible)
            item.setZValue(10 + (1 if selected else 0))
            self._scene.addItem(item)
            self._curve_items.append(item)

    def set_highlight(self, xs: np.ndarray | None, ys: np.ndarray | None, color: QColor | None = None) -> None:
        """Glow line through the given points (image pixels) to show which curve is selected; ``None`` clears it."""
        if self._highlight_item is not None:
            self._scene.removeItem(self._highlight_item)
            self._highlight_item = None
        if xs is None or len(xs) < 2 or not self.has_image():
            return
        path = QPainterPath(QPointF(float(xs[0]), float(ys[0])))
        for x, y in zip(xs[1:], ys[1:]):
            path.lineTo(float(x), float(y))
        item = self._scene.addPath(path)
        glow = QColor(255, 214, 0, 150)
        pen = QPen(glow, 11)
        pen.setCosmetic(True)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        item.setPen(pen)
        item.setZValue(9)
        self._highlight_item = item

    def set_points_visible(self, visible: bool) -> None:
        self._points_visible = visible
        for it in self._curve_items:
            it.setVisible(visible)
