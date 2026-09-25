"""Dialog to pick the page and the chart of a PDF (with a page preview showing where each chart is)."""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout, QLabel, QListWidget,
                               QListWidgetItem, QSizePolicy, QSpinBox, QVBoxLayout, QWidget)

from core.pdf_source import ChartInfo, PdfSource
from ui.image_view import ndarray_to_pixmap

WHOLE_PAGE = -1


class PageThumb(QWidget):
    """Page preview with a numbered box around every detected chart."""

    chartClicked = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pix: QPixmap | None = None
        self._page_size = (1.0, 1.0)
        self._charts: list[ChartInfo] = []
        self._selected = WHOLE_PAGE
        self.setMinimumSize(360, 500)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_page(self, pix: QPixmap, page_size_pt: tuple[float, float], charts: list[ChartInfo]) -> None:
        self._pix, self._page_size, self._charts, self._selected = pix, page_size_pt, charts, WHOLE_PAGE
        self.update()

    def set_selected(self, index: int) -> None:
        self._selected = index
        self.update()

    def _target(self) -> QRectF:
        if self._pix is None:
            return QRectF()
        avail = QRectF(self.rect()).adjusted(4, 4, -4, -4)
        scale = min(avail.width() / self._pix.width(), avail.height() / self._pix.height())
        w, h = self._pix.width() * scale, self._pix.height() * scale
        return QRectF(avail.center().x() - w / 2, avail.center().y() - h / 2, w, h)

    def _pt_to_widget(self, x: float, y: float) -> QPointF:
        t = self._target()
        return QPointF(t.x() + x / self._page_size[0] * t.width(), t.y() + y / self._page_size[1] * t.height())

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(60, 60, 64))
        if self._pix is None:
            return
        t = self._target()
        p.drawPixmap(t, self._pix, QRectF(self._pix.rect()))
        font = QFont(self.font())
        font.setBold(True)
        p.setFont(font)
        for i, c in enumerate(self._charts):
            a = self._pt_to_widget(c.region.x0, c.region.y0)
            b = self._pt_to_widget(c.region.x1, c.region.y1)
            r = QRectF(a, b)
            sel = i == self._selected
            colour = QColor(255, 140, 0) if sel else QColor(0, 120, 255)
            p.setPen(QPen(colour, 3 if sel else 1.5))
            p.setBrush(QColor(colour.red(), colour.green(), colour.blue(), 45 if sel else 20))
            p.drawRect(r)
            badge = QRectF(r.left(), r.top(), 20, 18)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(colour)
            p.drawRect(badge)
            p.setPen(QColor(255, 255, 255))
            p.drawText(badge, Qt.AlignmentFlag.AlignCenter, str(i + 1))

    def mousePressEvent(self, event) -> None:
        t = self._target()
        if not t.contains(event.position()):
            return
        px = (event.position().x() - t.x()) / t.width() * self._page_size[0]
        py = (event.position().y() - t.y()) / t.height() * self._page_size[1]
        hit = WHOLE_PAGE
        for i, c in enumerate(self._charts):
            if c.region.x0 <= px <= c.region.x1 and c.region.y0 <= py <= c.region.y1:
                hit = i
        self.chartClicked.emit(hit)


class PdfChartDialog(QDialog):
    """Choose page, chart (or the whole page) and render resolution."""

    def __init__(self, src: PdfSource, parent=None, initial_page: int = 0):
        super().__init__(parent)
        self.setWindowTitle("PDF: grafik seç")
        self._src = src
        self._charts: list[ChartInfo] = []

        self.thumb = PageThumb()
        self.page_spin = QSpinBox()
        self.page_spin.setRange(1, max(1, src.page_count))
        self.page_spin.setValue(initial_page + 1)
        self.page_spin.setEnabled(src.page_count > 1)
        self.list = QListWidget()
        self.dpi = QComboBox()
        for v in (150, 200, 300, 400, 600, 800, 1200):
            self.dpi.addItem(f"{v} DPI", v)
        self.dpi.setCurrentIndex(self.dpi.findData(600))
        self.size_label = QLabel()
        self.hint = QLabel("Listeden bir grafik seçin ya da önizlemede grafiğin kutusuna tıklayın.")
        self.hint.setWordWrap(True)

        form = QFormLayout()
        form.addRow("Sayfa:", self.page_spin)
        form.addRow("Çözünürlük:", self.dpi)
        form.addRow("", self.size_label)

        side = QVBoxLayout()
        side.addWidget(self.hint)
        side.addWidget(QLabel("Sayfadaki grafikler:"))
        side.addWidget(self.list, 1)
        side.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Aç")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("İptal")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        row = QHBoxLayout()
        row.addWidget(self.thumb, 3)
        row.addLayout(side, 2)
        root = QVBoxLayout(self)
        root.addLayout(row, 1)
        root.addWidget(buttons)
        self.resize(900, 640)

        self.page_spin.valueChanged.connect(self._load_page)
        self.list.currentRowChanged.connect(self._row_changed)
        self.thumb.chartClicked.connect(self._thumb_clicked)
        self.dpi.currentIndexChanged.connect(self._update_size_label)
        self._load_page()

    # ------------------------------------------------------------------ helpers
    def _load_page(self) -> None:
        page = self.page_spin.value() - 1
        self._charts = self._src.charts(page)
        thumb = self._src.render(page, None, dpi=110)
        self.thumb.set_page(ndarray_to_pixmap(thumb.image), self._src.page_size(page), self._charts)
        self.list.clear()
        for i, c in enumerate(self._charts):
            self.list.addItem(QListWidgetItem(f"{i + 1} · {c.title or 'Adsız grafik'}"))
        self.list.addItem(QListWidgetItem("Tüm sayfa"))
        self.list.setCurrentRow(0 if self._charts else 0)

    def _row_changed(self, row: int) -> None:
        idx = row if 0 <= row < len(self._charts) else WHOLE_PAGE
        self.thumb.set_selected(idx)
        # A whole page at 600 DPI is huge; default lower.
        if idx == WHOLE_PAGE and self.dpi.currentData() > 300:
            self.dpi.setCurrentIndex(self.dpi.findData(200))
        elif idx != WHOLE_PAGE and self.dpi.currentData() < 300:
            self.dpi.setCurrentIndex(self.dpi.findData(600))
        self._update_size_label()

    def _thumb_clicked(self, index: int) -> None:
        self.list.setCurrentRow(index if index != WHOLE_PAGE else len(self._charts))

    def _update_size_label(self) -> None:
        sel = self.selected_chart()
        page = self.page_spin.value() - 1
        w, h = self._src.page_size(page)
        if sel is not None:
            w, h = sel.region.width, sel.region.height
        s = self.dpi.currentData() / 72.0
        self.size_label.setText(f"Görüntü ≈ {int(w * s)} × {int(h * s)} piksel")

    # ------------------------------------------------------------------ result
    def selected_chart(self) -> ChartInfo | None:
        row = self.list.currentRow()
        return self._charts[row] if 0 <= row < len(self._charts) else None

    def selection(self) -> tuple[int, ChartInfo | None, float]:
        return self.page_spin.value() - 1, self.selected_chart(), float(self.dpi.currentData())
