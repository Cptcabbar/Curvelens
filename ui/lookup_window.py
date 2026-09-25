"""Main window: open a PDF -> gallery of its charts -> chart detail with curves -> lookup table.

Three simple screens in one stacked widget:

* welcome  - open (or drop) a PDF;
* gallery  - every chart found in the PDF, as picture + name;
* detail   - the chart, its curves with their meaning on the side, and the lookup table of the
             selected curve with CSV / PDF / print buttons.

All numbers come from the PDF's vector data (see :mod:`core.vector_charts`), read automatically
when the file is opened.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSize, Qt, QThread, Signal
from PySide6.QtGui import QAction, QColor, QIcon, QKeySequence, QPixmap
from PySide6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QDoubleSpinBox, QFileDialog, QFrame, QHBoxLayout,
                               QHeaderView, QInputDialog, QLabel, QListView, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
                               QProgressDialog, QPushButton, QSplitter, QStackedWidget, QTableView, QToolBar,
                               QVBoxLayout, QWidget)

from core import raster_charts as rcm
from core.calibration_store import CalibrationStore, chart_key, pdf_fingerprint
from core.lookup import LookupTable, auto_step, build_lookup, report_columns, to_html, uncertainty_text, write_csv
from core.pdf_source import PdfSource, RenderedRegion
from core.imageio import IMAGE_SUFFIXES
from core.vector_charts import ChartData, CurveResult, analyze_pdf
from ui import printing
from ui.calibration_dialog import CalibrationDialog
from ui.image_view import ImageView, Tool, ndarray_to_pixmap
from ui.query_panel import QueryPanel
from ui.table_model import ArrayTableModel, fmt

APP_NAME = "Plot Digitizer"
DETAIL_DPI = 400.0
THUMB_WIDTH = 320

EDIT_TOOLS = (
    (Tool.PAN, "Gezin", "Görüntüyü kaydır ve yakınlaştır"),
    (Tool.ADD_POINT, "Nokta ekle", "Eğrinin eksik ya da hatalı yerine tıklayarak nokta ekleyin"),
    (Tool.DELETE_POINT, "Nokta sil", "Silinecek noktaya tıklayın"),
    (Tool.ERASE, "Kutuyla sil", "Silinecek noktaların çevresine bir kutu çizin"),
    (Tool.MOVE_POINT, "Nokta taşı", "Bir noktayı tutup doğru yerine sürükleyin (boş yerde sürüklemek görüntüyü kaydırır)"),
)
PAN_HINT = "Tekerlek: yakınlaştır · sürükle: kaydır · seçili eğri görselde noktalarla işaretlenir"


def color_icon(rgb: tuple[int, int, int], dashed: bool = False, size: int = 18) -> QIcon:
    pm = QPixmap(size, size)
    pm.fill(QColor(255, 255, 255))
    from PySide6.QtGui import QPainter, QPen
    p = QPainter(pm)
    pen = QPen(QColor(*rgb), 3)
    if dashed:
        pen.setStyle(Qt.PenStyle.DashLine)
    p.setPen(pen)
    p.drawLine(1, size // 2, size - 2, size // 2)
    p.end()
    return QIcon(pm)


def is_image_path(path: str) -> bool:
    return str(path).lower().endswith(IMAGE_SUFFIXES)


def analyze_file(path: str, progress=None) -> list[ChartData]:
    """Charts of a PDF, or the one chart of an image file (PNG, JPG, BMP, TIFF, WebP)."""
    if is_image_path(path):
        return [rcm.chart_from_image(path)]
    return analyze_pdf(path, progress)


class AnalysisWorker(QThread):
    """Reads all charts of a PDF (or an image) off the GUI thread."""

    progress = Signal(int, int, str)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def run(self) -> None:
        try:
            charts = analyze_file(self.path, lambda i, n, text: self.progress.emit(i, n, text))
        except Exception as exc:                                        # corrupt / encrypted / unsupported files
            self.failed.emit(str(exc))
        else:
            self.done.emit(charts)


class LookupWindow(QMainWindow):
    def __init__(self, store: CalibrationStore | None = None) -> None:
        super().__init__()
        self.store = store or CalibrationStore()      # remembered axis calibrations of picture charts
        self._pdf_hash: str | None = None
        self.setWindowTitle(APP_NAME)
        self.resize(1400, 850)
        self.setAcceptDrops(True)
        self.charts: list[ChartData] = []
        self.pdf_path: Path | None = None
        self._src: PdfSource | None = None
        self._rendered: RenderedRegion | None = None
        self._chart: ChartData | None = None
        self._curve: CurveResult | None = None
        self._table: LookupTable | None = None
        self._worker: AnalysisWorker | None = None
        self._progress: QProgressDialog | None = None
        self._legacy: list[QMainWindow] = []
        self._loading = False
        self._cal: dict | None = None             # calibration clicks of a picture chart in progress

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        self.stack.addWidget(self._build_welcome())
        self.stack.addWidget(self._build_gallery())
        self.stack.addWidget(self._build_detail())
        self._build_actions()
        self.statusBar().showMessage("Bir PDF açın.")
        self.hover_label = QLabel("")
        self.statusBar().addPermanentWidget(self.hover_label)
        self._show_page(0)

    # ================================================================== building
    def _build_actions(self) -> None:
        self.act_open = QAction("PDF / görsel aç…", self, shortcut=QKeySequence.StandardKey.Open, triggered=self.open_pdf_dialog)
        self.act_back = QAction("← Grafikler", self, shortcut="Alt+Left", triggered=self.show_gallery)
        self.act_quit = QAction("Çıkış", self, shortcut="Ctrl+Q", triggered=self.close)
        self.act_manual = QAction("Görselden elle sayısallaştır (gelişmiş)…", self, triggered=self.open_manual_tool)
        self.act_help = QAction("Kullanım", self, shortcut="F1", triggered=self.show_help)
        self.act_undo = QAction("Geri al", self, shortcut=QKeySequence.StandardKey.Undo, triggered=self.undo_edit)
        self.act_redo = QAction("Yinele", self, shortcut="Ctrl+Y", triggered=self.redo_edit)
        self.act_reset = QAction("Eğriyi özgün haline döndür", self, triggered=self.reset_edits)
        self.act_rename = QAction("Eğrinin adını değiştir…", self, shortcut="F2", triggered=self.rename_curve)
        self.act_forget_cal = QAction("Bu grafiğin kayıtlı kalibrasyonunu sil", self, triggered=self.forget_calibration)
        self.act_forget_cal.setEnabled(False)

        mb = self.menuBar()
        m = mb.addMenu("&Dosya")
        m.addAction(self.act_open)
        m.addSeparator()
        m.addAction(self.act_quit)
        m = mb.addMenu("&Düzelt")
        m.addAction(self.act_undo)
        m.addAction(self.act_redo)
        m.addSeparator()
        m.addAction(self.act_reset)
        m.addAction(self.act_rename)
        m = mb.addMenu("&Araçlar")
        m.addAction(self.act_forget_cal)
        m.addSeparator()
        m.addAction(self.act_manual)
        m = mb.addMenu("&Yardım")
        m.addAction(self.act_help)

        tb = QToolBar("Ana")
        tb.setMovable(False)
        tb.setIconSize(QSize(16, 16))
        self.addToolBar(tb)
        tb.addAction(self.act_open)
        tb.addAction(self.act_back)
        self.act_back.setVisible(False)

    def _build_welcome(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addStretch(1)
        title = QLabel("Grafiklerden lookup tablosu")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 26px; font-weight: 600;")
        sub = QLabel("Bir PDF ya da grafik görseli (PNG, JPG, BMP…) açın: PDF'teki tüm grafikler otomatik bulunur (çizili\n"
                     "grafikler tam hassasiyetle, resim olanlar eksen kalibrasyonuyla okunur). Sonra bir eğri seçip\n"
                     "tabloyu, değer sorgusunu ve yazdırmayı kullanabilirsiniz.")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sub.setStyleSheet("font-size: 14px; color: gray;")
        self.welcome_btn = QPushButton("PDF / görsel aç…")
        self.welcome_btn.setMinimumHeight(48)
        self.welcome_btn.setMaximumWidth(260)
        self.welcome_btn.setStyleSheet("font-size: 16px;")
        self.welcome_btn.clicked.connect(self.open_pdf_dialog)
        hint = QLabel("PDF ya da görsel dosyasını bu pencereye sürükleyip bırakabilirsiniz.")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setStyleSheet("color: gray;")
        lay.addWidget(title)
        lay.addWidget(sub)
        lay.addSpacing(18)
        lay.addWidget(self.welcome_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(hint)
        lay.addStretch(2)
        return w

    def _build_gallery(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        self.gallery_title = QLabel("")
        self.gallery_title.setStyleSheet("font-size: 16px; font-weight: 600;")
        self.gallery_hint = QLabel("Bir grafiğe tıklayın.")
        self.gallery_hint.setStyleSheet("color: gray;")
        self.gallery = QListWidget()
        self.gallery.setViewMode(QListView.ViewMode.IconMode)
        self.gallery.setResizeMode(QListView.ResizeMode.Adjust)
        self.gallery.setMovement(QListView.Movement.Static)
        self.gallery.setWrapping(True)
        self.gallery.setSpacing(14)
        self.gallery.setIconSize(QSize(THUMB_WIDTH, 230))
        self.gallery.setWordWrap(True)
        self.gallery.setUniformItemSizes(True)
        self.gallery.itemActivated.connect(lambda it: self.show_chart(self.gallery.row(it)))
        self.gallery.itemClicked.connect(lambda it: self.show_chart(self.gallery.row(it)))
        lay.addWidget(self.gallery_title)
        lay.addWidget(self.gallery_hint)
        lay.addWidget(self.gallery, 1)
        return w

    def _build_detail(self) -> QWidget:
        w = QWidget()
        outer = QVBoxLayout(w)
        self.detail_title = QLabel("")
        self.detail_title.setStyleSheet("font-size: 16px; font-weight: 600;")
        outer.addWidget(self.detail_title)

        split = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(split, 1)

        # left: the chart picture
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        self.view = ImageView()
        self.view.cursorMoved.connect(self._on_cursor)
        self.view.clicked.connect(self._on_click)
        self.view.rectSelected.connect(self._on_rect)
        self.view.pointMoved.connect(self._on_point_moved)
        self.view.pointDeleteRequested.connect(self._on_point_delete)
        self.view.cancelled.connect(self._cancel_mode)
        self.hint_label = QLabel(PAN_HINT)
        self.hint_label.setStyleSheet("color: gray;")
        ll.addWidget(self.view, 1)
        ll.addWidget(self._build_edit_bar())
        ll.addWidget(self.hint_label)
        split.addWidget(left)

        # right: curves + notes + lookup table
        right = QWidget()
        right.setMinimumWidth(400)
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(self._heading("Eğriler ve anlamları"))
        rl.addWidget(self._build_raster_bar())
        self.curve_list = QListWidget()
        self.curve_list.setIconSize(QSize(22, 18))
        self.curve_list.setWordWrap(True)
        self.curve_list.setAlternatingRowColors(True)
        self.curve_list.setMinimumHeight(150)
        self.curve_list.currentRowChanged.connect(self._curve_selected)
        rl.addWidget(self.curve_list, 2)
        self.notes_label = QLabel("")
        self.notes_label.setWordWrap(True)
        self.notes_label.setStyleSheet("color: gray;")
        rl.addWidget(self.notes_label)

        rl.addWidget(self._heading("Lookup tablosu"))
        self.table_caption = QLabel("Bir eğri seçin.")
        self.table_caption.setWordWrap(True)
        rl.addWidget(self.table_caption)
        row = QHBoxLayout()
        row.addWidget(QLabel("X adımı:"))
        self.step_spin = QDoubleSpinBox()
        self.step_spin.setDecimals(6)
        self.step_spin.setRange(0.0, 1e9)
        self.step_spin.setSpecialValueText("Ham noktalar")
        self.step_spin.setKeyboardTracking(False)
        self.step_spin.setToolTip("0 = eğrinin kendi noktaları; aksi halde bu aralıkla doğrusal interpolasyon")
        self.step_spin.valueChanged.connect(self._step_changed)
        auto = QPushButton("Otomatik")
        auto.clicked.connect(self._auto_step)
        row.addWidget(self.step_spin, 1)
        row.addWidget(auto)
        rl.addLayout(row)
        self.table_model = ArrayTableModel()
        self.table = QTableView()
        self.table.setModel(self.table_model)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setDefaultSectionSize(22)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        rl.addWidget(self.table, 3)
        self.precision_label = QLabel("")
        self.precision_label.setWordWrap(True)
        self.precision_label.setStyleSheet("color: gray;")
        rl.addWidget(self.precision_label)
        self.query = QueryPanel()
        self.query.markersChanged.connect(self._show_markers)
        rl.addWidget(self.query)
        btns = QHBoxLayout()
        self.csv_btn = QPushButton("CSV")
        self.pdf_btn = QPushButton("PDF kaydet…")
        self.print_btn = QPushButton("Yazdır…")
        self.print_btn.setDefault(True)
        self.csv_btn.clicked.connect(self.export_csv)
        self.pdf_btn.clicked.connect(self.save_pdf)
        self.print_btn.clicked.connect(self.print_table)
        for b in (self.csv_btn, self.pdf_btn, self.print_btn):
            btns.addWidget(b)
        rl.addLayout(btns)
        split.addWidget(right)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        self._set_table_enabled(False)
        return w

    @staticmethod
    def _heading(text: str) -> QLabel:
        lb = QLabel(text)
        lb.setStyleSheet("font-size: 13px; font-weight: 600; margin-top: 6px;")
        return lb

    def _build_raster_bar(self) -> QWidget:
        """Tools of a picture chart (a chart pasted into the PDF as an image): axes, adding / renaming curves."""
        bar = QWidget()
        col = QVBoxLayout(bar)
        col.setContentsMargins(0, 0, 0, 0)
        self.raster_status = QLabel("")
        self.raster_status.setWordWrap(True)
        self.raster_status.setStyleSheet("color: #b9770e;")
        col.addWidget(self.raster_status)
        row = QHBoxLayout()
        self.calib_btn = QPushButton("Kalibre et")
        self.calib_btn.setToolTip("X ve Y eksenlerinden ikişer işaret (tick) gösterip değerlerini yazın")
        self.calib_btn.clicked.connect(self.start_calibration)
        self.add_curve_btn = QPushButton("＋ Eğri ekle")
        self.add_curve_btn.setToolTip("Görselde bir eğrinin üstüne tıklayın: o eğri okunur")
        self.add_curve_btn.clicked.connect(self.start_add_curve)
        self.plot_btn = QPushButton("Grafik alanı…")
        self.plot_btn.setToolTip("Eksenlerin çerçevesini (grafik alanını) elle çizin: bulunamadıysa ya da yanlışsa")
        self.plot_btn.clicked.connect(self.start_plot_area)
        self.dashed_check = QCheckBox("Kesikli")
        self.dashed_check.setToolTip("Eklenecek eğri kesikli/noktalı çizgiyse işaretleyin")
        self.rename_btn = QPushButton("Adı…")
        self.rename_btn.setToolTip("Eğrinin adını değiştir (F2)")
        self.rename_btn.clicked.connect(self.rename_curve)
        self.axis_btn = QPushButton("Eksen ⇄")
        self.axis_btn.setToolTip("Eğriyi sol/sağ Y eksenine ata")
        self.axis_btn.clicked.connect(self.switch_curve_axis)
        self.del_curve_btn = QPushButton("Eğriyi sil")
        self.del_curve_btn.clicked.connect(self.delete_curve)
        for w in (self.calib_btn, self.add_curve_btn, self.dashed_check, self.plot_btn):
            row.addWidget(w)
        row.addStretch(1)
        col.addLayout(row)
        row2 = QHBoxLayout()
        for w in (self.rename_btn, self.axis_btn, self.del_curve_btn):
            row2.addWidget(w)
        row2.addStretch(1)
        col.addLayout(row2)
        self._raster_bar = bar
        bar.setVisible(False)
        return bar

    def _build_edit_bar(self) -> QWidget:
        """Row of correction tools under the chart: add / delete / move points, undo, redo, reset."""
        bar = QWidget()
        row = QHBoxLayout(bar)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel("Düzelt:"))
        self.tool_group = QButtonGroup(self)
        self.tool_group.setExclusive(True)
        self.tool_buttons: dict[Tool, QPushButton] = {}
        for tool, text, tip in EDIT_TOOLS:
            b = QPushButton(text)
            b.setCheckable(True)
            b.setToolTip(tip)
            b.clicked.connect(lambda _checked=False, t=tool: self._set_edit_tool(t))
            self.tool_group.addButton(b)
            row.addWidget(b)
            self.tool_buttons[tool] = b
        self.tool_buttons[Tool.PAN].setChecked(True)
        row.addStretch(1)
        self.undo_btn = QPushButton("Geri al")
        self.redo_btn = QPushButton("Yinele")
        self.reset_btn = QPushButton("Özgün hale döndür")
        self.undo_btn.setToolTip("Son düzeltmeyi geri al (Ctrl+Z)")
        self.redo_btn.setToolTip("Geri alınanı yinele (Ctrl+Y)")
        self.reset_btn.setToolTip("Bu eğrideki tüm elle düzeltmeleri kaldır")
        self.undo_btn.clicked.connect(self.undo_edit)
        self.redo_btn.clicked.connect(self.redo_edit)
        self.reset_btn.clicked.connect(self.reset_edits)
        for b in (self.undo_btn, self.redo_btn, self.reset_btn):
            row.addWidget(b)
        self._refresh_edit_ui()
        return bar

    # ================================================================== pages
    def _show_page(self, index: int) -> None:
        self.stack.setCurrentIndex(index)
        self.act_back.setVisible(index == 2)
        if index != 2:
            self.hover_label.setText("")

    def show_gallery(self) -> None:
        self._show_page(1)
        self.statusBar().showMessage("Bir grafiğe tıklayın.")

    # ================================================================== opening
    def open_pdf_dialog(self) -> None:
        exts = " ".join(f"*{e}" for e in IMAGE_SUFFIXES)
        path, _ = QFileDialog.getOpenFileName(self, "PDF ya da grafik görseli aç", "",
                                              f"PDF ve görseller (*.pdf {exts});;PDF (*.pdf);;Görseller ({exts});;Tüm dosyalar (*)")
        if path:
            self.open_pdf(path)

    def open_pdf(self, path: str, blocking: bool = False) -> None:
        """Analyse a PDF (charts + exact curve data) or an image file (one picture chart).

        ``blocking=True`` skips the worker thread (tests, CLI).
        """
        if self._worker is not None:
            return
        if blocking:
            try:
                charts = analyze_file(path)
            except Exception as exc:
                self._analysis_failed(str(exc))
                return
            self._analysis_done(path, charts)
            return
        self._progress = QProgressDialog("Dosya analiz ediliyor…", None, 0, 0, self)
        self._progress.setWindowTitle(APP_NAME)
        self._progress.setWindowModality(Qt.WindowModality.WindowModal)
        self._progress.setMinimumDuration(0)
        self._progress.show()
        self._worker = AnalysisWorker(path)
        self._worker.progress.connect(self._on_progress)
        self._worker.done.connect(lambda charts, p=path: self._analysis_done(p, charts))
        self._worker.failed.connect(self._analysis_failed)
        self._worker.start()

    def _on_progress(self, i: int, n: int, text: str) -> None:
        if self._progress is not None:
            self._progress.setLabelText(text)

    def _finish_worker(self) -> None:
        if self._progress is not None:
            self._progress.close()
            self._progress = None
        if self._worker is not None:
            self._worker.wait()
            self._worker = None

    def _analysis_failed(self, message: str) -> None:
        self._finish_worker()
        QMessageBox.critical(self, APP_NAME, f"PDF açılamadı:\n{message}")

    def _analysis_done(self, path: str, charts: list[ChartData]) -> None:
        self._finish_worker()
        self.pdf_path = Path(path)
        self.charts = charts
        self._chart = self._curve = self._table = None
        if self._src is not None:
            self._src.close()
            self._src = None
        if not is_image_path(path):
            try:
                self._src = PdfSource(path)
            except Exception as exc:
                QMessageBox.critical(self, APP_NAME, f"PDF açılamadı:\n{exc}")
                return
        self.setWindowTitle(f"{APP_NAME} — {self.pdf_path.name}")
        try:
            self._pdf_hash = pdf_fingerprint(path) if any(c.kind == "raster" for c in charts) else None
        except OSError:
            self._pdf_hash = None
        self._fill_gallery()
        if not charts:
            QMessageBox.information(self, APP_NAME, "Bu PDF'te çizim çerçevesi olan bir grafik bulunamadı.\n"
                                    "Aranan: sayfaya çizilmiş (vektör) grafikler ve içinde çerçeveli bir grafik olan resimler. "
                                    "Tam sayfa taranmış PDF'lerde 'Araçlar → Görselden elle sayısallaştır' seçeneğini "
                                    "kullanabilirsiniz.")
            self._show_page(0)
            return
        self._show_page(1)
        n_curves = sum(len(c.curves) for c in charts)
        self.statusBar().showMessage(f"{len(charts)} grafik ve {n_curves} eğri bulundu. Bir grafiğe tıklayın.")

    def _fill_gallery(self) -> None:
        self.gallery.clear()
        self.gallery_title.setText(f"{self.pdf_path.name if self.pdf_path else ''} — {len(self.charts)} grafik")
        image_file = bool(self.charts) and self.charts[0].from_image
        for c in self.charts:
            if c.kind == "raster":
                need = "kalibrasyon kayıtlı" if self._saved(c) else "kalibrasyon gerekli"
                sub = "görsel grafik · " + (f"{len(c.curves)} eğri" if c.calibrated else need)
            else:
                sub = f"{len(c.curves)} eğri"
            where = "görsel dosya" if image_file else f"Sayfa {c.chart.page_index + 1}"
            item = QListWidgetItem(f"{c.title}\n{where} · {sub.replace('görsel grafik · ', '') if image_file else sub}")
            if c.thumbnail is not None:
                pm = ndarray_to_pixmap(c.thumbnail).scaledToWidth(THUMB_WIDTH, Qt.TransformationMode.SmoothTransformation)
                item.setIcon(QIcon(pm))
            item.setSizeHint(QSize(THUMB_WIDTH + 24, 300))
            item.setTextAlignment(Qt.AlignmentFlag.AlignHCenter)
            self.gallery.addItem(item)

    # drag & drop --------------------------------------------------------
    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls() and any(self._is_supported(u.toLocalFile()) for u in event.mimeData().urls()):
            event.acceptProposedAction()

    @staticmethod
    def _is_supported(path: str) -> bool:
        return path.lower().endswith(".pdf") or is_image_path(path)

    def dropEvent(self, event) -> None:
        for u in event.mimeData().urls():
            if self._is_supported(u.toLocalFile()):
                self.open_pdf(u.toLocalFile())
                break

    # ================================================================== chart detail
    def show_chart(self, index: int) -> None:
        if not (0 <= index < len(self.charts)):
            return
        chart = self.charts[index]
        if chart.kind != "raster" and self._src is None:
            return
        self._chart = chart
        self._cancel_calibration()
        if chart.kind == "raster" and not chart.calibrated and self._restore_calibration(chart):
            return                                    # apply_calibration has shown the chart again, calibrated
        if chart.kind == "raster":                    # a picture inside the PDF: show its own pixels
            self._rendered = chart.raster.rendered
        else:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                self._rendered = self._src.render(chart.chart.page_index, chart.chart.region, dpi=DETAIL_DPI)
            finally:
                QApplication.restoreOverrideCursor()
        self.view.set_image(self._rendered.image)
        self.view.set_plot_area(chart.raster.plot if chart.kind == "raster" else None)
        self.detail_title.setText(chart.title if chart.from_image else f"{chart.title}  ·  sayfa {chart.chart.page_index + 1}")
        notes = ("Grafik notları:\n" + "\n".join(f"• {n}" for n in chart.notes)) if chart.notes else ""
        self.notes_label.setText(notes)
        self._loading = True
        self._set_edit_tool(Tool.PAN)
        self.view.set_markers([])
        self.view.set_calibration_markers([])
        self.curve_list.clear()
        for cv in chart.curves:
            item = QListWidgetItem(color_icon(cv.color, cv.style == "dashed"), self._curve_item_text(chart, cv))
            self.curve_list.addItem(item)
        if not chart.curves:
            msg = ("Grafik görsel: önce «Kalibre et» ile eksenleri belirleyin."
                   if chart.kind == "raster" and not chart.calibrated
                   else "Bu grafikte eğri bulunamadı. «＋ Eğri ekle» ile görselden bir eğri seçebilirsiniz."
                   if chart.kind == "raster" else "Bu grafikte vektör eğri bulunamadı.")
            it = QListWidgetItem(msg)
            it.setFlags(Qt.ItemFlag.NoItemFlags)
            self.curve_list.addItem(it)
        self._loading = False
        self._curve = None
        self._refresh_raster_bar()
        self._show_page(2)
        if chart.curves:
            self.curve_list.setCurrentRow(0)
        else:
            self._curve_selected(-1)
        if chart.kind == "raster" and not chart.calibrated:
            self.statusBar().showMessage("Görsel grafik: «Kalibre et» ile X ve Y eksenlerinden ikişer işaret gösterin.")
        else:
            self.statusBar().showMessage("Bir eğri seçin: lookup tablosu yanda oluşur.")

    def _curve_selected(self, row: int) -> None:
        if self._loading:
            return
        chart = self._chart
        if chart is None or not (0 <= row < len(chart.curves)):
            self._curve = None
            self._table = None                       # nothing to print / export any more
            self.view.set_curves([])
            self.view.set_highlight(None, None)
            self.view.set_edit_points(None, None)
            self.view.set_markers([])
            self._set_table_enabled(False)
            self.table_model.set_data([], [])
            self.table_caption.setText("Bir eğri seçin." if chart and chart.curves else "")
            self.precision_label.setText("")
            self.query.set_curve(None)
            self._refresh_edit_ui()
            return
        self._curve = chart.curves[row]
        self._highlight_curve()
        data = self._curve.data
        if len(data):
            self.step_spin.blockSignals(True)
            self.step_spin.setValue(auto_step(float(data.x[0]), float(data.x[-1])))
            self.step_spin.blockSignals(False)
        yf = chart.y_fit_of(self._curve)
        self.query.set_curve(self._curve, chart.x_fit.title if chart.x_fit else "", yf.title if yf else "")
        self._rebuild_table()
        self._refresh_edit_ui()

    # ------------------------------------------------------------------ data <-> picture coordinates
    def _fits(self, cv: CurveResult | None = None):
        """``(x_fit, y_fit)`` of the current chart / curve, or ``None`` when a chart has no usable axes."""
        chart = self._chart
        cv = cv or self._curve
        if chart is None or cv is None or chart.x_fit is None:
            return None
        yf = chart.y_fit_of(cv)
        return None if yf is None else (chart.x_fit, yf)

    def _data_to_px(self, x, y, cv: CurveResult | None = None):
        fits = self._fits(cv)
        r = self._rendered
        if fits is None or r is None:
            return None
        xf, yf = fits
        return r.pt_to_px(xf.to_pos(np.asarray(x, float)), yf.to_pos(np.asarray(y, float)))

    def _px_to_data(self, x_px: float, y_px: float, cv: CurveResult | None = None):
        fits = self._fits(cv)
        r = self._rendered
        if fits is None or r is None:
            return None
        xf, yf = fits
        xp, yp = r.px_to_pt(x_px, y_px)
        return float(xf.to_value(xp)), float(yf.to_value(yp))

    def _click_uncertainty(self, x_px: float, y_px: float) -> tuple[float, float]:
        """+/- of a point placed by hand: half a pixel of the shown picture (never finer than the source)."""
        fits, r, chart = self._fits(), self._rendered, self._chart
        xf, yf = fits
        xp, yp = r.px_to_pt(x_px, y_px)
        floor = chart.pixel_pt if chart is not None else 0.06
        ux = float(xf.uncertainty(xp, pt=max(floor, 0.5 / r.sx)))
        uy = float(yf.uncertainty(yp, pt=max(floor, 0.5 / r.sy)))
        return ux, uy

    def _highlight_curve(self) -> None:
        cv, r = self._curve, self._rendered
        if cv is None or r is None:
            return
        px = self._data_to_px(cv.data.x, cv.data.y, cv)
        if px is None:
            xs, ys = (np.asarray(v) for v in r.pt_to_px(cv.x_pt, cv.y_pt))
        else:
            xs, ys = np.asarray(px[0]), np.asarray(px[1])
        order = np.argsort(xs, kind="stable")
        self.view.set_highlight(xs[order], ys[order])
        self.view.set_curves([(xs, ys, QColor(*cv.color), True)])
        self.view.set_edit_points(xs, ys)

    def _auto_step(self) -> None:
        if self._curve is not None and len(self._curve.data):
            d = self._curve.data
            self.step_spin.setValue(auto_step(float(d.x[0]), float(d.x[-1])))

    # ------------------------------------------------------------------ corrections (add / delete / move points)
    def _curve_item_text(self, chart: ChartData, cv: CurveResult) -> str:
        yf = chart.y_fit_of(cv)
        axis = f"{yf.title}" if yf and yf.title else "Y"
        edited = f" · düzeltildi ({cv.n_changes})" if cv.edited else ""
        side = f" · {cv.axis_note}" if chart.kind == "raster" and cv.axis_note else ""
        return f"{cv.label}\n{axis}{side} · {cv.style_label} · {cv.n_points} nokta{edited}"

    def _set_edit_tool(self, tool: Tool, hint: str = "") -> None:
        """Switch the picture's tool; the correction tools have buttons, the others (calibration, adding a curve) pass a ``hint``."""
        self.view.set_tool(tool)
        self.view.set_edit_mode(tool in (Tool.ADD_POINT, Tool.DELETE_POINT, Tool.ERASE, Tool.MOVE_POINT))
        if tool in self.tool_buttons:
            self.tool_buttons[tool].setChecked(True)
        else:
            self.tool_group.setExclusive(False)
            for b in self.tool_buttons.values():
                b.setChecked(False)
            self.tool_group.setExclusive(True)
        tip = hint or next((t for tl, _, t in EDIT_TOOLS if tl == tool), "")
        self.hint_label.setText(PAN_HINT if tool == Tool.PAN else f"{tip} · Esc: bitir")

    def _cancel_mode(self) -> None:
        """Esc: leave the current tool (and an unfinished calibration)."""
        if getattr(self, "_cal", None):
            self._cancel_calibration()
            self.statusBar().showMessage("Kalibrasyon iptal edildi.")
        self._set_edit_tool(Tool.PAN)

    def _need_curve_for_edit(self) -> bool:
        if self._curve is None or self._fits() is None:
            self.statusBar().showMessage("Önce düzeltilecek bir eğri seçin.")
            return False
        return True

    def _after_edit(self, message: str) -> None:
        """Refresh everything that shows the curve after a correction."""
        cv, chart = self._curve, self._chart
        if cv is None or chart is None:
            return
        self._highlight_curve()
        row = chart.curves.index(cv)
        item = self.curve_list.item(row)
        if item is not None:
            item.setText(self._curve_item_text(chart, cv))
        self._rebuild_table()
        self.query.refresh()
        self._refresh_edit_ui()
        self.statusBar().showMessage(message)

    def _refresh_edit_ui(self) -> None:
        cv = self._curve
        has = cv is not None and self._fits() is not None
        for b in self.tool_buttons.values():
            b.setEnabled(has or b is self.tool_buttons[Tool.PAN])
        undo, redo, reset = (bool(has and cv.can_undo), bool(has and cv.can_redo), bool(has and cv.edited))
        self.undo_btn.setEnabled(undo)
        self.redo_btn.setEnabled(redo)
        self.reset_btn.setEnabled(reset)
        if hasattr(self, "act_undo"):
            self.act_undo.setEnabled(undo)
            self.act_redo.setEnabled(redo)
            self.act_reset.setEnabled(reset)
        if hasattr(self, "_raster_bar"):
            self._refresh_raster_bar()

    def _on_click(self, x_px: float, y_px: float) -> None:
        tool = self.view.tool
        if tool == Tool.CALIB and self._cal:
            self._calibration_click(x_px, y_px)
        elif tool == Tool.PICK_CURVE and self._is_raster():
            self._add_curve_click(x_px, y_px)
        elif tool == Tool.ADD_POINT and self._need_curve_for_edit():
            xv, yv = self._px_to_data(x_px, y_px)
            ux, uy = self._click_uncertainty(x_px, y_px)
            self._curve.add_point(xv, yv, ux, uy)
            self._after_edit(f"Nokta eklendi: {fmt(xv, 6)}, {fmt(yv, 5)}")

    def _on_rect(self, rect) -> None:
        if self.view.tool == Tool.PLOT_AREA:
            self._plot_area_drawn(rect)
            return
        if self.view.tool != Tool.ERASE or not self._need_curve_for_edit():
            return
        a = self._px_to_data(rect.x0, rect.y0)
        b = self._px_to_data(rect.x1, rect.y1)
        n = self._curve.delete_in_box(a[0], b[0], a[1], b[1])
        if n:
            self._after_edit(f"{n} nokta silindi.")
        else:
            self.statusBar().showMessage("Kutunun içinde seçili eğriye ait nokta yok.")

    def _on_point_moved(self, index: int, x_px: float, y_px: float) -> None:
        if not self._need_curve_for_edit():
            return
        xv, yv = self._px_to_data(x_px, y_px)
        try:
            self._curve.move_point(index, xv, yv)
        except (IndexError, ValueError):
            return
        self._after_edit(f"Nokta taşındı: {fmt(xv, 6)}, {fmt(yv, 5)}")

    def _on_point_delete(self, index: int) -> None:
        if self._need_curve_for_edit() and self._curve.delete_points([index]):
            self._after_edit("Nokta silindi.")

    def undo_edit(self) -> None:
        if self._curve is not None and self._curve.undo():
            self._after_edit("Geri alındı.")

    def redo_edit(self) -> None:
        if self._curve is not None and self._curve.redo():
            self._after_edit("Yinelendi.")

    def reset_edits(self) -> None:
        if self._curve is None or not self._curve.edited:
            return
        if QMessageBox.question(self, APP_NAME, f"'{self._curve.label}' eğrisindeki tüm elle düzeltmeler kaldırılsın mı?\n"
                                "(Geri al ile vazgeçebilirsiniz.)") != QMessageBox.StandardButton.Yes:
            return
        if self._curve.reset_edits():
            self._after_edit("Eğri özgün haline döndürüldü.")

    def _show_markers(self, points) -> None:
        """Cross-hairs at the answers of the value query."""
        pts = []
        for x, y in points:
            px = self._data_to_px(x, y)
            if px is not None:
                pts.append((float(px[0]), float(px[1])))
        self.view.set_markers(pts)

    # ================================================================== picture charts (an image inside the PDF)
    def _is_raster(self) -> bool:
        return self._chart is not None and self._chart.kind == "raster"

    def _refresh_raster_bar(self) -> None:
        chart = self._chart
        on = chart is not None and chart.kind == "raster"
        self._raster_bar.setVisible(on)
        if not on:
            return
        cal = chart.calibrated
        saved = self._saved(chart)
        if not cal:
            self.raster_status.setStyleSheet("color: #b9770e;")
            if not chart.raster.frame_found:
                self.raster_status.setText("Grafiğin çerçevesi otomatik bulunamadı: «Grafik alanı…» ile eksenlerin çerçevesini "
                                           "çizin, sonra «Kalibre et».")
            else:
                self.raster_status.setText("Bu grafik bir görsel: değer okumak için önce eksenleri kalibre edin.")
        elif saved is not None:
            self.raster_status.setStyleSheet("color: gray;")
            what = "bu görsel" if chart.from_image else "bu PDF"
            self.raster_status.setText(f"Kalibrasyon {what} için kayıtlı ({saved.saved}): bu grafik bir daha sorulmaz.")
        else:
            self.raster_status.setStyleSheet("color: gray;")
            self.raster_status.setText("Kalibrasyon kayıtlı değil (bu oturumda geçerli).")
        self.raster_status.setVisible(True)
        self.act_forget_cal.setEnabled(saved is not None)
        self.calib_btn.setText("Kalibre et" if not cal else "Yeniden kalibre et")
        self.add_curve_btn.setEnabled(cal)
        self.plot_btn.setEnabled(True)
        self.dashed_check.setEnabled(cal)
        has = self._curve is not None
        self.rename_btn.setEnabled(has)
        self.del_curve_btn.setEnabled(has)
        self.axis_btn.setEnabled(has and len(chart.y_fits) > 1)

    # ---- calibration: two marks per axis, then their values
    def start_calibration(self) -> None:
        if not self._is_raster():
            return
        chart = self._chart
        if chart.calibrated and any(cv.edited for cv in chart.curves):
            if QMessageBox.question(self, APP_NAME, "Yeniden kalibrasyon eğrileri yeniden hesaplar; elle yaptığınız "
                                    "düzeltmeler silinir. Devam edilsin mi?") != QMessageBox.StandardButton.Yes:
                return
        steps = [("x", "X ekseninde 1. işarete (tick) tıklayın", "X1"), ("x", "X ekseninde 2. işarete tıklayın", "X2"),
                 ("y", "Y ekseninde 1. işarete tıklayın", "Y1"), ("y", "Y ekseninde 2. işarete tıklayın", "Y2")]
        self._cal = {"steps": steps, "clicks": [], "asked": False}
        self.view.set_calibration_markers([])
        self._set_edit_tool(Tool.CALIB, "Eksen üzerindeki işaretlere (tick) tıklayın; tıklama yakındaki işarete yapışır")
        self.statusBar().showMessage(steps[0][1] + " (birbirinden uzak iki işaret seçin).")

    def _cancel_calibration(self) -> None:
        self._cal = None
        if hasattr(self, "view"):
            self.view.set_calibration_markers([])

    def _calibration_click(self, x: float, y: float) -> None:
        cal, rc = self._cal, self._chart.raster
        kind, _prompt, name = cal["steps"][len(cal["clicks"])]
        ticks = rc.ticks.x if kind == "x" else (rc.ticks.y_right if name.startswith("R") else rc.ticks.y_left)
        cal["clicks"].append(rcm.snap(x if kind == "x" else y, ticks))
        self.view.set_calibration_markers([(cal["steps"][i][0], p, cal["steps"][i][2]) for i, p in enumerate(cal["clicks"])])
        if len(cal["clicks"]) == 4 and not cal["asked"]:
            cal["asked"] = True
            if QMessageBox.question(self, APP_NAME, "Grafikte sağ tarafta, farklı ölçekli ikinci bir Y ekseni var mı?") \
                    == QMessageBox.StandardButton.Yes:
                cal["steps"] += [("y", "Sağ Y ekseninde 1. işarete tıklayın", "R1"), ("y", "Sağ Y ekseninde 2. işarete tıklayın", "R2")]
        if len(cal["clicks"]) == len(cal["steps"]):
            self._finish_calibration()
        else:
            self.statusBar().showMessage(cal["steps"][len(cal["clicks"])][1] + ".")

    def _ask_axis_values(self, x_px, y_px, y2_px):
        """Dialog for the values at the clicked marks (a seam for tests)."""
        return CalibrationDialog.ask(self, x_px, y_px, y2_px)

    def _finish_calibration(self) -> None:
        c = self._cal["clicks"]
        answer = self._ask_axis_values((c[0], c[1]), (c[2], c[3]), (c[4], c[5]) if len(c) > 4 else None)
        if answer is None:
            self._cancel_calibration()
            self._set_edit_tool(Tool.PAN)
            self.statusBar().showMessage("Kalibrasyon iptal edildi.")
            return
        self.apply_calibration(*answer)

    # ---- the plot area of a picture chart (auto-detected frame, or drawn by hand)
    def start_plot_area(self) -> None:
        if self._is_raster():
            self._set_edit_tool(Tool.PLOT_AREA, "Eksenlerin oluşturduğu dikdörtgenin (çerçevenin) üstüne bir kutu sürükleyin")
            self.statusBar().showMessage("Grafik alanı: çerçevenin sol-üst köşesinden sağ-alt köşesine sürükleyin.")

    def _plot_area_drawn(self, rect) -> None:
        chart = self._chart
        if not self._is_raster() or rect.width < 40 or rect.height < 30:
            self.statusBar().showMessage("Grafik alanı çok küçük: çerçevenin tamamını kapsayan bir kutu çizin.")
            return
        rc = chart.raster
        if chart.calibrated and any(cv.edited for cv in chart.curves):
            if QMessageBox.question(self, APP_NAME, "Grafik alanı değişince eğriler yeniden okunur; elle yaptığınız "
                                    "düzeltmeler silinir. Devam edilsin mi?") != QMessageBox.StandardButton.Yes:
                return
        rcm.set_plot_area(rc, rect)
        self.view.set_plot_area(rc.plot)
        self._set_edit_tool(Tool.PAN)
        if chart.calibrated:                                # the axes stay; only the search area for the curves changed
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                rc.traces = rcm.discover_curves(rc)
                for tr in rc.traces:
                    tr.axis = min(rcm.guess_axis(rc, tr.rgb, len(chart.y_fits) > 1), len(chart.y_fits) - 1)
                rcm.rebuild_curves(chart)
            finally:
                QApplication.restoreOverrideCursor()
            self._reload_curve_list(0)
            self._fill_gallery()
            saved = self._saved(chart)
            if saved is not None and self._pdf_hash is not None:                 # keep the remembered axes, add the new area
                self._remember_calibration(chart, saved.x, saved.y, saved.y2)
            self.statusBar().showMessage(f"Grafik alanı ayarlandı: {len(chart.curves)} eğri okundu.")
        else:
            self.statusBar().showMessage("Grafik alanı ayarlandı; şimdi «Kalibre et» ile eksenleri gösterin.")
        self._refresh_raster_bar()

    # ---- remembered calibrations (per PDF content + chart): a chart is calibrated once, ever
    def _chart_key(self, chart: ChartData) -> str:
        rc = chart.raster
        return chart_key(chart.chart.page_index, rc.bbox, rc.image.shape)

    def _saved(self, chart: ChartData):
        """``(x, y, y2, saved_at)`` remembered for this chart of this PDF, or ``None``."""
        if self._pdf_hash is None or chart.kind != "raster":
            return None
        return self.store.get(self._pdf_hash, self._chart_key(chart))

    def _remember_calibration(self, chart: ChartData, x, y, y2) -> bool:
        if self._pdf_hash is None or self.pdf_path is None:
            return False
        rc = chart.raster
        plot = (rc.plot.x0, rc.plot.y0, rc.plot.x1, rc.plot.y1) if rc.plot_manual else None
        return self.store.save(self._pdf_hash, self.pdf_path.name, self._chart_key(chart), x, y, y2, plot)

    def _restore_calibration(self, chart: ChartData) -> bool:
        """Apply the remembered calibration of ``chart`` (``self._chart``); ``False`` when there is none / it is unusable."""
        saved = self._saved(chart)
        if saved is None:
            return False
        if saved.plot is not None:                           # the user had outlined the plot area: use it again
            rcm.set_plot_area(chart.raster, rcm.Rect(*saved.plot))
        return self.apply_calibration(saved.x, saved.y, saved.y2, remember=False, restored_at=saved.saved or "?")

    def forget_calibration(self) -> None:
        """Delete the remembered calibration of the shown chart (the chart stays calibrated until it is closed)."""
        chart = self._chart
        if chart is None or chart.kind != "raster" or self._pdf_hash is None:
            return
        if self.store.forget(self._pdf_hash, self._chart_key(chart)):
            self._fill_gallery()
            self._refresh_raster_bar()
            self.statusBar().showMessage("Kayıtlı kalibrasyon silindi; bu grafik bir sonraki açılışta yeniden kalibre edilecek.")
        else:
            self.statusBar().showMessage("Bu grafik için kayıtlı bir kalibrasyon yok.")

    def apply_calibration(self, x: rcm.AxisPoints, y: rcm.AxisPoints, y2: rcm.AxisPoints | None = None, *,
                          remember: bool = True, restored_at: str = "") -> bool:
        """Set the axes of the shown picture chart, read its curves (first time) and refresh the screen.

        ``remember`` stores the calibration for the next time this PDF is opened; ``restored_at`` marks a calibration
        that came from that store (nothing is asked or stored again; an unusable one is dropped quietly).
        """
        if not self._is_raster():
            return False
        chart = self._chart
        rc = chart.raster
        self._cancel_calibration()
        first = not rc.traces and not chart.curves
        try:
            rcm.calibrate(chart, x, y, y2)
        except ValueError as exc:
            if restored_at:
                if self._pdf_hash is not None:
                    self.store.forget(self._pdf_hash, self._chart_key(chart))
                return False
            QMessageBox.warning(self, APP_NAME, str(exc))
            return False
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            if first:
                rc.traces = rcm.discover_curves(rc)
                for tr in rc.traces:
                    tr.axis = rcm.guess_axis(rc, tr.rgb, y2 is not None)
            for tr in rc.traces:
                tr.axis = min(tr.axis, len(chart.y_fits) - 1)
            rcm.rebuild_curves(chart)
        finally:
            QApplication.restoreOverrideCursor()
        kept = remember and self._remember_calibration(chart, x, y, y2)
        self._fill_gallery()
        self.show_chart(self.charts.index(chart))
        if restored_at:
            msg = (f"Kayıtlı kalibrasyon uygulandı ({restored_at}): {len(chart.curves)} eğri okundu. "
                   "Yeniden kalibre etmeniz gerekmez.")
        else:
            msg = (f"Kalibre edildi: {len(chart.curves)} eğri okundu. "
                   + ("Kalibrasyon kaydedildi: bu PDF'in bu grafiği bir daha sorulmaz. " if kept else
                      "(Kalibrasyon kaydedilemedi.) " if remember else "")
                   + "Eksik ya da fazla eğri için «＋ Eğri ekle» / «Eğriyi sil», hatalı noktalar için «Düzelt» araçlarını kullanın.")
        self.statusBar().showMessage(msg)
        return True

    # ---- adding, renaming, removing curves of a picture chart
    def _reload_curve_list(self, select: int | None = None) -> None:
        chart = self._chart
        self._loading = True
        self.curve_list.clear()
        for cv in chart.curves:
            self.curve_list.addItem(QListWidgetItem(color_icon(cv.color, cv.style == "dashed"), self._curve_item_text(chart, cv)))
        if not chart.curves:
            it = QListWidgetItem("Bu grafikte eğri yok. «＋ Eğri ekle» ile görselden bir eğri seçin.")
            it.setFlags(Qt.ItemFlag.NoItemFlags)
            self.curve_list.addItem(it)
        self._loading = False
        self._curve = None
        if chart.curves:
            self.curve_list.setCurrentRow(0 if select is None else min(select, len(chart.curves) - 1))
        else:
            self._curve_selected(-1)

    def start_add_curve(self) -> None:
        if self._is_raster() and self._chart.calibrated:
            self._set_edit_tool(Tool.PICK_CURVE, "Görselde eklemek istediğiniz eğrinin kalın çizgisinin üstüne tıklayın")
            self.statusBar().showMessage("Eğrinin üstüne tıklayın (kesikli çizgi ise önce «Kesikli»yi işaretleyin).")

    def _add_curve_click(self, x: float, y: float) -> None:
        chart = self._chart
        rc = chart.raster
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            tr = rcm.trace_curve_at(rc, x, y, dashed=self.dashed_check.isChecked())
        finally:
            QApplication.restoreOverrideCursor()
        if tr is None:
            self.statusBar().showMessage("Burada izlenecek bir eğri bulunamadı: eğrinin tam üstüne (çizginin kalın yerine) tıklayın.")
            return
        tr.label = f"Eğri {len(rc.traces) + 1} ({rcm.colour_name(tr.rgb)})"
        tr.axis = rcm.guess_axis(rc, tr.rgb, len(chart.y_fits) > 1)
        rc.traces.append(tr)
        chart.curves.append(rcm.curve_from_trace(len(chart.curves), tr, chart, rc))
        self._reload_curve_list(len(chart.curves) - 1)
        self._fill_gallery()
        self._set_edit_tool(Tool.PAN)
        self.statusBar().showMessage(f"Eğri eklendi: {tr.label} ({len(tr.x_px)} nokta).")

    def _selected_row(self) -> int | None:
        row = self.curve_list.currentRow()
        chart = self._chart
        return row if chart is not None and 0 <= row < len(chart.curves) else None

    def rename_curve(self) -> None:
        row = self._selected_row()
        if row is None:
            return
        cv = self._chart.curves[row]
        name, ok = QInputDialog.getText(self, APP_NAME, "Eğrinin adı:", text=cv.label)
        name = name.strip()
        if not ok or not name:
            return
        cv.label = name
        if self._is_raster():
            self._chart.raster.traces[row].label = name
        self.curve_list.item(row).setText(self._curve_item_text(self._chart, cv))
        self._rebuild_table()

    def delete_curve(self) -> None:
        row = self._selected_row()
        if row is None or not self._is_raster():
            return
        chart = self._chart
        if QMessageBox.question(self, APP_NAME, f"'{chart.curves[row].label}' eğrisi listeden silinsin mi?") \
                != QMessageBox.StandardButton.Yes:
            return
        del chart.curves[row]
        del chart.raster.traces[row]
        for i, cv in enumerate(chart.curves):
            cv.index = i
        self._reload_curve_list(row)
        self._fill_gallery()

    def switch_curve_axis(self) -> None:
        row = self._selected_row()
        chart = self._chart
        if row is None or not self._is_raster() or len(chart.y_fits) < 2:
            return
        cv = chart.curves[row]
        if cv.edited and QMessageBox.question(self, APP_NAME, "Eksen değişince eğri yeniden hesaplanır; elle düzeltmeleri "
                                              "silinir. Devam edilsin mi?") != QMessageBox.StandardButton.Yes:
            return
        rc = chart.raster
        rc.traces[row].axis = 1 - min(rc.traces[row].axis, 1)
        chart.curves[row] = rcm.curve_from_trace(row, rc.traces[row], chart, rc)
        self._reload_curve_list(row)

    def _step_changed(self, _value: float) -> None:
        self._rebuild_table()

    def _rebuild_table(self) -> None:
        if self._chart is None or self._curve is None:
            return
        step = self.step_spin.value()
        table = build_lookup(self._chart, self._curve, step if step > 0 else None,
                             source=self.pdf_path.name if self.pdf_path else "")
        self._table = table
        self.table_model.set_data([table.x_header, table.y_header], [[a for a, _ in table.rows()], [b for _, b in table.rows()]])
        hdr = self.table.horizontalHeader()                       # the X column is as wide as its (possibly long) name
        self.table.setColumnWidth(0, max(110, hdr.fontMetrics().horizontalAdvance(table.x_header) + 34))
        yf = self._chart.y_fit_of(self._curve)
        self.table_caption.setText(f"<b>{self._curve.label}</b> — {table.y_header} / {table.x_header} · {len(table)} satır")
        self.precision_label.setText(f"Değerler {table.method_text}. Olası belirsizlik: "
                                     + uncertainty_text(table)
                                     + (" · kesikli çizgide boşluklar doğrusal interpolasyonla doldurulur"
                                        if self._curve.style == "dashed" else ""))
        self._set_table_enabled(len(table) > 0)

    def _set_table_enabled(self, on: bool) -> None:
        for w in (self.csv_btn, self.pdf_btn, self.print_btn, self.step_spin):
            w.setEnabled(on)

    def _on_cursor(self, x_px: float, y_px: float) -> None:
        r, chart = self._rendered, self._chart
        if math.isnan(x_px) or r is None or chart is None or chart.x_fit is None:
            self.hover_label.setText("")
            return
        xp, yp = r.px_to_pt(x_px, y_px)
        xv = float(chart.x_fit.to_value(xp))
        cv = self._curve
        yf = chart.y_fit_of(cv) if cv is not None else (chart.y_fits[0] if chart.y_fits else None)
        text = f"{chart.x_fit.title or 'X'} = {fmt(xv, 6)}"
        if yf is not None:
            text += f"   {yf.title or 'Y'} = {fmt(float(yf.to_value(yp)), 5)}"
        if cv is not None and len(cv.data) and cv.data.x[0] <= xv <= cv.data.x[-1]:
            text += f"   |   {cv.label}: {fmt(float(np.interp(xv, cv.data.x, cv.data.y)), 6)}"
        self.hover_label.setText(text)

    # ================================================================== output
    def _report_html(self) -> str | None:
        if self._table is None:
            return None
        return to_html(self._table, columns=report_columns(len(self._table)))

    def _default_name(self, ext: str) -> str:
        base = f"{self._chart.title if self._chart else 'tablo'}_{self._curve.label if self._curve else ''}"
        safe = "".join(ch if ch.isalnum() or ch in "-_ ." else "_" for ch in base).strip(" ._") or "lookup"
        return f"{safe}.{ext}"

    def export_csv(self) -> None:
        if self._table is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "CSV kaydet", self._default_name("csv"), "CSV (*.csv)")
        if path:
            write_csv(self._table, path)
            self.statusBar().showMessage(f"Kaydedildi: {path}")

    def save_pdf(self) -> None:
        html_text = self._report_html()
        if html_text is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "PDF kaydet", self._default_name("pdf"), "PDF (*.pdf)")
        if path:
            printing.save_pdf(html_text, path)
            self.statusBar().showMessage(f"Kaydedildi: {path}")

    def print_table(self) -> None:
        html_text = self._report_html()
        if html_text is None:
            return
        if printing.print_with_dialog(html_text, self):
            self.statusBar().showMessage("Yazıcıya gönderildi.")

    # ================================================================== misc
    def open_manual_tool(self) -> None:
        """The older image-based digitiser (manual calibration) for scanned charts."""
        from ui.main_window import MainWindow

        win = MainWindow()
        win.show()
        self._legacy.append(win)

    def show_help(self) -> None:
        QMessageBox.information(self, "Kullanım", (
            "1. PDF ya da grafik görseli (PNG, JPG, BMP, TIFF, WebP) açın (Ctrl+O ya da sürükle-bırak).\n"
            "    PDF'teki tüm grafikler otomatik bulunur; sayfaya çizilmiş\n"
            "    (vektör) grafiklerin değerleri tam hassasiyetle okunur.\n"
            "2. Galeriden bir grafiğe tıklayın.\n"
            "3. Sağdaki listede eğrilerin ne anlama geldiği yazar (legend + eksen). Bir eğri seçin:\n"
            "    lookup tablosu (X → Y) oluşur; adımı değiştirebilirsiniz.\n"
            "4. Değer sorgula: X yazın, Y'yi; Y yazın, X'i (varsa tüm çözümleri) görün. Grafiğe artı işaretiyle gösterilir.\n"
            "5. Hatalı okuma varsa 'Düzelt' araçları: nokta ekle, sil, kutuyla sil, taşı (Ctrl+Z / Ctrl+Y).\n"
            "6. CSV, PDF kaydet veya Yazdır ile çıktı alın.\n\n"
            "Grafik PDF'e resim olarak yapıştırılmışsa ('görsel grafik'): 'Kalibre et' ile X ve Y eksenlerinden ikişer\n"
            "işaret (tick) gösterip değerlerini yazın; eğriler renklerinden otomatik okunur. Eksik eğri için\n"
            "'＋ Eğri ekle' ile eğrinin üstüne tıklayın. Değerler resmin piksel çözünürlüğüyle sınırlıdır.\n"
            "Kalibrasyon kaydedilir: aynı PDF bir daha açıldığında o grafik sorulmadan hazır gelir.\n\n"
            "Görsel dosyalarda (PNG/JPG/BMP…) grafiğin çerçevesi otomatik aranır; bulunamazsa 'Grafik alanı…' ile\n"
            "çizin. Sonrası resim grafiklerle aynıdır.\n\n"
            "Her şeyi elle yapmak için: Araçlar → Görselden elle sayısallaştır (gelişmiş)."))

    def closeEvent(self, event) -> None:
        if self._worker is not None:
            self._worker.wait()
        if self._src is not None:
            self._src.close()
        for w in self._legacy:
            w.close()
        super().closeEvent(event)
