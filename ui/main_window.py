"""Main window: a thin Turkish-language shell over :class:`core.project.Project`."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QSettings, Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QColor, QIcon, QKeySequence, QPixmap
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDockWidget, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QMainWindow, QMessageBox, QPushButton, QRadioButton, QScrollArea,
                               QSlider, QSpinBox, QTableView, QToolBar, QVBoxLayout, QWidget)

from core.extraction import pick_reference_color
from core.imageio import IMAGE_SUFFIXES, load_image
from core.models import Rect
from core.plotarea import detect_plot_area
from core.pdf_source import ChartInfo, PdfSource, RenderedRegion
from core.project import AxisSetup, Project, nice_step
from ui.image_view import ImageView, Tool
from ui.pdf_dialog import PdfChartDialog
from ui.table_model import ArrayTableModel, fmt

APP_NAME = "Curvelens"

TOOL_HINTS = {
    Tool.PAN: "Kaydırmak için sürükleyin, yakınlaştırmak için fare tekerleğini kullanın.",
    Tool.PICK_COLOR: "Eğrinin üzerine tıklayın: tıkladığınız pikselin rengi referans alınır. (Esc: bitir)",
    Tool.PLOT_AREA: "Çizim alanını sürükleyerek seçin (eksen çerçevesinin içi).",
    Tool.EXCLUDE: "Yok sayılacak alanı (legend kutusu, metin bloğu) sürükleyerek seçin. (Esc: bitir)",
    Tool.ERASE: "Silinecek noktaların üzerine bir kutu çizin. (Esc: bitir)",
    Tool.ADD_POINT: "Seçili eğriye nokta eklemek için tıklayın. (Esc: bitir)",
}
CALIB_HINT = "{axis} ekseni, nokta {n}: eksen üzerinde değerini bildiğiniz bir işarete (tick/ızgara) tıklayın."


def color_icon(rgb: tuple[int, int, int], size: int = 14) -> QIcon:
    pm = QPixmap(size, size)
    pm.fill(QColor(*rgb))
    return QIcon(pm)


def parse_float(text: str) -> float | None:
    text = text.strip().replace(",", ".")
    if not text:
        return None
    try:
        v = float(text)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


class AxisBox(QGroupBox):
    """Two reference points (pixel picked with the mouse + typed value), scale and label of one axis."""

    def __init__(self, axis: str, window: "MainWindow"):
        super().__init__(f"{axis.upper()} ekseni")
        self.axis, self.win = axis, window
        self._loading = False
        grid = QGridLayout()
        self.pick_btns, self.px_labels, self.value_edits = [], [], []
        for i in range(2):
            btn = QPushButton(f"Nokta {i + 1} seç")
            btn.setToolTip("Görselde eksen üzerinde değerini bildiğiniz bir noktaya tıklayın.")
            btn.clicked.connect(lambda _=False, k=i: self.win.begin_calibration(self.axis, k))
            px = QLabel("—")
            px.setMinimumWidth(62)
            ed = QLineEdit()
            ed.setPlaceholderText("değer")
            ed.setMaximumWidth(90)
            ed.editingFinished.connect(self._changed)
            grid.addWidget(btn, i, 0)
            grid.addWidget(px, i, 1)
            grid.addWidget(ed, i, 2)
            self.pick_btns.append(btn)
            self.px_labels.append(px)
            self.value_edits.append(ed)
        self.log_check = QCheckBox("Logaritmik eksen")
        self.log_check.toggled.connect(self._changed)
        self.label_edit = QLineEdit()
        self.label_edit.setPlaceholderText("Ad (ör. Zaman)")
        self.unit_edit = QLineEdit()
        self.unit_edit.setPlaceholderText("Birim (ör. s)")
        self.unit_edit.setMaximumWidth(90)
        self.label_edit.editingFinished.connect(self._changed)
        self.unit_edit.editingFinished.connect(self._changed)
        names = QHBoxLayout()
        names.addWidget(self.label_edit, 2)
        names.addWidget(self.unit_edit, 1)
        lay = QVBoxLayout(self)
        lay.addLayout(grid)
        lay.addWidget(self.log_check)
        lay.addLayout(names)

    @property
    def setup(self) -> AxisSetup:
        return self.win.project.x_axis if self.axis == "x" else self.win.project.y_axis

    def load(self) -> None:
        """Show the project's state in the widgets (without echoing it back)."""
        self._loading = True
        s = self.setup
        for i in range(2):
            self.px_labels[i].setText("—" if s.pixel[i] is None else f"{s.pixel[i]:.2f}")
            self.value_edits[i].setText("" if s.value[i] is None else fmt(s.value[i], 10))
        self.log_check.setChecked(s.log)
        self.label_edit.setText(s.label)
        self.unit_edit.setText(s.unit)
        self._loading = False

    def _changed(self, *_) -> None:
        if self._loading:
            return
        s = self.setup
        for i in range(2):
            s.value[i] = parse_float(self.value_edits[i].text())
        s.log = self.log_check.isChecked()
        s.label = self.label_edit.text().strip()
        s.unit = self.unit_edit.text().strip()
        self.win.on_calibration_changed()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.project = Project()
        self._pdf: PdfSource | None = None
        self._chart: ChartInfo | None = None
        self._rendered: RenderedRegion | None = None
        self._calib_slot: tuple[str, int] | None = None
        self._loading = False
        self._selected = -1
        self._extract_timer = QTimer(self)
        self._extract_timer.setSingleShot(True)
        self._extract_timer.setInterval(250)
        self._extract_timer.timeout.connect(self._extract_selected)

        self.setWindowTitle(APP_NAME)
        self.resize(1400, 860)
        self.view = ImageView()
        self.setCentralWidget(self.view)
        self._build_actions()
        self._build_panel()
        self._build_table_dock()
        self.statusBar().showMessage("Bir görsel veya PDF açın (Ctrl+O).")
        self.coord_label = QLabel("")
        self.statusBar().addPermanentWidget(self.coord_label)

        self.view.clicked.connect(self._on_click)
        self.view.rectSelected.connect(self._on_rect)
        self.view.cursorMoved.connect(self._on_cursor)
        self.view.cancelled.connect(lambda: self.set_tool(Tool.PAN))
        self.set_tool(Tool.PAN)
        self._refresh_all()

    # ================================================================== building
    def _build_actions(self) -> None:
        self.act_open = QAction("&Aç…", self, shortcut=QKeySequence.StandardKey.Open, triggered=self.open_file)
        self.act_export = QAction("CSV &dışa aktar…", self, shortcut="Ctrl+E", triggered=self.export_csv)
        self.act_quit = QAction("Çı&kış", self, shortcut="Ctrl+Q", triggered=self.close)
        self.act_undo = QAction("&Geri al", self, shortcut=QKeySequence.StandardKey.Undo, triggered=self.undo)
        self.act_redo = QAction("&Yinele", self, shortcut="Ctrl+Y", triggered=self.redo)
        self.act_zoom_in = QAction("Yakınlaştır", self, shortcut="Ctrl+=", triggered=lambda: self.view.zoom_by(1.25))
        self.act_zoom_out = QAction("Uzaklaştır", self, shortcut="Ctrl+-", triggered=lambda: self.view.zoom_by(0.8))
        self.act_fit = QAction("Sığdır", self, shortcut="Ctrl+0", triggered=self.view.fit_in_view)
        self.act_points = QAction("Çıkarılan noktaları göster", self, checkable=True, checked=True,
                                  triggered=lambda on: self.view.set_points_visible(on))
        self.act_help = QAction("Kullanım ipuçları", self, shortcut="F1", triggered=self.show_help)

        mb = self.menuBar()
        m = mb.addMenu("&Dosya")
        m.addActions([self.act_open, self.act_export])
        m.addSeparator()
        m.addAction(self.act_quit)
        m = mb.addMenu("&Düzen")
        m.addActions([self.act_undo, self.act_redo])
        m = mb.addMenu("&Görünüm")
        m.addActions([self.act_zoom_in, self.act_zoom_out, self.act_fit])
        m.addSeparator()
        m.addAction(self.act_points)
        m = mb.addMenu("&Yardım")
        m.addAction(self.act_help)

        tb = QToolBar("Araçlar")
        tb.setMovable(False)
        self.addToolBar(tb)
        self.tool_group = QActionGroup(self)
        self.tool_group.setExclusionPolicy(QActionGroup.ExclusionPolicy.ExclusiveOptional)
        self.tool_actions: dict[Tool, QAction] = {}
        tb.addAction(self.act_open)
        tb.addSeparator()
        for tool, text, tip in (
            (Tool.PAN, "Kaydır", "Görseli sürükleyerek kaydırın"),
            (Tool.PLOT_AREA, "Çizim alanı", "Eğrilerin aranacağı dikdörtgeni seçin; dışı yok sayılır"),
            (Tool.EXCLUDE, "Hariç tut", "Legend, metin gibi yok sayılacak dikdörtgenler ekleyin"),
            (Tool.PICK_COLOR, "Eğri ekle", "Eğriye tıklayın: rengi referans alınır"),
            (Tool.ERASE, "Noktaları sil", "Kutu çizerek noktaları silin"),
            (Tool.ADD_POINT, "Nokta ekle", "Seçili eğriye tek tek nokta ekleyin"),
        ):
            a = QAction(text, self, checkable=True)
            a.setToolTip(tip)
            a.triggered.connect(lambda checked, t=tool: self.set_tool(t if checked else Tool.PAN))
            self.tool_group.addAction(a)
            self.tool_actions[tool] = a
            tb.addAction(a)
        tb.addSeparator()
        tb.addActions([self.act_undo, self.act_redo])
        tb.addSeparator()
        tb.addActions([self.act_zoom_in, self.act_zoom_out, self.act_fit])
        tb.addSeparator()
        tb.addAction(self.act_export)

    def _group(self, title: str) -> tuple[QGroupBox, QVBoxLayout]:
        box = QGroupBox(title)
        lay = QVBoxLayout(box)
        return box, lay

    def _build_panel(self) -> None:
        panel = QWidget()
        col = QVBoxLayout(panel)

        # --- source ----------------------------------------------------------------
        box, lay = self._group("Kaynak")
        self.source_label = QLabel("Dosya açılmadı")
        self.source_label.setWordWrap(True)
        self.chart_btn = QPushButton("PDF'te başka grafik seç…")
        self.chart_btn.clicked.connect(self.choose_pdf_chart)
        self.y_axis_combo = QComboBox()
        self.y_axis_combo.setToolTip("Grafiğin birden fazla Y ekseni varsa hangisinin kullanılacağı")
        self.y_axis_combo.currentIndexChanged.connect(self._pdf_axis_changed)
        self.pdf_calib_btn = QPushButton("Kalibrasyonu PDF'ten öner")
        self.pdf_calib_btn.setToolTip("Eksen değerlerini PDF'in metin katmanından (tick etiketleri) okur.")
        self.pdf_calib_btn.clicked.connect(lambda: self._apply_pdf_calibration(0, max(0, self.y_axis_combo.currentIndex())))
        lay.addWidget(self.source_label)
        lay.addWidget(self.chart_btn)
        lay.addWidget(self.y_axis_combo)
        lay.addWidget(self.pdf_calib_btn)
        col.addWidget(box)

        # --- calibration -----------------------------------------------------------
        self.axis_boxes = {"x": AxisBox("x", self), "y": AxisBox("y", self)}
        col.addWidget(self.axis_boxes["x"])
        col.addWidget(self.axis_boxes["y"])
        self.calib_status = QLabel("")
        self.calib_status.setWordWrap(True)
        col.addWidget(self.calib_status)

        # --- areas -----------------------------------------------------------------
        box, lay = self._group("Çizim alanı")
        self.area_label = QLabel("")
        self.area_label.setWordWrap(True)
        row = QHBoxLayout()
        b1 = QPushButton("Alanı seç")
        b1.clicked.connect(lambda: self.set_tool(Tool.PLOT_AREA))
        b2 = QPushButton("Hariç tutma ekle")
        b2.clicked.connect(lambda: self.set_tool(Tool.EXCLUDE))
        row.addWidget(b1)
        row.addWidget(b2)
        row2 = QHBoxLayout()
        b3 = QPushButton("Alanı sıfırla")
        b3.clicked.connect(self.reset_plot_area)
        b4 = QPushButton("Hariç tutmaları temizle")
        b4.clicked.connect(self.clear_exclusions)
        row2.addWidget(b3)
        row2.addWidget(b4)
        lay.addWidget(self.area_label)
        lay.addLayout(row)
        lay.addLayout(row2)
        col.addWidget(box)

        # --- curves ----------------------------------------------------------------
        box, lay = self._group("Eğriler")
        self.curve_list = QListWidget()
        self.curve_list.setMinimumHeight(110)
        self.curve_list.currentRowChanged.connect(self._curve_selected)
        row = QHBoxLayout()
        b = QPushButton("Eğri ekle (tıkla)")
        b.clicked.connect(lambda: self.set_tool(Tool.PICK_COLOR))
        self.remove_btn = QPushButton("Sil")
        self.remove_btn.clicked.connect(self.remove_curve)
        row.addWidget(b)
        row.addWidget(self.remove_btn)
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Eğri adı (ör. 0.56A)")
        self.name_edit.editingFinished.connect(self._rename_curve)
        self.de_slider = QSlider(Qt.Orientation.Horizontal)
        self.de_slider.setRange(2, 100)
        self.de_spin = QSpinBox()
        self.de_spin.setRange(2, 100)
        self.de_spin.setToolTip("Referans renge izin verilen Lab renk farkı (ΔE). Büyütmek daha fazla piksel yakalar.")
        self.de_slider.valueChanged.connect(self.de_spin.setValue)
        self.de_spin.valueChanged.connect(self.de_slider.setValue)
        self.de_spin.valueChanged.connect(self._de_changed)
        de_row = QHBoxLayout()
        de_row.addWidget(QLabel("Renk eşiği ΔE:"))
        de_row.addWidget(self.de_slider, 1)
        de_row.addWidget(self.de_spin)
        self.grid_check = QCheckBox("Izgara çizgilerini temizle")
        self.grid_check.setToolTip("Kesikli/sürekli ızgara ve çerçeve çizgilerini maskeden çıkarır (siyah eğriler için önemli).")
        self.grid_check.toggled.connect(self._grid_toggled)
        self.dashed_check = QCheckBox("Bu eğri kesikli çizgi")
        self.dashed_check.setToolTip("Eğrinin kendisi kesikli/noktalıysa açın: kısa parçalar korunur.\n"
                                     "Aynı renkte sürekli ve kesikli eğri varsa, eğriyi tıkladığınız noktadan izlenir.")
        self.dashed_check.toggled.connect(self._dashed_toggled)
        row2 = QHBoxLayout()
        self.extract_btn = QPushButton("Yeniden çıkar")
        self.extract_btn.clicked.connect(self._extract_selected)
        self.extract_all_btn = QPushButton("Hepsini çıkar")
        self.extract_all_btn.clicked.connect(self.extract_all)
        row2.addWidget(self.extract_btn)
        row2.addWidget(self.extract_all_btn)
        self.curve_info = QLabel("")
        self.curve_info.setWordWrap(True)
        for w in (self.curve_list,):
            lay.addWidget(w)
        lay.addLayout(row)
        lay.addWidget(self.name_edit)
        lay.addLayout(de_row)
        lay.addWidget(self.grid_check)
        lay.addWidget(self.dashed_check)
        lay.addLayout(row2)
        lay.addWidget(self.curve_info)
        col.addWidget(box)

        # --- editing ---------------------------------------------------------------
        box, lay = self._group("Düzeltme")
        row = QHBoxLayout()
        b1 = QPushButton("Noktaları sil (kutu)")
        b1.clicked.connect(lambda: self.set_tool(Tool.ERASE))
        b2 = QPushButton("Nokta ekle (tıkla)")
        b2.clicked.connect(lambda: self.set_tool(Tool.ADD_POINT))
        row.addWidget(b1)
        row.addWidget(b2)
        self.erase_all_check = QCheckBox("Silme tüm eğrilere uygulansın")
        lay.addLayout(row)
        lay.addWidget(self.erase_all_check)
        col.addWidget(box)

        # --- post processing -------------------------------------------------------
        box, lay = self._group("Son işleme")
        self.sg_check = QCheckBox("Savitzky–Golay yumuşatma")
        self.sg_window = QSpinBox()
        self.sg_window.setRange(3, 999)
        self.sg_window.setSingleStep(2)
        self.sg_window.setValue(self.project.post.window)
        self.sg_window.setToolTip("Pencere genişliği (nokta sayısı, tek sayı)")
        self.sg_order = QSpinBox()
        self.sg_order.setRange(1, 7)
        self.sg_order.setValue(self.project.post.polyorder)
        self.rs_check = QCheckBox("Düzgün X ızgarasına yeniden örnekle")
        self.rs_step = QDoubleSpinBox()
        self.rs_step.setDecimals(6)
        self.rs_step.setRange(1e-9, 1e12)
        self.rs_step.setValue(1.0)
        self.rs_suggest = QPushButton("Öner")
        self.rs_suggest.clicked.connect(self._suggest_step)
        form = QFormLayout()
        form.addRow("Pencere:", self.sg_window)
        form.addRow("Polinom derecesi:", self.sg_order)
        step_row = QHBoxLayout()
        step_row.addWidget(self.rs_step, 1)
        step_row.addWidget(self.rs_suggest)
        form.addRow("Adım (X birimi):", step_row)
        lay.addWidget(self.sg_check)
        lay.addLayout(form)
        lay.addWidget(self.rs_check)
        for w in (self.sg_check, self.rs_check):
            w.toggled.connect(self._post_changed)
        for w in (self.sg_window, self.sg_order, self.rs_step):
            w.valueChanged.connect(self._post_changed)
        col.addWidget(box)

        # --- export ----------------------------------------------------------------
        box, lay = self._group("Dışa aktar")
        self.rb_combined = QRadioButton("Tek dosya (ortak X ızgarası, çok sütun)")
        self.rb_separate = QRadioButton("Eğri başına ayrı dosya")
        self.rb_combined.setChecked(True)
        self.unc_check = QCheckBox("Noktaya göre belirsizlik sütunları ekle")
        self.unc_check.setToolTip("Belirsizlik her durumda dosya başlığına yazılır; bu seçenek sütun olarak da ekler.")
        btn = QPushButton("CSV olarak dışa aktar…")
        btn.clicked.connect(self.export_csv)
        lay.addWidget(self.rb_combined)
        lay.addWidget(self.rb_separate)
        lay.addWidget(self.unc_check)
        lay.addWidget(btn)
        col.addWidget(box)
        col.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        scroll.setMinimumWidth(360)
        dock = QDockWidget("Kontroller", self)
        dock.setWidget(scroll)
        dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)

    def _build_table_dock(self) -> None:
        self.table_model = ArrayTableModel()
        self.table = QTableView()
        self.table.setModel(self.table_model)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setDefaultSectionSize(20)
        dock = QDockWidget("Veri önizleme (seçili eğri)", self)
        dock.setWidget(self.table)
        dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable | QDockWidget.DockWidgetFeature.DockWidgetClosable)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self.table_dock = dock
        self.act_table = dock.toggleViewAction()
        self.act_table.setText("Veri önizleme tablosu")
        self.menuBar().actions()[2].menu().addAction(self.act_table)

    # ================================================================== tools
    def set_tool(self, tool: Tool, slot: tuple[str, int] | None = None) -> None:
        self._calib_slot = slot if tool == Tool.CALIB else None
        self.view.set_tool(tool)
        for t, a in self.tool_actions.items():
            a.setChecked(t == tool)
        if tool == Tool.CALIB and slot:
            self.statusBar().showMessage(CALIB_HINT.format(axis=slot[0].upper(), n=slot[1] + 1))
        else:
            self.statusBar().showMessage(TOOL_HINTS.get(tool, ""))

    def begin_calibration(self, axis: str, index: int) -> None:
        if self.project.image is None:
            return self._info("Önce bir görsel veya PDF açın.")
        self.set_tool(Tool.CALIB, (axis, index))

    # ================================================================== file handling
    def open_file(self, path: str | None = None) -> None:
        if not path:
            exts = " ".join(f"*{e}" for e in IMAGE_SUFFIXES)
            settings = QSettings("Curvelens", "Curvelens")
            path, _ = QFileDialog.getOpenFileName(self, "Görsel veya PDF aç", settings.value("last_dir", ""),
                                                  f"Görsel ve PDF ({exts} *.pdf);;Tüm dosyalar (*)")
            if not path:
                return
            settings.setValue("last_dir", str(Path(path).parent))
        try:
            if path.lower().endswith(".pdf"):
                self._open_pdf(path)
            else:
                self._open_image(path)
        except Exception as exc:                                    # unreadable / corrupt files
            QMessageBox.critical(self, APP_NAME, f"Dosya açılamadı:\n{exc}")

    def _reset_source(self) -> None:
        if self._pdf is not None:
            self._pdf.close()
        self._pdf, self._chart, self._rendered = None, None, None

    def _open_image(self, path: str) -> None:
        img = load_image(path)
        self._reset_source()
        self._install_image(img, Path(path).name)
        self.source_label.setText(f"{Path(path).name}\n{img.shape[1]} × {img.shape[0]} piksel")
        area = detect_plot_area(img)
        if area is not None:
            self.project.set_plot_area(area)
            self._refresh_all()
            self.statusBar().showMessage("Çizim alanı çerçeveden otomatik önerildi (yeşil kesikli çizgi); "
                                         "gerekirse 'Alanı seç' ile değiştirin.")

    def _open_pdf(self, path: str) -> None:
        src = PdfSource(path)
        dlg = PdfChartDialog(src, self)
        if dlg.exec() != PdfChartDialog.DialogCode.Accepted:
            src.close()
            return
        page, chart, dpi = dlg.selection()
        self._load_pdf(src, page, chart, dpi)

    def _load_pdf(self, src: PdfSource, page: int, chart: ChartInfo | None, dpi: float) -> None:
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            rendered = src.render(page, chart.region if chart else None, dpi)
        finally:
            QApplication.restoreOverrideCursor()
        if self._pdf is not src:
            self._reset_source()
        self._pdf, self._chart, self._rendered = src, chart, rendered
        what = chart.title if chart else f"sayfa {page + 1}"
        self._install_image(rendered.image, f"{src.path.name} / {what}")
        self.source_label.setText(f"{src.path.name}\n{chart.label if chart else f'Sayfa {page + 1} (tamamı)'}\n"
                                  f"{rendered.image.shape[1]} × {rendered.image.shape[0]} piksel · {dpi:g} DPI")
        if chart is not None:
            self.project.set_plot_area(rendered.rect_pt_to_px(chart.plot_rect))
            self._fill_y_axis_combo(chart)
            self._apply_pdf_calibration(0, 0)
        self._refresh_all()

    def choose_pdf_chart(self) -> None:
        if self._pdf is None:
            return
        page = self._chart.page_index if self._chart else 0
        dlg = PdfChartDialog(self._pdf, self, page)
        if dlg.exec() == PdfChartDialog.DialogCode.Accepted:
            p, chart, dpi = dlg.selection()
            self._load_pdf(self._pdf, p, chart, dpi)

    def _install_image(self, img: np.ndarray, source: str) -> None:
        self.project.set_image(img, source)
        self.view.set_image(img)
        self._selected = -1
        self.setWindowTitle(f"{APP_NAME} — {source}")
        self.set_tool(Tool.PAN)
        self._refresh_all()

    # ================================================================== PDF calibration
    def _fill_y_axis_combo(self, chart: ChartInfo) -> None:
        self.y_axis_combo.blockSignals(True)
        self.y_axis_combo.clear()
        for a in chart.y_axes:
            side = "sol" if a.side == "left" else "sağ"
            self.y_axis_combo.addItem(f"Y: {a.title or 'adsız'} ({side})")
        self.y_axis_combo.blockSignals(False)

    def _pdf_axis_changed(self) -> None:
        if self._chart is not None and self.y_axis_combo.currentIndex() >= 0 and not self._loading:
            self._apply_pdf_calibration(0, self.y_axis_combo.currentIndex(), keep_x=True)

    def _apply_pdf_calibration(self, xi: int, yi: int, keep_x: bool = False) -> None:
        """Fill both axes from the PDF's tick labels (snapped to grid lines)."""
        chart, rendered, src = self._chart, self._rendered, self._pdf
        if not (chart and rendered and src) or xi >= len(chart.x_axes) or yi >= len(chart.y_axes):
            self._info("Bu grafik için PDF'ten eksen bilgisi okunamadı; noktaları elle seçin.")
            return
        for axis, hint in (("x", chart.x_axes[xi]), ("y", chart.y_axes[yi])):
            if axis == "x" and keep_x:
                continue
            cal = src.axis_from_hint(hint, rendered)
            if cal is None:
                continue
            setup = self.project.x_axis if axis == "x" else self.project.y_axis
            setup.pixel, setup.value, setup.log = [cal.p1, cal.p2], [cal.v1, cal.v2], cal.log
            setup.unit = hint.unit
            setup.label = hint.title.replace(f"({hint.unit})", "").strip() if hint.unit else hint.title
        self._refresh_all()
        self.statusBar().showMessage("Kalibrasyon PDF'in tick etiketlerinden önerildi; gerekirse noktaları elle değiştirin.")

    # ================================================================== mouse handlers
    def _on_click(self, x: float, y: float) -> None:
        tool = self.view.tool
        if tool == Tool.CALIB and self._calib_slot:
            axis, idx = self._calib_slot
            self.project.set_axis_point(axis, idx, pixel=x if axis == "x" else y)
            self.set_tool(Tool.PAN)
            self._refresh_all()
            self.axis_boxes[axis].value_edits[idx].setFocus()
        elif tool == Tool.PICK_COLOR:
            if min(pick_reference_color(self.project.image, x, y)) >= 235:
                self.statusBar().showMessage("Tıkladığınız nokta arka plan (beyaz) gibi görünüyor; "
                                             "eğrinin tam üzerine tıklayın (yakınlaştırmak yardımcı olur).")
                return
            curve = self.project.add_curve(x, y)
            self._selected = len(self.project.curves) - 1
            self._extract(self._selected)
            self._refresh_all()
            self.name_edit.setFocus()
            self.name_edit.selectAll()
            self.statusBar().showMessage(f"'{curve.name}' eklendi (renk {curve.color_rgb}). Adını yazabilir, "
                                         "eşiği ayarlayabilir veya başka eğriye tıklayabilirsiniz.")
        elif tool == Tool.ADD_POINT:
            if not self._need_curve():
                return
            self.project.add_point(self._selected, x, y)
            self._refresh_all()

    def _on_rect(self, rect: Rect) -> None:
        tool = self.view.tool
        if tool == Tool.PLOT_AREA:
            self.project.set_plot_area(rect)
            self.set_tool(Tool.PAN)
            self._reextract_unedited()
        elif tool == Tool.EXCLUDE:
            self.project.add_exclusion(rect)
            self._reextract_unedited()
        elif tool == Tool.ERASE:
            targets = range(len(self.project.curves)) if self.erase_all_check.isChecked() else \
                ([self._selected] if self._need_curve() else [])
            n = sum(self.project.erase(i, rect) for i in targets)
            self.statusBar().showMessage(f"{n} nokta silindi.")
        self._refresh_all()

    def _on_cursor(self, x: float, y: float) -> None:
        if math.isnan(x):
            self.coord_label.setText("")
            return
        text = f"piksel ({x:.1f}, {y:.1f})"
        try:
            cal = self.project.calibration()
        except ValueError:
            cal = None
        if cal is not None:
            vx, vy = cal.pixel_to_data(x, y)
            s = self.project
            text += f"   →   {s.x_axis.label or 'X'} = {fmt(float(vx), 7)} {s.x_axis.unit}   " \
                    f"{s.y_axis.label or 'Y'} = {fmt(float(vy), 7)} {s.y_axis.unit}"
        self.coord_label.setText(text)

    # ================================================================== extraction
    def _extract(self, index: int) -> None:
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            res = self.project.extract(index)
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, APP_NAME, f"Çıkarma başarısız: {exc}")
            return
        QApplication.restoreOverrideCursor()
        if len(res) == 0:
            reason = {"no_match": "Referans renge uyan piksel bulunamadı: renk eşiğini (ΔE) artırın veya alanı kontrol edin.",
                      "no_curve": "Bir eğri izlenemedi: renk eşiğini artırın ya da çizim alanını daraltın."}
            self.statusBar().showMessage(reason.get(res.stats.get("reason", ""), "Eğri bulunamadı."))

    def _extract_selected(self) -> None:
        if 0 <= self._selected < len(self.project.curves):
            self._extract(self._selected)
            self._refresh_all()

    def extract_all(self) -> None:
        for i in range(len(self.project.curves)):
            self._extract(i)
        self._refresh_all()

    def _reextract_unedited(self) -> None:
        if not self.project.curves:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            done = self.project.reextract_unedited()
        finally:
            QApplication.restoreOverrideCursor()
        kept = len(self.project.curves) - len(done)
        if kept:
            self.statusBar().showMessage(f"{len(done)} eğri yeniden çıkarıldı; elle düzeltilen {kept} eğri korundu "
                                         "('Yeniden çıkar' ile isteğe bağlı yenileyebilirsiniz).")

    # ================================================================== curve panel
    def _need_curve(self) -> bool:
        if 0 <= self._selected < len(self.project.curves):
            return True
        self._info("Önce listeden bir eğri seçin veya 'Eğri ekle' ile yeni bir eğri oluşturun.")
        return False

    def _curve_selected(self, row: int) -> None:
        if self._loading:
            return
        self._selected = row
        self._refresh_curve_controls()
        self._refresh_overlays()
        self._refresh_table()

    def _rename_curve(self) -> None:
        if 0 <= self._selected < len(self.project.curves):
            name = self.name_edit.text().strip() or self.project.curves[self._selected].name
            self.project.curves[self._selected].name = name
            self._refresh_curve_list()
            self._refresh_table()

    def _de_changed(self, value: int) -> None:
        if self._loading or not (0 <= self._selected < len(self.project.curves)):
            return
        self.project.curves[self._selected].params.delta_e = float(value)
        self._extract_timer.start()

    def _grid_toggled(self, on: bool) -> None:
        if self._loading or not (0 <= self._selected < len(self.project.curves)):
            return
        self.project.curves[self._selected].params.remove_grid = on
        self._extract_timer.start()

    def _dashed_toggled(self, on: bool) -> None:
        if self._loading or not (0 <= self._selected < len(self.project.curves)):
            return
        self.project.curves[self._selected].params.dashed = on
        self._extract_timer.start()

    def remove_curve(self) -> None:
        if self._need_curve():
            self.project.remove_curve(self._selected)
            self._selected = min(self._selected, len(self.project.curves) - 1)
            self._refresh_all()

    # ================================================================== calibration + post + export
    def on_calibration_changed(self) -> None:
        self._refresh_calibration_status()
        self._refresh_overlays()
        self._refresh_curve_controls()
        self._refresh_table()

    def _post_changed(self, *_) -> None:
        if self._loading:
            return
        p = self.project.post
        p.smooth, p.window, p.polyorder = self.sg_check.isChecked(), self.sg_window.value() | 1, self.sg_order.value()
        p.resample, p.step = self.rs_check.isChecked(), self.rs_step.value()
        self._refresh_table()

    def _suggest_step(self) -> None:
        step = self.project.suggested_step()
        if step is None:
            return self._info("Adım önerisi için kalibrasyonu tamamlayın ve bir eğri çıkarın.")
        self.rs_step.setValue(step)

    def export_csv(self) -> None:
        proj = self.project
        try:
            if proj.calibration() is None:
                return self._info("Dışa aktarmadan önce iki eksenin kalibrasyonunu tamamlayın (X ve Y için 2'şer nokta + değer).")
        except ValueError as exc:
            return self._info(str(exc))
        combined = self.rb_combined.isChecked()
        if combined and not (proj.post.resample and proj.post.step > 0):
            step = proj.suggested_step() or 1.0
            self.rs_step.setValue(step)
            self.rs_check.setChecked(True)
            self._post_changed()
            self.statusBar().showMessage(f"Ortak X ızgarası için yeniden örnekleme etkinleştirildi (adım {step:g}).")
        settings = QSettings("Curvelens", "Curvelens")
        start = settings.value("export_dir", "")
        if combined:
            base = Path(proj.source).stem or "egriler"
            target, _ = QFileDialog.getSaveFileName(self, "CSV kaydet", str(Path(start) / f"{base}.csv"), "CSV (*.csv)")
        else:
            target = QFileDialog.getExistingDirectory(self, "Klasör seç (eğri başına bir CSV yazılır)", start)
        if not target:
            return
        try:
            paths = proj.export(target, combined, self.unc_check.isChecked())
        except Exception as exc:
            return self._info(str(exc), warn=True)
        settings.setValue("export_dir", str(Path(paths[0]).parent))
        self.statusBar().showMessage(f"{len(paths)} dosya yazıldı: {paths[0].parent}")

    # ================================================================== undo
    def undo(self) -> None:
        if self.project.undo() is not None:
            self._refresh_all()

    def redo(self) -> None:
        if self.project.redo() is not None:
            self._refresh_all()

    # ================================================================== area actions
    def reset_plot_area(self) -> None:
        if self._rendered is not None and self._chart is not None:
            self.project.set_plot_area(self._rendered.rect_pt_to_px(self._chart.plot_rect))
        elif self.project.image is not None:
            self.project.set_plot_area(detect_plot_area(self.project.image))
        else:
            self.project.set_plot_area(None)
        self._reextract_unedited()
        self._refresh_all()

    def clear_exclusions(self) -> None:
        self.project.clear_exclusions()
        self._reextract_unedited()
        self._refresh_all()

    # ================================================================== refreshing
    def _refresh_all(self) -> None:
        self._loading = True
        try:
            for box in self.axis_boxes.values():
                box.load()
            has_pdf_chart = self._chart is not None
            self.chart_btn.setVisible(self._pdf is not None)
            self.y_axis_combo.setVisible(has_pdf_chart and self.y_axis_combo.count() > 1)
            self.pdf_calib_btn.setVisible(has_pdf_chart)
            self._refresh_calibration_status()
            self._refresh_area_label()
            self._refresh_curve_list()
            self._refresh_curve_controls()
            p = self.project.post
            self.sg_check.setChecked(p.smooth)
            self.rs_check.setChecked(p.resample)
        finally:
            self._loading = False
        self._refresh_overlays()
        self._refresh_table()
        self.act_undo.setEnabled(self.project.can_undo())
        self.act_redo.setEnabled(self.project.can_redo())

    def _refresh_calibration_status(self) -> None:
        proj = self.project
        err = proj.calibration_error()
        if err:
            self.calib_status.setText(f"⚠ {err}")
        elif proj.calibration() is not None:
            cal = proj.calibration()
            self.calib_status.setText(
                f"✔ Kalibrasyon tamam · çözünürlük: {fmt(cal.x.resolution(), 4)} {proj.x_axis.unit}/px (X), "
                f"{fmt(cal.y.resolution(), 4)} {proj.y_axis.unit}/px (Y)")
        else:
            self.calib_status.setText("Kalibrasyon için her eksende 2 nokta seçip değerlerini girin.")

    def _refresh_area_label(self) -> None:
        a = self.project.plot_area
        txt = "Çizim alanı: tüm görsel (seçilmedi)" if a is None else \
            f"Çizim alanı: {a.width:.0f} × {a.height:.0f} px"
        self.area_label.setText(f"{txt}\nHariç tutulan alan: {len(self.project.exclusions)}")

    def _refresh_curve_list(self) -> None:
        was = self._loading
        self._loading = True
        self.curve_list.clear()
        for c in self.project.curves:
            item = QListWidgetItem(color_icon(c.color_rgb), f"{c.name}  ({len(c)} nokta)")
            self.curve_list.addItem(item)
        if 0 <= self._selected < self.curve_list.count():
            self.curve_list.setCurrentRow(self._selected)
        self._loading = was

    def _refresh_curve_controls(self) -> None:
        was = self._loading
        self._loading = True
        ok = 0 <= self._selected < len(self.project.curves)
        for w in (self.name_edit, self.de_slider, self.de_spin, self.grid_check, self.dashed_check, self.extract_btn,
                  self.remove_btn):
            w.setEnabled(ok)
        self.extract_all_btn.setEnabled(bool(self.project.curves))
        if ok:
            c = self.project.curves[self._selected]
            self.name_edit.setText(c.name)
            self.de_spin.setValue(int(round(c.params.delta_e)))
            self.de_slider.setValue(int(round(c.params.delta_e)))
            self.grid_check.setChecked(c.params.remove_grid)
            self.dashed_check.setChecked(c.params.dashed)
            self.curve_info.setText(self._curve_summary(self._selected))
        else:
            self.name_edit.clear()
            self.curve_info.setText("Bir eğri eklemek için 'Eğri ekle'ye basıp eğrinin üzerine tıklayın.")
        self._loading = was

    def _curve_summary(self, i: int) -> str:
        c = self.project.curves[i]
        if len(c) == 0:
            return "Nokta yok: renk eşiğini artırın veya çizim alanını kontrol edin."
        lines = [f"{len(c)} nokta · referans renk RGB{c.color_rgb}"]
        try:
            d = self.project.curve_data(i, post=False)
        except ValueError as exc:
            return "\n".join(lines + [str(exc)])
        if d is None:
            lines.append("Değer aralığı ve belirsizlik için kalibrasyonu tamamlayın.")
        else:
            s = self.project
            lines.append(f"X: {fmt(d.x[0], 6)} … {fmt(d.x[-1], 6)} {s.x_axis.unit}")
            lines.append(f"Y: {fmt(float(np.min(d.y)), 6)} … {fmt(float(np.max(d.y)), 6)} {s.y_axis.unit}")
            xl, xh = float(np.min(d.x_unc)), float(np.max(d.x_unc))
            yl, yh = float(np.min(d.y_unc)), float(np.max(d.y_unc))
            fx = f"±{fmt(xh, 3)}" if math.isclose(xl, xh, rel_tol=1e-3) else f"±{fmt(xl, 3)}…±{fmt(xh, 3)}"
            fy = f"±{fmt(yh, 3)}" if math.isclose(yl, yh, rel_tol=1e-3) else f"±{fmt(yl, 3)}…±{fmt(yh, 3)}"
            lines.append(f"Belirsizlik (±0,5 px): {fx} {s.x_axis.unit} · {fy} {s.y_axis.unit}")
        quality = self.project.curve_quality(i)
        if quality is not None:
            extent, density = quality
            if extent < 0.5:
                lines.append(f"⚠ Eğri çizim alanının yalnızca %{extent * 100:.0f}'ini kaplıyor: kısmen başka bir eğrinin altında "
                             "kalmış olabilir veya renk eşiği (ΔE) düşük.")
            if density < 0.85 and not c.params.dashed:
                lines.append(f"⚠ Eğride boşluklar var (sütunların %{density * 100:.0f}'i dolu): başka bir çizginin altında "
                             "kalmış ya da kesikli bir eğri olabilir ('Bu eğri kesikli çizgi' seçeneğini deneyin).")
        if c.edited:
            lines.append("Elle düzeltildi.")
        return "\n".join(lines)

    def _refresh_overlays(self) -> None:
        proj = self.project
        markers = []
        for axis, setup in (("x", proj.x_axis), ("y", proj.y_axis)):
            for i in range(2):
                if setup.pixel[i] is not None:
                    markers.append((axis, setup.pixel[i], f"{axis.upper()}{i + 1}"))
        self.view.set_calibration_markers(markers)
        self.view.set_plot_area(proj.plot_area)
        self.view.set_exclusions(proj.exclusions)
        self.view.set_curves([(c.x_px, c.y_px, QColor(*c.color_rgb), i == self._selected)
                              for i, c in enumerate(proj.curves)])

    def _refresh_table(self) -> None:
        if not (0 <= self._selected < len(self.project.curves)):
            self.table_model.set_data([], [])
            return
        try:
            d = self.project.curve_data(self._selected)
        except ValueError:
            d = None
        s = self.project
        if d is None:
            self.table_model.set_data([], [])
            return
        xl = s.x_axis.info().full_label or "X"
        yl = s.y_axis.info().full_label or "Y"
        self.table_model.set_data([xl, yl, "±X", "±Y"], [d.x, d.y, d.x_unc, d.y_unc])

    # ================================================================== misc
    def _info(self, text: str, warn: bool = False) -> None:
        (QMessageBox.warning if warn else QMessageBox.information)(self, APP_NAME, text)

    def show_help(self) -> None:
        QMessageBox.information(self, "Kullanım ipuçları", (
            "1. Dosya → Aç: PNG/JPG ya da PDF. PDF'te grafiği listeden veya önizlemeden seçin;\n"
            "    eksen değerleri PDF'in metin katmanından otomatik önerilir.\n"
            "2. Kalibrasyon: her eksen için 'Nokta seç' ile değerini bildiğiniz iki işarete tıklayın,\n"
            "    değerlerini yazın (log eksen için işaretleyin). Uzak iki nokta daha doğrudur.\n"
            "3. Çizim alanı: eğrilerin bulunduğu çerçevenin içini seçin; legend/metin için 'Hariç tut'.\n"
            "4. Eğri ekle: eğrinin üzerine tıklayın, adını yazın, gerekirse renk eşiğini (ΔE) ayarlayın.\n"
            "5. Doğrulama: noktalar eğrinin renginde bindirilir. Hatalı noktaları kutuyla silin,\n"
            "    eksikleri tek tek ekleyin. Ctrl+Z ile geri alın.\n"
            "6. Son işleme ve CSV: yumuşatma/yeniden örnekleme, sonra Dışa aktar.\n\n"
            "Fare: tekerlek = yakınlaştır, sürükle = kaydır, orta tuş veya Boşluk+sürükle = her araçta kaydır,\n"
            "Esc = aracı bırak."))

    def closeEvent(self, event) -> None:
        self._reset_source()
        super().closeEvent(event)
