"""'Tablo karşılaştırma' tab: import a lookup table (CSV/TXT) and look at its points on a chart.

The table is matched with a chart of the open PDF / image (and one of its curves): its points are drawn on that chart
and compared with the curve.  When the user gives no chart to match (or nothing is open), the table gets a chart of its
own, drawn from its numbers.
"""
from __future__ import annotations

import html
import math
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QFileDialog, QHBoxLayout, QHeaderView, QLabel,
                               QMessageBox, QPushButton, QRadioButton, QSplitter, QTableView, QVBoxLayout, QWidget)

from core.table_import import Comparison, ImportedTable, compare_to_curve, read_table, unit_in, units_conflict
from ui.image_view import ImageView
from ui.plot_canvas import PALETTE, GraphImage, render_graph
from ui.table_model import ArrayTableModel, fmt

TABLE_COLOR = (220, 0, 140)


# --------------------------------------------------------------------------- #
# Pairing dialog
# --------------------------------------------------------------------------- #
class PairDialog(QDialog):
    """'Which chart is this table for?' - a chart + curve of the open document, or a new chart of its own."""

    def __init__(self, parent, table: ImportedTable, charts: list, current: tuple[int, int] | None):
        super().__init__(parent)
        self.setWindowTitle("Tabloyu bir grafikle eşleştir")
        self.setMinimumWidth(520)
        self._charts = charts
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(f"<b>{html.escape(table.name)}</b>: {len(table)} satır, sütunlar: "
                             f"{html.escape(table.x_header)} → {html.escape(', '.join(table.y_headers))}"))
        lay.addWidget(QLabel("Tablodaki noktalar hangi grafik üzerinde gösterilsin?"))
        self.r_chart = QRadioButton("Açık grafikle eşleştir (noktalar o grafiğin üstüne çizilir, eğriyle karşılaştırılır)")
        self.chart_box, self.curve_box = QComboBox(), QComboBox()
        self.r_new = QRadioButton("Eşleştirme yapma: tablodan yeni bir grafik oluştur")
        usable = []
        for i, c in enumerate(charts):
            where = "görsel dosya" if c.from_image else f"sayfa {c.chart.page_index + 1}"
            why = " — önce kalibre edin" if c.kind == "raster" and not c.calibrated else " — eğri yok" if not c.curves else ""
            self.chart_box.addItem(f"{c.title} ({where}){why}", i)
            ok = not why
            usable.append(ok)
            self.chart_box.model().item(i).setEnabled(ok)
        self.chart_box.currentIndexChanged.connect(self._fill_curves)
        self.curve_box.currentIndexChanged.connect(self._check_units)
        self.warn = QLabel("")
        self.warn.setWordWrap(True)
        self.warn.setStyleSheet("color: #b9770e;")
        self._table = table
        row = QHBoxLayout()
        row.addSpacing(24)
        row.addWidget(QLabel("Grafik:"))
        row.addWidget(self.chart_box, 2)
        row.addWidget(QLabel("Eğri:"))
        row.addWidget(self.curve_box, 2)
        lay.addWidget(self.r_chart)
        lay.addLayout(row)
        lay.addWidget(self.warn)
        lay.addWidget(self.r_new)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.result_choice: tuple | None = None

        if any(usable):
            start = current[0] if current and usable[current[0]] else usable.index(True)
            self.chart_box.setCurrentIndex(start)
            self._fill_curves()
            if current and current[0] == start and 0 <= current[1] < self.curve_box.count():
                self.curve_box.setCurrentIndex(current[1])
            self.r_chart.setChecked(True)
        else:
            self.r_chart.setEnabled(False)
            self.chart_box.setEnabled(False)
            self.curve_box.setEnabled(False)
            self.r_new.setChecked(True)
        self.r_chart.toggled.connect(lambda _on: (self.chart_box.setEnabled(self.r_chart.isChecked()),
                                                  self.curve_box.setEnabled(self.r_chart.isChecked())))
        self._check_units()

    def _fill_curves(self) -> None:
        i = self.chart_box.currentData()
        self.curve_box.clear()
        if i is not None:
            for cv in self._charts[i].curves:
                self.curve_box.addItem(cv.label)
        self._check_units()

    def _check_units(self) -> None:
        i = self.chart_box.currentData()
        if i is None or not self.chart_box.isEnabled():
            self.warn.setText("")
            return
        c = self._charts[i]
        msgs = []
        if c.x_fit is not None and units_conflict(self._table.x_header, c.x_fit.title):
            msgs.append(f"X birimi: tabloda {unit_in(self._table.x_header)}, grafikte {unit_in(c.x_fit.title)}")
        k = self.curve_box.currentIndex()
        if 0 <= k < len(c.curves):
            yf = c.y_fit_of(c.curves[k])
            if yf is not None and units_conflict(self._table.y_headers[0], yf.title):
                msgs.append(f"Y birimi: tabloda {unit_in(self._table.y_headers[0])}, grafikte {unit_in(yf.title)}")
        self.warn.setText("Uyarı: " + "; ".join(msgs) + ". Noktalar yine de yerleştirilir, ama karşılaştırma anlamsız olabilir."
                          if msgs else "")

    def accept(self) -> None:
        if self.r_chart.isChecked() and self.chart_box.isEnabled():
            i, k = self.chart_box.currentData(), self.curve_box.currentIndex()
            if i is None or k < 0:
                return
            self.result_choice = ("chart", int(i), int(k))
        else:
            self.result_choice = ("new",)
        super().accept()

    @staticmethod
    def ask(parent, table, charts, current):
        dlg = PairDialog(parent, table, charts, current)
        return dlg.result_choice if dlg.exec() == QDialog.DialogCode.Accepted else None


# --------------------------------------------------------------------------- #
# The tab
# --------------------------------------------------------------------------- #
class CompareTab(QWidget):
    """Import + match + view.  ``window`` is the main :class:`ui.lookup_window.LookupWindow` (charts and rendering)."""

    changed = Signal()

    def __init__(self, window):
        super().__init__()
        self.window = window
        self.table: ImportedTable | None = None
        self.mode = "none"                                  # "none" | "chart" | "new"
        self.chart = None
        self.curve = None
        self.rendered = None
        self.graph: GraphImage | None = None
        self.cmp: Comparison | None = None
        self.out_of_view = 0

        root = QVBoxLayout(self)
        top = QHBoxLayout()
        self.import_btn = QPushButton("Tablo içe aktar…")
        self.import_btn.setToolTip("CSV / TXT / TSV: X ve Y sütunlu bir lookup tablosu (bu programın CSV çıktısı da olur)")
        self.pair_btn = QPushButton("Eşleştir…")
        self.pair_btn.setToolTip("Tabloyu açık bir grafik ve eğriyle eşleştir, ya da yeni grafik oluştur")
        self.col_label = QLabel("Y sütunu:")
        self.col_combo = QComboBox()
        self.png_btn = QPushButton("Grafiği PNG kaydet…")
        self.csv_btn = QPushButton("Karşılaştırmayı CSV kaydet…")
        self.file_label = QLabel("Henüz tablo içe aktarılmadı.")
        self.file_label.setStyleSheet("color: gray;")
        for w in (self.import_btn, self.pair_btn, self.col_label, self.col_combo, self.png_btn, self.csv_btn):
            top.addWidget(w)
        top.addStretch(1)
        root.addLayout(top)
        root.addWidget(self.file_label)

        split = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        self.view = ImageView()
        self.hint = QLabel("Tekerlek: yakınlaştır · sürükle (ya da sağ tuşla sürükle): kaydır · tabloda bir satır seçince o nokta grafikte halkayla gösterilir")
        self.hint.setStyleSheet("color: gray;")
        ll.addWidget(self.view, 1)
        ll.addWidget(self.hint)
        split.addWidget(left)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        self.summary.setTextFormat(Qt.TextFormat.RichText)
        self.model = ArrayTableModel()
        self.grid = QTableView()
        self.grid.setModel(self.model)
        self.grid.setAlternatingRowColors(True)
        self.grid.verticalHeader().setDefaultSectionSize(22)
        self.grid.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.grid.horizontalHeader().setStretchLastSection(True)
        self.grid.selectionModel().selectionChanged.connect(self._rows_selected)
        rl.addWidget(self.summary)
        rl.addWidget(self.grid, 1)
        split.addWidget(right)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        root.addWidget(split, 1)

        self.import_btn.clicked.connect(lambda: self.import_table())
        self.pair_btn.clicked.connect(self.pair)
        self.col_combo.currentIndexChanged.connect(lambda _i: self.refresh())
        self.png_btn.clicked.connect(self.save_png)
        self.csv_btn.clicked.connect(self.save_csv)
        self._set_enabled()

    # ------------------------------------------------------------------ seams (tests replace them)
    def _pick_file(self) -> str:
        path, _ = QFileDialog.getOpenFileName(self, "Lookup tablosu aç", "",
                                              "Tablolar (*.csv *.txt *.tsv *.dat);;Tüm dosyalar (*)")
        return path

    def _choose_pairing(self, table, charts, current):
        return PairDialog.ask(self, table, charts, current)

    # ------------------------------------------------------------------ state
    def _set_enabled(self) -> None:
        has = self.table is not None
        multi = has and len(self.table.ys) > 1
        for w in (self.pair_btn, self.png_btn, self.csv_btn):
            w.setEnabled(has)
        self.col_label.setVisible(multi)
        self.col_combo.setVisible(multi)

    def _usable_charts(self) -> list:
        return list(self.window.charts)

    def _current_pair(self):
        w = self.window
        if w._chart is None or w._curve is None or w._chart not in w.charts:
            return None
        return w.charts.index(w._chart), w._chart.curves.index(w._curve)

    # ------------------------------------------------------------------ import + pairing
    def import_table(self, path: str | None = None) -> bool:
        """Read a table file, ask which chart it belongs to, show it.  ``False`` when nothing was loaded."""
        path = path or self._pick_file()
        if not path:
            return False
        try:
            table = read_table(path)
        except ValueError as exc:
            QMessageBox.critical(self, "Plot Digitizer", f"Tablo açılamadı:\n{exc}")
            return False
        charts = self._usable_charts()
        if charts and any(c.kind == "vector" or c.calibrated for c in charts):
            choice = self._choose_pairing(table, charts, self._current_pair())
            if choice is None:
                return False                                            # cancelled: the previous state stays
        else:
            choice = ("new",)                                           # nothing to match: make a chart of its own
        self.table = table
        self._apply(choice)
        info = f"{table.name}: {len(table)} nokta"
        if table.skipped:
            info += f" ({table.skipped} sayısal olmayan satır atlandı)"
        if choice[0] == "new" and not (charts and any(c.kind == "vector" or c.calibrated for c in charts)):
            info += " · eşleştirilecek açık grafik olmadığı için tablodan yeni grafik oluşturuldu"
        self.window.statusBar().showMessage(info)
        self.changed.emit()
        return True

    def pair(self) -> None:
        if self.table is None:
            return
        charts = self._usable_charts()
        if not any(c.kind == "vector" or c.calibrated for c in charts):
            self.window.statusBar().showMessage("Eşleştirilecek açık grafik yok: önce bir PDF ya da görsel açın.")
            return
        here = None
        if self.mode == "chart" and self.chart in self.window.charts and self.curve in self.chart.curves:
            here = (self.window.charts.index(self.chart), self.chart.curves.index(self.curve))
        choice = self._choose_pairing(self.table, charts, here or self._current_pair())
        if choice is not None:
            self._apply(choice)
            self.changed.emit()

    def _apply(self, choice: tuple) -> None:
        if choice[0] == "chart":
            self.chart = self.window.charts[choice[1]]
            self.curve = self.chart.curves[choice[2]]
            self.mode = "chart"
        else:
            self.chart = self.curve = self.rendered = None
            self.mode = "new"
        self.col_combo.blockSignals(True)
        self.col_combo.clear()
        if self.table is not None:
            for h in self.table.y_headers:
                self.col_combo.addItem(h)
        self.col_combo.blockSignals(False)
        self._set_enabled()
        self.refresh()

    def document_changed(self) -> None:
        """A different PDF / image was opened: a match with the old chart is gone."""
        if self.mode == "chart":
            self.mode = "new"
            self.chart = self.curve = self.rendered = None
            self.refresh()
            self.file_label.setText(self.file_label.text() + " — önceki eşleşme, dosya değiştiği için kaldırıldı")

    # ------------------------------------------------------------------ drawing
    def _column(self) -> int:
        return max(0, self.col_combo.currentIndex())

    def refresh(self) -> None:
        """Redraw the chart, the numbers and the table (also after the matched curve was corrected elsewhere)."""
        t = self.table
        if t is None:
            self.view.set_image(None)
            self.model.set_data([], [])
            self.summary.setText("")
            return
        if self.mode == "chart" and (self.chart is None or self.curve is None or self.curve not in self.chart.curves
                                     or self.chart not in self.window.charts):
            self.mode = "new"                                           # the curve is gone (deleted / re-read)
        if self.mode == "chart":
            self._draw_on_chart()
        else:
            self._draw_new_graph()
        self.file_label.setText(f"{t.name} · {len(t)} satır · " + (
            f"eşleşen grafik: {self.chart.title} → {self.curve.label}" if self.mode == "chart"
            else "eşleşme yok: tablodan yeni grafik"))
        self._rows_selected()

    def _draw_on_chart(self) -> None:
        w, t, chart, cv = self.window, self.table, self.chart, self.curve
        self.rendered = w.render_for(chart)
        self.graph = None
        xf, yf = chart.x_fit, chart.y_fit_of(cv)
        j = self._column()
        x, y = t.x, t.ys[j]
        self.cmp = compare_to_curve(x, y, cv.data)
        self.view.set_image(self.rendered.image)
        with np.errstate(all="ignore"):
            px, py = self.rendered.pt_to_px(xf.to_pos(x), yf.to_pos(y))
        h, wd = self.rendered.image.shape[:2]
        ok = np.isfinite(px) & np.isfinite(py)
        inside = ok & (px >= 0) & (px <= wd) & (py >= 0) & (py <= h)
        self.out_of_view = int((ok & ~inside).sum())
        self._px, self._py = np.where(ok, px, np.nan), np.where(ok, py, np.nan)
        self.view.set_curves([(px[inside], py[inside], QColor(*TABLE_COLOR), True)])
        with np.errstate(all="ignore"):
            cx, cy = self.rendered.pt_to_px(xf.to_pos(cv.data.x), yf.to_pos(cv.data.y))
        order = np.argsort(cx, kind="stable")
        self.view.set_highlight(np.asarray(cx)[order], np.asarray(cy)[order])
        c = self.cmp
        d = 6
        self.model.set_data([t.x_header, f"{t.y_headers[j]} (tablo)", f"{cv.label} (eğri)", "Fark", "Fark %"],
                            [self._strs(c.x, 6), self._strs(c.y, d), self._strs(c.y_curve, d), self._strs(c.delta, d),
                             self._strs(c.rel_pct, 3)])
        self._fit_columns()
        self.summary.setText(self._chart_summary())

    def _draw_new_graph(self) -> None:
        t = self.table
        series = [(h, t.x, y) for h, y in zip(t.y_headers, t.ys)]
        y_title = t.y_headers[0] if len(t.y_headers) == 1 else "Değer"
        self.graph = render_graph(series, t.x_header, y_title, title=t.name)
        self.rendered, self.cmp, self.out_of_view = None, None, 0
        self.view.set_image(self.graph.image)
        self.view.set_highlight(None, None)
        curves = []
        self._px, self._py = None, None
        for i, (_n, x, y) in enumerate(series):
            px, py = self.graph.to_px(x, y)
            ok = np.isfinite(px) & np.isfinite(py)
            curves.append((px[ok], py[ok], QColor(*PALETTE[i % len(PALETTE)]), i == self._column()))
            if i == self._column():
                self._px, self._py = np.where(ok, px, np.nan), np.where(ok, py, np.nan)
        self.view.set_curves(curves)
        self.model.set_data([t.x_header] + list(t.y_headers), [self._strs(t.x, 6)] + [self._strs(y, 6) for y in t.ys])
        self._fit_columns()
        self.summary.setText(self._new_summary())

    def _fit_columns(self) -> None:
        """Every column as wide as its header (long names such as 'Voltage (V) (tablo)' stay readable)."""
        hdr = self.grid.horizontalHeader()
        for j in range(self.model.columnCount()):
            name = str(self.model.headerData(j, Qt.Orientation.Horizontal) or "")
            self.grid.setColumnWidth(j, max(90, hdr.fontMetrics().horizontalAdvance(name) + 34))

    @staticmethod
    def _strs(values, digits: int) -> list[str]:
        return [fmt(float(v), digits) for v in values]

    def _unc_text(self) -> str:
        cv = self.curve
        if cv is None or not len(cv.data):
            return ""
        u = float(np.nanmax(cv.data.y_unc))
        yf = self.chart.y_fit_of(cv)
        unit = unit_in(yf.title) if yf else ""
        return f"Eğrinin belirsizliği ±{u:.2g} {unit}".strip()

    def _chart_summary(self) -> str:
        c, t, cv = self.cmp, self.table, self.curve
        e = html.escape
        yf = self.chart.y_fit_of(cv)
        unit = unit_in(yf.title) if yf else ""
        u = f" {e(unit)}" if unit else ""
        lines = [f"<b>Grafikte {c.n_total - self.out_of_view} nokta işaretli</b> (tablodaki {c.n_total} noktanın "
                 f"tamamı; grafik alanı dışında kalan: {self.out_of_view})."]
        if c.n_compared:
            lines.append(f"Eğrinin X aralığında {c.n_compared} nokta var. Tablo − eğri farkı: ortalama {c.mean:+.4g}{u}, "
                         f"ortalama mutlak {c.mean_abs:.4g}{u}, RMS {c.rms:.4g}{u}, en büyük {c.max_abs:.4g}{u} "
                         f"(X = {c.x_of_max:.6g}). Bağıl: ortalama %{c.mean_rel_pct:.3g}, en büyük %{c.max_rel_pct:.3g}.")
            lines.append(e(self._unc_text()) + ".")
        else:
            lines.append("Tablonun X değerleri eğrinin X aralığında değil: karşılaştırma yapılamadı.")
        if units_conflict(t.x_header, self.chart.x_fit.title) or (
                yf is not None and units_conflict(t.y_headers[self._column()], yf.title)):
            lines.append("<span style='color:#b9770e'>Uyarı: tablo ve grafik birimleri uyuşmuyor.</span>")
        return "<br>".join(lines)

    def _new_summary(self) -> str:
        t = self.table
        parts = [f"<b>Grafikte {len(t)} nokta işaretli</b> (tablodaki satırların tamamı). X: {t.x.min():.6g} … {t.x.max():.6g}."]
        for h, y in zip(t.y_headers, t.ys):
            ok = y[np.isfinite(y)]
            parts.append(f"{html.escape(h)}: {len(ok)} değer, {ok.min():.6g} … {ok.max():.6g}." if ok.size else f"{html.escape(h)}: değer yok.")
        parts.append("Bir grafikle karşılaştırmak için <b>Eşleştir…</b> düğmesini kullanın.")
        return "<br>".join(parts)

    # ------------------------------------------------------------------ selection <-> chart
    def _rows_selected(self, *_args) -> None:
        pts = []
        if self.table is not None and self._px is not None:
            rows = sorted({i.row() for i in self.grid.selectionModel().selectedIndexes()})
            for r in rows[:300]:
                if 0 <= r < len(self._px) and np.isfinite(self._px[r]) and np.isfinite(self._py[r]):
                    pts.append((float(self._px[r]), float(self._py[r])))
        self.view.set_row_markers(pts, reveal=bool(pts))

    _px = None
    _py = None

    # ------------------------------------------------------------------ output
    def save_png(self) -> None:
        if self.table is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Grafiği kaydet", str(Path(self.table.name).with_suffix(".png")), "PNG (*.png)")
        if path:
            ok = self.view.viewport().grab().save(path)
            self.window.statusBar().showMessage(f"Kaydedildi: {path}" if ok else "Grafik kaydedilemedi.")

    def save_csv(self) -> None:
        if self.table is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Karşılaştırmayı kaydet", str(Path(self.table.name).stem + "_karsilastirma.csv"),
                                              "CSV (*.csv)")
        if not path:
            return
        t, j = self.table, self._column()
        with open(path, "w", encoding="utf-8-sig", newline="") as fh:
            if self.mode == "chart" and self.cmp is not None:
                c = self.cmp
                fh.write(f"# Tablo: {t.name}; eşleşen eğri: {self.chart.title} / {self.curve.label}\n")
                fh.write(f"{t.x_header},{t.y_headers[j]} (tablo),{self.curve.label} (eğri),Fark,Fark %\n")
                for i in range(len(c.x)):
                    fh.write(",".join("" if not math.isfinite(v) else f"{v:.10g}" for v in
                                      (c.x[i], c.y[i], c.y_curve[i], c.delta[i], c.rel_pct[i])) + "\n")
            else:
                fh.write(f"# Tablo: {t.name}\n")
                fh.write(",".join([t.x_header] + list(t.y_headers)) + "\n")
                for i in range(len(t.x)):
                    fh.write(",".join("" if not math.isfinite(v) else f"{v:.10g}" for v in
                                      [t.x[i]] + [y[i] for y in t.ys]) + "\n")
        self.window.statusBar().showMessage(f"Kaydedildi: {path}")
