"""Values of the axis marks the user clicked on a picture chart (two per axis) and the axis names."""
from __future__ import annotations

from PySide6.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox, QGridLayout, QGroupBox, QLabel, QLineEdit,
                               QVBoxLayout, QWidget)

from core.raster_charts import AxisPoints
from ui.query_panel import parse_number


class _AxisBox(QGroupBox):
    def __init__(self, title: str, px: tuple[float, float], unit_hint: str, mark_names: tuple[str, str]):
        super().__init__(title)
        g = QGridLayout(self)
        self.px = px
        self.name = QLineEdit()
        self.name.setPlaceholderText(unit_hint)
        self.v1, self.v2 = QLineEdit(), QLineEdit()
        for e in (self.v1, self.v2):
            e.setPlaceholderText("değer")
        self.log = QCheckBox("Logaritmik eksen")
        g.addWidget(QLabel("Eksen adı:"), 0, 0)
        g.addWidget(self.name, 0, 1, 1, 2)
        for row, (label, edit, p) in enumerate(((mark_names[0], self.v1, px[0]), (mark_names[1], self.v2, px[1])), start=1):
            g.addWidget(QLabel(f"{label} (piksel {p:.1f}) değeri:"), row, 0, 1, 2)
            g.addWidget(edit, row, 2)
        g.addWidget(self.log, 3, 0, 1, 3)
        g.setColumnStretch(2, 1)

    def points(self) -> AxisPoints | str:
        a, b = parse_number(self.v1.text()), parse_number(self.v2.text())
        if a is None or b is None:
            return "iki işaretin değerini de sayı olarak yazın"
        if a == b:
            return "iki işaretin değeri aynı olamaz"
        if self.log.isChecked() and (a <= 0 or b <= 0):
            return "logaritmik eksende değerler pozitif olmalı"
        return AxisPoints(self.px, (a, b), self.name.text().strip(), self.log.isChecked())


class CalibrationDialog(QDialog):
    """Asks for the values written at the clicked marks (and names for the axes)."""

    def __init__(self, parent: QWidget | None, x_px, y_px, y2_px=None):
        super().__init__(parent)
        self.setWindowTitle("Eksen kalibrasyonu")
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Tıkladığınız işaretlerin yanında yazan değerleri girin.\n"
                             "Eksen adına birimi parantez içinde yazarsanız tablo başlığında görünür: Gerilim (V)"))
        self.x_box = _AxisBox("X ekseni", tuple(x_px), "örn. Kapasite (mAh)", ("1. işaret", "2. işaret"))
        self.y_box = _AxisBox("Y ekseni (sol)" if y2_px else "Y ekseni", tuple(y_px), "örn. Gerilim (V)", ("1. işaret", "2. işaret"))
        lay.addWidget(self.x_box)
        lay.addWidget(self.y_box)
        self.y2_box = None
        if y2_px:
            self.y2_box = _AxisBox("Y ekseni (sağ)", tuple(y2_px), "örn. Sıcaklık (°C)", ("1. işaret", "2. işaret"))
            lay.addWidget(self.y2_box)
        self.error = QLabel("")
        self.error.setStyleSheet("color: #c0392b;")
        lay.addWidget(self.error)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self._try_accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.result_points: tuple[AxisPoints, AxisPoints, AxisPoints | None] | None = None

    def _try_accept(self) -> None:
        boxes = [("X ekseni", self.x_box), ("Y ekseni", self.y_box)] + ([("Sağ Y ekseni", self.y2_box)] if self.y2_box else [])
        out = []
        for name, box in boxes:
            p = box.points()
            if isinstance(p, str):
                self.error.setText(f"{name}: {p}.")
                return
            out.append(p)
        self.result_points = (out[0], out[1], out[2] if len(out) > 2 else None)
        self.accept()

    @staticmethod
    def ask(parent, x_px, y_px, y2_px=None):
        """``(x, y, y2)`` :class:`AxisPoints`, or ``None`` when the dialog is cancelled."""
        dlg = CalibrationDialog(parent, x_px, y_px, y2_px)
        return dlg.result_points if dlg.exec() == QDialog.DialogCode.Accepted else None
