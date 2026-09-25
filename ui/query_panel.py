"""'Değer sorgula' box: type an X to get Y, or a Y to get X, for the selected curve (no reading off the chart)."""
from __future__ import annotations

import html
import math

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QGridLayout, QGroupBox, QLabel, QLineEdit

from core.lookup import decimals_for, name_of, unit_of
from core.query import Answer, solve_x, value_at


def parse_number(text: str) -> float | None:
    """'1500', '3,65', ' 2.5e3 ' -> float; anything else -> None."""
    t = text.strip().replace(" ", "").replace(" ", "")
    if not t:
        return None
    if "," in t and "." in t:                       # 1.234,5  ->  1234.5
        t = t.replace(".", "").replace(",", ".") if t.rfind(",") > t.rfind(".") else t.replace(",", "")
    else:
        t = t.replace(",", ".")
    try:
        v = float(t)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def fmt_answer(a: Answer) -> str:
    if math.isfinite(a.unc) and a.unc > 0:
        d = decimals_for(a.unc, cap=6)
        v = f"{a.value:.{d}f}"
        return f"<b>{v}</b> ± {a.unc:.2g}"
    return f"<b>{a.value:.6g}</b>" + (" ± belirsiz" if not math.isfinite(a.unc) else "")


class QueryPanel(QGroupBox):
    """Works on ``curve.data`` (edited data included).  ``markersChanged`` carries the answers as data-unit points."""

    markersChanged = Signal(object)          # list[(x, y)] to mark on the chart; [] clears

    def __init__(self, parent=None):
        super().__init__("Değer sorgula", parent)
        self._curve = None
        self._x_name, self._x_unit = "X", ""
        self._y_name, self._y_unit = "Y", ""
        g = QGridLayout(self)
        g.setContentsMargins(8, 6, 8, 6)
        self.x_lbl, self.y_lbl = QLabel("X ="), QLabel("Y =")
        self.x_edit, self.y_edit = QLineEdit(), QLineEdit()
        for e in (self.x_edit, self.y_edit):
            e.setClearButtonEnabled(True)
        self.y_out = QLabel("")               # answer of the X box
        self.x_out = QLabel("")               # answer of the Y box
        self._answers: dict[str, list[tuple[float, float]]] = {"x": [], "y": []}
        for lb in (self.y_out, self.x_out):
            lb.setWordWrap(True)
            lb.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        g.setVerticalSpacing(3)
        g.addWidget(self.x_lbl, 0, 0)
        g.addWidget(self.x_edit, 0, 1)
        g.addWidget(self.y_out, 1, 0, 1, 2)
        g.addWidget(self.y_lbl, 2, 0)
        g.addWidget(self.y_edit, 2, 1)
        g.addWidget(self.x_out, 3, 0, 1, 2)
        g.setColumnStretch(1, 1)
        for lb in (self.y_out, self.x_out):
            lb.setVisible(False)                 # answers take room only while there is one
        self.x_edit.textChanged.connect(lambda _t: self._compute("x"))
        self.y_edit.textChanged.connect(lambda _t: self._compute("y"))
        self.setEnabled(False)

    # ------------------------------------------------------------------ public
    def set_curve(self, curve, x_title: str = "", y_title: str = "") -> None:
        """Select the curve to query (``None`` disables the box).  Typed values are kept and re-evaluated."""
        self._curve = curve
        self.setEnabled(curve is not None and len(curve.data) > 0)
        self._x_name, self._x_unit = (name_of(x_title) or "X"), unit_of(x_title)
        self._y_name, self._y_unit = (name_of(y_title) or "Y"), unit_of(y_title)
        self.x_lbl.setText(f"{self._x_name} =")
        self.y_lbl.setText(f"{self._y_name} =")
        self.x_edit.setPlaceholderText(f"{self._x_unit or 'X'} değeri yazın")
        self.y_edit.setPlaceholderText(f"{self._y_unit or 'Y'} değeri yazın")
        self.refresh()

    def refresh(self) -> None:
        """Re-evaluate both boxes (the curve changed, e.g. after a correction)."""
        self._compute("x", emit=False)
        self._compute("y", emit=False)
        self._emit_markers()

    # ------------------------------------------------------------------ internals
    def _fmt_unit(self, unit: str) -> str:
        return f" {html.escape(unit)}" if unit else ""

    def _compute(self, which: str, emit: bool = True) -> None:
        edit, out = (self.x_edit, self.y_out) if which == "x" else (self.y_edit, self.x_out)
        self._answers[which] = []
        cv = self._curve
        text = edit.text()
        if cv is None or len(cv.data) == 0 or not text.strip():
            out.setText("")
        else:
            v = parse_number(text)
            d = cv.data
            grey = "<span style='color:gray'>{}</span>"
            if v is None:
                out.setText(grey.format("bir sayı yazın"))
            elif which == "x":
                a = value_at(d, v)
                if a is None:
                    out.setText(grey.format(f"eğrinin {self._x_name} aralığı dışında ({d.x[0]:.6g} … {d.x[-1]:.6g})"))
                else:
                    note = f"<br>{grey.format(html.escape(a.note))}" if a.note else ""
                    out.setText(f"→ {html.escape(self._y_name)} = {fmt_answer(a)}{self._fmt_unit(self._y_unit)}{note}")
                    self._answers["x"] = [(v, a.value)]
            else:
                sols = solve_x(d, v)
                if not sols:
                    out.setText(grey.format(f"eğri bu değeri almıyor ({d.y.min():.6g} … {d.y.max():.6g} arasında)"))
                else:
                    lines = []
                    for a in sols:
                        line = f"{fmt_answer(a)}{self._fmt_unit(self._x_unit)}"
                        if a.note:
                            line += f" <span style='color:gray'>({html.escape(a.note)})</span>"
                        lines.append(line)
                    head = f"→ {html.escape(self._x_name)} = " if len(sols) == 1 else f"→ {len(sols)} çözüm ({html.escape(self._x_name)}):<br>"
                    out.setText(head + "<br>".join(lines))
                    self._answers["y"] = [(a.value, v) for a in sols]
        out.setVisible(bool(out.text()))
        if emit:
            self._emit_markers()

    def _emit_markers(self) -> None:
        pts = list(self._answers.get("x", [])) + list(self._answers.get("y", []))
        self.markersChanged.emit(pts)
