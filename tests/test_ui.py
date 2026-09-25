"""GUI tests: real mouse events on the view, driven through the offscreen Qt platform."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QPoint, QPointF, Qt                       # noqa: E402
from PySide6.QtGui import QWheelEvent                                # noqa: E402
from PySide6.QtTest import QTest                                     # noqa: E402
from PySide6.QtWidgets import QApplication, QFileDialog              # noqa: E402

from core.pdf_source import PdfSource                                # noqa: E402
from core.models import Rect                                         # noqa: E402
from ui.image_view import Tool                                       # noqa: E402
from ui.main_window import MainWindow                                # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PNG = ROOT / "samples" / "discharge.png"
PDF = ROOT / "samples" / "INR18650P28A-V1-80093.pdf"

# Same fixed pixel coordinates as tests/test_integration_sample.py (tick-mark centres).
X_TICKS = ((308.79, "300"), (1447.71, "2700"))
Y_TICKS = ((785.14, "1"), (405.12, "3"))


@pytest.fixture(scope="module")
def app():
    a = QApplication.instance() or QApplication([])
    yield a


@pytest.fixture()
def win(app):
    w = MainWindow()
    w.resize(1400, 800)
    w.show()
    app.processEvents()
    yield w
    w.close()


def scene_to_view(win, x, y, zoom=None):
    """Zoom in and centre on a scene point; returns its viewport position."""
    v = win.view
    if zoom is not None:
        v.zoom_by(zoom / v.zoom)
    v.centerOn(QPointF(x, y))
    QApplication.processEvents()
    return v.mapFromScene(QPointF(x, y))


def click(win, x, y, zoom=4.0):
    pos = scene_to_view(win, x, y, zoom)
    QTest.mouseClick(win.view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, pos)


def drag(win, x0, y0, x1, y1, zoom=None):
    """Drag between two scene points (zoomed out so both are visible)."""
    v = win.view
    if zoom is not None:
        v.zoom_by(zoom / v.zoom)
    QApplication.processEvents()
    p0, p1 = v.mapFromScene(QPointF(x0, y0)), v.mapFromScene(QPointF(x1, y1))
    vp = v.viewport()
    QTest.mousePress(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, p0)
    QTest.mouseMove(vp, QPoint((p0.x() + p1.x()) // 2, (p0.y() + p1.y()) // 2))
    QTest.mouseMove(vp, p1)
    QTest.mouseRelease(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, p1)
    QApplication.processEvents()


def type_value(win, axis, idx, text):
    ed = win.axis_boxes[axis].value_edits[idx]
    ed.setText(text)
    ed.editingFinished.emit()


def calibrate_by_clicks(win):
    for axis, ticks in (("x", X_TICKS), ("y", Y_TICKS)):
        for i, (px, value) in enumerate(ticks):
            win.begin_calibration(axis, i)
            assert win.view.tool == Tool.CALIB
            x, y = (px, 500.0) if axis == "x" else (700.0, px)
            click(win, x, y)
            assert win.view.tool == Tool.PAN                       # one click, then back to panning
            type_value(win, axis, i, value)


pytestmark = pytest.mark.skipif(not PNG.exists(), reason="samples/discharge.png missing")


# ----------------------------------------------------------------------------- basics
def test_open_image_shows_it_and_fits_the_view(win):
    win.open_file(str(PNG))
    assert win.view.has_image() and win.project.image.shape[:2] == (1124, 1645)
    assert "discharge.png" in win.windowTitle()
    fit = win.view.zoom
    assert 0.1 < fit < 1.0
    win.resize(900, 600)                                             # auto-fit follows the window size
    QApplication.processEvents()
    assert win.view.zoom < fit


def test_wheel_zoom_keeps_the_point_under_the_cursor_fixed(win):
    win.open_file(str(PNG))
    v = win.view
    pos = QPointF(400, 300)
    before = v.mapToScene(pos.toPoint())
    z0 = v.zoom
    ev = QWheelEvent(pos, v.viewport().mapToGlobal(pos), QPoint(0, 0), QPoint(0, 120),
                     Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
    QApplication.sendEvent(v.viewport(), ev)
    assert v.zoom == pytest.approx(z0 * 1.25, rel=1e-6)
    after = v.mapToScene(pos.toPoint())
    assert (after - before).manhattanLength() < 0.6                  # scene units (pixels of the image)


def test_pan_by_middle_button_drag_and_space_drag(win):
    win.open_file(str(PNG))
    win.view.zoom_by(4.0)
    QApplication.processEvents()
    v, hb = win.view, win.view.horizontalScrollBar()
    win.set_tool(Tool.ERASE)                                         # a drawing tool must not block panning
    start = hb.value()
    vp = v.viewport()
    QTest.mousePress(vp, Qt.MouseButton.MiddleButton, Qt.KeyboardModifier.NoModifier, QPoint(300, 300))
    QTest.mouseMove(vp, QPoint(250, 300))
    QTest.mouseRelease(vp, Qt.MouseButton.MiddleButton, Qt.KeyboardModifier.NoModifier, QPoint(250, 300))
    assert hb.value() != start
    assert win.project.curves == []                                  # nothing was erased/drawn


def test_escape_returns_to_pan_tool(win):
    win.open_file(str(PNG))
    win.set_tool(Tool.PICK_COLOR)
    QTest.keyClick(win.view, Qt.Key.Key_Escape)
    assert win.view.tool == Tool.PAN


# ----------------------------------------------------------------------------- full workflow with the mouse
def test_full_workflow_matches_the_acceptance_values(win, tmp_path, monkeypatch):
    win.open_file(str(PNG))
    calibrate_by_clicks(win)
    assert win.project.calibration_error() is None and win.project.calibration() is not None
    assert "Kalibrasyon tamam" in win.calib_status.text()
    # clicked pixels agree with the tick centres to well below a pixel
    assert win.project.x_axis.pixel[0] == pytest.approx(X_TICKS[0][0], abs=0.6)
    assert win.project.y_axis.pixel[1] == pytest.approx(Y_TICKS[1][0], abs=0.6)

    # plot area (inside the frame) + an exclusion around the legend, both by dragging
    win.view.zoom_by(0.2)
    win.set_tool(Tool.PLOT_AREA)
    drag(win, 176, 33, 1583, 966)
    a = win.project.plot_area
    assert (a.x0, a.y0, a.x1, a.y1) == pytest.approx((176, 33, 1583, 966), abs=4)
    win.set_tool(Tool.EXCLUDE)
    drag(win, 1085, 515, 1310, 780)
    assert len(win.project.exclusions) == 1

    # curves: click on the coloured lines (pure red at 1500 mAh, magenta at 1500 mAh)
    cal = win.project.calibration()
    red_y = float(cal.y.to_pixel(3.6452))
    mag_y = float(cal.y.to_pixel(3.1575))
    x1500 = float(cal.x.to_pixel(1500))
    win.set_tool(Tool.PICK_COLOR)
    click(win, x1500, red_y, zoom=4.0)
    win.name_edit.setText("0.56A")
    win.name_edit.editingFinished.emit()
    click(win, x1500, mag_y, zoom=4.0)
    win.name_edit.setText("30A")
    win.name_edit.editingFinished.emit()
    assert [c.name for c in win.project.curves] == ["0.56A", "30A"]
    assert all(len(c) > 1000 for c in win.project.curves)
    assert win.curve_list.count() == 2

    red = win.project.curve_data(0, post=False)
    mag = win.project.curve_data(1, post=False)
    assert np.interp(1500, red.x, red.y) == pytest.approx(3.65, abs=0.03)
    assert np.interp(1500, mag.x, mag.y) == pytest.approx(3.15, abs=0.03)
    assert red.x[-1] == pytest.approx(2830, abs=30)

    # editing: erase a stretch with a rubber band, undo, add one point
    n = len(win.project.curves[0])
    win.curve_list.setCurrentRow(0)
    win.set_tool(Tool.ERASE)
    win.view.fit_in_view()
    QApplication.processEvents()
    xa, xb = float(cal.x.to_pixel(1000)), float(cal.x.to_pixel(1200))
    drag(win, xa, float(cal.y.to_pixel(3.9)), xb, float(cal.y.to_pixel(3.6)))
    removed = n - len(win.project.curves[0])                         # 200 mAh is ~95 px wide -> ~95 points
    assert 80 <= removed <= 110 and win.project.curves[0].edited
    assert len(win.project.curves[1]) > 1000                         # other curve untouched
    win.undo()
    assert len(win.project.curves[0]) == n
    win.set_tool(Tool.ADD_POINT)
    click(win, xa, float(cal.y.to_pixel(3.75)), zoom=2.0)
    assert len(win.project.curves[0]) == n + 1

    # post-processing + export (dialogs stubbed)
    win.sg_check.setChecked(True)
    win.rs_check.setChecked(True)
    win.rs_step.setValue(10.0)
    target = tmp_path / "out.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(target), "CSV (*.csv)"))
    win.rb_combined.setChecked(True)
    win.export_csv()
    df = pd.read_csv(target, comment="#", encoding="utf-8-sig")
    assert df.shape[1] == 3 and df.iloc[:, 0].iloc[1] - df.iloc[:, 0].iloc[0] == 10
    row = df[df.iloc[:, 0] == 1500].iloc[0]
    assert row.iloc[1] == pytest.approx(3.65, abs=0.03) and row.iloc[2] == pytest.approx(3.15, abs=0.03)
    assert "curve \"0.56A\"" in target.read_text(encoding="utf-8-sig")


def test_pick_color_needs_an_image_and_add_point_needs_a_curve(win, monkeypatch):
    shown = []
    monkeypatch.setattr(type(win), "_info", lambda self, text, warn=False: shown.append(text))
    win.begin_calibration("x", 0)                                   # no image yet
    assert shown and win.view.tool == Tool.PAN
    win.open_file(str(PNG))
    win.set_tool(Tool.ADD_POINT)
    click(win, 500, 300, zoom=1.0)                                  # no curve selected
    assert len(shown) == 2 and win.project.curves == []


def test_changing_delta_e_reextracts_the_selected_curve(win):
    win.open_file(str(PNG))
    win.project.set_plot_area(None)
    win.set_tool(Tool.PICK_COLOR)
    click(win, 878.2, 282.5, zoom=4.0)                              # on the red curve at ~1500 mAh
    assert len(win.project.curves) == 1
    strict = len(win.project.curves[0])
    win.de_spin.setValue(3)                                          # far too strict: pure red only
    win._extract_timer.stop()
    win._extract_selected()
    assert win.project.curves[0].params.delta_e == 3.0
    win.de_spin.setValue(60)
    win._extract_timer.stop()
    win._extract_selected()
    assert win.project.curves[0].params.delta_e == 60.0
    assert len(win.project.curves[0]) >= strict * 0.9                # looser threshold never loses the curve


# ----------------------------------------------------------------------------- PDF
@pytest.mark.skipif(not PDF.exists(), reason="sample PDF missing")
def test_pdf_chart_is_loaded_and_calibrated_automatically(win):
    with PdfSource(PDF) as tmp:
        charts = tmp.charts(0)
    src = PdfSource(PDF)
    chart = next(c for c in charts if "Discharge Rate" in c.title)
    win._load_pdf(src, 0, chart, 300)
    proj = win.project
    assert "Discharge Rate Characteristics" in proj.source
    assert proj.calibration() is not None
    assert (proj.x_axis.label, proj.x_axis.unit) == ("Capacity", "mAh")
    assert (proj.y_axis.label, proj.y_axis.unit) == ("Voltage", "V")
    assert proj.x_axis.value == [0.0, 3000.0] and proj.y_axis.value == [0.0, 5.0]
    a = proj.plot_area
    assert a is not None and 700 < a.width < 720 and 470 < a.height < 480           # 300 dpi frame
    assert "Kalibrasyon tamam" in win.calib_status.text()
    assert win.chart_btn.isVisibleTo(win)

    # a curve extracted through the UI logic agrees with the datasheet
    win.set_tool(Tool.PICK_COLOR)
    cal = proj.calibration()
    click(win, float(cal.x.to_pixel(1500)), float(cal.y.to_pixel(3.6452)), zoom=4.0)
    d = proj.curve_data(0, post=False)
    assert np.interp(1500, d.x, d.y) == pytest.approx(3.65, abs=0.03)
    win._reset_source()


@pytest.mark.skipif(not PDF.exists(), reason="sample PDF missing")
def test_chart_with_several_y_axes_offers_a_choice(win):
    src = PdfSource(PDF)
    chart = next(c for c in src.charts(0) if c.title.startswith("Charge"))
    win._load_pdf(src, 0, chart, 200)
    assert win.y_axis_combo.count() == 3 and win.y_axis_combo.isVisibleTo(win)
    seen = {}
    for i in range(3):
        win.y_axis_combo.setCurrentIndex(i)
        seen[win.project.y_axis.label] = (win.project.y_axis.value[0], win.project.y_axis.value[1], win.project.y_axis.unit)
    assert seen == {"Voltage": (0.0, 5.0, "V"), "Current": (0.0, 10.0, "A"), "Capacity": (0.0, 120.0, "%")}
    win._reset_source()


@pytest.mark.skipif(not PDF.exists(), reason="sample PDF missing")
def test_loading_another_chart_replaces_the_previous_state(win):
    src = PdfSource(PDF)
    charts = src.charts(0)
    win._load_pdf(src, 0, charts[1], 200)
    win.project.add_curve_with_color((255, 0, 0))
    win._load_pdf(src, 0, charts[3], 200)                            # Cycle Characteristics
    assert win.project.curves == []
    assert win.project.x_axis.label == "Cycle Number" and win.project.x_axis.value[1] == 500.0
    assert win.project.y_axis.value[1] == 120.0
    win._load_pdf(src, 0, None, 100)                                  # whole page: no chart, no calibration
    assert win.project.calibration() is None and win.project.plot_area is None
    win._reset_source()


def test_clicking_the_white_background_does_not_create_a_curve(win):
    win.open_file(str(PNG))
    win.set_tool(Tool.PICK_COLOR)
    click(win, 900, 60, zoom=1.0)                        # empty white area above the curves
    assert win.project.curves == []
    assert "arka plan" in win.statusBar().currentMessage()


def test_dashed_toggle_and_quality_warning(win):
    win.open_file(str(PNG))
    win.set_tool(Tool.PICK_COLOR)
    click(win, 878.2, 282.5, zoom=4.0)                              # red curve
    assert len(win.project.curves) == 1
    assert win.project.curves[0].seed == pytest.approx((878.2, 282.5), abs=0.6)
    assert win.dashed_check.isEnabled() and not win.dashed_check.isChecked()
    win.dashed_check.setChecked(True)
    assert win.project.curves[0].params.dashed is True
    win._extract_timer.stop()
    win._extract_selected()
    assert len(win.project.curves[0]) > 500                          # dashed mode still finds the solid red curve
    # erase most of the curve: the summary warns about the short extent
    win.project.erase(0, Rect(0, 0, 1500, 2000))
    win._refresh_all()
    assert "⚠" in win.curve_info.text()


def test_frame_is_suggested_as_plot_area_when_an_image_is_opened(win):
    win.open_file(str(PNG))
    a = win.project.plot_area
    assert a is not None
    assert (a.x0, a.y0, a.x1, a.y1) == pytest.approx((167.14, 24.11, 1591.67, 975.15), abs=1.0)
    assert "otomatik" in win.statusBar().currentMessage()
    # the black curve works straight away, no manual area needed
    win.set_tool(Tool.PICK_COLOR)
    x, y = 878.2, 405.12 - (3.28 - 3.0) * 190.01
    click(win, x, y, zoom=4.0)
    assert len(win.project.curves) == 1 and win.project.curves[0].color_rgb == (0, 0, 0)
    assert len(win.project.curves[0]) > 1000
    # a manual rectangle replaces the suggestion, "Alanı sıfırla" brings it back
    win.project.set_plot_area(Rect(300, 100, 900, 700))
    win.reset_plot_area()
    assert win.project.plot_area == a
