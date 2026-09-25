"""GUI flow for a chart that is a picture inside the PDF: calibrate with clicks, curves appear, edit, query, add/rename/delete."""
import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")
from PySide6.QtWidgets import QApplication, QFileDialog, QInputDialog, QMessageBox      # noqa: E402

from core.query import value_at                                                          # noqa: E402
from core.calibration_store import CalibrationStore                                     # noqa: E402
from core.raster_charts import AxisPoints                                                # noqa: E402
from ui.image_view import Tool                                                           # noqa: E402
from ui.lookup_window import LookupWindow                                                # noqa: E402
from tests import make_pdfs                                                              # noqa: E402

YES, NO = QMessageBox.StandardButton.Yes, QMessageBox.StandardButton.No


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def picture(tmp_path_factory):
    path, truth = make_pdfs.picture_chart(tmp_path_factory.mktemp("rui") / "pic.pdf", dual=True)
    return path, truth


@pytest.fixture()
def win(app, picture, tmp_path):
    path, truth = picture
    w = LookupWindow(store=CalibrationStore(tmp_path / "cal.json"))
    w.resize(1400, 850)
    w.show()
    app.processEvents()
    w.open_pdf(str(path), blocking=True)
    w.show_chart(0)
    w.truth = truth
    yield w
    w.close()


def px(truth, key, i):
    return truth[key][i][1]


def click_calibration(w, truth, monkeypatch, offset=3.0, right=True):
    """Drive the real click sequence: 4 clicks (+2 for a right axis) a few pixels beside the tick marks, then the dialog."""
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: YES if right else NO)
    seen = {}

    def dialog(x_px, y_px, y2_px):
        seen["px"] = (x_px, y_px, y2_px)
        return (AxisPoints(x_px, (0.0, 2000.0), "Capacity (mAh)"), AxisPoints(y_px, (2.0, 4.5), "Voltage (V)"),
                AxisPoints(y2_px, (0.0, 100.0), "Temperature (C)") if y2_px else None)

    monkeypatch.setattr(w, "_ask_axis_values", dialog)
    w.start_calibration()
    assert w.view.tool == Tool.CALIB
    marks = [("x", px(truth, "x_ticks", 0) + offset), ("x", px(truth, "x_ticks", 4) - offset),
             ("y", px(truth, "y_ticks", 0) + offset), ("y", px(truth, "y_ticks", 5) - offset)]
    if right:
        marks += [("y", px(truth, "y2_ticks", 0) - offset), ("y", px(truth, "y2_ticks", 5) + offset)]
    frame = truth["frame"]
    for kind, pos in marks:
        w._on_click(pos, 300.0) if kind == "x" else w._on_click(600.0, pos)
    return seen


# ------------------------------------------------------------------------------- before calibration
def test_gallery_marks_the_picture_chart_and_the_detail_asks_for_calibration(win):
    assert win.gallery.item(0).text() == "Picture chart\nSayfa 1 · görsel grafik · kalibrasyon gerekli"
    assert win._raster_bar.isVisibleTo(win) and win.raster_status.isVisibleTo(win)
    assert win.calib_btn.text() == "Kalibre et" and not win.add_curve_btn.isEnabled()
    assert win.curve_list.count() == 1 and "Kalibre et" in win.curve_list.item(0).text()
    assert win._curve is None and not win.print_btn.isEnabled() and not win.query.isEnabled()
    assert "Discharge at 25 C" in win.notes_label.text()
    win._on_cursor(100.0, 100.0)                                       # hovering without axes must not raise
    assert win.hover_label.text() == ""


def test_vector_charts_do_not_show_the_picture_tools(app):
    from tests.test_edit_ui import PDF
    w = LookupWindow()
    w.open_pdf(str(PDF), blocking=True)
    w.show_chart(1)
    assert not w._raster_bar.isVisibleTo(w)
    w.close()


# ------------------------------------------------------------------------------- calibration by clicking
def test_clicks_snap_onto_ticks_and_the_dialog_values_calibrate_the_chart(win, monkeypatch):
    truth = win.truth
    seen = click_calibration(win, truth, monkeypatch)
    x_px, y_px, y2_px = seen["px"]
    # every click landed 3 px beside a tick mark and was moved exactly onto it
    assert x_px == pytest.approx((px(truth, "x_ticks", 0), px(truth, "x_ticks", 4)), abs=1.0)
    assert y_px == pytest.approx((px(truth, "y_ticks", 0), px(truth, "y_ticks", 5)), abs=1.0)
    assert y2_px == pytest.approx((px(truth, "y2_ticks", 0), px(truth, "y2_ticks", 5)), abs=1.0)
    ch = win._chart
    assert ch.calibrated and len(ch.y_fits) == 2 and win._cal is None and win.view.tool == Tool.PAN
    assert win.stack.currentIndex() == 2 and "Kalibre edildi" in win.statusBar().currentMessage()
    assert win.gallery.item(0).text().endswith(f"{len(ch.curves)} eğri")
    assert win.calib_btn.text() == "Yeniden kalibre et" and win.add_curve_btn.isEnabled()


def test_curves_are_read_and_assigned_to_their_axes(win, monkeypatch):
    click_calibration(win, win.truth, monkeypatch)
    ch = win._chart
    assert [cv.label.split("(")[1] for cv in ch.curves] == ["yeşil)", "kırmızı)", "mavi)", "siyah)"][:0] or len(ch.curves) == 4
    by = {cv.label.split("(")[1].rstrip(")"): cv for cv in ch.curves}
    assert set(by) == {"yeşil", "kırmızı", "mavi", "siyah"}
    assert by["yeşil"].axis_note == "sağ eksen" and by["kırmızı"].axis_note == "sol eksen"
    truth = win.truth["curves"]
    assert value_at(by["kırmızı"].data, 1000.0).value == pytest.approx(3.6, abs=0.02)
    assert value_at(by["yeşil"].data, 1000.0).value == pytest.approx(50.0, abs=1.0)        # read on the 0..100 axis
    assert value_at(by["siyah"].data, 1500.0).value == pytest.approx(float(np.interp(1500.0, *truth["black"])), abs=0.02)
    # the list shows what each curve is
    texts = [win.curve_list.item(i).text() for i in range(win.curve_list.count())]
    assert any("Temperature (C) · sağ eksen" in t for t in texts) and any("Voltage (V) · sol eksen" in t for t in texts)


def test_lookup_table_query_and_export_work_on_a_picture_curve(win, monkeypatch, tmp_path):
    click_calibration(win, win.truth, monkeypatch)
    row = next(i for i in range(win.curve_list.count()) if "kırmızı" in win.curve_list.item(i).text())
    win.curve_list.setCurrentRow(row)
    assert win.table_model.rowCount() > 10 and "grafik görselinden" in win.precision_label.text()
    assert "vektör" not in win.precision_label.text()
    win.query.x_edit.setText("1000")
    assert "3.6" in win.query.y_out.text() and "Voltage" in win.query.y_out.text()
    win.query.y_edit.setText("3,6")                                     # comma decimal: the red curve is 3.6 V at 1000 mAh
    shown = float(win.query.x_out.text().split("<b>")[1].split("</b>")[0])
    assert shown == pytest.approx(1000.0, abs=8.0) and "mAh" in win.query.x_out.text()
    target = tmp_path / "pic.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(target), "CSV (*.csv)"))
    win.export_csv()
    assert "grafik görselinden" in target.read_text(encoding="utf-8-sig")
    assert "grafik görselinden" in win._report_html()


def test_a_chart_without_a_right_axis_is_calibrated_with_four_clicks(win, monkeypatch):
    seen = click_calibration(win, win.truth, monkeypatch, right=False)
    assert seen["px"][2] is None and len(win._chart.y_fits) == 1 and win._chart.calibrated
    assert all(cv.axis_note == "" for cv in win._chart.curves)


def test_escape_cancels_a_calibration_in_progress(win, monkeypatch):
    win.start_calibration()
    win._on_click(px(win.truth, "x_ticks", 0), 300.0)
    assert win._cal is not None and len(win._cal["clicks"]) == 1
    win.view.cancelled.emit()
    assert win._cal is None and win.view.tool == Tool.PAN and not win._chart.calibrated
    assert "iptal" in win.statusBar().currentMessage()


def test_a_cancelled_values_dialog_leaves_the_chart_uncalibrated(win, monkeypatch):
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: NO)
    monkeypatch.setattr(win, "_ask_axis_values", lambda *a: None)
    win.start_calibration()
    for pos in (px(win.truth, "x_ticks", 0), px(win.truth, "x_ticks", 4)):
        win._on_click(pos, 300.0)
    for pos in (px(win.truth, "y_ticks", 0), px(win.truth, "y_ticks", 5)):
        win._on_click(600.0, pos)
    assert not win._chart.calibrated and win._cal is None and "iptal" in win.statusBar().currentMessage()


def test_bad_calibration_values_are_reported_not_applied(win, monkeypatch):
    shown = []
    monkeypatch.setattr("ui.lookup_window.QMessageBox.warning", lambda *a, **k: shown.append(a[2]))
    t = win.truth
    win.apply_calibration(AxisPoints((px(t, "x_ticks", 0), px(t, "x_ticks", 0) + 0.5), (0, 1), "X"),
                          AxisPoints((px(t, "y_ticks", 0), px(t, "y_ticks", 5)), (2.0, 4.5), "Y"))
    assert shown and "yakın" in shown[0] and not win._chart.calibrated


# ------------------------------------------------------------------------------- curves and corrections
def test_add_delete_rename_and_move_curves_between_axes(win, monkeypatch):
    click_calibration(win, win.truth, monkeypatch)
    ch = win._chart
    n = len(ch.curves)
    # delete the black curve, then add it back with a click on it
    row = next(i for i, cv in enumerate(ch.curves) if "siyah" in cv.label)
    win.curve_list.setCurrentRow(row)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: YES)
    win.delete_curve()
    assert len(ch.curves) == n - 1 and len(ch.raster.traces) == n - 1
    win.start_add_curve()
    assert win.view.tool == Tool.PICK_CURVE
    x, y = win.truth["to_px"]("black", 1500.0)
    win._on_click(x, y)
    assert len(ch.curves) == n and win.view.tool == Tool.PAN and "Eğri eklendi" in win.statusBar().currentMessage()
    new = ch.curves[-1]
    assert new.color[0] < 80 and value_at(new.data, 1500.0).value == pytest.approx(2.8, abs=0.02)
    assert win._curve is new and win.curve_list.currentRow() == n - 1
    # rename
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("Referans hücre", True))
    win.rename_curve()
    assert new.label == "Referans hücre" and ch.raster.traces[-1].label == "Referans hücre"
    assert win.curve_list.item(n - 1).text().startswith("Referans hücre")
    # move it to the right axis: same pixels, other scale
    before = value_at(new.data, 1500.0).value
    win.switch_curve_axis()
    assert ch.curves[-1].axis_note == "sağ eksen" and value_at(ch.curves[-1].data, 1500.0).value != pytest.approx(before, abs=0.5)
    win.switch_curve_axis()
    assert value_at(ch.curves[-1].data, 1500.0).value == pytest.approx(before, abs=0.02)


def test_clicking_where_there_is_no_curve_adds_nothing(win, monkeypatch):
    click_calibration(win, win.truth, monkeypatch)
    n = len(win._chart.curves)
    win.start_add_curve()
    x0, y0, x1, y1 = win.truth["frame"]
    win._on_click(x0 + 40, y0 + 30)                      # empty paper in the corner
    assert len(win._chart.curves) == n and "bulunamadı" in win.statusBar().currentMessage()


def test_point_corrections_work_on_a_picture_curve(win, monkeypatch):
    click_calibration(win, win.truth, monkeypatch)
    row = next(i for i, cv in enumerate(win._chart.curves) if "kırmızı" in cv.label)
    win.curve_list.setCurrentRow(row)
    cv = win._curve
    n = cv.n_points
    win._set_edit_tool(Tool.ADD_POINT)
    x, y = win._data_to_px(1000.0, 3.0)
    win._on_click(float(x), float(y))
    assert cv.n_points == n + 1 and cv.edited and "düzeltildi (1)" in win.curve_list.item(row).text()
    assert value_at(cv.data, 1000.0).value == pytest.approx(3.0, abs=0.05)
    assert cv.data.x_unc[np.argmin(np.abs(cv.data.x - 1000.0))] > 0
    win.undo_edit()
    assert not cv.edited


def test_recalibration_asks_before_dropping_hand_corrections(win, monkeypatch):
    click_calibration(win, win.truth, monkeypatch)
    win._on_point_delete(5)
    assert win._curve.edited
    asked = []
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: asked.append(a[2]) or NO)
    win.start_calibration()
    assert asked and "düzeltmeler silinir" in asked[0] and win._cal is None and win._curve.edited
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: YES)
    win.start_calibration()
    assert win._cal is not None
