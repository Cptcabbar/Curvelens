"""GUI tests of the hand-correction tools (add / delete / move points, undo) and the value query box."""
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QPoint, Qt                                # noqa: E402
from PySide6.QtTest import QTest                                     # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox              # noqa: E402

from core.models import Rect                                         # noqa: E402
from ui.image_view import Tool                                       # noqa: E402
from ui.lookup_window import LookupWindow                            # noqa: E402
from ui.query_panel import parse_number                              # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "samples" / "INR18650P28A-V1-80093.pdf"
pytestmark = pytest.mark.skipif(not PDF.exists(), reason="sample datasheet PDF missing")


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def rate(app):
    """Window on the 'Discharge Rate Characteristics' chart, first curve (0.56A) selected."""
    w = LookupWindow()
    w.resize(1400, 850)
    w.show()
    app.processEvents()
    w.open_pdf(str(PDF), blocking=True)
    w.show_chart(1)
    yield w
    w.close()


def table_map(w):
    m = w.table_model
    return {float(m.data(m.index(r, 0))): float(m.data(m.index(r, 1))) for r in range(m.rowCount())}


def px_of(w, x, y):
    px = w._data_to_px(x, y)
    return float(px[0]), float(px[1])


# ------------------------------------------------------------------------------- adding
def test_edit_tools_are_enabled_for_a_selected_curve_and_start_in_pan_mode(rate):
    w = rate
    assert w.view.tool == Tool.PAN and w.tool_buttons[Tool.PAN].isChecked()
    assert all(b.isEnabled() for b in w.tool_buttons.values())
    assert not w.undo_btn.isEnabled() and not w.redo_btn.isEnabled() and not w.reset_btn.isEnabled()


def test_clicking_with_the_add_tool_inserts_a_point_and_changes_the_table(rate):
    w = rate
    cv = w._curve
    before = cv.n_points
    old = table_map(w)[1500.0]
    w._set_edit_tool(Tool.ADD_POINT)
    assert w.view.tool == Tool.ADD_POINT and "Esc" in w.hint_label.text()
    px, py = px_of(w, 1500.0, 3.2)                       # far below the curve (3.646 V)
    w._on_click(px, py)
    assert cv.n_points == before + 1 and cv.edited and cv.n_changes == 1
    assert table_map(w)[1500.0] == pytest.approx(3.2, abs=0.01) and table_map(w)[1500.0] != old
    assert "düzeltildi (1)" in w.curve_list.item(0).text()
    assert "elle düzeltildi" in w._table.method_text and "1 değişiklik" in w._table.method_text
    assert w.undo_btn.isEnabled() and w.reset_btn.isEnabled()
    assert "Nokta eklendi" in w.statusBar().currentMessage()


def test_added_point_carries_the_click_resolution_as_uncertainty(rate):
    w = rate
    w._set_edit_tool(Tool.ADD_POINT)
    px, py = px_of(w, 1500.0, 3.2)
    w._on_click(px, py)
    d = w._curve.data
    i = int(np.argmin(np.abs(d.x - 1500.0) + np.abs(d.y - 3.2)))
    assert d.x_unc[i] > 0 and d.y_unc[i] > 0
    assert d.y_unc[i] == pytest.approx(0.5 / w._rendered.sy * abs(w._chart.y_fits[0].slope), rel=0.05)


def test_add_tool_is_ignored_when_no_curve_is_selected(rate):
    w = rate
    w._curve_selected(-1)
    w._set_edit_tool(Tool.ADD_POINT)
    w._on_click(100.0, 100.0)                            # must not raise
    assert not w.tool_buttons[Tool.ADD_POINT].isEnabled() or w._curve is None


# ------------------------------------------------------------------------------- deleting
def test_deleting_one_point_and_a_box_of_points(rate):
    w = rate
    cv = w._curve
    n0 = cv.n_points
    w._on_point_delete(10)
    assert cv.n_points == n0 - 1 and "Nokta silindi" in w.statusBar().currentMessage()
    # box around the curve between x = 500 and 700 mAh
    a = px_of(w, 500.0, 4.5)
    b = px_of(w, 700.0, 3.0)
    n1 = cv.n_points
    w._set_edit_tool(Tool.ERASE)
    w._on_rect(Rect(min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])))
    removed = n1 - cv.n_points
    assert removed > 5 and not np.any((cv.data.x >= 500) & (cv.data.x <= 700))
    # the hole is bridged by the lookup table's linear interpolation
    assert len(table_map(w)) == 56 and 3.6 < table_map(w)[600.0] < 4.0
    # an empty box says so
    w._on_rect(Rect(0, 0, 5, 5))
    assert "yok" in w.statusBar().currentMessage()


def test_deleting_everything_leaves_a_disabled_empty_table(rate):
    w = rate
    cv = w._curve
    w._on_point_delete(0)
    cv.delete_points(range(cv.n_points))
    w._after_edit("x")
    assert cv.n_points == 0 and w.table_model.rowCount() == 0 and not w.print_btn.isEnabled()
    w.undo_edit()
    assert cv.n_points > 0 and w.table_model.rowCount() == 56


# ------------------------------------------------------------------------------- moving
def test_moving_a_point_updates_the_curve(rate):
    w = rate
    cv = w._curve
    i = int(np.argmin(np.abs(cv.data.x - 1000.0)))
    x0, y0 = float(cv.data.x[i]), float(cv.data.y[i])
    px, py = px_of(w, x0, y0 - 0.3)                      # drag it down by 0.3 V
    w._on_point_moved(i, px, py)
    d = cv.data
    j = int(np.argmin(np.abs(d.x - x0)))
    assert d.y[j] == pytest.approx(y0 - 0.3, abs=0.003) and cv.edited
    assert table_map(w)[1000.0] < y0 - 0.2
    assert "Nokta taşındı" in w.statusBar().currentMessage()


# ------------------------------------------------------------------------------- undo / redo / reset
def test_undo_redo_and_reset_with_buttons_and_shortcuts(rate, monkeypatch):
    w = rate
    cv = w._curve
    original = cv.data
    w._on_point_delete(3)
    w._on_point_delete(3)
    assert cv.n_changes == 2
    w.undo_btn.click()
    assert cv.n_changes == 1 and w.redo_btn.isEnabled()
    w.act_redo.trigger()
    assert cv.n_changes == 2 and not w.redo_btn.isEnabled()
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No)
    w.reset_btn.click()
    assert cv.edited                                     # declined
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    w.reset_btn.click()
    assert not cv.edited and np.array_equal(cv.data.x, original.x)
    assert "düzeltildi" not in w.curve_list.item(0).text() and not w.reset_btn.isEnabled()
    w.act_undo.trigger()                                 # the reset itself can be taken back
    assert cv.edited


def test_edits_belong_to_their_curve_and_survive_switching(rate):
    w = rate
    w._on_point_delete(0)
    first = w._curve
    w.curve_list.setCurrentRow(2)
    assert w._curve is not first and not w._curve.edited and not w.reset_btn.isEnabled()
    w.curve_list.setCurrentRow(0)
    assert w._curve is first and first.edited and w.reset_btn.isEnabled()


def test_escape_ends_the_edit_mode(rate):
    w = rate
    w._set_edit_tool(Tool.MOVE_POINT)
    QTest.keyClick(w.view, Qt.Key.Key_Escape)
    assert w.view.tool == Tool.PAN and w.tool_buttons[Tool.PAN].isChecked()


# ------------------------------------------------------------------------------- real mouse events
def test_mouse_click_on_a_point_with_the_delete_tool(rate, app):
    w = rate
    cv = w._curve
    n0 = cv.n_points
    i = 40
    x, y = px_of(w, float(cv.data.x[i]), float(cv.data.y[i]))
    vp = w.view.viewport()
    pos = w.view.mapFromScene(x, y)
    w._set_edit_tool(Tool.DELETE_POINT)
    QTest.mouseClick(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, pos)
    assert cv.n_points == n0 - 1
    QTest.mouseClick(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(2, 2))   # empty spot
    assert cv.n_points == n0 - 1


def test_mouse_drag_of_a_point_with_the_move_tool(rate, app):
    w = rate
    cv = w._curve
    i = 60
    x0, y0 = float(cv.data.x[i]), float(cv.data.y[i])
    x, y = px_of(w, x0, y0)
    vp = w.view.viewport()
    start = w.view.mapFromScene(x, y)
    tx, ty = px_of(w, x0, y0 - 0.25)
    end = w.view.mapFromScene(tx, ty)
    w._set_edit_tool(Tool.MOVE_POINT)
    QTest.mousePress(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start)
    QTest.mouseMove(vp, end)
    assert w.view._move_ring is not None                 # a ring follows the cursor
    QTest.mouseRelease(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, end)
    assert w.view._move_ring is None and cv.edited
    d = cv.data
    j = int(np.argmin(np.abs(d.x - x0)))
    assert d.y[j] == pytest.approx(y0 - 0.25, abs=0.02)


def test_mouse_click_with_the_add_tool(rate, app):
    w = rate
    cv = w._curve
    n0 = cv.n_points
    x, y = px_of(w, 2000.0, 3.0)
    w._set_edit_tool(Tool.ADD_POINT)
    QTest.mouseClick(w.view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                     w.view.mapFromScene(x, y))
    assert cv.n_points == n0 + 1


# ------------------------------------------------------------------------------- value query
def test_parse_number_accepts_comma_and_point():
    assert parse_number("1500") == 1500.0 and parse_number(" 3,65 ") == 3.65 and parse_number("2.5e3") == 2500.0
    assert parse_number("1.234,5") == 1234.5 and parse_number("1,234.5") == 1234.5
    assert parse_number("") is None and parse_number("abc") is None and parse_number("nan") is None


def test_x_to_y_query_gives_the_curve_value_with_uncertainty(rate):
    w = rate
    q = w.query
    assert q.isEnabled()
    q.x_edit.setText("1500")
    assert "Voltage" in q.y_out.text() and "3.646" in q.y_out.text() and "±" in q.y_out.text() and " V" in q.y_out.text()
    assert len(w.view._marker_items) == 2                # a cross-hair (halo + line) at the answer
    q.x_edit.setText("1500,5")                           # comma decimal
    assert "3.64" in q.y_out.text()
    q.x_edit.setText("99999")
    assert "aralığı dışında" in q.y_out.text() and w.view._marker_items == []
    q.x_edit.setText("abc")
    assert "sayı" in q.y_out.text()
    q.x_edit.clear()
    assert q.y_out.text() == ""


def test_y_to_x_query_lists_the_capacity_at_a_voltage(rate):
    w = rate
    q = w.query
    q.y_edit.setText("3.5")
    txt = q.x_out.text()
    assert "Capacity" in txt and "mAh" in txt and "çözüm" not in txt          # one answer
    shown = float(txt.split("<b>")[1].split("</b>")[0])
    assert 2040.0 < shown < 2070.0                                            # the datasheet's 0.56A curve is 3.5 V there
    assert float(np.interp(shown, w._curve.data.x, w._curve.data.y)) == pytest.approx(3.5, abs=0.01)
    assert len(w.view._marker_items) == 2
    q.y_edit.setText("9")
    assert "almıyor" in q.x_out.text() and w.view._marker_items == []


def test_query_follows_the_selected_curve_and_hand_corrections(rate):
    w = rate
    w.query.x_edit.setText("1500")
    first = w.query.y_out.text()
    w.curve_list.setCurrentRow(4)                        # 30A
    assert w.query.y_out.text() != first and "3.15" in w.query.y_out.text()
    w.curve_list.setCurrentRow(0)
    w._set_edit_tool(Tool.ADD_POINT)
    px, py = px_of(w, 1500.0, 3.0)
    w._on_click(px, py)                                  # a point at (1500, 3.0) is now on the curve
    assert "3.0" in w.query.y_out.text() and "3.646" not in w.query.y_out.text()


def test_query_box_is_disabled_without_a_curve(rate):
    w = rate
    w._curve_selected(-1)
    assert not w.query.isEnabled() and w.query.y_out.text() == ""
