"""Right (and middle) mouse button drags the picture in every tool; the left button keeps clicking."""
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")
from PySide6.QtCore import QPoint, Qt                                      # noqa: E402
from PySide6.QtTest import QTest                                           # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox                    # noqa: E402

from core.calibration_store import CalibrationStore                        # noqa: E402
from ui.image_view import Tool                                             # noqa: E402
from ui.lookup_window import LookupWindow                                  # noqa: E402
from tests import make_pdfs                                                # noqa: E402

RIGHT, MIDDLE, LEFT = Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton, Qt.MouseButton.LeftButton
NOMOD = Qt.KeyboardModifier.NoModifier


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def win(app, tmp_path):
    path, truth = make_pdfs.picture_chart(tmp_path / "pic.pdf")
    w = LookupWindow(store=CalibrationStore(tmp_path / "cal.json"))
    w.resize(1400, 850)
    w.show()
    app.processEvents()
    w.open_pdf(str(path), blocking=True)
    w.show_chart(0)
    w.truth = truth
    yield w
    w.close()


def zoomed(w):
    """Zoom in so that the picture is larger than the view (there is something to scroll)."""
    v = w.view
    v.zoom_by(4.0)
    QApplication.processEvents()
    v.centerOn(600.0, 400.0)                              # the middle of the picture: clicks land on it
    QApplication.processEvents()
    assert v.horizontalScrollBar().maximum() > 0
    return v


def drag(v, button, start=QPoint(900, 300), end=QPoint(820, 240)):
    vp = v.viewport()
    QTest.mousePress(vp, button, NOMOD, start)
    QTest.mouseMove(vp, end)
    QTest.mouseRelease(vp, button, NOMOD, end)


@pytest.mark.parametrize("button", [RIGHT, MIDDLE])
def test_dragging_with_the_right_or_middle_button_scrolls_the_picture(win, button):
    v = zoomed(win)
    h0, v0 = v.horizontalScrollBar().value(), v.verticalScrollBar().value()
    drag(v, button)                                       # drag left/up by (80, 60) px: the picture follows the hand
    assert v.horizontalScrollBar().value() == h0 + 80 and v.verticalScrollBar().value() == v0 + 60
    assert not v._panning


def test_right_drag_works_while_calibrating_and_does_not_place_a_point(win, monkeypatch):
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No)
    win.start_calibration()
    assert win.view.tool == Tool.CALIB and "sağ tuşla" in win.hint_label.text() and "sağ tuşla" in win.statusBar().currentMessage()
    v = zoomed(win)
    h0 = v.horizontalScrollBar().value()
    drag(v, RIGHT)
    assert v.horizontalScrollBar().value() != h0 and len(win._cal["clicks"]) == 0
    QTest.mouseClick(v.viewport(), RIGHT, NOMOD, QPoint(800, 300))                    # a plain right click: nothing either
    assert len(win._cal["clicks"]) == 0
    # the left button still places the points
    QTest.mouseClick(v.viewport(), LEFT, NOMOD, QPoint(800, 300))
    assert len(win._cal["clicks"]) == 1
    # ... and a right drag in between does not disturb the sequence
    drag(v, RIGHT)
    assert len(win._cal["clicks"]) == 1
    QTest.mouseClick(v.viewport(), LEFT, NOMOD, QPoint(900, 320))
    assert len(win._cal["clicks"]) == 2


@pytest.mark.parametrize("tool", [Tool.ADD_POINT, Tool.ERASE, Tool.DELETE_POINT, Tool.PICK_CURVE, Tool.PAN])
def test_right_drag_scrolls_in_every_tool_and_makes_no_edit(win, tool):
    v = zoomed(win)
    win._set_edit_tool(tool)
    h0 = v.horizontalScrollBar().value()
    drag(v, RIGHT)
    assert v.horizontalScrollBar().value() != h0
    assert v._rubber is None and v._drag_start is None          # no selection box was started


def test_left_drag_with_a_click_tool_still_does_not_scroll(win):
    v = zoomed(win)
    win._set_edit_tool(Tool.ADD_POINT)
    h0 = v.horizontalScrollBar().value()
    drag(v, LEFT)
    assert v.horizontalScrollBar().value() == h0


def test_the_hints_mention_the_right_button(win):
    assert "sağ tuşla sürükle" in win.hint_label.text()
    assert "sağ tuşla sürükle" in win.compare.hint.text()
