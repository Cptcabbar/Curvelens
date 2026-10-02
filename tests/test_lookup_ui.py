"""GUI tests of the PDF -> gallery -> chart -> curve -> lookup table window (offscreen Qt)."""
from pathlib import Path

import time

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QPoint, QPointF, Qt                       # noqa: E402
from PySide6.QtTest import QTest                                     # noqa: E402
from PySide6.QtWidgets import QApplication, QFileDialog              # noqa: E402

from ui import printing                                              # noqa: E402
from ui.lookup_window import LookupWindow                            # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "samples" / "INR18650P28A-V1-80093.pdf"
pytestmark = pytest.mark.skipif(not PDF.exists(), reason="sample datasheet PDF missing")


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def win(app):
    w = LookupWindow()
    w.resize(1400, 850)
    w.show()
    app.processEvents()
    yield w
    w.close()


@pytest.fixture()
def opened(win):
    win.open_pdf(str(PDF), blocking=True)
    return win


def wait_until(condition, timeout=20.0):
    """Run the event loop until ``condition()`` holds.  (QTest.qWait does not deliver signals that a worker
    thread queues to the GUI thread; the normal event loop - and the real application - does.)"""
    end = time.time() + timeout
    while time.time() < end and not condition():
        QApplication.processEvents()
        time.sleep(0.02)
    return condition()


def table_rows(w):
    m = w.table_model
    return [[m.data(m.index(r, c)) for c in range(m.columnCount())] for r in range(m.rowCount())]


# ------------------------------------------------------------------------------- flow
def test_starts_on_the_welcome_page(win):
    assert win.stack.currentIndex() == 0
    assert win.welcome_btn.isVisible() and not win.act_back.isVisible()
    assert win.print_btn.isEnabled() is False


def test_opening_a_pdf_shows_a_gallery_of_all_charts_with_names(opened):
    w = opened
    assert w.stack.currentIndex() == 1 and w.gallery.count() == 4
    names = [w.gallery.item(i).text() for i in range(4)]
    assert [n.splitlines()[0] for n in names] == [
        "Charge Characteristics", "Discharge Rate Characteristics", "Discharge Temperature Characteristics",
        "Cycle Characteristics"]
    assert [n.splitlines()[1] for n in names] == ["Sayfa 1 · 6 eğri", "Sayfa 1 · 5 eğri", "Sayfa 1 · 7 eğri", "Sayfa 1 · 2 eğri"]
    assert all(not w.gallery.item(i).icon().isNull() for i in range(4))          # a picture for every chart
    assert "sample" not in w.windowTitle() and "INR18650P28A" in w.windowTitle()
    assert "4 grafik ve 20 eğri" in w.statusBar().currentMessage()


def test_choosing_a_chart_shows_its_curves_with_their_meaning(opened):
    w = opened
    w.show_chart(1)
    assert w.stack.currentIndex() == 2 and w.act_back.isVisible() and w.view.has_image()
    assert "Discharge Rate Characteristics" in w.detail_title.text()
    items = [w.curve_list.item(i).text() for i in range(w.curve_list.count())]
    assert [t.splitlines()[0] for t in items] == ["0.56A", "2.8A", "10A", "20A", "30A"]
    assert all("Voltage (V)" in t and "sürekli çizgi" in t for t in items)
    assert "Temperature: 23°C." in w.notes_label.text()
    # the first curve is selected right away and its table exists
    assert w.curve_list.currentRow() == 0 and w._curve.label == "0.56A"


def test_selecting_a_curve_builds_the_lookup_table(opened):
    w = opened
    w.show_chart(1)
    rows = table_rows(w)
    assert len(rows) == 56 and rows[0] == ["50", "4.140"]                # auto step 50 mAh, 3 decimals
    assert dict((int(a), b) for a, b in rows)[1500] == "3.646"           # the acceptance value 3.65 V
    assert [w.table_model.headerData(i, Qt.Orientation.Horizontal) for i in (0, 1)] == ["Capacity (mAh)", "Voltage (V)"]
    assert "0.56A" in w.table_caption.text() and "56 satır" in w.table_caption.text()
    assert "vektör" in w.precision_label.text() and "±0.0026 V" in w.precision_label.text()
    assert w.csv_btn.isEnabled() and w.pdf_btn.isEnabled() and w.print_btn.isEnabled()

    w.curve_list.setCurrentRow(4)                                        # 30A
    assert w._curve.label == "30A"
    assert float(dict((int(a), b) for a, b in table_rows(w))[1500]) == pytest.approx(3.1575, abs=0.001)
    assert w.view._highlight_item is not None                            # selected curve is marked in the picture


def test_step_can_be_changed_and_raw_points_shown(opened):
    w = opened
    w.show_chart(1)
    w.step_spin.setValue(100.0)
    assert len(table_rows(w)) == 28 and table_rows(w)[0][0] == "100"
    w.step_spin.setValue(0.0)                                            # "Ham noktalar"
    rows = table_rows(w)
    assert len(rows) == 307 and w._table.step is None
    w._auto_step()
    assert w.step_spin.value() == 50.0 and len(table_rows(w)) == 56


def test_dashed_curve_of_the_charge_chart_is_listed_and_interpolated(opened):
    w = opened
    w.show_chart(0)
    texts = [w.curve_list.item(i).text() for i in range(w.curve_list.count())]
    assert len(texts) == 6
    assert sum("kesikli çizgi" in t for t in texts) == 3
    assert any(t.startswith("5.6A charge\nCurrent (A) · kesikli çizgi") for t in texts)
    row = next(i for i, t in enumerate(texts) if t.startswith("5.6A charge\nCurrent (A)"))
    w.curve_list.setCurrentRow(row)
    assert [w.table_model.headerData(i, Qt.Orientation.Horizontal) for i in (0, 1)] == ["Time (hrs)", "Current (A)"]
    assert table_rows(w)[0][1] == "5.600" and "interpolasyon" in w.precision_label.text()


def test_marker_series_of_the_cycle_chart(opened):
    w = opened
    w.show_chart(3)
    assert w.curve_list.count() == 2
    assert w.curve_list.item(0).text().startswith("10A discharge\nCapacity (%) · işaretçi")
    assert len(table_rows(w)) >= 30


def test_back_button_returns_to_the_gallery_and_another_chart_can_be_opened(opened):
    w = opened
    w.show_chart(2)
    w.act_back.trigger()
    assert w.stack.currentIndex() == 1
    w.gallery.itemClicked.emit(w.gallery.item(3))
    assert w.stack.currentIndex() == 2 and "Cycle" in w.detail_title.text()


def test_hover_shows_data_values_and_the_curve_value(opened):
    w = opened
    w.show_chart(1)
    r, chart = w._rendered, w._chart
    xpt = float(chart.x_fit.to_pos(1500.0))
    ypt = float(chart.y_fits[0].to_pos(3.0))
    px, py = r.pt_to_px(xpt, ypt)
    w._on_cursor(float(px), float(py))
    text = w.hover_label.text()
    assert "Capacity (mAh) = 1500" in text and "Voltage (V) = 3" in text and "0.56A: 3.645" in text
    w._on_cursor(float("nan"), float("nan"))
    assert w.hover_label.text() == ""


# ------------------------------------------------------------------------------- output
def test_csv_export(opened, tmp_path, monkeypatch):
    w = opened
    w.show_chart(1)
    target = tmp_path / "rate.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(target), "CSV (*.csv)"))
    w.export_csv()
    df = pd.read_csv(target, comment="#", encoding="utf-8-sig")
    assert list(df.columns) == ["Capacity (mAh)", "Voltage (V)"] and len(df) == 56
    assert df.loc[df["Capacity (mAh)"] == 1500, "Voltage (V)"].iloc[0] == pytest.approx(3.646, abs=0.0006)
    assert "Kaydedildi" in w.statusBar().currentMessage()


def test_pdf_button_writes_the_table_as_pdf(opened, tmp_path, monkeypatch):
    w = opened
    w.show_chart(1)
    target = tmp_path / "rate.pdf"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(target), "PDF (*.pdf)"))
    w.save_pdf()
    assert target.read_bytes().startswith(b"%PDF") and target.stat().st_size > 1000


def test_print_button_hands_the_report_to_the_print_dialog(opened, monkeypatch):
    w = opened
    w.show_chart(1)
    seen = {}

    def fake(html_text, parent=None):
        seen["html"], seen["parent"] = html_text, parent
        return True

    monkeypatch.setattr(printing, "print_with_dialog", fake)
    w.print_table()
    assert "Lookup tablosu — 0.56A" in seen["html"] and "3.646" in seen["html"] and seen["parent"] is w
    assert "Yazıcıya gönderildi" in w.statusBar().currentMessage()
    # cancelled dialog: no message
    monkeypatch.setattr(printing, "print_with_dialog", lambda *a, **k: False)
    w.statusBar().showMessage("")
    w.print_table()
    assert w.statusBar().currentMessage() == ""


def test_buttons_are_disabled_without_a_curve(opened):
    w = opened
    w.show_chart(1)
    w._curve_selected(-1)
    assert not w.print_btn.isEnabled() and not w.csv_btn.isEnabled() and w.table_model.rowCount() == 0
    w.print_table()                                                      # must not raise
    w.export_csv()
    w.save_pdf()


# ------------------------------------------------------------------------------- robustness
def test_non_pdf_and_missing_files_show_an_error_instead_of_crashing(win, monkeypatch, tmp_path):
    shown = []
    monkeypatch.setattr("ui.lookup_window.QMessageBox.critical", lambda *a, **k: shown.append(a[2]))
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a pdf at all")
    win.open_pdf(str(bad), blocking=True)
    assert shown and "PDF açılamadı" in shown[0] and win.stack.currentIndex() == 0
    win.open_pdf(str(tmp_path / "missing.pdf"), blocking=True)
    assert len(shown) == 2


def test_pdf_without_charts_explains_what_to_do(win, monkeypatch, tmp_path):
    import pypdfium2 as pdfium

    blank = tmp_path / "blank.pdf"
    doc = pdfium.PdfDocument.new()
    doc.new_page(200, 100)
    doc.save(str(blank))
    doc.close()
    shown = []
    monkeypatch.setattr("ui.lookup_window.QMessageBox.information", lambda *a, **k: shown.append(a[2]))
    win.open_pdf(str(blank), blocking=True)
    assert shown and "çizim çerçevesi olan bir grafik bulunamadı" in shown[0]
    assert win.stack.currentIndex() == 0


def test_opening_another_pdf_replaces_everything(opened, tmp_path):
    pytest.importorskip("matplotlib")
    from tests import make_pdfs

    path, _ = make_pdfs.simple_lines(tmp_path / "m.pdf")
    w = opened
    w.show_chart(1)
    w.open_pdf(str(path), blocking=True)
    assert w.stack.currentIndex() == 1 and w.gallery.count() == 1
    assert w._curve is None and "m.pdf" in w.windowTitle()
    w.show_chart(0)
    assert [w.curve_list.item(i).text().splitlines()[0] for i in range(3)] == ["Motor B", "Motor A", "Motor C"]
    assert table_rows(w)                                                 # a table exists for the first curve


def test_background_analysis_thread_delivers_the_same_result(win, app):
    win.open_pdf(str(PDF))                                               # non-blocking: worker thread + progress dialog
    assert win._worker is not None and win._progress is not None
    assert wait_until(lambda: win.gallery.count() == 4)
    assert win._worker is None and win._progress is None and win.stack.currentIndex() == 1


def test_drop_of_a_pdf_file_opens_it(win, app):
    from PySide6.QtCore import QMimeData, QUrl
    from PySide6.QtGui import QDropEvent

    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(PDF))])
    ev = QDropEvent(QPointF(10, 10), Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    win.dropEvent(ev)
    assert wait_until(lambda: win.gallery.count() == 4)


def test_manual_tool_is_still_reachable_from_the_menu(win):
    win.act_manual.trigger()
    assert len(win._legacy) == 1 and win._legacy[0].windowTitle().startswith("Curvelens")
