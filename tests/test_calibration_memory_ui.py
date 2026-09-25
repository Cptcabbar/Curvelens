"""A picture chart is calibrated once: the next time the same PDF (or a copy of it) is opened, the chart is ready."""
import shutil

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")
from PySide6.QtWidgets import QApplication, QMessageBox              # noqa: E402

from core.calibration_store import CalibrationStore                  # noqa: E402
from core.query import value_at                                      # noqa: E402
from core.raster_charts import AxisPoints                            # noqa: E402
from ui.lookup_window import LookupWindow                            # noqa: E402
from tests import make_pdfs                                          # noqa: E402

NO = QMessageBox.StandardButton.No


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def pdfs(tmp_path_factory):
    d = tmp_path_factory.mktemp("mem")
    path, truth = make_pdfs.picture_chart(d / "chart.pdf", dual=True)
    return path, truth, d


@pytest.fixture()
def store(tmp_path):
    return CalibrationStore(tmp_path / "cal.json")


def make_window(app, store):
    w = LookupWindow(store=store)
    w.resize(1400, 850)
    w.show()
    app.processEvents()
    return w


def axes(truth):
    xt, yt, y2t = truth["x_ticks"], truth["y_ticks"], truth["y2_ticks"]
    return (AxisPoints((xt[0][1], xt[-1][1]), (0.0, 2000.0), "Capacity (mAh)"),
            AxisPoints((yt[0][1], yt[-1][1]), (2.0, 4.5), "Voltage (V)"),
            AxisPoints((y2t[0][1], y2t[-1][1]), (0.0, 100.0), "Temperature (C)"))


def open_chart(app, store, path):
    w = make_window(app, store)
    w.open_pdf(str(path), blocking=True)
    return w


def test_a_calibration_is_saved_when_the_user_finishes_it(app, pdfs, store):
    path, truth, _ = pdfs
    w = open_chart(app, store, path)
    w.show_chart(0)
    assert not w._chart.calibrated and w._saved(w._chart) is None
    w.apply_calibration(*axes(truth))
    assert store.path.exists() and w._saved(w._chart) is not None
    assert "Kalibrasyon kaydedildi" in w.statusBar().currentMessage()
    assert "kayıtlı" in w.raster_status.text() and w.act_forget_cal.isEnabled()
    w.close()


def test_reopening_the_pdf_needs_no_calibration_any_more(app, pdfs, store):
    path, truth, _ = pdfs
    w = open_chart(app, store, path)
    w.show_chart(0)
    w.apply_calibration(*axes(truth))
    first = {cv.label: value_at(cv.data, 1000.0).value for cv in w._chart.curves}
    w.close()

    w2 = open_chart(app, store, path)                                     # a fresh window: nothing in memory
    assert w2.gallery.item(0).text().endswith("kalibrasyon kayıtlı")     # the gallery says so before the chart is opened
    w2.show_chart(0)
    ch = w2._chart
    assert ch.calibrated and len(ch.y_fits) == 2 and w2.stack.currentIndex() == 2
    assert "Kayıtlı kalibrasyon uygulandı" in w2.statusBar().currentMessage()
    assert "kayıtlı" in w2.raster_status.text() and not w2.calib_btn.text().startswith("Kalibre et")
    second = {cv.label: value_at(cv.data, 1000.0).value for cv in ch.curves}
    assert second == pytest.approx(first)                                 # identical values, no clicks
    assert w2.curve_list.currentRow() == 0 and w2.table_model.rowCount() > 10
    assert w2.gallery.item(0).text().endswith(f"{len(ch.curves)} eğri")
    w2.close()


def test_a_copy_of_the_pdf_under_another_name_is_recognised(app, pdfs, store):
    path, truth, d = pdfs
    w = open_chart(app, store, path)
    w.show_chart(0)
    w.apply_calibration(*axes(truth))
    w.close()
    copy = d / "copy of chart (2).pdf"
    shutil.copy(path, copy)
    w2 = open_chart(app, store, copy)
    w2.show_chart(0)
    assert w2._chart.calibrated and "Kayıtlı kalibrasyon uygulandı" in w2.statusBar().currentMessage()
    w2.close()


def test_a_different_pdf_is_not_touched(app, pdfs, store):
    path, truth, d = pdfs
    w = open_chart(app, store, path)
    w.show_chart(0)
    w.apply_calibration(*axes(truth))
    w.close()
    other, _ = make_pdfs.picture_chart(d / "other.pdf")                   # same look, other file content (no right axis)
    w2 = open_chart(app, store, other)
    assert w2.gallery.item(0).text().endswith("kalibrasyon gerekli")
    w2.show_chart(0)
    assert not w2._chart.calibrated
    w2.close()


def test_recalibrating_replaces_the_remembered_values(app, pdfs, store, monkeypatch):
    path, truth, _ = pdfs
    w = open_chart(app, store, path)
    w.show_chart(0)
    w.apply_calibration(*axes(truth))
    x, y, y2 = axes(truth)
    w.apply_calibration(AxisPoints(x.px, (0.0, 4000.0), "Capacity (mAh)"), y, y2)        # deliberately different
    w.close()
    w2 = open_chart(app, store, path)
    w2.show_chart(0)
    assert w2._chart.x_fit.to_value(w2._chart.raster.rendered.px_to_pt(x.px[1], 0.0)[0]) == pytest.approx(4000.0, abs=1.0)
    w2.close()


def test_forgetting_deletes_it_for_the_next_time(app, pdfs, store):
    path, truth, _ = pdfs
    w = open_chart(app, store, path)
    w.show_chart(0)
    w.apply_calibration(*axes(truth))
    w.forget_calibration()
    assert w._saved(w._chart) is None and not w.act_forget_cal.isEnabled()
    assert "silindi" in w.statusBar().currentMessage() and "kayıtlı değil" in w.raster_status.text()
    assert w._chart.calibrated                                            # this session's chart stays usable
    w.forget_calibration()                                                # nothing left: says so, no error
    assert "yok" in w.statusBar().currentMessage()
    w.close()
    w2 = open_chart(app, store, path)
    w2.show_chart(0)
    assert not w2._chart.calibrated and w2.gallery.item(0).text().endswith("kalibrasyon gerekli")
    w2.close()


def test_a_cancelled_or_rejected_calibration_saves_nothing(app, pdfs, store, monkeypatch):
    path, truth, _ = pdfs
    w = open_chart(app, store, path)
    w.show_chart(0)
    monkeypatch.setattr("ui.lookup_window.QMessageBox.warning", lambda *a, **k: None)
    x, y, y2 = axes(truth)
    w.apply_calibration(AxisPoints((x.px[0], x.px[0] + 0.5), (0, 1), "X"), y)      # marks too close together
    assert not w._chart.calibrated and not store.path.exists()
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: NO)
    monkeypatch.setattr(w, "_ask_axis_values", lambda *a: None)
    w.start_calibration()
    for pos in (x.px[0], x.px[1]):
        w._on_click(pos, 300.0)
    for pos in (y.px[0], y.px[1]):
        w._on_click(600.0, pos)
    assert not store.path.exists()
    w.close()


def test_a_stale_remembered_calibration_is_dropped_quietly(app, pdfs, store, monkeypatch):
    path, truth, _ = pdfs
    w = open_chart(app, store, path)
    w.show_chart(0)
    key, h = w._chart_key(w._chart), w._pdf_hash
    x, y, y2 = axes(truth)
    store.save(h, "chart.pdf", key, AxisPoints((5.0, 5.5), (0, 1), "X"), y, None)      # unusable: marks 0.5 px apart
    w.close()
    shown = []
    monkeypatch.setattr("ui.lookup_window.QMessageBox.warning", lambda *a, **k: shown.append(a[2]))
    w2 = open_chart(app, store, path)
    w2.show_chart(0)
    assert not w2._chart.calibrated and shown == []                       # no error dialog for a stored value
    assert store.get(h, key) is None                                      # and the bad entry is gone
    w2.close()


def test_a_read_only_data_folder_only_costs_the_memory(app, pdfs, tmp_path):
    path, truth, _ = pdfs
    blocker = tmp_path / "file"
    blocker.write_text("x")
    w = open_chart(app, CalibrationStore(blocker / "sub" / "cal.json"), path)
    w.show_chart(0)
    w.apply_calibration(*axes(truth))
    assert w._chart.calibrated and "kaydedilemedi" in w.statusBar().currentMessage()
    w.close()


def test_vector_pdfs_never_touch_the_store(app, store):
    from tests.test_edit_ui import PDF

    w = open_chart(app, store, PDF)
    w.show_chart(1)
    assert w._pdf_hash is None and not store.path.exists() and not w.act_forget_cal.isEnabled()
    w.close()
