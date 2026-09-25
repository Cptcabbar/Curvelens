"""Chart images opened directly (PNG, JPG, BMP, TIFF): same flow as a picture chart inside a PDF."""
import shutil

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")
from PySide6.QtCore import QMimeData, QPointF, QUrl, Qt              # noqa: E402
from PySide6.QtGui import QDropEvent                                 # noqa: E402
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox  # noqa: E402

from core.calibration_store import CalibrationStore                  # noqa: E402
from core.query import value_at                                      # noqa: E402
from core.raster_charts import AxisPoints, chart_from_image         # noqa: E402
from ui.image_view import Tool                                       # noqa: E402
from ui.lookup_window import LookupWindow, is_image_path             # noqa: E402
from core.models import Rect                                         # noqa: E402
from tests import make_pdfs                                          # noqa: E402

NO = QMessageBox.StandardButton.No


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def store(tmp_path):
    return CalibrationStore(tmp_path / "cal.json")


def open_image(app, store, path):
    w = LookupWindow(store=store)
    w.resize(1400, 850)
    w.show()
    app.processEvents()
    w.open_pdf(str(path), blocking=True)
    return w


def calibrate(w, truth, right=False):
    xt, yt = truth["x_ticks"], truth["y_ticks"]
    y2 = None
    if right:
        y2t = truth["y2_ticks"]
        y2 = AxisPoints((y2t[0][1], y2t[-1][1]), (0.0, 100.0), "Temperature (C)")
    w.apply_calibration(AxisPoints((xt[0][1], xt[-1][1]), (0.0, 2000.0), "Capacity (mAh)"),
                        AxisPoints((yt[0][1], yt[-1][1]), (2.0, 4.5), "Voltage (V)"), y2)


def close_to_truth(cv, xs, ys, tol_v):
    d = cv.data
    inside = (xs > d.x[0] + 30) & (xs < d.x[-1] - 30)
    return float(np.max(np.abs(np.interp(xs[inside], d.x, d.y) - ys[inside]))) <= tol_v


def test_image_suffixes():
    for name in ("a.png", "A.JPG", "b.jpeg", "c.bmp", "d.tif", "e.tiff", "f.webp"):
        assert is_image_path(name)
    assert not is_image_path("a.pdf") and not is_image_path("a.txt")


@pytest.mark.parametrize("suffix", [".png", ".jpg", ".bmp", ".tiff"])
def test_an_image_file_opens_as_one_chart_and_is_read_like_a_picture_chart(app, store, tmp_path, suffix):
    path, truth = make_pdfs.picture_file(tmp_path / f"chart{suffix}")
    w = open_image(app, store, path)
    assert w.stack.currentIndex() == 1 and w.gallery.count() == 1 and w.windowTitle().endswith(f"chart{suffix}")
    assert w.gallery.item(0).text() == "chart\ngörsel dosya · kalibrasyon gerekli"
    w.show_chart(0)
    ch = w._chart
    assert ch.kind == "raster" and ch.from_image and ch.raster.frame_found and not ch.calibrated
    assert w.detail_title.text() == "chart" and "sayfa" not in w.detail_title.text()
    x0, y0, x1, y1 = truth["frame"]
    p = ch.raster.plot
    assert (p.x0, p.y0, p.x1, p.y1) == pytest.approx((x0, y0, x1, y1), abs=2.0)
    calibrate(w, truth)
    assert len(ch.curves) == 3 and w.stack.currentIndex() == 2
    tol = 0.02 if suffix == ".jpg" else 0.012                                # JPEG blurs the strokes a little
    for cv, name in zip(ch.curves, ("red", "blue", "black")):
        assert close_to_truth(cv, *truth["curves"][name], tol_v=tol), (suffix, name)
    w.close()


def test_a_bmp_with_a_second_axis_and_queries(app, store, tmp_path):
    path, truth = make_pdfs.picture_file(tmp_path / "dual.bmp", dual=True)
    w = open_image(app, store, path)
    w.show_chart(0)
    monkey = QMessageBox
    calibrate(w, truth, right=True)
    by = {cv.label.split("(")[1].rstrip(")"): cv for cv in w._chart.curves}
    assert by["yeşil"].axis_note == "sağ eksen"
    assert value_at(by["yeşil"].data, 1000.0).value == pytest.approx(50.0, abs=1.0)
    w.curve_list.setCurrentRow(list(w._chart.curves).index(by["kırmızı"]))
    w.query.x_edit.setText("1000")
    assert "3.6" in w.query.y_out.text()
    assert monkey is not None
    w.close()


def test_exports_do_not_mention_a_page_for_an_image_file(app, store, tmp_path, monkeypatch):
    path, truth = make_pdfs.picture_file(tmp_path / "chart.png")
    w = open_image(app, store, path)
    w.show_chart(0)
    calibrate(w, truth)
    w.curve_list.setCurrentRow(0)
    html = w._report_html()
    assert "chart.png" in html and "sayfa" not in html and "grafik görselinden" in html
    target = tmp_path / "out.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(target), "CSV (*.csv)"))
    w.export_csv()
    text = target.read_text(encoding="utf-8-sig")
    assert "chart.png (" in text and "sayfa" not in text
    w.close()


def test_the_calibration_of_an_image_is_remembered_by_its_content(app, store, tmp_path):
    path, truth = make_pdfs.picture_file(tmp_path / "chart.png", dual=True)
    w = open_image(app, store, path)
    w.show_chart(0)
    calibrate(w, truth, right=True)
    first = {cv.label: value_at(cv.data, 1000.0).value for cv in w._chart.curves}
    assert "bu görsel için kayıtlı" in w.raster_status.text()
    w.close()
    copy = tmp_path / "renamed.png"
    shutil.copy(path, copy)
    w2 = open_image(app, store, copy)
    assert w2.gallery.item(0).text().endswith("kalibrasyon kayıtlı")
    w2.show_chart(0)
    assert w2._chart.calibrated and "Kayıtlı kalibrasyon uygulandı" in w2.statusBar().currentMessage()
    assert {cv.label: value_at(cv.data, 1000.0).value for cv in w2._chart.curves} == pytest.approx(first)
    w2.close()


def test_a_chart_without_a_frame_asks_for_the_plot_area_and_remembers_it(app, store, tmp_path):
    path, truth = make_pdfs.picture_file(tmp_path / "open.png", frame=False)
    w = open_image(app, store, path)
    w.show_chart(0)
    rc = w._chart.raster
    assert not rc.frame_found and "Grafik alanı" in w.raster_status.text()
    assert w.plot_btn.isEnabled() and w.view._area_item is not None            # the guess is outlined
    w.start_plot_area()
    assert w.view.tool == Tool.PLOT_AREA
    x0, y0, x1, y1 = truth["frame"]
    w.view.rectSelected.emit(Rect(x0 - 6, y0 - 6, x1 + 6, y1 + 6))             # the user drags a box around the axes
    assert rc.frame_found and rc.plot_manual and w.view.tool == Tool.PAN
    assert (rc.plot.x0, rc.plot.y1) == pytest.approx((x0 - 6, y1 + 6), abs=0.5)
    # ticks are looked up along the new edges: a click near the 0 mark snaps onto it
    assert min(abs(v - truth["x_ticks"][0][1]) for v in rc.ticks.x) < 8.0
    calibrate(w, truth)
    assert len(w._chart.curves) == 3
    for cv, name in zip(w._chart.curves, ("red", "blue", "black")):
        assert close_to_truth(cv, *truth["curves"][name], tol_v=0.03), name
    w.close()
    w2 = open_image(app, store, path)                                            # next time: area and axes come back
    w2.show_chart(0)
    r2 = w2._chart.raster
    assert w2._chart.calibrated and r2.plot_manual and r2.plot.x0 == pytest.approx(x0 - 6, abs=0.5)
    assert len(w2._chart.curves) == 3
    w2.close()


def test_a_tiny_plot_area_is_refused(app, store, tmp_path):
    path, _ = make_pdfs.picture_file(tmp_path / "open.png", frame=False)
    w = open_image(app, store, path)
    w.show_chart(0)
    w.start_plot_area()
    w.view.rectSelected.emit(Rect(10, 10, 30, 25))
    assert not w._chart.raster.plot_manual and "çok küçük" in w.statusBar().currentMessage()
    w.close()


def test_changing_the_plot_area_of_a_calibrated_chart_rereads_the_curves(app, store, tmp_path, monkeypatch):
    path, truth = make_pdfs.picture_file(tmp_path / "chart.png")
    w = open_image(app, store, path)
    w.show_chart(0)
    calibrate(w, truth)
    w._on_point_delete(5)
    assert w._curve.edited
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: NO)
    w.start_plot_area()
    x0, y0, x1, y1 = truth["frame"]
    w.view.rectSelected.emit(Rect(x0, y0, x1, y1))
    assert w._curve.edited and not w._chart.raster.plot_manual                   # declined: nothing changed
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    w.start_plot_area()
    w.view.rectSelected.emit(Rect(x0, y0, x1, y1))
    assert w._chart.raster.plot_manual and len(w._chart.curves) == 3 and not w._curve.edited
    w.close()


def test_unreadable_and_tiny_images_are_reported(app, store, tmp_path, monkeypatch):
    shown = []
    monkeypatch.setattr("ui.lookup_window.QMessageBox.critical", lambda *a, **k: shown.append(a[2]))
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    w = LookupWindow(store=store)
    w.open_pdf(str(bad), blocking=True)
    assert shown and "açılamadı" in shown[0] and w.stack.currentIndex() == 0
    import cv2

    tiny = tmp_path / "tiny.png"
    cv2.imwrite(str(tiny), np.full((30, 30, 3), 255, np.uint8))
    with pytest.raises(ValueError):
        chart_from_image(tiny)
    w.close()


def test_dropping_an_image_opens_it_and_other_files_are_ignored(app, store, tmp_path):
    path, _ = make_pdfs.picture_file(tmp_path / "chart.jpg")
    w = LookupWindow(store=store)
    w.show()
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(tmp_path / "notes.txt"))])
    ev = QDropEvent(QPointF(5, 5), Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    w.dropEvent(ev)
    assert w.gallery.count() == 0
    mime.setUrls([QUrl.fromLocalFile(str(path))])
    w.dropEvent(QDropEvent(QPointF(5, 5), Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton,
                           Qt.KeyboardModifier.NoModifier))
    from tests.test_lookup_ui import wait_until

    assert wait_until(lambda: w.gallery.count() == 1)
    w.close()


def test_a_path_with_non_ascii_characters_works(app, store, tmp_path):
    d = tmp_path / "grafikler çşğ"
    d.mkdir()
    path, truth = make_pdfs.picture_file(d / "deşarj eğrisi.png")
    w = open_image(app, store, path)
    w.show_chart(0)
    calibrate(w, truth)
    assert len(w._chart.curves) == 3 and "deşarj eğrisi" in w.windowTitle()
    w.close()
