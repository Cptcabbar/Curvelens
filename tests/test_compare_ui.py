"""Chart dots follow the lookup table; the 'Tablo karşılaştırma' tab (import, matching, new chart)."""
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox      # noqa: E402

from core.calibration_store import CalibrationStore                       # noqa: E402
from core.lookup import build_lookup, write_csv                           # noqa: E402
from ui.compare_tab import PairDialog                                     # noqa: E402
from ui.image_view import Tool                                            # noqa: E402
from ui.lookup_window import LookupWindow                                 # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "samples" / "INR18650P28A-V1-80093.pdf"
pytestmark = pytest.mark.skipif(not PDF.exists(), reason="sample datasheet PDF missing")


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def win(app, tmp_path):
    w = LookupWindow(store=CalibrationStore(tmp_path / "cal.json"))
    w.resize(1400, 850)
    w.show()
    app.processEvents()
    w.open_pdf(str(PDF), blocking=True)
    w.show_chart(1)                                        # Discharge Rate Characteristics, curve 0.56A selected
    yield w
    w.close()


def dots(view, i=0):
    poly = view._curve_items[i]._poly
    return np.array([(poly.at(k).x(), poly.at(k).y()) for k in range(poly.count())])


def table_xy(w):
    t = w._table
    ok = np.isfinite(t.y)
    return t.x[ok], t.y[ok]


# ------------------------------------------------------------------------------- dots = table rows
def test_dots_on_the_chart_are_the_rows_of_the_lookup_table(win):
    w = win
    x, y = table_xy(w)
    d = dots(w.view)
    assert len(d) == len(x) == 56
    px = w._data_to_px(x, y)
    assert np.allclose(d[:, 0], px[0], atol=1e-3) and np.allclose(d[:, 1], px[1], atol=1e-3)
    back = np.array([w._px_to_data(a, b) for a, b in d])
    assert np.allclose(back[:, 0], x, atol=1e-6) and np.allclose(back[:, 1], y, atol=1e-6)
    assert "56 nokta" in w.points_note.text() and "lookup tablosunun satırlarıyla aynı" in w.points_note.text()


def test_changing_the_step_changes_the_dots_count_and_values(win):
    w = win
    seen = []
    for step, rows in ((100.0, 28), (25.0, None), (200.0, 14)):
        w.step_spin.setValue(step)
        x, y = table_xy(w)
        d = dots(w.view)
        rows = rows or w.table_model.rowCount()
        seen.append(rows)
        assert len(d) == len(x) == rows == w.table_model.rowCount()
        back = np.array([w._px_to_data(a, b) for a, b in d])
        assert np.allclose(back[:, 0], x, atol=1e-6) and np.allclose(back[:, 1], y, atol=1e-6)
        assert f"{rows} nokta" in w.points_note.text()
    assert seen[1] > 100                                   # a finer step really gives more rows
    w.step_spin.setValue(0.0)                              # "Ham noktalar": the curve's own points
    assert len(dots(w.view)) == len(w._table) > 100


def test_table_rows_and_dots_agree_in_the_displayed_numbers(win):
    w = win
    w.step_spin.setValue(300.0)
    m = w.table_model
    rows = [(float(m.data(m.index(r, 0))), float(m.data(m.index(r, 1)))) for r in range(m.rowCount())]
    back = [w._px_to_data(a, b) for a, b in dots(w.view)]
    assert len(rows) == len(back)
    for (tx, ty), (bx, by) in zip(rows, back):
        assert bx == pytest.approx(tx, abs=1e-6) and by == pytest.approx(ty, abs=1e-3)      # table shows 3 decimals


def test_correction_tools_show_the_raw_points_and_leaving_them_returns_to_the_table(win):
    w = win
    w.step_spin.setValue(100.0)
    assert len(dots(w.view)) == 28
    w._set_edit_tool(Tool.MOVE_POINT)
    assert len(dots(w.view)) == w._curve.n_points and "ham nokta" in w.points_note.text()
    w._set_edit_tool(Tool.PAN)
    assert len(dots(w.view)) == 28 and "28 nokta" in w.points_note.text()


def test_a_correction_updates_the_dots_and_the_table_together(win):
    w = win
    w.step_spin.setValue(100.0)
    before = table_xy(w)[1].copy()
    w._on_point_delete(len(w._curve.data.x) // 2)
    w._set_edit_tool(Tool.PAN)
    x, y = table_xy(w)
    assert len(dots(w.view)) == len(x)
    assert w._curve.edited
    w.step_spin.setValue(50.0)
    assert len(dots(w.view)) == 56 and before is not None


def test_selecting_table_rows_rings_those_points_on_the_chart(win):
    w = win
    w.step_spin.setValue(100.0)
    x, y = table_xy(w)
    w.table.selectRow(5)
    assert len(w.view._row_items) == 1
    px = w._data_to_px(x[5], y[5])
    pos = w.view._row_items[0].pos()
    assert pos.x() == pytest.approx(float(px[0]), abs=1e-3) and pos.y() == pytest.approx(float(px[1]), abs=1e-3)
    w.table.clearSelection()
    assert w.view._row_items == []
    w.step_spin.setValue(200.0)                             # the table is rebuilt: no stale ring
    assert w.view._row_items == []


# ------------------------------------------------------------------------------- import + pairing
def export_table(w, tmp_path, offset=0.0, name="lookup.csv", step=100.0):
    t = build_lookup(w._chart, w._curve, step)
    t.y = t.y + offset
    return write_csv(t, tmp_path / name)


def pair_with(w, chart_i, curve_i):
    w.compare._choose_pairing = lambda table, charts, current: ("chart", chart_i, curve_i)


def test_import_matches_the_table_with_the_chart_and_draws_its_points(win, tmp_path):
    w = win
    csv = export_table(w, tmp_path)
    pair_with(w, 1, 0)
    assert w.compare.import_table(str(csv))
    c = w.compare
    assert c.mode == "chart" and c.chart is w._chart and c.curve is w._curve
    assert len(dots(c.view)) == 28
    assert c.cmp.n_total == 28 and c.cmp.n_compared == 28
    assert c.cmp.max_abs < 6e-4 and c.cmp.mean_abs < 6e-4                    # the table IS the curve (3 decimals in the file)
    assert "Grafikte 28 nokta işaretli" in c.summary.text() and "Eğrinin belirsizliği ±0.0026" in c.summary.text()
    assert c.model.rowCount() == 28 and c.model.columnCount() == 5
    assert "eşleşen grafik: Discharge Rate Characteristics → 0.56A" in c.file_label.text()


def test_the_import_menu_switches_to_the_comparison_tab(win, tmp_path, monkeypatch):
    w = win
    csv = export_table(w, tmp_path)
    pair_with(w, 1, 0)
    monkeypatch.setattr(w.compare, "_pick_file", lambda: str(csv))
    assert w.tabs.currentIndex() == 0
    w.act_import.trigger()
    assert w.tabs.currentIndex() == 1 and w.compare.table is not None
    assert not w.act_back.isVisible()                                          # the gallery button belongs to tab 1


def test_differences_from_the_curve_are_measured(win, tmp_path):
    w = win
    csv = export_table(w, tmp_path, offset=0.05)
    pair_with(w, 1, 0)
    w.compare.import_table(str(csv))
    c = w.compare.cmp
    assert c.mean == pytest.approx(0.05, abs=6e-4) and c.max_abs == pytest.approx(0.05, abs=6e-4)
    assert 1.0 < c.mean_rel_pct < 2.0                                          # 0.05 V of about 3.5 V
    import re

    assert re.search(r"ortalama \+0\.0(49|50)", w.compare.summary.text())
    m = w.compare.model
    assert float(m.data(m.index(3, 3))) == pytest.approx(0.05, abs=6e-4)


def test_points_beyond_the_chart_area_are_counted_not_drawn(win, tmp_path):
    w = win
    p = tmp_path / "far.csv"
    p.write_text("Capacity (mAh),Voltage (V)\n500,3.9\n1000,3.7\n5000,3.0\n9000,2.0\n", encoding="utf-8")
    pair_with(w, 1, 0)
    w.compare.import_table(str(p))
    c = w.compare
    assert c.cmp.n_total == 4 and c.cmp.n_compared == 2 and c.out_of_view == 2
    assert len(dots(c.view)) == 2 and "grafik alanı dışında kalan: 2" in c.summary.text()


def test_no_open_chart_means_a_new_chart_without_asking(win, tmp_path, monkeypatch):
    w = win
    w.charts = []                                                              # nothing to match with
    called = []
    monkeypatch.setattr(w.compare, "_choose_pairing", lambda *a: called.append(a))
    p = tmp_path / "t.csv"
    p.write_text("Zaman (s);Sıcaklık (C)\n0;20,5\n10;25,0\n20;31,5\n", encoding="utf-8")
    assert w.compare.import_table(str(p)) and called == []
    c = w.compare
    assert c.mode == "new" and c.graph is not None and c.chart is None
    assert len(dots(c.view)) == 3 and "eşleştirilecek açık grafik olmadığı için" in w.statusBar().currentMessage()
    px = c.graph.to_px(c.table.x, c.table.ys[0])
    assert np.allclose(dots(c.view)[:, 0], px[0], atol=1e-3) and np.allclose(dots(c.view)[:, 1], px[1], atol=1e-3)
    back = c.graph.to_data(*px)
    assert np.allclose(back[0], [0, 10, 20]) and np.allclose(back[1], [20.5, 25.0, 31.5])
    assert c.graph.image.shape == (760, 1200, 3) and c.graph.image.std() > 5          # something was drawn
    assert "Grafikte 3 nokta işaretli" in c.summary.text() and "Eşleştir" in c.summary.text()


def test_choosing_new_chart_in_the_dialog_gives_a_chart_of_its_own(win, tmp_path):
    w = win
    csv = export_table(w, tmp_path)
    w.compare._choose_pairing = lambda *a: ("new",)
    w.compare.import_table(str(csv))
    assert w.compare.mode == "new" and w.compare.cmp is None and len(dots(w.compare.view)) == 28
    assert w.compare.model.columnCount() == 2


def test_several_y_columns_are_drawn_and_the_column_can_be_chosen(win, tmp_path):
    w = win
    p = tmp_path / "multi.csv"
    p.write_text("Capacity (mAh),A (V),B (V)\n0,4.1,3.9\n1000,3.8,3.5\n2000,3.4,3.0\n", encoding="utf-8")
    w.compare._choose_pairing = lambda *a: ("new",)
    w.compare.import_table(str(p))
    c = w.compare
    assert c.col_combo.count() == 2 and not c.col_combo.isHidden()
    assert len(c._curve_items_count()) == 2 if hasattr(c, "_curve_items_count") else len(c.view._curve_items) == 2
    assert c.model.columnCount() == 3
    pair_with(w, 1, 0)
    c.pair()
    assert c.mode == "chart" and c.col_combo.count() == 2
    first = c.cmp.y.copy()
    c.col_combo.setCurrentIndex(1)
    assert not np.array_equal(first, c.cmp.y) and c.cmp.y[0] == pytest.approx(3.9)
    assert "B (V) (tablo)" in [c.model.headerData(1, __import__("PySide6.QtCore", fromlist=["Qt"]).Qt.Orientation.Horizontal)]


def test_a_unit_mismatch_is_reported(win, tmp_path):
    w = win
    p = tmp_path / "ah.csv"
    p.write_text("Capacity (Ah)\tVoltage (V)\n0,5\t3,9\n1,0\t3,7\n", encoding="utf-8")
    pair_with(w, 1, 0)
    w.compare.import_table(str(p))
    assert "birimleri uyuşmuyor" in w.compare.summary.text()


def test_row_selection_in_the_comparison_table_rings_the_point(win, tmp_path):
    w = win
    csv = export_table(w, tmp_path)
    pair_with(w, 1, 0)
    w.compare.import_table(str(csv))
    c = w.compare
    c.grid.selectRow(3)
    assert len(c.view._row_items) == 1
    d = dots(c.view)[3]
    pos = c.view._row_items[0].pos()
    assert (pos.x(), pos.y()) == pytest.approx((d[0], d[1]), abs=1e-3)


def test_cancelling_the_pairing_dialog_leaves_everything_as_it_was(win, tmp_path):
    w = win
    csv = export_table(w, tmp_path)
    w.compare._choose_pairing = lambda *a: None
    assert not w.compare.import_table(str(csv))
    assert w.compare.table is None and w.compare.mode == "none" and w.tabs.currentIndex() == 0
    pair_with(w, 1, 0)
    w.compare.import_table(str(csv))
    w.compare._choose_pairing = lambda *a: None
    w.compare.pair()
    assert w.compare.mode == "chart"                                           # unchanged


def test_unreadable_tables_are_reported(win, tmp_path, monkeypatch):
    shown = []
    monkeypatch.setattr("ui.compare_tab.QMessageBox.critical", lambda *a, **k: shown.append(a[2]))
    p = tmp_path / "bad.csv"
    p.write_text("no numbers here\nat all\n", encoding="utf-8")
    assert not win.compare.import_table(str(p)) and shown and "Tablo açılamadı" in shown[0]
    assert win.compare.table is None


def test_the_comparison_follows_corrections_made_in_the_other_tab(win, tmp_path):
    w = win
    csv = export_table(w, tmp_path)
    pair_with(w, 1, 0)
    w.compare.import_table(str(csv))
    assert w.compare.cmp.max_abs < 6e-4
    n = w._curve.delete_in_box(1800, 2200, 0, 10)                                # hand-correct the curve: a hole
    w._curve.add_point(2000.0, 3.0, 1.0, 0.01)
    assert n > 0
    w.tabs.setCurrentIndex(0)
    w.tabs.setCurrentIndex(1)
    assert w.compare.cmp.max_abs > 0.3 and "en büyük" in w.compare.summary.text()


def test_opening_another_document_drops_the_match(win, tmp_path):
    w = win
    csv = export_table(w, tmp_path)
    pair_with(w, 1, 0)
    w.compare.import_table(str(csv))
    assert w.compare.mode == "chart"
    w.open_pdf(str(PDF), blocking=True)
    assert w.compare.mode == "new" and w.compare.chart is None and len(dots(w.compare.view)) == 28
    assert "kaldırıldı" in w.compare.file_label.text()


def test_save_buttons_write_files(win, tmp_path, monkeypatch):
    w = win
    csv = export_table(w, tmp_path)
    pair_with(w, 1, 0)
    w.compare.import_table(str(csv))
    out_csv, out_png = tmp_path / "cmp.csv", tmp_path / "cmp.png"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(out_csv), "CSV (*.csv)"))
    w.compare.save_csv()
    text = out_csv.read_text(encoding="utf-8-sig").splitlines()
    assert text[0].startswith("# Tablo: lookup.csv") and text[1].startswith("Capacity (mAh),Voltage (V) (tablo)") and len(text) == 30
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(out_png), "PNG (*.png)"))
    w.compare.save_png()
    assert out_png.read_bytes()[:4] == b"\x89PNG"


# ------------------------------------------------------------------------------- the pairing dialog
def test_pair_dialog_offers_only_usable_charts_and_defaults_to_the_current_curve(win, app):
    w = win
    from core.table_import import ImportedTable

    t = ImportedTable("t.csv", "Capacity (mAh)", ["Voltage (V)"], np.array([0.0, 1.0]), [np.array([1.0, 2.0])])
    dlg = PairDialog(None, t, w.charts, (1, 2))
    assert dlg.chart_box.count() == 4 and dlg.chart_box.currentData() == 1 and dlg.curve_box.currentIndex() == 2
    assert dlg.r_chart.isChecked() and dlg.r_new.isEnabled()
    dlg.accept()
    assert dlg.result_choice == ("chart", 1, 2)
    dlg2 = PairDialog(None, t, w.charts, None)
    dlg2.r_new.setChecked(True)
    dlg2.accept()
    assert dlg2.result_choice == ("new",)


def test_pair_dialog_disables_uncalibrated_picture_charts_and_offers_new_chart(app, tmp_path):
    pytest.importorskip("matplotlib")
    from tests import make_pdfs
    from core.table_import import ImportedTable

    path, truth = make_pdfs.picture_chart(tmp_path / "pic.pdf")
    w = LookupWindow(store=CalibrationStore(tmp_path / "cal.json"))
    w.open_pdf(str(path), blocking=True)
    t = ImportedTable("t.csv", "X", ["Y"], np.array([0.0, 1.0]), [np.array([1.0, 2.0])])
    dlg = PairDialog(None, t, w.charts, None)
    assert "önce kalibre edin" in dlg.chart_box.itemText(0) and not dlg.r_chart.isEnabled() and dlg.r_new.isChecked()
    dlg.accept()
    assert dlg.result_choice == ("new",)
    called = []
    w.compare._choose_pairing = lambda *a: called.append(1)
    p = tmp_path / "t.csv"
    p.write_text("X,Y\n1,2\n2,3\n", encoding="utf-8")
    w.compare.import_table(str(p))
    assert called == [] and w.compare.mode == "new"                            # no usable chart: no question
    w.close()


def test_pair_dialog_warns_about_units(win):
    from core.table_import import ImportedTable

    t = ImportedTable("t.csv", "Capacity (Ah)", ["Voltage (mV)"], np.array([0.0, 1.0]), [np.array([1.0, 2.0])])
    dlg = PairDialog(None, t, win.charts, (1, 0))
    assert "X birimi: tabloda Ah, grafikte mAh" in dlg.warn.text() and "Y birimi: tabloda mV, grafikte V" in dlg.warn.text()
