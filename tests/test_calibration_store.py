"""Remembered calibrations: keys, file handling, damaged files."""
import json
from types import SimpleNamespace

import pytest

from core.calibration_store import CalibrationStore, chart_key, data_dir, pdf_fingerprint
from core.raster_charts import AxisPoints


def axes():
    return (AxisPoints((10.5, 900.25), (0.0, 3000.0), "Capacity (mAh)"),
            AxisPoints((700.0, 40.0), (2.0, 4.5), "Voltage (V)"),
            AxisPoints((700.0, 40.0), (1.0, 1000.0), "Gain", log=True))


def test_round_trip_keeps_every_number_name_and_the_log_flag(tmp_path):
    st = CalibrationStore(tmp_path / "c.json")
    x, y, y2 = axes()
    assert st.get("h1", "k1") is None and not st.has("h1", "k1")
    assert st.save("h1", "a.pdf", "k1", x, y, y2)
    got = st.get("h1", "k1")
    assert got is not None
    gx, gy, gy2, saved, plot = got
    assert gx == x and gy == y and gy2 == y2 and gy2.log and len(saved) == 16 and plot is None
    assert st.get("h1", "other") is None and st.get("h2", "k1") is None


def test_no_second_axis_is_stored_as_none(tmp_path):
    st = CalibrationStore(tmp_path / "c.json")
    x, y, _ = axes()
    st.save("h", "a.pdf", "k", x, y, None)
    assert st.get("h", "k")[2] is None


def test_saving_again_replaces_and_other_charts_of_the_pdf_are_kept(tmp_path):
    st = CalibrationStore(tmp_path / "c.json")
    x, y, _ = axes()
    st.save("h", "a.pdf", "k1", x, y, None)
    st.save("h", "a.pdf", "k2", x, y, None)
    x2 = AxisPoints((1.0, 2.0), (5.0, 6.0), "New")
    st.save("h", "a.pdf", "k1", x2, y, None)
    assert st.get("h", "k1")[0] == x2 and st.get("h", "k2")[0] == x


def test_forget_removes_one_chart_and_then_the_empty_pdf(tmp_path):
    st = CalibrationStore(tmp_path / "c.json")
    x, y, _ = axes()
    st.save("h", "a.pdf", "k1", x, y, None)
    st.save("h", "a.pdf", "k2", x, y, None)
    assert st.forget("h", "k1") and st.get("h", "k1") is None and st.has("h", "k2")
    assert not st.forget("h", "k1")                                       # already gone
    assert st.forget("h", "k2")
    assert json.loads((tmp_path / "c.json").read_text(encoding="utf-8"))["pdfs"] == {}


def test_a_damaged_file_is_set_aside_and_the_store_keeps_working(tmp_path):
    p = tmp_path / "c.json"
    p.write_text("{ not json", encoding="utf-8")
    st = CalibrationStore(p)
    assert st.get("h", "k") is None
    assert (tmp_path / "c.bad").exists()
    x, y, _ = axes()
    assert st.save("h", "a.pdf", "k", x, y, None) and st.has("h", "k")


def test_wrong_version_or_garbled_entries_are_ignored(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"version": 99, "pdfs": {"h": {"charts": {"k": {}}}}}), encoding="utf-8")
    st = CalibrationStore(p)
    assert st.get("h", "k") is None
    p.write_text(json.dumps({"version": 1, "pdfs": {"h": {"charts": {"k": {"x": {"px": [1], "value": [1, 2]}, "y": None}}}}}),
                 encoding="utf-8")
    assert st.get("h", "k") is None


def test_unwritable_location_does_not_raise(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    st = CalibrationStore(blocker / "sub" / "c.json")                     # a *file* where the folder should be
    x, y, _ = axes()
    assert st.save("h", "a.pdf", "k", x, y, None) is False and st.get("h", "k") is None


def test_the_oldest_pdfs_are_dropped_beyond_the_limit(tmp_path, monkeypatch):
    import core.calibration_store as cs

    monkeypatch.setattr(cs, "MAX_PDFS", 3)
    st = CalibrationStore(tmp_path / "c.json")
    x, y, _ = axes()
    for i in range(5):
        st.save(f"h{i}", f"{i}.pdf", "k", x, y, None)
    kept = [h for h in ("h0", "h1", "h2", "h3", "h4") if st.has(h, "k")]
    assert len(kept) == 3 and "h4" in kept


def test_fingerprint_follows_the_content_not_the_name(tmp_path):
    a, b, c = tmp_path / "a.pdf", tmp_path / "renamed copy.pdf", tmp_path / "c.pdf"
    a.write_bytes(b"%PDF-1.4 same bytes")
    b.write_bytes(b"%PDF-1.4 same bytes")
    c.write_bytes(b"%PDF-1.4 other bytes")
    assert pdf_fingerprint(a) == pdf_fingerprint(b) != pdf_fingerprint(c)


def test_chart_key_uses_page_place_and_size():
    bbox = SimpleNamespace(x0=71.4, y0=88.6, x1=524.0, y1=374.7)
    assert chart_key(0, bbox, (760, 1200, 3)) == "p0:71,89,524,375:1200x760"
    assert chart_key(1, bbox, (760, 1200, 3)) != chart_key(0, bbox, (760, 1200, 3))


def test_data_dir_can_be_redirected_and_tests_do_not_use_the_real_one(monkeypatch, tmp_path):
    assert "plotdigitizer-test-" in str(data_dir())                       # set by tests/conftest.py
    monkeypatch.setenv("PLOT_DIGITIZER_DATA", str(tmp_path))
    assert data_dir() == tmp_path
    monkeypatch.delenv("PLOT_DIGITIZER_DATA")
    assert data_dir().name == "PlotDigitizer"


def test_a_hand_drawn_plot_area_is_remembered_too(tmp_path):
    st = CalibrationStore(tmp_path / "c.json")
    x, y, _ = axes()
    st.save("h", "a.png", "k", x, y, None, plot=(10.0, 20.0, 300.5, 400.25))
    assert st.get("h", "k").plot == (10.0, 20.0, 300.5, 400.25)
    st.save("h", "a.png", "k", x, y, None)
    assert st.get("h", "k").plot is None
