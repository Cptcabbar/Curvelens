"""Lookup tables read from text files, and their comparison with a curve."""
import math

import numpy as np
import pytest

from core.lookup import LookupTable, write_csv
from core.postprocess import CurveData
from core.table_import import compare_to_curve, parse_number, read_table, units_conflict


def write(tmp_path, name, text, enc="utf-8"):
    p = tmp_path / name
    p.write_bytes(text.encode(enc))
    return p


def test_parse_number_variants():
    assert parse_number("1500") == 1500 and parse_number(" 3,65 ") == 3.65 and parse_number("2.5e3") == 2500
    assert parse_number("1.234,5") == 1234.5 and parse_number("1,234.5") == 1234.5 and parse_number("−") != parse_number("−")
    for empty in ("", "—", "nan", "NA", "n/a", "abc", "inf"):
        assert math.isnan(parse_number(empty))


def test_the_applications_own_csv_export_round_trips(tmp_path):
    x = np.array([50.0, 100.0, 150.0])
    t = LookupTable("Chart", "Curve", "solid", "Capacity (mAh)", "Voltage (V)", x, np.array([4.14, 4.11, np.nan]), 50.0, 0, 3,
                    1.2, 0.0026, source="a.pdf", page=1)
    p = write_csv(t, tmp_path / "out.csv")
    r = read_table(p)
    assert r.x_header == "Capacity (mAh)" and r.y_headers == ["Voltage (V)"]
    assert list(r.x) == [50.0, 100.0, 150.0]
    assert r.ys[0][0] == 4.14 and math.isnan(r.ys[0][2])                          # the empty cell stays a gap
    assert r.comments and "Lookup tablosu" in r.comments[0]


@pytest.mark.parametrize("text,expect", [
    ("X,Y\n1,10\n2,20\n3,30\n", ("X", [1, 2, 3], [10, 20, 30])),
    ("X;Y\n1,5;10,5\n2,5;20,5\n", ("X", [1.5, 2.5], [10.5, 20.5])),                  # decimal comma with ;
    ("Zaman (s)\tGerilim (V)\n0\t4,2\n10\t4,1\n", ("Zaman (s)", [0, 10], [4.2, 4.1])),  # tab + decimal comma
    ("1 10\n2 20\n3 30\n", ("X", [1, 2, 3], [10, 20, 30])),                            # blanks, no header
    ("1 10,5\n2 20,5\n", ("X", [1, 2], [10.5, 20.5])),                                  # blanks + decimal comma
    ("1|10\n2|20\n", ("X", [1, 2], [10, 20])),
    ("1.234,5;2\n2.234,5;3\n", ("X", [1234.5, 2234.5], [2, 3])),                        # thousands dots
])
def test_common_layouts(tmp_path, text, expect):
    r = read_table(write(tmp_path, "t.csv", text))
    xh, xs, ys = expect
    assert r.x_header == xh and list(r.x) == xs and list(r.ys[0]) == ys


def test_several_y_columns_and_unsorted_rows_are_sorted_by_x(tmp_path):
    r = read_table(write(tmp_path, "t.csv", "T,A,B\n3,30,300\n1,10,100\n2,20,\n"))
    assert list(r.x) == [1, 2, 3] and r.y_headers == ["A", "B"]
    assert list(r.ys[0]) == [10, 20, 30] and r.ys[1][0] == 100 and math.isnan(r.ys[1][1])


def test_a_column_without_any_number_is_dropped(tmp_path):
    r = read_table(write(tmp_path, "t.csv", "X,A,B\n1,10,\n2,20,\n"))
    assert r.y_headers == ["A"]


def test_bad_lines_are_counted_not_fatal(tmp_path):
    r = read_table(write(tmp_path, "t.csv", "X,Y\n1,10\nnotes here,\n2,20\n3\n4,40\n"))
    assert list(r.x) == [1, 2, 4] and r.skipped == 2


def test_turkish_windows_encoding(tmp_path):
    p = write(tmp_path, "t.csv", "Süre (sn);Gerilim (V)\n1;4,2\n2;4,1\n", enc="cp1254")
    r = read_table(p)
    assert r.x_header == "Süre (sn)" and r.y_headers == ["Gerilim (V)"]
    p2 = write(tmp_path, "t2.csv", "Süre (sn);Gerilim (V)\n1;4,2\n2;4,1\n", enc="utf-8-sig")
    assert read_table(p2).x_header == "Süre (sn)"


@pytest.mark.parametrize("text,msg", [
    ("", "satırı yok"), ("a,b\n", "satırı yok"), ("a,b\nc,d\ne,f\n", "sayısal"), ("1\n2\n3\n", "ayrılamadı"),
    ("X,Y\n1,\n2,\n", "sayı yok"), ("X,Y\n1,5\n", "iki satır")])
def test_unusable_files_give_a_clear_error(tmp_path, text, msg):
    with pytest.raises(ValueError, match=msg):
        read_table(write(tmp_path, "t.csv", text))


def test_missing_file_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="okunamadı"):
        read_table(tmp_path / "nope.csv")


def curve(xs, ys):
    xs = np.asarray(xs, float)
    return CurveData("c", xs, np.asarray(ys, float), np.zeros(len(xs)), np.zeros(len(xs)))


def test_comparison_statistics():
    c = curve([0, 10, 20], [0, 10, 20])                                   # y = x
    x = np.array([-5.0, 5.0, 10.0, 15.0, 25.0])
    y = np.array([0.0, 6.0, 10.0, 14.0, 30.0])
    r = compare_to_curve(x, y, c)
    assert r.n_total == 5 and r.n_compared == 3                            # -5 and 25 are outside the curve
    assert np.isnan(r.y_curve[0]) and np.isnan(r.y_curve[4])
    assert list(r.delta[1:4]) == [1.0, 0.0, -1.0]
    assert r.mean == 0.0 and r.mean_abs == pytest.approx(2 / 3) and r.max_abs == 1.0 and r.x_of_max in (5.0, 15.0)
    assert r.rms == pytest.approx(math.sqrt(2 / 3))
    assert r.max_rel_pct == pytest.approx(20.0)                            # |1| / 5
    assert r.mean_rel_pct == pytest.approx((20.0 + 0.0 + 100 / 15) / 3)


def test_comparison_without_overlap_or_with_gaps():
    r = compare_to_curve([100.0, 200.0], [1.0, 2.0], curve([0, 10], [0, 1]))
    assert r.n_compared == 0 and math.isnan(r.mean_abs)
    r2 = compare_to_curve([5.0, 6.0], [np.nan, 3.0], curve([0, 10], [0, 10]))
    assert r2.n_compared == 1 and r2.delta[1] == pytest.approx(-3.0)
    assert compare_to_curve([1.0], [1.0], curve([0], [0])).n_compared == 0     # a one-point curve has no line


def test_units_conflict():
    assert units_conflict("Capacity (mAh)", "Capacity (Ah)")
    assert not units_conflict("Capacity (mAh)", "Kapasite (MAH)")
    assert not units_conflict("Capacity", "Kapasite (mAh)") and not units_conflict("X", "Y")
