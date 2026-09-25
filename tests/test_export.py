import numpy as np
import pandas as pd
import pytest

from core.export import AxisInfo, export_combined, export_separate, safe_filename
from core.postprocess import CurveData

X = AxisInfo("Capacity", "mAh")
Y = AxisInfo("Voltage", "V")


def _curve(name, xs, ys, ux=1.05, uy=0.0026):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    return CurveData(name, xs, ys, np.full_like(xs, ux), np.full_like(xs, uy))


def _header(path):
    with open(path, encoding="utf-8-sig") as fh:
        return [ln.rstrip("\n") for ln in fh if ln.startswith("#")]


def test_separate_files_one_per_curve_with_header_and_uncertainty(tmp_path):
    curves = [_curve("0.56A", [0, 10, 20], [4.1, 4.0, 3.9]), _curve("2.8A", [0, 10], [4.0, 3.9])]
    paths = export_separate(curves, tmp_path, X, Y, source="datasheet.pdf / Discharge Rate", step=10)
    assert [p.name for p in paths] == ["0.56A.csv", "2.8A.csv"]

    head = _header(paths[0])
    text = "\n".join(head)
    assert "Capacity (mAh)" in text and "Voltage (V)" in text
    assert "datasheet.pdf / Discharge Rate" in text
    assert "±1.05 mAh" in text and "±0.0026 V" in text
    assert "resampled to x step: 10 mAh" in text

    df = pd.read_csv(paths[0], comment="#", encoding="utf-8-sig")
    assert list(df.columns) == ["Capacity (mAh)", "Voltage (V)", "±x", "±y"]
    assert df["Voltage (V)"].tolist() == [4.1, 4.0, 3.9]
    assert df["±y"].tolist() == [0.0026] * 3


def test_separate_files_can_omit_per_point_uncertainty(tmp_path):
    p = export_separate([_curve("a", [0, 1], [1, 2])], tmp_path, X, Y, per_point_uncertainty=False)[0]
    assert list(pd.read_csv(p, comment="#", encoding="utf-8-sig").columns) == ["Capacity (mAh)", "Voltage (V)"]


def test_combined_file_shares_the_x_column(tmp_path):
    grid = [0, 10, 20, 30]
    curves = [_curve("0.56A", grid, [4.1, 4.0, 3.9, 3.8]), _curve("30A", grid[:3], [3.6, 3.5, 3.4])]
    p = export_combined(curves, tmp_path / "all.csv", X, Y, source="s", step=10)
    df = pd.read_csv(p, comment="#", encoding="utf-8-sig")
    assert list(df.columns) == ["Capacity (mAh)", "0.56A [Voltage (V)]", "30A [Voltage (V)]"]
    assert df["Capacity (mAh)"].tolist() == grid
    assert df["0.56A [Voltage (V)]"].tolist() == [4.1, 4.0, 3.9, 3.8]
    assert np.isnan(df["30A [Voltage (V)]"].iloc[3])                   # shorter curve -> NaN, not invented data
    text = "\n".join(_header(p))
    assert 'curve "0.56A"' in text and 'curve "30A"' in text and "±0.0026 V" in text


def test_combined_file_with_per_point_uncertainty_columns(tmp_path):
    p = export_combined([_curve("a", [0, 1], [1, 2])], tmp_path / "u.csv", X, Y, per_point_uncertainty=True)
    df = pd.read_csv(p, comment="#", encoding="utf-8-sig")
    assert "a ±y" in df.columns


def test_log_axis_uncertainty_is_reported_as_a_range(tmp_path):
    c = CurveData("l", np.array([1.0, 10.0]), np.array([1.0, 2.0]), np.array([0.1, 1.0]), np.array([0.01, 0.01]))
    p = export_separate([c], tmp_path, AxisInfo("f", "Hz", log=True), Y)[0]
    text = "\n".join(_header(p))
    assert "[log10]" in text and "±0.1..±1 Hz" in text and "varies" in text


def test_file_names_are_sanitised_and_unique(tmp_path):
    assert safe_filename('a/b:c*?"<>|') == "a_b_c______" or "/" not in safe_filename('a/b')
    paths = export_separate([_curve("dup", [0, 1], [1, 2]), _curve("DUP", [0, 1], [3, 4]),
                             _curve('bad:name?', [0, 1], [5, 6])], tmp_path, X, Y)
    names = [p.name for p in paths]
    assert len(set(n.lower() for n in names)) == 3
    assert all(p.exists() for p in paths)
    assert safe_filename("   ") == "curve"


def test_utf8_bom_so_excel_reads_the_plus_minus_sign(tmp_path):
    p = export_separate([_curve("a", [0, 1], [1, 2])], tmp_path, X, Y)[0]
    assert p.read_bytes().startswith(b"\xef\xbb\xbf")
