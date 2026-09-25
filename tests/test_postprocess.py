import numpy as np
import pytest

from core.calibration import AxisCalibration, Calibration
from core.postprocess import CurveData, resample, smooth_savgol, to_curve_data, uncertainty_summary


def _curve(x, y, name="c"):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return CurveData(name, x, y, np.full_like(x, 0.5), np.full_like(x, 0.01))


# ---------------------------------------------------------------------- smoothing
def test_savgol_reduces_noise_and_keeps_shape():
    rng = np.random.default_rng(1)
    x = np.arange(0.5, 600.5)
    clean = 200 + 30 * np.sin(x / 90)
    noisy = clean + rng.normal(0, 1.0, x.size)
    sm = smooth_savgol(noisy, x, window=31, polyorder=3)
    assert np.std(sm - clean) < 0.4 * np.std(noisy - clean)
    assert np.max(np.abs(sm - clean)) < 1.0


def test_savgol_preserves_a_straight_line_exactly():
    x = np.arange(0.5, 100.5)
    y = 3.0 + 0.7 * x
    assert smooth_savgol(y, x, window=11, polyorder=2) == pytest.approx(y, abs=1e-9)


def test_savgol_handles_gaps_short_input_and_even_windows():
    x = np.array([1.5, 2.5, 3.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5])   # hole at 4.5..6.5
    y = 2 * x
    out = smooth_savgol(y, x, window=6, polyorder=2)                         # even window -> made odd
    assert out == pytest.approx(y, abs=1e-8)
    assert smooth_savgol(np.array([1.0, 2.0]), np.array([0.5, 1.5])) == pytest.approx([1.0, 2.0])
    huge = smooth_savgol(y, x, window=999, polyorder=9)                       # clamped to the data length
    assert huge.shape == y.shape and np.all(np.isfinite(huge))


# ---------------------------------------------------------------------- calibration -> data
def test_to_curve_data_applies_calibration_sorts_and_computes_uncertainty():
    cal = Calibration(AxisCalibration(100, 0, 500, 3000), AxisCalibration(400, 0, 100, 5))
    x_px = np.array([300.0, 100.0, 500.0])
    y_px = np.array([250.0, 400.0, 100.0])
    cd = to_curve_data("a", x_px, y_px, cal)
    assert list(cd.x) == pytest.approx([0, 1500, 3000])
    assert cd.y[0] == pytest.approx(0.0) and cd.y[1] == pytest.approx(2.5) and cd.y[2] == pytest.approx(5.0)
    assert cd.x_unc == pytest.approx([0.5 * 7.5] * 3)          # 3000 / 400 px = 7.5 per px
    assert cd.y_unc == pytest.approx([0.5 * 5 / 300] * 3)


def test_log_axis_uncertainty_varies_along_the_curve():
    cal = Calibration(AxisCalibration(0, 0, 100, 10), AxisCalibration(400, 1, 0, 1e4, log=True))
    cd = to_curve_data("l", [10, 50, 90], [400, 200, 40], cal)
    assert cd.y_unc[0] < cd.y_unc[1] < cd.y_unc[2]
    lo, hi = uncertainty_summary(cd)[2:]
    assert lo == pytest.approx(cd.y_unc[0]) and hi == pytest.approx(cd.y_unc[2])


# ---------------------------------------------------------------------- resampling
def test_resample_uses_multiples_of_the_step_inside_the_range_only():
    c = _curve([3, 12, 27, 41], [0, 9, 24, 38])                   # y = x - 3
    r = resample(c, 10)
    assert list(r.x) == [10, 20, 30, 40]
    assert r.y == pytest.approx([7, 17, 27, 37])
    assert r.x.min() >= c.x.min() and r.x.max() <= c.x.max()      # never extrapolates


def test_resample_step_finer_than_data_and_exact_endpoints():
    c = _curve([0, 10], [0, 100])
    r = resample(c, 2.5)
    assert list(r.x) == [0, 2.5, 5, 7.5, 10]
    assert r.y == pytest.approx([0, 25, 50, 75, 100])


def test_resample_marks_large_holes_as_nan():
    c = _curve([0, 1, 2, 30, 31, 32], [0, 1, 2, 30, 31, 32])
    r = resample(c, 5, max_gap=5)
    inside_hole = (r.x > 2) & (r.x < 30)
    assert np.all(np.isnan(r.y[inside_hole]))
    assert np.all(np.isfinite(r.y[~inside_hole]))
    assert np.all(np.isfinite(resample(c, 5).y))                  # without max_gap: plain interpolation


def test_resample_merges_duplicate_x_and_respects_window():
    c = _curve([0, 5, 5, 10], [0, 4, 6, 10])
    r = resample(c, 5)
    assert r.y == pytest.approx([0, 5, 10])
    w = resample(_curve(np.arange(0, 101, 1.0), np.arange(0, 101, 1.0)), 10, x_start=25, x_stop=65)
    assert list(w.x) == [30, 40, 50, 60]


def test_resample_edge_cases():
    with pytest.raises(ValueError):
        resample(_curve([0, 1], [0, 1]), 0)
    assert len(resample(_curve([], []), 5)) == 0
    assert len(resample(_curve([1, 2], [1, 2]), 10)) == 0         # no multiple of 10 inside [1, 2]
