"""Acceptance tests on the datasheet's *Discharge Rate Characteristics* chart.

The sample is ``samples/discharge.png`` (that chart rendered from the PDF at 600 DPI).
Its X axis is capacity in mAh (0-3000), Y is voltage in V (0-5); the "0.56A ... 30A"
curves are red, green, blue, black and magenta.

Calibration points and rectangles below are fixed pixel coordinates that were measured
on the image itself (centres of the axis tick marks, sub-pixel), not derived from the PDF.
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from core.calibration import AxisCalibration, Calibration
from core.export import AxisInfo, export_combined, export_separate
from core.extraction import extract_curve, pick_reference_color
from core.models import ExtractionParams, Rect
from core.postprocess import resample, smooth_savgol, to_curve_data

ROOT = Path(__file__).resolve().parents[1]
PNG = ROOT / "samples" / "discharge.png"
PDF = ROOT / "samples" / "INR18650P28A-V1-80093.pdf"
pytestmark = pytest.mark.skipif(not PNG.exists(), reason="samples/discharge.png missing")

# --- calibration: tick-mark centres measured on samples/discharge.png ----------------
X_TICKS = ((308.79, 300.0), (1447.71, 2700.0))       # (pixel x, mAh)   ticks "300" and "2700"
Y_TICKS = ((785.14, 1.0), (405.12, 3.0))             # (pixel y, V)     ticks "1.0" and "3.0"
CAL = Calibration(
    x=AxisCalibration(X_TICKS[0][0], X_TICKS[0][1], X_TICKS[1][0], X_TICKS[1][1]),
    y=AxisCalibration(Y_TICKS[0][0], Y_TICKS[0][1], Y_TICKS[1][0], Y_TICKS[1][1]),
)

# --- rectangles (pixels) --------------------------------------------------------------
PLOT_FRAME = Rect(167.1, 24.1, 1591.7, 975.2)        # centre-lines of the black frame
PLOT_INNER = Rect(176.0, 33.0, 1583.0, 966.0)        # what a user drags inside the frame
LEGEND = Rect(1085, 515, 1310, 780)                  # legend swatches + labels
NOTE = Rect(295, 790, 880, 945)                      # "Charge: CC-CV ..." text block

CURVES = {                                           # name -> RGB of the curve
    "0.56A": (255, 0, 0),
    "2.8A": (0, 255, 0),
    "10A": (0, 0, 255),
    "20A": (0, 0, 0),
    "30A": (255, 0, 255),
}


@pytest.fixture(scope="module")
def image():
    img = cv2.imread(str(PNG))
    assert img is not None
    return img


@pytest.fixture(scope="module")
def extracted(image):
    """Curves extracted the way the app does it: colour picked from the image, plot area + exclusions."""
    out = {}
    for name, rgb in CURVES.items():
        res = extract_curve(image, rgb, PLOT_INNER, [LEGEND, NOTE], ExtractionParams())
        assert len(res) > 0, f"{name}: nothing extracted ({res.stats})"
        out[name] = to_curve_data(name, res.x_px, res.y_px, CAL)
    return out


def value_at(curve, x):
    return float(np.interp(x, curve.x, curve.y))


# ------------------------------------------------------------------------ calibration sanity
def test_calibration_reproduces_the_axis_extremes(image):
    """Frame edges (not used for the calibration) must map to 0/3000 mAh and 0/5 V.

    The datasheet is drawn on integer 1/600 inch device coordinates, so tick and frame
    positions jitter by +/-0.5 px; extrapolating from two inner ticks to the frame
    therefore agrees to about 1 px (2.1 mAh, 0.005 V) - ten times below the tolerances
    of the acceptance criteria.
    """
    px_x, px_y = 1.5, 1.5
    assert float(CAL.x.to_value(167.14)) == pytest.approx(0, abs=px_x * CAL.x.resolution())
    assert float(CAL.x.to_value(1591.24)) == pytest.approx(3000, abs=px_x * CAL.x.resolution())
    assert float(CAL.y.to_value(975.15)) == pytest.approx(0.0, abs=px_y * CAL.y.resolution())
    assert float(CAL.y.to_value(24.11)) == pytest.approx(5.0, abs=px_y * CAL.y.resolution())
    assert image.shape[:2] == (1124, 1645)


# ------------------------------------------------------------------------ acceptance criteria
def test_all_five_curves_including_black_are_extracted(extracted):
    assert set(extracted) == set(CURVES)
    for name, c in extracted.items():
        assert len(c) > 1000, name
        assert c.x[0] < 40 and c.x[-1] > 2600, f"{name}: covers {c.x[0]:.0f}..{c.x[-1]:.0f} mAh"
        assert np.all(np.diff(c.x) > 0)


def test_red_curve_control_values(extracted):
    red = extracted["0.56A"]
    assert value_at(red, 20) == pytest.approx(4.15, abs=0.03)            # t ~ 0
    assert value_at(red, 1500) == pytest.approx(3.65, abs=0.03)
    end_x = red.x[-1]
    assert end_x == pytest.approx(2830, abs=30)
    assert red.y[-1] == pytest.approx(2.5, abs=0.03)


def test_magenta_curve_control_value(extracted):
    assert value_at(extracted["30A"], 1500) == pytest.approx(3.15, abs=0.03)


def test_black_curve_is_where_the_datasheet_draws_it(extracted):
    black = extracted["20A"]
    assert value_at(black, 1500) == pytest.approx(3.28, abs=0.03)
    assert black.y[-1] == pytest.approx(2.5, abs=0.03) and black.x[-1] == pytest.approx(2780, abs=30)


def test_curves_keep_their_order_at_every_x(extracted):
    order = ["0.56A", "2.8A", "10A", "20A", "30A"]                        # highest voltage first
    for x in (100, 500, 1000, 1500, 2000, 2400):
        vals = [value_at(extracted[n], x) for n in order]
        assert vals == sorted(vals, reverse=True), (x, vals)
        assert all(2.5 < v < 4.25 for v in vals)


def test_uncertainty_is_half_a_pixel_of_resolution(extracted):
    c = extracted["0.56A"]
    assert c.x_unc[0] == pytest.approx(0.5 * CAL.x.resolution())
    assert c.y_unc[0] == pytest.approx(0.5 * CAL.y.resolution())
    assert c.x_unc[0] == pytest.approx(1.05, abs=0.02)                    # mAh
    assert c.y_unc[0] == pytest.approx(0.00263, abs=0.0001)               # V


# ------------------------------------------------------------------------ robustness on the real sample
@pytest.mark.parametrize("area", [PLOT_INNER, PLOT_FRAME], ids=["inside-frame", "on-frame"])
@pytest.mark.parametrize("use_exclusions", [True, False], ids=["with-exclusions", "no-exclusions"])
def test_control_values_hold_for_any_reasonable_user_selection(image, area, use_exclusions):
    """Whether or not the legend/note are excluded and the frame is inside the area."""
    ex = [LEGEND, NOTE] if use_exclusions else []
    for name, x, v in (("0.56A", 1500, 3.65), ("30A", 1500, 3.15)):
        res = extract_curve(image, CURVES[name], area, ex, ExtractionParams())
        c = to_curve_data(name, res.x_px, res.y_px, CAL)
        assert value_at(c, x) == pytest.approx(v, abs=0.03)
    black = to_curve_data("20A", *(lambda r: (r.x_px, r.y_px))(extract_curve(image, CURVES["20A"], area, ex)), CAL)
    assert value_at(black, 1500) == pytest.approx(3.28, abs=0.03)
    assert black.x[-1] > 2700


def test_colour_picked_from_a_clicked_pixel_gives_the_same_curve(image, extracted):
    """The GUI picks the colour under the mouse; the datasheet's curves are pure colours."""
    # a pixel in the middle of the red curve at 1500 mAh
    x_px = float(CAL.x.to_pixel(1500))
    y_px = float(CAL.y.to_pixel(3.6452))
    rgb = pick_reference_color(image, x_px, y_px)
    assert rgb[0] > 200 and rgb[1] < 60 and rgb[2] < 60
    res = extract_curve(image, rgb, PLOT_INNER, [LEGEND, NOTE])
    c = to_curve_data("red", res.x_px, res.y_px, CAL)
    assert value_at(c, 1500) == pytest.approx(value_at(extracted["0.56A"], 1500), abs=0.005)


# ------------------------------------------------------------------------ post-processing + export
def test_resample_smooth_and_export_pipeline(image, extracted, tmp_path):
    step = 10.0
    grids = []
    for name, rgb in CURVES.items():
        res = extract_curve(image, rgb, PLOT_INNER, [LEGEND, NOTE])
        y_smooth = smooth_savgol(res.y_px, res.x_px, window=15, polyorder=3)
        c = to_curve_data(name, res.x_px, y_smooth, CAL)
        g = resample(c, step, max_gap=5 * step)
        assert np.allclose(np.diff(g.x), step)
        assert g.x[0] % step == 0 and g.x[-1] % step == 0
        # smoothing + resampling leave the values where the raw extraction had them
        raw = extracted[name]
        common = g.x[(g.x > raw.x[0] + 30) & (g.x < raw.x[-1] - 60)]
        assert np.max(np.abs(np.interp(common, g.x, g.y) - np.interp(common, raw.x, raw.y))) < 0.01
        grids.append(g)

    xa, ya = AxisInfo("Capacity", "mAh"), AxisInfo("Voltage", "V")
    combined = export_combined(grids, tmp_path / "discharge.csv", xa, ya, source="discharge.png", step=step)
    df = pd.read_csv(combined, comment="#", encoding="utf-8-sig")
    assert list(df.columns)[0] == "Capacity (mAh)" and df.shape[1] == 6
    row = df[df["Capacity (mAh)"] == 1500].iloc[0]
    assert row["0.56A [Voltage (V)]"] == pytest.approx(3.65, abs=0.03)
    assert row["30A [Voltage (V)]"] == pytest.approx(3.15, abs=0.03)
    head = combined.read_text(encoding="utf-8-sig").splitlines()
    assert any(l.startswith("# curve \"0.56A\"") and "±" in l for l in head)

    files = export_separate(grids, tmp_path / "each", xa, ya, source="discharge.png", step=step)
    assert sorted(p.name for p in files) == sorted(f"{n}.csv" for n in CURVES)


# ------------------------------------------------------------------------ against the PDF's vector layer
@pytest.mark.skipif(not PDF.exists(), reason="sample PDF missing")
@pytest.mark.parametrize("dpi", [200, 600])
def test_pdf_end_to_end_matches_the_vector_curves(dpi):
    """PDF -> chart detection -> render -> calibration hint -> extraction, checked against the
    exact polylines stored in the PDF (independent of any pixel analysis)."""
    from core.pdf_source import PdfSource
    from tests.vector_truth import curve_polylines, distance_to_polyline

    with PdfSource(PDF) as src:
        chart = next(c for c in src.charts(0) if "Discharge Rate" in c.title)
        r = src.render(0, chart.region, dpi=dpi)
        cal = src.suggest_calibration(chart, r)
    truth = curve_polylines(PDF, 0, chart.plot_rect)
    plot = r.rect_pt_to_px(chart.plot_rect)
    for name, rgb in CURVES.items():
        res = extract_curve(r.image, rgb, plot)
        assert len(res) > 0
        tx, ty = r.pt_to_px(*truth[rgb])
        d = distance_to_polyline(res.x_px, res.y_px, np.asarray(tx), np.asarray(ty), step=0.1)
        assert d.mean() < 0.35, f"{name}: mean distance {d.mean():.2f} px"
        assert np.percentile(d, 95) < 0.8, f"{name}: p95 {np.percentile(d, 95):.2f} px"
        covered = (res.x_px.max() - res.x_px.min()) / (np.max(tx) - np.min(tx))
        assert covered > 0.9, f"{name}: only {covered:.0%} of the curve found"
        # in data units: voltage error along the curve is far below the 0.03 V tolerance
        X, Y = cal.pixel_to_data(res.x_px, res.y_px)
        TX, TY = cal.pixel_to_data(np.asarray(tx), np.asarray(ty))
        lo, hi = max(X.min(), TX.min()) + 10, min(X.max(), TX.max()) - 60      # steep end excluded
        g = np.linspace(lo, hi, 300)
        assert np.max(np.abs(np.interp(g, X, Y) - np.interp(g, TX, TY))) < 0.02


# ------------------------------------------------------------------------ Charge chart: solid + dashed curves of one colour
@pytest.mark.skipif(not PDF.exists(), reason="sample PDF missing")
def test_charge_chart_solid_and_dashed_curves_of_the_same_colour_are_told_apart():
    """The charge chart draws a solid (2.8 A) and a dashed (5.6 A) curve in each of red/blue/black.

    Values are checked against physics rather than pixels: charging 2.8 Ah cells at 2.8 A / 5.6 A puts
    the capacity curve at 25 % after 0.25 h (2.8 A) and 50 % after 0.25 h (5.6 A), and the current
    curves at 2.8 A / 5.6 A during the constant-current phase.
    """
    from core.extraction import delta_e_map
    from core.pdf_source import PdfSource

    with PdfSource(PDF) as src:
        chart = next(c for c in src.charts(0) if c.title.startswith("Charge"))
        r = src.render(0, chart.region, dpi=300)
        cals = {a.title: src.suggest_calibration(chart, r, y_index=i) for i, a in enumerate(chart.y_axes)}
    plot = r.rect_pt_to_px(chart.plot_rect)
    col_x = float(cals["Voltage (V)"].x.to_pixel(0.5))

    def traces(rgb, axis):
        de = delta_e_map(r.image[:, int(col_x):int(col_x) + 1], rgb)[:, 0]
        ys = np.flatnonzero(de < 25)
        ys = ys[(ys > plot.y0 + 8) & (ys < plot.y1 - 8)]
        runs = [rn for rn in np.split(ys, np.flatnonzero(np.diff(ys) > 3) + 1) if len(rn) >= 3]
        upper, lower = sorted(float(np.mean(rn)) + 0.5 for rn in runs)[0], sorted(float(np.mean(rn)) + 0.5 for rn in runs)[-1]
        out = {}
        for kind, yc, dashed in (("dashed", upper, True), ("solid", lower, False)):
            res = extract_curve(r.image, rgb, plot, params=ExtractionParams(dashed=dashed), seed=(col_x, yc))
            out[kind] = to_curve_data(kind, res.x_px, res.y_px, cals[axis])
        return out

    red = traces((255, 0, 0), "Capacity (%)")
    assert value_at(red["solid"], 0.25) == pytest.approx(25.0, abs=1.5)
    assert value_at(red["solid"], 0.5) == pytest.approx(50.0, abs=2.0)
    assert value_at(red["dashed"], 0.25) == pytest.approx(50.0, abs=2.0)
    assert red["solid"].x[-1] == pytest.approx(1.32, abs=0.03)             # solid curve runs on to 1.3 h ...
    assert red["dashed"].x[-1] == pytest.approx(0.89, abs=0.03)            # ... the 5.6 A one ends at 0.89 h

    blue = traces((0, 0, 255), "Current (A)")
    assert value_at(blue["solid"], 0.25) == pytest.approx(2.8, abs=0.05)
    assert value_at(blue["dashed"], 0.25) == pytest.approx(5.6, abs=0.08)
    assert blue["dashed"].x[-1] == pytest.approx(0.89, abs=0.03)
