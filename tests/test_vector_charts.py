from pathlib import Path

import numpy as np
import pytest

from core.pdf_source import AxisHint, TickLabel
from core.vector_charts import (AxisFit, VPath, _chains, _compose, analyze_pdf, clean_text, fit_axis)

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "samples" / "INR18650P28A-V1-80093.pdf"


# ----------------------------------------------------------------------------------- unit tests
def test_clean_text_repairs_superscript_degree_signs():
    assert clean_text("23oC") == "23°C"
    assert clean_text("-40oC") == "-40°C"
    assert clean_text("0 oC") == "0°C"
    assert clean_text("Temperature: 23oC.") == "Temperature: 23°C."
    assert clean_text("  two   spaces ") == "two spaces"
    assert clean_text("2.8A charge") == "2.8A charge"          # nothing to repair


def _hint(pairs, orientation="y", title="T (u)"):
    return AxisHint(orientation, "left", title, [TickLabel(str(v), v, p, True) for v, p in pairs])


def test_fit_axis_linear_uses_all_ticks_and_reports_rms():
    hint = _hint([(0, 100.0), (1, 80.0), (2, 60.05), (3, 39.95), (4, 20.0)])      # 20 pt per unit, y down
    f = fit_axis(hint)
    assert not f.log and f.slope == pytest.approx(-0.05, rel=1e-3)
    assert float(f.to_value(60.0)) == pytest.approx(2.0, abs=0.003)
    assert float(f.to_pos(2.5)) == pytest.approx(50.0, abs=0.1)
    assert 0 < f.rms_pt < 0.1
    assert f.unit == "u"


def test_fit_axis_detects_a_log_axis_even_with_a_bad_zero_label():
    # decades 0.1 .. 1000 at 58 pt spacing; the producer printed the first label as "0"
    pos = [(1000, 36.0), (100, 94.0), (10, 152.0), (1, 210.0), (0, 268.0)]
    f = fit_axis(_hint(pos))
    assert f.log
    assert float(f.to_value(94.0)) == pytest.approx(100.0, rel=1e-6)
    assert float(f.to_value(268.0)) == pytest.approx(0.1, rel=1e-6)


def test_fit_axis_needs_two_ticks():
    assert fit_axis(_hint([(0, 1.0)])) is None
    assert fit_axis(_hint([])) is None


def test_axis_fit_uncertainty_is_linear_or_value_proportional():
    lin = fit_axis(_hint([(0, 100.0), (10, 0.0)]))
    assert lin.uncertainty(np.array([10.0, 50.0])) == pytest.approx([0.006, 0.006])       # 0.06 pt * 0.1 unit/pt
    log = fit_axis(_hint([(1, 100.0), (10, 50.0), (100, 0.0)]))
    unc = log.uncertainty(np.array([100.0, 0.0]))
    assert unc[1] > 50 * unc[0]                                  # 100x larger value -> ~100x larger uncertainty


def test_compose_applies_inner_then_outer_matrix():
    inner = (2.0, 0.0, 0.0, 2.0, 1.0, 1.0)                      # x' = 2x + 1
    outer = (1.0, 0.0, 0.0, 1.0, 10.0, 20.0)                    # translate
    a, b, c, d, e, f = _compose(inner, outer)
    x, y = a * 3 + c * 4 + e, b * 3 + d * 4 + f
    assert (x, y) == (2 * 3 + 1 + 10, 2 * 4 + 1 + 20)
    rot = (0.0, 1.0, -1.0, 0.0, 0.0, 0.0)                       # 90 degrees
    a, b, c, d, e, f = _compose(rot, inner)
    assert (a * 1 + c * 0 + e, b * 1 + d * 0 + f) == (2 * 0 + 1, 2 * 1 + 1)


def _piece(*pts, dashed=False):
    return VPath(np.array(pts, float), False, (0, 0, 0), None, 1.0, dashed)


def test_chains_join_touching_pieces_and_keep_gaps_apart():
    a, b = _piece((0, 0), (5, 1)), _piece((5, 1), (10, 3))            # touch
    c = _piece((30, 9), (32, 9.5))                                    # a separate dash
    chains = _chains([a, b, c])
    sizes = sorted(len(ch) for ch in chains)
    assert sizes == [1, 2]
    joined = next(ch for ch in chains if len(ch) == 2)
    assert {id(p) for p in joined} == {id(a), id(b)}


# ----------------------------------------------------------------------------------- the datasheet
pytestmark_pdf = pytest.mark.skipif(not PDF.exists(), reason="sample datasheet PDF missing")


@pytest.fixture(scope="module")
def sheet():
    return {c.title: c for c in analyze_pdf(PDF)}


def curve(chart, label, style=None, color=None, axis=None):
    hits = [c for c in chart.curves if c.label == label and (style is None or c.style == style)
            and (color is None or c.color == color)
            and (axis is None or chart.y_fits[c.y_axis].title == axis)]
    assert len(hits) == 1, [(c.label, c.style, c.color) for c in chart.curves]
    return hits[0]


def at(cv, x):
    return float(np.interp(x, cv.data.x, cv.data.y))


@pytestmark_pdf
def test_all_charts_and_curves_are_found(sheet):
    counts = {t: len(c.curves) for t, c in sheet.items()}
    assert counts == {"Charge Characteristics": 6, "Discharge Rate Characteristics": 5,
                      "Discharge Temperature Characteristics": 7, "Cycle Characteristics": 2}


@pytestmark_pdf
def test_meaning_of_the_discharge_rate_curves(sheet):
    ch = sheet["Discharge Rate Characteristics"]
    names = {c.label: c.color for c in ch.curves}
    assert names == {"0.56A": (255, 0, 0), "2.8A": (0, 255, 0), "10A": (0, 0, 255), "20A": (0, 0, 0), "30A": (255, 0, 255)}
    assert all(c.style == "solid" and c.from_legend for c in ch.curves)
    assert ch.x_fit.title == "Capacity (mAh)" and ch.y_fits[0].title == "Voltage (V)"
    assert ch.notes == ["Charge: CC-CV, 2.8A, 4.2V, 50mA.", "Discharge: Cut 0ff at 2.5V.", "Temperature: 23°C."]


@pytestmark_pdf
def test_meaning_of_the_temperature_curves(sheet):
    ch = sheet["Discharge Temperature Characteristics"]
    names = {c.label: c.color for c in ch.curves}
    assert names == {"-40°C": (255, 101, 0), "-30°C": (0, 255, 0), "-20°C": (0, 204, 255), "0°C": (255, 0, 255),
                     "23°C": (0, 0, 255), "45°C": (0, 0, 0), "60°C": (255, 0, 0)}
    assert "Charge: CC-CV, 2.8A, 4.2V, 50mA at 23°C." in ch.notes


@pytestmark_pdf
def test_solid_and_dashed_curves_of_one_colour_and_their_axes_in_the_charge_chart(sheet):
    ch = sheet["Charge Characteristics"]
    combos = {(c.label, c.style, c.color, ch.y_fits[c.y_axis].title) for c in ch.curves}
    assert combos == {
        ("2.8A charge", "solid", (0, 0, 0), "Voltage (V)"), ("5.6A charge", "dashed", (0, 0, 0), "Voltage (V)"),
        ("2.8A charge", "solid", (0, 0, 255), "Current (A)"), ("5.6A charge", "dashed", (0, 0, 255), "Current (A)"),
        ("2.8A charge", "solid", (255, 0, 0), "Capacity (%)"), ("5.6A charge", "dashed", (255, 0, 0), "Capacity (%)"),
    }


@pytestmark_pdf
def test_cycle_chart_marker_series(sheet):
    ch = sheet["Cycle Characteristics"]
    a, b = curve(ch, "10A discharge"), curve(ch, "20A discharge")
    assert (a.style, a.color, b.style, b.color) == ("markers", (0, 0, 0), "markers", (0, 0, 255))
    assert a.n_points == b.n_points == 500                           # the legend glyph is not counted
    assert a.data.x[-1] == pytest.approx(500.0, abs=0.2)
    assert 84.0 < a.data.y[-1] < 84.6 and 79.2 < b.data.y[-1] < 79.9
    assert 78.4 < b.data.y.min() < 78.9
    assert a.data.y[0] == pytest.approx(100.0, abs=0.2)


@pytestmark_pdf
def test_discharge_values_are_exact_and_agree_with_the_pixel_extraction(sheet):
    """Vector values against the independent raster extraction of the same chart (600 dpi)."""
    from core.extraction import extract_curve
    from core.pdf_source import PdfSource

    ch = sheet["Discharge Rate Characteristics"]
    red = curve(ch, "0.56A")
    assert at(red, 1500) == pytest.approx(3.6452, abs=0.0015)
    assert red.data.x[0] == pytest.approx(0.0, abs=2.0) and red.data.x[-1] == pytest.approx(2848.7, abs=2.0)
    with PdfSource(PDF) as src:
        chart = next(c for c in src.charts(0) if "Discharge Rate" in c.title)
        r = src.render(0, chart.region, dpi=600)
        cal = src.suggest_calibration(chart, r)
    res = extract_curve(r.image, (255, 0, 0), r.rect_pt_to_px(chart.plot_rect))
    X, Y = cal.pixel_to_data(res.x_px, res.y_px)
    grid = np.linspace(50, 2600, 200)
    diff = np.interp(grid, X, Y) - np.interp(grid, red.data.x, red.data.y)
    assert np.abs(diff).max() < 0.006                                # pixel method is ~0.5 px = 0.0026 V coarse


@pytestmark_pdf
def test_curve_hidden_under_another_one_is_recovered_completely(sheet):
    """In the datasheet the 45 C curve lies under the 60 C curve for ~93 % of its length: pixels cannot see it."""
    ch = sheet["Discharge Temperature Characteristics"]
    c45, c60, c23 = curve(ch, "45°C"), curve(ch, "60°C"), curve(ch, "23°C")
    assert c45.n_points > 200 and c45.data.x[-1] > 2800
    assert abs(at(c45, 1500) - at(c60, 1500)) < 0.03                 # nearly the same curve...
    assert at(c45, 1500) > at(c23, 1500) - 0.05                      # ...and consistent with the others
    assert c45.data.y[0] == pytest.approx(4.185, abs=0.01)


@pytestmark_pdf
def test_charge_chart_values_follow_physics(sheet):
    """Constant-current charging: capacity grows linearly with time, twice as fast at twice the current,
    and the current curves sit at 2.8 A / 5.6 A."""
    ch = sheet["Charge Characteristics"]
    cap_solid = curve(ch, "2.8A charge", "solid", (255, 0, 0))
    cap_dash = curve(ch, "5.6A charge", "dashed", (255, 0, 0))
    assert at(cap_solid, 0.5) / at(cap_solid, 0.25) == pytest.approx(2.0, abs=0.02)      # linear in time
    assert at(cap_dash, 0.25) / at(cap_solid, 0.25) == pytest.approx(2.0, abs=0.06)      # twice the current
    assert 25.0 < at(cap_solid, 0.25) < 26.0                                             # ~2.75 Ah cell
    assert cap_dash.data.x[-1] == pytest.approx(0.89, abs=0.01) and cap_dash.data.y.max() == pytest.approx(100.0, abs=0.2)
    cur_solid = curve(ch, "2.8A charge", "solid", (0, 0, 255))
    cur_dash = curve(ch, "5.6A charge", "dashed", (0, 0, 255))
    assert at(cur_solid, 0.4) == pytest.approx(2.8, abs=0.005)
    assert at(cur_dash, 0.2) == pytest.approx(5.6, abs=0.005)
    volt = curve(ch, "2.8A charge", "solid", (0, 0, 0))
    assert volt.data.y.max() == pytest.approx(4.2, abs=0.01)
    assert not any(c.data.x.max() > 1.35 for c in ch.curves)         # no axis line / tick of the frame leaked in


@pytestmark_pdf
def test_uncertainty_reflects_the_drawing_resolution(sheet):
    red = curve(sheet["Discharge Rate Characteristics"], "0.56A")
    assert red.data.x_unc[0] == pytest.approx(1.05, abs=0.4)          # mAh per 0.06 pt
    assert red.data.y_unc[0] == pytest.approx(0.0026, abs=0.001)      # V


@pytestmark_pdf
def test_chart_thumbnails_are_rendered(sheet):
    for ch in sheet.values():
        assert ch.thumbnail is not None and ch.thumbnail.ndim == 3 and ch.thumbnail.shape[1] > 300


# ----------------------------------------------------------------------------------- generated PDFs with a known answer
mpl = pytest.importorskip("matplotlib")
from tests import make_pdfs                                          # noqa: E402


def max_abs_error(cv, x, y, lo=0.0, hi=1.0):
    """Largest |curve(x) - y| over the middle part of the truth's x range, in data units.

    matplotlib snaps grid and frame lines to whole 1/72 inch positions, so calibration from them carries
    a fixed offset of up to ~0.1 pt (0.03 % of the axis span) that comes from the PDF itself.
    """
    span = float(np.ptp(x))
    sel = (x >= x.min() + lo * span) & (x <= x.min() + hi * span)
    return float(np.max(np.abs(np.interp(x[sel], cv.data.x, cv.data.y) - y[sel])))


def by_label(chart):
    return {c.label: c for c in chart.curves}


def test_generated_lines_legend_dash_and_values(tmp_path):
    path, truth = make_pdfs.simple_lines(tmp_path / "a.pdf")
    (chart,) = analyze_pdf(path)
    assert chart.title == "Motor test"
    assert (chart.x_fit.title, chart.y_fits[0].title) == ("Time (s)", "Voltage (V)")
    cur = by_label(chart)
    assert set(cur) == {"Motor A", "Motor B", "Motor C"}
    assert [cur[k].style for k in ("Motor A", "Motor B", "Motor C")] == ["solid", "dashed", "solid"]
    for name, (x, y) in truth.items():
        assert max_abs_error(cur[name], x, y, 0.02, 0.98) < 0.004, name       # axis span 6 V -> 0.07 %


def test_generated_log_log_chart(tmp_path):
    path, truth = make_pdfs.log_axes(tmp_path / "b.pdf")
    (chart,) = analyze_pdf(path)
    assert chart.x_fit.log and chart.y_fits[0].log
    (cv,) = chart.curves
    x, y = truth["Gain"]
    inside = (y >= 0.11) & (y <= 990)                              # the part drawn inside the frame
    got = np.interp(x[inside], cv.data.x, cv.data.y)
    assert np.max(np.abs(got / y[inside] - 1)) < 0.02               # within 2 % everywhere on a log scale


def test_generated_twin_axes_use_the_axis_of_the_matching_colour(tmp_path):
    path, truth = make_pdfs.twin_axes(tmp_path / "c.pdf")
    (chart,) = analyze_pdf(path)
    assert [f.title for f in chart.y_fits] == ["Current (A)", "Temperature (C)"]
    cur = by_label(chart)
    assert set(cur) == {"Current (A)", "Temperature (C)"}
    assert cur["Current (A)"].y_axis == 0 and cur["Temperature (C)"].y_axis == 1
    for label, key in (("Current (A)", "Current"), ("Temperature (C)", "Temperature")):
        x, y = truth[key]
        assert max_abs_error(cur[label], x, y, 0.02, 0.98) < (0.004 if key == 'Current' else 0.07)   # 6 A / 100 C spans


def test_generated_markers_come_from_form_xobjects(tmp_path):
    path, truth = make_pdfs.markers(tmp_path / "d.pdf")
    (chart,) = analyze_pdf(path)
    cur = by_label(chart)
    assert set(cur) == {"Cell 1", "Cell 2"}
    for name, (x, y) in truth.items():
        cv = cur[name]
        assert cv.style == "markers" and cv.n_points == len(x)      # legend marker excluded
        assert np.max(np.abs(cv.data.x - x)) < 0.05
        assert np.max(np.abs(cv.data.y - y)) < 0.1                  # 0.1 % of the axis span


def test_generated_two_charts_on_one_page_and_two_pages(tmp_path):
    path, truth = make_pdfs.two_charts(tmp_path / "e.pdf")
    charts = analyze_pdf(path)
    assert [c.title for c in charts] == ["First chart", "Second chart"]
    for chart, key in zip(charts, ("P", "Q")):
        (cv,) = chart.curves
        assert cv.label == key
        assert max_abs_error(cv, *truth[key], 0.02, 0.98) < (0.12 if key == 'P' else 0.08)         # 100 W / 60 L spans

    path, truth = make_pdfs.two_pages(tmp_path / "f.pdf")
    charts = analyze_pdf(path)
    assert [(c.title, c.chart.page_index) for c in charts] == [("Page one", 0), ("Page two", 1)]
    assert [c.curves[0].label for c in charts] == ["S1", "S2"]


def test_pdf_without_charts_or_with_progress_callback(tmp_path):
    import pypdfium2 as pdfium

    blank = tmp_path / "blank.pdf"
    doc = pdfium.PdfDocument.new()
    doc.new_page(200, 100)
    doc.save(str(blank))
    doc.close()
    seen = []
    assert analyze_pdf(blank, lambda i, n, text: seen.append((i, n))) == []
    assert seen and seen[-1] == (1, 1)
