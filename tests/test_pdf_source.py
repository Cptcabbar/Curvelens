from pathlib import Path

import cv2
import numpy as np
import pypdfium2 as pdfium
import pytest

from core.models import Rect
from core.pdf_source import PdfSource

PDF = Path(__file__).resolve().parents[1] / "samples" / "INR18650P28A-V1-80093.pdf"
pytestmark = pytest.mark.skipif(not PDF.exists(), reason="sample datasheet PDF missing")


@pytest.fixture(scope="module")
def src():
    with PdfSource(PDF) as s:
        yield s


def test_all_four_charts_are_found_and_told_apart_by_title(src):
    charts = src.charts(0)
    assert [c.title for c in charts] == [
        "Charge Characteristics",
        "Discharge Rate Characteristics",
        "Discharge Temperature Characteristics",
        "Cycle Characteristics",
    ]
    assert [c.index for c in charts] == [0, 1, 2, 3]
    assert charts[1].label == "Sayfa 1 · Discharge Rate Characteristics"


def test_plot_frames_match_the_drawn_frames(src):
    rate = src.charts(0)[1].plot_rect
    assert (rate.x0, rate.y0, rate.x1, rate.y1) == pytest.approx((360.6, 283.68, 531.6, 397.8), abs=0.2)
    # Same frame geometry for the two discharge charts, stacked vertically.
    temp = src.charts(0)[2].plot_rect
    assert temp.width == pytest.approx(rate.width, abs=0.2) and temp.height == pytest.approx(rate.height, abs=0.2)
    assert temp.y0 > rate.y1


def test_discharge_rate_axes_titles_units_and_ticks(src):
    c = src.charts(0)[1]
    assert len(c.x_axes) == 1 and len(c.y_axes) == 1
    ax, ay = c.x_axes[0], c.y_axes[0]
    assert (ax.title, ax.unit) == ("Capacity (mAh)", "mAh")
    assert (ay.title, ay.unit) == ("Voltage (V)", "V")
    assert [t.value for t in ax.ticks] == [0, 300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000]
    assert [t.value for t in ay.ticks] == [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]          # top of the page first
    assert all(t.snapped for t in ax.ticks + ay.ticks)
    # x ticks are evenly spaced, y ticks too
    assert np.allclose(np.diff([t.pos for t in ax.ticks]), 17.1, atol=0.2)
    assert np.allclose(np.diff([t.pos for t in ay.ticks]), 22.86, atol=0.2)


def test_charge_chart_has_three_y_axes_that_are_not_mixed_up(src):
    c = src.charts(0)[0]
    by_title = {a.title: a for a in c.y_axes}
    assert set(by_title) == {"Voltage (V)", "Current (A)", "Capacity (%)"}
    assert [t.value for t in by_title["Voltage (V)"].ticks] == [5, 4, 3, 2, 1, 0]
    assert [t.value for t in by_title["Current (A)"].ticks] == [10, 8, 6, 4, 2, 0]
    assert [t.value for t in by_title["Capacity (%)"].ticks] == [120, 100, 80, 60, 40, 20, 0]
    assert by_title["Capacity (%)"].side == "right"
    assert c.x_axes[0].title == "Time (hrs)" and [t.value for t in c.x_axes[0].ticks] == [0, 0.5, 1.0, 1.5]
    # right-hand axis labels never landed on the left axes' grid lines (100 % sits between grid lines)
    pos = {t.value: t.pos for t in by_title["Capacity (%)"].ticks}
    grid = {t.value: t.pos for t in by_title["Voltage (V)"].ticks}
    assert abs(pos[100] - grid[4]) > 3


def test_cycle_chart_uses_its_own_axis_titles(src):
    c = src.charts(0)[3]
    assert c.x_axes[0].title == "Cycle Number (times)"
    assert c.y_axes[0].title == "Capacity (%)"
    assert [t.value for t in c.x_axes[0].ticks] == [0, 100, 200, 300, 400, 500]


def test_chart_regions_do_not_swallow_neighbouring_headings(src):
    charts = src.charts(0)
    for upper, lower in zip(charts[1:-1], charts[2:]):
        assert upper.region.y1 < lower.region.y0
    for c in charts:
        r = c.region
        assert r.x0 < c.plot_rect.x0 and r.y1 > c.plot_rect.y1          # tick labels are inside
        assert r.y0 <= c.plot_rect.y0 and r.x1 >= c.plot_rect.x1


@pytest.mark.parametrize("dpi", [150, 300, 600])
def test_render_transform_is_exact(src, dpi):
    """Frame edges found in the raster sit where the point->pixel transform says."""
    chart = src.charts(0)[1]
    r = src.render(0, chart.region, dpi=dpi)
    fr = r.rect_pt_to_px(chart.plot_rect)
    g = (255 - cv2.cvtColor(r.image, cv2.COLOR_BGR2GRAY).astype(float))
    half = 2.2 * dpi / 300

    def centre(profile, at):
        lo, hi = int(at - half - 1), int(at + half + 2)
        w = profile[lo:hi]
        return float((w * (np.arange(lo, hi) + 0.5)).sum() / w.sum())

    rows = slice(int(fr.y0) + 20, int(fr.y1) - 20)
    cols = slice(int(fr.x0) + 20, int(fr.x1) - 20)
    col_profile, row_profile = g[rows, :].mean(0), np.median(g[:, cols], axis=1)
    assert centre(col_profile, fr.x0) == pytest.approx(fr.x0, abs=0.35)
    assert centre(col_profile, fr.x1) == pytest.approx(fr.x1, abs=0.35)
    assert centre(row_profile, fr.y0) == pytest.approx(fr.y0, abs=0.35)
    assert centre(row_profile, fr.y1) == pytest.approx(fr.y1, abs=0.35)


def test_point_pixel_roundtrip_and_covered_region(src):
    r = src.render(0, src.charts(0)[1].region, dpi=300)
    x, y = r.pt_to_px(400.0, 350.0)
    assert r.px_to_pt(x, y) == pytest.approx((400.0, 350.0))
    reg = r.region
    h, w = r.image.shape[:2]
    assert reg.width * r.sx == pytest.approx(w) and reg.height * r.sy == pytest.approx(h)


def test_suggested_calibration_maps_tick_labels_to_their_values(src):
    chart = src.charts(0)[1]
    r = src.render(0, chart.region, dpi=600)
    cal = src.suggest_calibration(chart, r)
    assert cal is not None and not cal.x.log and not cal.y.log
    plot = r.rect_pt_to_px(chart.plot_rect)
    # 0 mAh / 0 V at the frame's lower-left corner, 3000 mAh / 5 V at the upper right (sub-pixel)
    assert float(cal.x.to_pixel(0)) == pytest.approx(plot.x0, abs=0.6)
    assert float(cal.x.to_pixel(3000)) == pytest.approx(plot.x1, abs=1.5)
    assert float(cal.y.to_pixel(0.0)) == pytest.approx(plot.y1, abs=0.8)
    assert float(cal.y.to_pixel(5.0)) == pytest.approx(plot.y0, abs=0.8)
    assert cal.x.resolution() == pytest.approx(2.1, abs=0.05)          # mAh per pixel at 600 dpi
    # a non-default axis of the chart with three y axes
    c0 = src.charts(0)[0]
    r0 = src.render(0, c0.region, dpi=300)
    volt = next(i for i, a in enumerate(c0.y_axes) if a.title == "Voltage (V)")
    amps = next(i for i, a in enumerate(c0.y_axes) if a.title == "Current (A)")
    cv, ca = src.suggest_calibration(c0, r0, y_index=volt), src.suggest_calibration(c0, r0, y_index=amps)
    assert cv.y.v2 == 5.0 and ca.y.v2 == 10.0
    assert float(cv.y.to_pixel(5.0)) == pytest.approx(float(ca.y.to_pixel(10.0)), abs=0.1)   # same top gridline


def test_whole_page_render_and_page_without_charts(tmp_path, src):
    r = src.render(0, None, dpi=72)
    assert r.image.shape[:2] == (842, 595)
    assert src.page_count == 1

    blank = tmp_path / "blank.pdf"
    doc = pdfium.PdfDocument.new()
    doc.new_page(200, 100)
    doc.save(str(blank))
    doc.close()
    with PdfSource(blank) as b:
        assert b.charts(0) == []
        assert b.render(0, Rect(10, 10, 60, 40), dpi=144).image.shape[:2] == (60, 100)
