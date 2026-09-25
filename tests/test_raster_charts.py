"""Charts that are pictures inside a PDF: find them, calibrate, read the curves (known answer from matplotlib)."""
import numpy as np
import pytest

pytest.importorskip("matplotlib")
from core.postprocess import CurveData                                                    # noqa: E402
from core.query import value_at                                                           # noqa: E402
from core.raster_charts import (AxisPoints, calibrate, colour_name, detect_ticks, discover_curves,  # noqa: E402
                                find_image_charts, snap, trace_curve_at)
from core.vector_charts import analyze_pdf                                                # noqa: E402
from tests import make_pdfs                                                               # noqa: E402

X_NAME, Y_NAME, Y2_NAME = "Capacity (mAh)", "Voltage (V)", "Temperature (C)"


@pytest.fixture(scope="module")
def plain(tmp_path_factory):
    path, truth = make_pdfs.picture_chart(tmp_path_factory.mktemp("pic") / "plain.pdf")
    return path, truth, analyze_pdf(str(path))


@pytest.fixture(scope="module")
def dual(tmp_path_factory):
    path, truth = make_pdfs.picture_chart(tmp_path_factory.mktemp("pic") / "dual.pdf", dual=True)
    return path, truth, analyze_pdf(str(path))


@pytest.fixture(scope="module")
def same(tmp_path_factory):
    path, truth = make_pdfs.picture_chart(tmp_path_factory.mktemp("pic") / "same.pdf", dual=True, same_colour=True)
    return path, truth, analyze_pdf(str(path))


def calibrate_exactly(chart, truth, two_axes=False):
    """Calibration from the exact tick positions (the user clicks them; the click snaps onto the tick)."""
    xt, yt = truth["x_ticks"], truth["y_ticks"]
    x = AxisPoints((xt[0][1], xt[-1][1]), (xt[0][0], xt[-1][0]), X_NAME)
    y = AxisPoints((yt[0][1], yt[-1][1]), (yt[0][0], yt[-1][0]), Y_NAME)
    y2 = None
    if two_axes:
        y2t = truth["y2_ticks"]
        y2 = AxisPoints((y2t[0][1], y2t[-1][1]), (y2t[0][0], y2t[-1][0]), Y2_NAME)
    calibrate(chart, x, y, y2)
    return chart


def assert_close_to_truth(curve, xs, ys, px_tol=2.0):
    """The traced curve stays within ``px_tol`` picture pixels of the plotted one (vertically), on most of its length."""
    d: CurveData = curve.data
    inside = (xs > d.x[0] + 30) & (xs < d.x[-1] - 30)
    got = np.interp(xs[inside], d.x, d.y)
    tol = px_tol * float(np.nanmax(d.y_unc)) * 2.0             # y_unc is half a pixel
    assert np.max(np.abs(got - ys[inside])) <= tol, (np.max(np.abs(got - ys[inside])), tol)


# ------------------------------------------------------------------------------- finding the chart
def test_a_picture_chart_is_found_named_by_its_heading_and_left_uncalibrated(plain):
    _, truth, charts = plain
    assert len(charts) == 1
    c = charts[0]
    assert c.kind == "raster" and not c.calibrated and c.x_fit is None and c.y_fits == [] and c.curves == []
    assert c.title == "Picture chart" and c.notes == ["Discharge at 25 C (2.5 V)"]
    assert c.thumbnail is not None and c.thumbnail.shape[:2] == truth["shape"]
    assert c.pixel_pt > 0.1


def test_frame_is_the_axes_box(plain):
    _, truth, charts = plain
    plot = charts[0].raster.plot
    x0, y0, x1, y1 = truth["frame"]
    assert (plot.x0, plot.y0, plot.x1, plot.y1) == pytest.approx((x0, y0, x1, y1), abs=1.5)


def test_tick_marks_are_found_within_a_pixel(plain):
    _, truth, charts = plain
    t = charts[0].raster.ticks
    for _, p in truth["x_ticks"]:
        assert min(abs(v - p) for v in t.x) < 1.0
    for _, p in truth["y_ticks"]:
        assert min(abs(v - p) for v in t.y_left) < 1.0
    # nothing far from a real tick or the frame is offered: a click would snap onto a curve otherwise
    real = [p for _, p in truth["x_ticks"]]
    assert all(min(abs(v - p) for p in real) < 1.0 for v in t.x)


def test_right_ticks_of_a_second_axis_are_found(dual):
    _, truth, charts = dual
    t = charts[0].raster.ticks
    for _, p in truth["y2_ticks"]:
        assert min(abs(v - p) for v in t.y_right) < 1.0


def test_snap_moves_a_click_onto_the_nearest_tick_only_when_close():
    assert snap(101.0, [100.0, 200.0]) == 100.0
    assert snap(150.0, [100.0, 200.0]) == 150.0
    assert snap(5.0, []) == 5.0


def test_logos_and_pictures_without_a_frame_are_not_charts(tmp_path):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(6, 4), dpi=72)
    fig.figimage(np.random.default_rng(0).integers(0, 255, (300, 400, 3), dtype=np.uint8), 20, 20)
    p = tmp_path / "noise.pdf"
    fig.savefig(p, format="pdf")
    plt.close(fig)
    assert analyze_pdf(str(p)) == []


def test_vector_charts_are_not_duplicated_as_pictures(tmp_path):
    path, _ = make_pdfs.simple_lines(tmp_path / "v.pdf")
    charts = analyze_pdf(str(path))
    assert [c.kind for c in charts] == ["vector"]


# ------------------------------------------------------------------------------- reading the curves
def test_colours_have_turkish_names():
    assert colour_name((238, 67, 67)) == "kırmızı" and colour_name((57, 83, 163)) == "mavi"
    assert colour_name((0, 0, 0)) == "siyah" and colour_name((146, 214, 233)) == "camgöbeği"
    assert colour_name((242, 234, 34)) == "sarı" and colour_name((77, 184, 77)) == "yeşil"


def test_calibrated_chart_reads_all_three_curves_at_the_plotted_values(plain):
    _, truth, charts = plain
    chart = calibrate_exactly(charts[0], truth)
    rc = chart.raster
    rc.traces = discover_curves(rc)
    from core.raster_charts import rebuild_curves

    rebuild_curves(chart)
    assert chart.calibrated and len(chart.curves) == 3
    assert [t.label.split("(")[1] for t in rc.traces] == ["kırmızı)", "mavi)", "siyah)"]        # top curve first
    for cv, name in zip(chart.curves, ("red", "blue", "black")):
        xs, ys = truth["curves"][name]
        assert cv.data.x[0] < 60 and cv.data.x[-1] > 1940                                        # the whole width
        assert_close_to_truth(cv, xs, ys)
        assert cv.data.x_unc[0] > 0 and cv.data.y_unc[0] > 0
    assert chart.x_fit.title == X_NAME and chart.y_fits[0].title == Y_NAME


def test_calibration_errors_are_reported(plain):
    _, truth, charts = plain
    xt = truth["x_ticks"]
    yt = truth["y_ticks"]
    y = AxisPoints((yt[0][1], yt[-1][1]), (yt[0][0], yt[-1][0]), Y_NAME)
    with pytest.raises(ValueError):
        calibrate(charts[0], AxisPoints((xt[0][1], xt[0][1] + 1.0), (0, 100), "X"), y)      # marks too close
    with pytest.raises(ValueError):
        calibrate(charts[0], AxisPoints((xt[0][1], xt[-1][1]), (5.0, 5.0), "X"), y)          # same value twice
    with pytest.raises(ValueError):
        calibrate(charts[0], AxisPoints((xt[0][1], xt[-1][1]), (-1.0, 100.0), "X", log=True), y)


def test_a_curve_can_be_added_from_a_click_on_it(plain):
    _, truth, charts = plain
    chart = calibrate_exactly(charts[0], truth)
    rc = chart.raster
    px, py = truth["to_px"]("black", 1000.0)
    tr = trace_curve_at(rc, px, py)
    assert tr is not None and max(tr.rgb) < 80
    from core.raster_charts import curve_from_trace

    tr.label = "Siyah"
    rc.traces.append(tr)
    cv = curve_from_trace(0, tr, chart, rc)
    assert_close_to_truth(cv, *truth["curves"]["black"])
    # a click on empty paper finds nothing
    assert trace_curve_at(rc, rc.plot.x0 + 40, rc.plot.y0 + 30) is None


def test_second_axis_curves_are_assigned_to_the_axis_of_their_colour(dual):
    _, truth, charts = dual
    chart = calibrate_exactly(charts[0], truth, two_axes=True)
    rc = chart.raster
    from core.raster_charts import guess_axis, rebuild_curves

    rc.traces = discover_curves(rc)
    for tr in rc.traces:
        tr.axis = guess_axis(rc, tr.rgb, True)
    rebuild_curves(chart)
    by_colour = {colour_name(t.rgb): (t, cv) for t, cv in zip(rc.traces, chart.curves)}
    assert set(by_colour) >= {"yeşil", "kırmızı", "mavi", "siyah"}
    assert by_colour["yeşil"][0].axis == 1 and by_colour["kırmızı"][0].axis == 0
    xs, ys = truth["curves"]["temp"]
    assert_close_to_truth(by_colour["yeşil"][1], xs, ys)                                   # read on the right axis
    assert chart.y_fits[1].title == Y2_NAME and by_colour["yeşil"][1].axis_note == "sağ eksen"
    assert by_colour["kırmızı"][1].axis_note == "sol eksen"


def test_two_curves_of_one_colour_are_both_found(same):
    _, truth, charts = same
    chart = calibrate_exactly(charts[0], truth, two_axes=True)
    rc = chart.raster
    traces = discover_curves(rc)
    blues = [t for t in traces if colour_name(t.rgb) == "mavi"]
    assert len(blues) == 2                                                                # voltage curve + temperature curve
    ys = sorted(float(np.mean(t.y_px)) for t in blues)
    assert ys[1] - ys[0] > 20


def test_curves_read_from_a_picture_can_be_edited_and_queried_like_any_other(plain):
    _, truth, charts = plain
    chart = calibrate_exactly(charts[0], truth)
    rc = chart.raster
    rc.traces = discover_curves(rc)
    from core.raster_charts import rebuild_curves
    from core.query import solve_x

    rebuild_curves(chart)
    cv = chart.curves[0]                                                                   # red
    n = cv.n_points
    assert cv.delete_in_box(900, 1100, 0, 10) > 5 and cv.edited and cv.n_points < n
    a = value_at(cv.data, 1000.0)
    assert a is not None and a.value == pytest.approx(4.2 - 0.6, abs=0.03)                 # the hole is bridged
    cv.add_point(1000.0, 3.0, 1.0, 0.01)
    assert value_at(cv.data, 1000.0).value == pytest.approx(3.0, abs=0.01)
    hits = solve_x(cv.data, 3.6)
    assert len(hits) >= 1
    assert cv.undo() and cv.undo() and not cv.edited
    assert find_image_charts is not None and detect_ticks is not None
