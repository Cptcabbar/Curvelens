import numpy as np
import pandas as pd
import pytest

from core.models import Rect
from core.project import Project, nice_step
from tests.synth import draw_curve, make_plot

CURVE_Y = lambda x: 300 - 0.4 * (x - 50)          # pixel y of the synthetic red curve


@pytest.fixture()
def project():
    """Synthetic chart: x 0..1000 mAh over px 50..590, y 0..5 V over px 370..50, one red curve."""
    img, (x0, y0, x1, y1) = make_plot()
    xs = np.linspace(x0 + 4, x1 - 4, 900)
    draw_curve(img, xs, CURVE_Y(xs), (0, 0, 255), 3)                            # red (BGR)
    p = Project()
    p.set_image(img, "synthetic")
    p.x_axis.label, p.x_axis.unit = "Capacity", "mAh"
    p.y_axis.label, p.y_axis.unit = "Voltage", "V"
    p.set_axis_point("x", 0, pixel=50, value=0)
    p.set_axis_point("x", 1, pixel=590, value=1000)
    p.set_axis_point("y", 0, pixel=370, value=0)
    p.set_axis_point("y", 1, pixel=50, value=5)
    p.set_plot_area(Rect(x0 + 2, y0 + 2, x1 - 2, y1 - 2))
    return p


def test_calibration_becomes_available_only_when_complete():
    p = Project()
    assert p.calibration() is None
    p.set_axis_point("x", 0, pixel=0, value=0)
    p.set_axis_point("x", 1, pixel=100, value=10)
    p.set_axis_point("y", 0, pixel=100, value=0)
    assert p.calibration() is None
    p.set_axis_point("y", 1, pixel=0, value=5)
    cal = p.calibration()
    assert cal is not None and float(cal.x.to_value(50)) == pytest.approx(5)


def test_invalid_calibration_reports_a_readable_error():
    p = Project()
    p.x_axis.log = True
    p.set_axis_point("x", 0, pixel=0, value=0)          # zero on a log axis
    p.set_axis_point("x", 1, pixel=100, value=10)
    p.set_axis_point("y", 0, pixel=100, value=0)
    p.set_axis_point("y", 1, pixel=0, value=5)
    with pytest.raises(ValueError):
        p.calibration()
    assert "pozitif" in p.calibration_error()


def test_add_curve_picks_colour_and_extract_fills_points(project):
    c = project.add_curve(300.0, CURVE_Y(300.0), "0.56A")
    assert c.name == "0.56A" and c.color_rgb[0] > 200 and c.color_rgb[1] < 80
    res = project.extract(0)
    assert len(res) > 400 and len(c) == len(res)
    data = project.curve_data(0, post=False)
    assert data.x[0] < 20 and data.x[-1] > 950
    # x = 500 mAh -> px 320 -> V = (370 - y_px) / 320 * 5
    assert np.interp(500, data.x, data.y) == pytest.approx((370 - CURVE_Y(320.0)) / 320 * 5, abs=0.01)


def test_default_curve_names_are_numbered(project):
    project.add_curve_with_color((255, 0, 0))
    project.add_curve_with_color((0, 255, 0))
    assert [c.name for c in project.curves] == ["Eğri 1", "Eğri 2"]


def test_erase_add_undo_redo(project):
    project.add_curve(300.0, CURVE_Y(300.0))
    project.extract(0)
    n_extracted = len(project.curves[0])
    project.undo()                                        # undo the extraction itself
    assert len(project.curves[0]) == 0
    project.redo()
    assert len(project.curves[0]) == n_extracted

    removed = project.erase(0, Rect(200, 0, 300, 400))
    assert removed > 50 and len(project.curves[0]) == n_extracted - removed
    assert project.erase(0, Rect(0, 0, 5, 5)) == 0        # nothing there -> no undo entry
    project.add_point(0, 250.5, 190.5)
    assert len(project.curves[0]) == n_extracted - removed + 1
    assert project.undo() == 0 and len(project.curves[0]) == n_extracted - removed
    assert project.undo() == 0 and len(project.curves[0]) == n_extracted
    assert project.redo() == 0 and len(project.curves[0]) == n_extracted - removed
    project.add_point(0, 1, 1)                            # a new edit clears the redo stack
    assert not project.can_redo()


def test_undo_when_nothing_to_undo(project):
    assert project.undo() is None and project.redo() is None


def test_undo_survives_removal_of_another_curve(project):
    project.add_curve_with_color((255, 0, 0))
    project.add_curve_with_color((0, 255, 0))
    project.curves[1].set_points([1, 2, 3], [1, 2, 3])
    project.add_point(1, 4, 4)
    project.remove_curve(0)
    assert project.undo() == 0                            # former curve 1 is now curve 0
    assert len(project.curves[0]) == 3


def test_post_processing_resamples_to_the_grid_and_smooths(project):
    project.add_curve(300.0, CURVE_Y(300.0))
    project.extract(0)
    project.post.resample, project.post.step = True, 50.0
    d = project.curve_data(0)
    assert np.allclose(np.diff(d.x), 50.0) and d.x[0] % 50 == 0
    project.post.smooth = True
    d2 = project.curve_data(0)
    assert np.allclose(d2.x, d.x) and np.max(np.abs(d2.y - d.y)) < 0.02
    assert project.suggested_step() in (5.0, 10.0)
    assert "Savitzky" in project.smoothing_note()


def test_curve_data_is_none_until_calibrated_and_extracted(project):
    project.add_curve_with_color((255, 0, 0))
    assert project.curve_data(0) is None                  # no points yet
    project.curves[0].set_points([1, 2], [1, 2])
    project.x_axis.value[1] = None
    assert project.curve_data(0) is None                  # calibration incomplete


def test_export_combined_and_separate(project, tmp_path):
    project.add_curve(300.0, CURVE_Y(300.0), "A")
    project.add_curve_with_color((0, 255, 0), "B")
    project.extract(0)
    project.curves[1].set_points([60, 300, 500], [200, 150, 100])
    project.post.resample, project.post.step = True, 100.0

    paths = project.export(tmp_path / "all.csv", combined=True)
    df = pd.read_csv(paths[0], comment="#", encoding="utf-8-sig")
    assert df.columns[0] == "Capacity (mAh)" and df.shape[1] == 3
    head = paths[0].read_text(encoding="utf-8-sig")
    assert "source: synthetic" in head and "uncertainty" in head and "resampled to x step: 100 mAh" in head

    paths = project.export(tmp_path / "each", combined=False)
    assert sorted(p.name for p in paths) == ["A.csv", "B.csv"]


def test_export_guards(project, tmp_path):
    with pytest.raises(ValueError, match="veri yok"):
        project.export(tmp_path / "x", combined=False)
    project.add_curve(300.0, CURVE_Y(300.0))
    project.extract(0)
    with pytest.raises(ValueError, match="Ortak X"):
        project.export(tmp_path / "x.csv", combined=True)  # resampling not enabled
    project.x_axis.value[0] = None
    with pytest.raises(ValueError, match="kalibrasyon"):
        project.export(tmp_path / "x", combined=False)


def test_set_image_resets_everything(project):
    project.add_curve_with_color((255, 0, 0))
    project.set_image(np.zeros((10, 10, 3), np.uint8), "other")
    assert project.curves == [] and project.calibration() is None and project.plot_area is None
    assert project.source == "other" and not project.can_undo()


def test_exclusions_are_forwarded_to_extraction(project):
    project.add_curve(300.0, CURVE_Y(300.0))
    project.extract(0)
    n_all = len(project.curves[0])
    project.add_exclusion(Rect(250, 0, 350, 400))
    project.extract(0)
    assert len(project.curves[0]) < n_all - 50
    project.clear_exclusions()
    project.extract(0)
    assert len(project.curves[0]) == n_all


def test_nice_step():
    assert nice_step(3000) in (10.0, 20.0)
    assert nice_step(1.0) == 0.005
    assert nice_step(0) == 1.0


def test_logarithmic_y_axis_end_to_end():
    """Curve that is a straight line in semi-log space: v(x) = 10**(3 - 3 x / 1000)."""
    img, (x0, y0, x1, y1) = make_plot(grid=False)
    xs = np.linspace(x0 + 6, x1 - 6, 900)
    # y pixel: 3 decades (10^3 .. 10^0) map onto px 50..370 (top = 1000, bottom = 1)
    y_px = lambda x: 50 + (x - 50) * (320.0 / 540.0)
    draw_curve(img, xs, y_px(xs), (0, 0, 255), 3)
    p = Project()
    p.set_image(img, "log")
    p.set_axis_point("x", 0, pixel=50, value=0)
    p.set_axis_point("x", 1, pixel=590, value=1000)
    p.set_axis_point("y", 0, pixel=370, value=1)              # bottom: 1
    p.set_axis_point("y", 1, pixel=50, value=1000)            # top: 1000
    p.y_axis.log = True
    p.set_plot_area(Rect(x0 + 2, y0 + 2, x1 - 2, y1 - 2))
    p.add_curve(300.0, y_px(300.0))
    p.extract(0)
    d = p.curve_data(0, post=False)
    assert d is not None and len(d) > 400
    # v = 1000 at x = 0 (px 50) falling to 1 at px 590 (x = 1000): v(x) = 10**(3 - 3*x/1000)
    expected = 10 ** (3 - 3 * d.x / 1000)
    inner = (d.x > 30) & (d.x < 970)
    assert np.max(np.abs(d.y[inner] / expected[inner] - 1)) < 0.02          # within 2 % everywhere
    # uncertainty grows with the value on a log axis
    assert d.y_unc[np.argmax(d.y)] > 50 * d.y_unc[np.argmin(d.y)]


def test_clicking_selects_the_curve_among_same_coloured_ones():
    from tests.synth import draw_curve as dc
    img, (x0, y0, x1, y1) = make_plot(grid=False)
    xs = np.linspace(x0 + 4, x1 - 4, 900)
    top = lambda x: 120 + 0.1 * (x - 50)
    bottom = lambda x: 300 - 0.1 * (x - 50)
    dc(img, xs, top(xs), (0, 0, 255), 3)
    dc(img, xs, bottom(xs), (0, 0, 255), 3)
    p = Project()
    p.set_image(img)
    p.set_plot_area(Rect(x0, y0, x1, y1))
    a = p.add_curve(300.5, top(300.5))
    b = p.add_curve(300.5, bottom(300.5))
    assert a.seed == (300.5, top(300.5))
    p.extract(0)
    p.extract(1)
    assert np.median(np.abs(a.y_px - top(a.x_px))) < 0.3
    assert np.median(np.abs(b.y_px - bottom(b.x_px))) < 0.3


def test_curve_quality_reports_extent_and_density(project):
    assert project.curve_quality(0) if project.curves else True
    project.add_curve(300.0, CURVE_Y(300.0))
    assert project.curve_quality(0) is None                       # not extracted yet
    project.extract(0)
    extent, density = project.curve_quality(0)
    assert extent > 0.95 and density > 0.95
    project.erase(0, Rect(150, 0, 350, 400))                       # a hole in the middle
    extent2, density2 = project.curve_quality(0)
    assert extent2 == pytest.approx(extent, abs=0.02) and density2 < 0.75
    project.erase(0, Rect(200, 0, 700, 400))
    assert project.curve_quality(0)[0] < 0.2
