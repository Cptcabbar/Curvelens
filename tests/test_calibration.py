import numpy as np
import pytest

from core.calibration import AxisCalibration, Calibration


def test_linear_roundtrip_and_known_values():
    ax = AxisCalibration(p1=100, v1=0, p2=500, v2=3000)
    assert ax.to_value(100) == pytest.approx(0)
    assert ax.to_value(500) == pytest.approx(3000)
    assert ax.to_value(300) == pytest.approx(1500)
    assert ax.to_value(700) == pytest.approx(4500)  # extrapolation is allowed
    px = np.linspace(0, 800, 17)
    assert ax.to_pixel(ax.to_value(px)) == pytest.approx(px)


def test_linear_inverted_axis_like_image_y():
    # Image y grows downwards: pixel 400 is 0 V, pixel 100 is 5 V.
    ay = AxisCalibration(p1=400, v1=0.0, p2=100, v2=5.0)
    assert ay.to_value(250) == pytest.approx(2.5)
    assert ay.to_value(100) == pytest.approx(5.0)
    assert ay.resolution() == pytest.approx(5.0 / 300)


def test_log_axis_decades_are_equidistant():
    ax = AxisCalibration(p1=0, v1=1, p2=300, v2=1000, log=True)
    assert ax.to_value(100) == pytest.approx(10)
    assert ax.to_value(200) == pytest.approx(100)
    assert ax.to_value(150) == pytest.approx(10 ** 1.5)
    assert ax.to_pixel(100) == pytest.approx(200)
    px = np.array([10.0, 55.5, 240.0])
    assert ax.to_pixel(ax.to_value(px)) == pytest.approx(px)


def test_log_axis_with_non_decade_reference_points():
    ax = AxisCalibration(p1=50, v1=2, p2=450, v2=2000, log=True)
    assert ax.to_value(50) == pytest.approx(2)
    assert ax.to_value(450) == pytest.approx(2000)
    # geometric mean sits halfway
    assert ax.to_value(250) == pytest.approx(np.sqrt(2 * 2000))


def test_resolution_linear_is_constant_and_log_scales_with_value():
    lin = AxisCalibration(0, 0, 200, 100)
    assert lin.resolution() == pytest.approx(0.5)
    assert lin.resolution(np.array([0, 10, 199])) == pytest.approx([0.5, 0.5, 0.5])
    assert lin.uncertainty() == pytest.approx(0.25)  # +/-0.5 px

    log = AxisCalibration(0, 1, 100, 100, log=True)   # 2 decades / 100 px
    px = np.array([0.0, 50.0, 100.0])
    res = log.resolution(px)
    # dv/dpx = ln(10) * v * (2/100)
    assert res == pytest.approx(np.log(10) * np.array([1, 10, 100]) * 0.02)
    assert res[2] / res[0] == pytest.approx(100)


def test_uncertainty_matches_half_pixel_resolution():
    ax = AxisCalibration(0, 0, 1000, 5)
    assert ax.uncertainty() == pytest.approx(0.5 * 0.005)
    assert ax.uncertainty(pixel_error=1.0) == pytest.approx(0.005)


@pytest.mark.parametrize("kwargs", [
    dict(p1=10, v1=0, p2=10, v2=5),              # same pixel
    dict(p1=0, v1=3, p2=10, v2=3),               # same value
    dict(p1=0, v1=0, p2=10, v2=5, log=True),     # non-positive on log axis
    dict(p1=0, v1=-1, p2=10, v2=5, log=True),
    dict(p1=0, v1=float("nan"), p2=10, v2=5),
])
def test_invalid_axis_raises(kwargs):
    with pytest.raises(ValueError):
        AxisCalibration(**kwargs)


def test_axes_are_independent_of_each_other():
    """X depends only on the x pixel, Y only on the y pixel."""
    cal = Calibration(x=AxisCalibration(100, 0, 600, 3000),
                      y=AxisCalibration(500, 0, 50, 5))
    xv, yv = cal.pixel_to_data(350, 275)
    assert xv == pytest.approx(1500)
    assert yv == pytest.approx(2.5)
    # Changing y pixels leaves X untouched and vice versa.
    assert cal.pixel_to_data(350, 100)[0] == pytest.approx(1500)
    assert cal.pixel_to_data(100, 275)[1] == pytest.approx(2.5)
    px, py = cal.data_to_pixel(1500, 2.5)
    assert (px, py) == pytest.approx((350, 275))


def test_mixed_linear_x_log_y():
    cal = Calibration(x=AxisCalibration(0, 0, 100, 10),
                      y=AxisCalibration(400, 1, 0, 1e4, log=True))
    xv, yv = cal.pixel_to_data(50, 200)
    assert xv == pytest.approx(5)
    assert yv == pytest.approx(100)
    ux, uy = cal.uncertainty(x_px=50, y_px=200)
    assert ux == pytest.approx(0.05)
    assert uy == pytest.approx(0.5 * np.log(10) * 100 * (4 / 400))


def test_dict_roundtrip():
    cal = Calibration(x=AxisCalibration(1, 2, 3, 4), y=AxisCalibration(5, 1, 9, 100, log=True))
    assert Calibration.from_dict(cal.to_dict()) == cal
