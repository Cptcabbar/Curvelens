"""X -> Y and Y -> X lookups on a curve."""
import math

import numpy as np
import pytest

from core.postprocess import CurveData
from core.query import solve_x, value_at


def make(x, y, xu=0.0, yu=0.0):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    return CurveData("c", x, y, np.full(len(x), xu), np.full(len(x), yu))


def test_value_at_interpolates_linearly_and_hits_the_points_exactly():
    d = make([0, 10, 20], [0, 5, 25])
    assert value_at(d, 5).value == pytest.approx(2.5)
    assert value_at(d, 15).value == pytest.approx(15.0)
    assert value_at(d, 10).value == pytest.approx(5.0)
    assert value_at(d, 0).value == 0.0 and value_at(d, 20).value == 25.0


def test_value_at_outside_the_curve_is_none():
    d = make([0, 10], [0, 5])
    assert value_at(d, -0.1) is None and value_at(d, 10.1) is None
    assert value_at(d, float("nan")) is None


def test_value_at_averages_duplicate_x():
    d = make([0, 5, 5, 10], [0, 4, 6, 10])
    assert value_at(d, 5).value == pytest.approx(5.0)


def test_value_at_uncertainty_grows_with_the_slope():
    x = np.arange(0.0, 100.0)
    flat = make(x, np.zeros(100), xu=1.0, yu=0.01)
    steep = make(x, 5.0 * x, xu=1.0, yu=0.01)
    uf, us = value_at(flat, 50).unc, value_at(steep, 50).unc
    assert uf == pytest.approx(0.01, abs=1e-6)
    assert us == pytest.approx(math.hypot(0.01, 5.0 * 1.0), rel=1e-3)


def test_value_at_notes_a_gap():
    x = np.concatenate([np.arange(0, 10.0), np.arange(50, 60.0)])
    d = make(x, x * 2)
    assert "boşluk" in value_at(d, 30).note and value_at(d, 5).note == ""


def test_solve_x_monotonic_curve_single_answer():
    d = make([0, 10, 20], [0, 5, 25], xu=0.1, yu=0.05)
    a = solve_x(d, 15.0)
    assert len(a) == 1 and a[0].value == pytest.approx(15.0)
    assert a[0].unc == pytest.approx(math.hypot(0.1, 0.05 / 2.0), rel=0.3)


def test_solve_x_returns_every_crossing_of_a_non_monotonic_curve():
    x = np.linspace(0, 10, 101)
    d = make(x, np.sin(x))                                   # crosses 0.5 four times in 0..10
    a = solve_x(d, 0.5)
    got = [r.value for r in a]
    expect = [math.asin(0.5), math.pi - math.asin(0.5), 2 * math.pi + math.asin(0.5), 3 * math.pi - math.asin(0.5)]
    assert len(a) == 4 and np.allclose(got, expect, atol=0.02)
    assert got == sorted(got)


def test_solve_x_level_the_curve_never_reaches_is_empty():
    d = make([0, 10], [1, 2])
    assert solve_x(d, 0.5) == [] and solve_x(d, 2.5) == []


def test_solve_x_at_a_vertex_and_at_the_ends():
    d = make([0, 10, 20], [0, 5, 0])
    a = solve_x(d, 5.0)
    assert len(a) == 1 and a[0].value == pytest.approx(10.0)
    b = solve_x(make([0, 10], [3, 7]), 3.0)
    assert len(b) == 1 and b[0].value == 0.0


def test_solve_x_flat_stretch_is_one_answer_with_a_note():
    d = make([0, 10, 20, 30], [0, 4, 4, 9])
    a = solve_x(d, 4.0)
    assert len(a) == 1 and a[0].value == pytest.approx(15.0) and "düz" in a[0].note


def test_solve_x_noisy_plateau_crossings_are_merged():
    rng = np.random.default_rng(1)
    x = np.linspace(0, 100, 401)
    y = 3.0 + 0.001 * rng.standard_normal(401)              # a plateau at 3.0 with noise
    y[:20] += np.linspace(2, 0, 20)                          # ... entered from above
    a = solve_x(make(x, y, yu=0.004), 3.0)                   # the points' own uncertainty covers the noise
    assert len(a) == 1 and "düz" in a[0].note
    assert 5.0 < a[0].value < 95.0 and a[0].unc > 30.0       # anywhere on the plateau: a wide +/- band


def test_solve_x_flat_piece_reports_its_x_range_as_uncertainty():
    x = np.linspace(0, 10, 50)
    d = make(x, np.concatenate([np.linspace(5, 3, 25), np.full(25, 3.0)]), yu=0.01)
    a = solve_x(d, 3.0)
    assert len(a) == 1 and "düz" in a[0].note and a[0].unc > 2.0


def test_solve_x_noisy_steep_crossing_is_still_one_answer():
    rng = np.random.default_rng(3)
    x = np.linspace(0, 10, 201)
    y = 2.0 * x + 0.05 * rng.standard_normal(201)            # noise 0.05 = 0.5 step of x
    a = solve_x(make(x, y, xu=0.01, yu=0.05), 10.0)
    assert len(a) == 1 and a[0].value == pytest.approx(5.0, abs=0.1)


def test_empty_and_single_point_curves():
    e = CurveData("e", *(np.empty(0) for _ in range(4)))
    assert value_at(e, 1.0) is None and solve_x(e, 1.0) == []
    one = make([2.0], [7.0])
    assert value_at(one, 2.0).value == 7.0 and solve_x(one, 7.0) == []
