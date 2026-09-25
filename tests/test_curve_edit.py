"""Hand corrections of a curve: add / delete / move points, undo, redo, reset."""
import numpy as np
import pytest

from core.curve_edit import CurveEditor
from core.postprocess import CurveData


def curve(n=11):
    x = np.linspace(0.0, 10.0, n)
    return CurveData("c", x, x ** 2, np.full(n, 0.1), np.full(n, 0.5))


def test_untouched_curve_is_not_edited_and_data_is_a_copy():
    src = curve()
    ed = CurveEditor(src)
    assert not ed.is_edited and ed.n_changes == 0 and not ed.can_undo and not ed.can_redo
    ed.data.y[0] = 99.0                                       # the editor never writes into the caller's arrays
    assert src.y[0] == 0.0


def test_add_point_keeps_x_sorted_and_returns_its_index():
    ed = CurveEditor(curve())
    i = ed.add_point(2.5, 7.0)
    d = ed.data
    assert len(d) == 12 and d.x[i] == 2.5 and d.y[i] == 7.0
    assert np.all(np.diff(d.x) >= 0)
    assert ed.is_edited and ed.n_changes == 1


def test_added_point_gets_median_uncertainty_unless_given():
    ed = CurveEditor(curve())
    i = ed.add_point(2.5, 7.0)
    assert ed.data.x_unc[i] == pytest.approx(0.1) and ed.data.y_unc[i] == pytest.approx(0.5)
    j = ed.add_point(3.5, 9.0, x_unc=0.3, y_unc=0.2)
    assert ed.data.x_unc[j] == pytest.approx(0.3) and ed.data.y_unc[j] == pytest.approx(0.2)


def test_add_point_at_an_existing_x_goes_after_the_old_one():
    ed = CurveEditor(curve())
    i = ed.add_point(4.0, 100.0)
    assert ed.data.x[i] == 4.0 and ed.data.y[i] == 100.0 and ed.data.y[i - 1] == pytest.approx(16.0)


def test_non_finite_coordinates_are_refused():
    ed = CurveEditor(curve())
    with pytest.raises(ValueError):
        ed.add_point(float("nan"), 1.0)
    with pytest.raises(ValueError):
        ed.move_point(0, 1.0, float("inf"))
    assert not ed.is_edited


def test_delete_points_and_out_of_range_indices_are_ignored():
    ed = CurveEditor(curve())
    assert ed.delete([2, 3, 3, 99, -1]) == 2
    assert len(ed) == 9 and 2.0 not in ed.data.x and 3.0 not in ed.data.x
    assert ed.delete([]) == 0 and ed.n_changes == 1          # nothing to do -> no history entry


def test_delete_in_box_uses_data_units_and_borders_are_inclusive():
    ed = CurveEditor(curve())
    n = ed.delete_in_box(3.0, 5.0, 0.0, 100.0)               # x = 3, 4, 5
    assert n == 3 and list(ed.data.x[:4]) == [0.0, 1.0, 2.0, 6.0]
    assert ed.delete_in_box(5.0, 3.0, 1000.0, 2000.0) == 0   # swapped corners are fine, nothing inside


def test_move_point_resorts_and_returns_new_index():
    ed = CurveEditor(curve())
    i = ed.move_point(1, 7.5, 50.0)                            # x = 1 -> 7.5
    d = ed.data
    assert np.all(np.diff(d.x) >= 0) and d.x[i] == 7.5 and d.y[i] == 50.0
    assert 1.0 not in d.x and len(d) == 11
    with pytest.raises(IndexError):
        ed.move_point(50, 0.0, 0.0)


def test_move_point_keeps_the_points_own_uncertainty():
    src = curve()
    src.y_unc[3] = 2.0
    ed = CurveEditor(src)
    i = ed.move_point(3, 9.5, 1.0)
    assert ed.data.y_unc[i] == 2.0


def test_undo_redo_walk_through_the_history():
    ed = CurveEditor(curve())
    ed.add_point(2.5, 7.0)
    ed.delete([0])
    ed.move_point(0, 1.5, 3.0)
    assert ed.n_changes == 3
    assert ed.undo() and ed.undo() and ed.undo()
    assert not ed.undo() and not ed.is_edited and len(ed) == 11
    assert ed.redo() and ed.redo() and ed.redo()
    assert not ed.redo() and ed.is_edited and ed.n_changes == 3


def test_new_edit_clears_the_redo_stack():
    ed = CurveEditor(curve())
    ed.add_point(2.5, 7.0)
    ed.undo()
    assert ed.can_redo
    ed.add_point(3.5, 8.0)
    assert not ed.can_redo


def test_undoing_every_edit_means_not_edited_again():
    ed = CurveEditor(curve())
    ed.delete([4])
    ed.undo()
    assert not ed.is_edited


def test_reset_restores_the_original_and_can_be_undone():
    src = curve()
    ed = CurveEditor(src)
    ed.delete([0, 1, 2])
    ed.add_point(9.5, 1.0)
    assert ed.reset() and not ed.is_edited
    assert np.array_equal(ed.data.x, src.x) and np.array_equal(ed.data.y, src.y)
    assert not ed.reset()                                     # nothing left to reset
    assert ed.undo() and ed.is_edited and len(ed) == 9        # the edits come back


def test_history_is_bounded():
    ed = CurveEditor(curve())
    for k in range(250):
        ed.add_point(0.001 * k, 1.0)
    assert ed.n_changes == 200


def test_nearest_measures_in_scaled_units():
    ed = CurveEditor(curve())
    i, dist = ed.nearest(3.1, 9.2)
    assert i == 3 and dist < 0.5
    # a 100x larger y scale makes y differences matter less: the closest x wins
    j, _ = ed.nearest(3.4, 1000.0, x_scale=1.0, y_scale=1e6)
    assert j == 3


def test_empty_curve_is_handled():
    e = CurveData("e", *(np.empty(0) for _ in range(4)))
    ed = CurveEditor(e)
    assert ed.nearest(1, 1) is None and ed.delete([0]) == 0
    i = ed.add_point(1.0, 2.0)
    assert i == 0 and len(ed) == 1
