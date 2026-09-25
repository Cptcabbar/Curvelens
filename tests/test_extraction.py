import cv2
import numpy as np
import pytest

from core.extraction import (bgr_to_lab, delta_e_map, extract_curve, pick_reference_color, rgb_to_lab)
from core.models import Curve, ExtractionParams, Rect
from tests.synth import dashed_line, draw_curve, make_plot

RED, GREEN, BLUE, BLACK, MAGENTA = (255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0), (255, 0, 255)


def bgr(rgb):
    return (rgb[2], rgb[1], rgb[0])


def _curve(xs, level, amp, decay=0.0):
    return level - amp * (xs - xs[0]) / (xs[-1] - xs[0]) - decay * np.exp((xs - xs[-1]) / 15.0)


CAP = 4   # px at both ends of a drawn stroke are affected by its round cap


def _errors(res, xs, ys_fn, cap=CAP):
    """Vertical errors against the analytic curve, ignoring the stroke's end caps."""
    keep = (res.x_px > xs[0] + cap) & (res.x_px < xs[-1] - cap)
    return np.abs(res.y_px[keep] - ys_fn(res.x_px[keep]))


def _max_error(res, xs, ys_fn):
    return float(np.max(_errors(res, xs, ys_fn)))


# --------------------------------------------------------------------------- colour
def test_lab_of_pure_red_matches_reference():
    lab = rgb_to_lab(RED)
    assert lab == pytest.approx([53.24, 80.09, 67.20], abs=0.1)


def test_delta_e_zero_for_same_colour_and_large_for_others():
    img = np.zeros((2, 2, 3), np.uint8)
    img[0, 0] = bgr(RED)
    img[0, 1] = bgr(GREEN)
    img[1, 0] = bgr((250, 10, 10))
    img[1, 1] = (255, 255, 255)
    de = delta_e_map(img, RED)
    assert de[0, 0] == pytest.approx(0, abs=1e-3)
    assert de[1, 0] < 10
    assert de[0, 1] > 100 and de[1, 1] > 50


def test_pick_reference_color_uses_pixel_under_position():
    img = np.zeros((10, 10, 3), np.uint8)
    img[4, 7] = bgr((12, 34, 56))
    assert pick_reference_color(img, 7.5, 4.5) == (12, 34, 56)
    assert pick_reference_color(img, 99, 99) == (0, 0, 0)          # clamped


# --------------------------------------------------------------------------- core extraction
def test_smooth_coloured_curve_is_recovered_with_subpixel_accuracy():
    img, (x0, y0, x1, y1) = make_plot()
    xs = np.linspace(x0 + 4, x1 - 4, 900)
    f = lambda x: 150 + 0.25 * (x - x0) + 40 * np.sin((x - x0) / 80)
    draw_curve(img, xs, f(xs), bgr(RED), 3)
    res = extract_curve(img, RED, Rect(x0, y0, x1, y1))
    assert len(res) > 0.9 * (x1 - x0 - 8)
    # Sub-pixel: edge pixels of partial coverage weigh in slightly differently
    # depending on the stroke's phase, hence a few 1/10 px, never a whole pixel.
    assert _max_error(res, xs, f) < 0.4
    assert np.mean(_errors(res, xs, f)) < 0.12
    # x positions are pixel centres
    assert np.allclose(res.x_px % 1, 0.5)


def test_black_curve_survives_black_dashed_grid_and_frame():
    """Black curve on a plot whose grid and frame are black too."""
    img = np.full((420, 640, 3), 255, np.uint8)
    x0, y0, x1, y1 = 50, 50, 590, 370
    for i in range(1, 6):
        dashed_line(img, (x0 + (x1 - x0) * i / 6, y0), (x0 + (x1 - x0) * i / 6, y1), (0, 0, 0), 5, 3)
        dashed_line(img, (x0, y0 + (y1 - y0) * i / 6), (x1, y0 + (y1 - y0) * i / 6), (0, 0, 0), 5, 3)
    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 0), 2)
    xs = np.linspace(x0 + 5, x1 - 5, 800)
    f = lambda x: 120 + 0.35 * (x - x0) + 30 * np.sin((x - x0) / 60)
    draw_curve(img, xs, f(xs), (0, 0, 0), 3)
    res = extract_curve(img, BLACK, Rect(x0, y0, x1, y1), params=ExtractionParams(delta_e=40))
    assert len(res) > 0.9 * (x1 - x0 - 10)
    # only columns crossing a grid line / the frame may be off; the bulk must be exact
    err = _errors(res, xs, f)
    assert np.percentile(err, 95) < 0.3
    assert err.max() < 1.5


def test_text_and_legend_swatch_are_ignored_by_component_filter():
    img, (x0, y0, x1, y1) = make_plot()
    xs = np.linspace(x0 + 4, x1 - 4, 900)
    f = lambda x: 200 - 0.2 * (x - x0)
    draw_curve(img, xs, f(xs), bgr(BLACK), 3)
    cv2.putText(img, "Charge: CC-CV, 2.8A", (x0 + 30, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.line(img, (x1 - 120, 90), (x1 - 60, 90), (0, 0, 0), 3, cv2.LINE_AA)        # legend swatch
    res = extract_curve(img, BLACK, Rect(x0 + 4, y0 + 4, x1 - 4, y1 - 4), params=ExtractionParams(delta_e=40))
    assert _max_error(res, xs, f) < 0.3


def test_exclusion_rectangle_removes_a_same_coloured_distractor():
    img, (x0, y0, x1, y1) = make_plot()
    xs = np.linspace(x0 + 4, x1 - 4, 900)
    f = lambda x: 250 - 0.2 * (x - x0)
    draw_curve(img, xs, f(xs), bgr(BLUE), 3)
    # A long horizontal blue "legend" line that is as long as the curve itself.
    cv2.line(img, (x0 + 20, 100), (x1 - 20, 100), bgr(BLUE), 3, cv2.LINE_AA)
    area = Rect(x0, y0, x1, y1)
    with_ex = extract_curve(img, BLUE, area, [Rect(x0, 90, x1, 110)])
    assert _max_error(with_ex, xs, f) < 0.3
    assert len(with_ex) > 0.9 * (x1 - x0 - 8)


def test_flat_curve_spanning_the_whole_width_is_not_mistaken_for_grid():
    """A constant-value curve (e.g. constant current) covers the full width like a grid line."""
    img, (x0, y0, x1, y1) = make_plot()
    xs = np.linspace(x0 + 4, x1 - 4, 900)
    draw_curve(img, xs, np.full_like(xs, 233.0), bgr(BLUE), 3)
    res = extract_curve(img, BLUE, Rect(x0, y0, x1, y1))
    assert len(res) > 0.9 * (x1 - x0 - 8)
    assert np.max(np.abs(res.y_px - 233.0)[(res.x_px > xs[0] + CAP) & (res.x_px < xs[-1] - CAP)]) < 0.3


def test_dashed_black_grid_is_removed_but_frame_and_ticks_do_not_win():
    img = np.full((420, 640, 3), 255, np.uint8)
    x0, y0, x1, y1 = 50, 50, 590, 370
    for i in range(1, 6):
        dashed_line(img, (x0 + (x1 - x0) * i / 6, y0), (x0 + (x1 - x0) * i / 6, y1), (0, 0, 0), 5, 3)
        dashed_line(img, (x0, y0 + (y1 - y0) * i / 6), (x1, y0 + (y1 - y0) * i / 6), (0, 0, 0), 5, 3)
    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 0), 3)
    mask = np.all(img[y0:y1 + 1, x0:x1 + 1] < 60, axis=2)
    from core.extraction import clean_mask
    cleaned = clean_mask(mask, ExtractionParams())
    assert cleaned.sum() < 0.02 * mask.sum()          # nothing curve-like remains


def test_plot_area_limits_the_search():
    img, (x0, y0, x1, y1) = make_plot()
    xs = np.linspace(x0 + 4, x1 - 4, 900)
    draw_curve(img, xs, 200 - 0.1 * (xs - x0), bgr(RED), 3)
    left_half = Rect(x0, y0, (x0 + x1) / 2, y1)
    res = extract_curve(img, RED, left_half)
    assert res.x_px.max() <= left_half.x1
    assert res.x_px.min() >= x0


def test_crossing_curves_of_different_colours_stay_separate():
    img, (x0, y0, x1, y1) = make_plot()
    xs = np.linspace(x0 + 4, x1 - 4, 1000)
    f_red = lambda x: 120 + 0.5 * (x - x0)              # descending on screen (y grows)
    f_blue = lambda x: 300 - 0.35 * (x - x0)            # crosses the red one
    draw_curve(img, xs, f_red(xs), bgr(RED), 3)
    draw_curve(img, xs, f_blue(xs), bgr(BLUE), 3)
    area = Rect(x0, y0, x1, y1)
    r = extract_curve(img, RED, area)
    b = extract_curve(img, BLUE, area)
    assert np.median(_errors(r, xs, f_red)) < 0.25
    assert np.median(_errors(b, xs, f_blue)) < 0.25
    # No point may have jumped onto the other curve (a jump would be tens of px).
    assert _max_error(r, xs, f_red) < 1.5
    assert _max_error(b, xs, f_blue) < 1.5


def test_two_runs_in_a_column_pick_the_one_continuing_the_trend():
    """Two same-coloured curves: tracking must stay on the one it started with."""
    img, (x0, y0, x1, y1) = make_plot(grid=False)
    xs = np.linspace(x0 + 4, x1 - 4, 900)
    f1 = lambda x: 150 + 0.1 * (x - x0)
    f2 = lambda x: 260 - 0.05 * (x - x0)                # never closer than ~30 px
    draw_curve(img, xs, f1(xs), bgr(RED), 3)
    draw_curve(img, xs, f2(xs), bgr(RED), 3)
    res = extract_curve(img, RED, Rect(x0, y0, x1, y1))
    inner = (res.x_px > xs[0] + CAP) & (res.x_px < xs[-1] - CAP)
    on1 = (np.abs(res.y_px - f1(res.x_px)) < 1.0)[inner]
    on2 = (np.abs(res.y_px - f2(res.x_px)) < 1.0)[inner]
    # Every point belongs to exactly one of the two curves and the trace never zig-zags between them.
    assert np.all(on1 | on2)
    assert on1.all() or on2.all()


def test_steep_section_and_sharp_knee_are_followed():
    img, (x0, y0, x1, y1) = make_plot()
    xs = np.linspace(x0 + 4, x1 - 4, 2000)
    f = lambda x: 100 + 0.15 * (x - x0) + 200 * np.exp((x - (x1 - 30)) / 6.0)
    ys = np.clip(f(xs), y0 + 3, y1 - 3)
    draw_curve(img, xs, ys, bgr(GREEN), 3)
    res = extract_curve(img, GREEN, Rect(x0, y0, x1, y1))
    flat = res.x_px < x1 - 60
    flat &= res.x_px > xs[0] + CAP
    assert np.max(np.abs(res.y_px[flat] - f(res.x_px[flat]))) < 0.3
    assert res.x_px.max() > x1 - 25                      # reached the steep end
    assert res.y_px[-1] > res.y_px[flat][-1] + 30        # ... and followed it downwards


def test_gap_from_occlusion_is_bridged_and_far_side_is_reached():
    img, (x0, y0, x1, y1) = make_plot(grid=False)
    xs = np.linspace(x0 + 4, x1 - 4, 900)
    f = lambda x: 180 + 0.2 * (x - x0)
    draw_curve(img, xs, f(xs), bgr(RED), 3)
    cv2.rectangle(img, (300, 100), (315, 300), (255, 255, 255), -1)      # 16 px hole
    res = extract_curve(img, RED, Rect(x0, y0, x1, y1))
    assert res.x_px.min() < 320 and res.x_px.max() > 500
    assert _max_error(res, xs, f) < 0.3


def test_thin_one_pixel_line_is_extracted():
    img, (x0, y0, x1, y1) = make_plot(grid=False)
    xs = np.linspace(x0 + 4, x1 - 4, 900)
    f = lambda x: 180 + 0.2 * (x - x0) + 20 * np.sin((x - x0) / 50)
    draw_curve(img, xs, f(xs), bgr(BLUE), 1)
    res = extract_curve(img, BLUE, Rect(x0, y0, x1, y1), params=ExtractionParams(delta_e=45))
    assert len(res) > 0.9 * (x1 - x0 - 8)
    assert np.median(_errors(res, xs, f)) < 0.3


def test_jpeg_compression_keeps_accuracy():
    img, (x0, y0, x1, y1) = make_plot()
    xs = np.linspace(x0 + 4, x1 - 4, 900)
    f = lambda x: 180 + 0.2 * (x - x0) + 20 * np.sin((x - x0) / 50)
    draw_curve(img, xs, f(xs), bgr(MAGENTA), 3)
    img = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 60])[1], 1)
    ref = pick_reference_color(img, 320, f(320))
    res = extract_curve(img, ref, Rect(x0, y0, x1, y1), params=ExtractionParams(delta_e=35))
    assert len(res) > 0.9 * (x1 - x0 - 8)
    assert np.median(_errors(res, xs, f)) < 0.3


def test_unmatched_colour_returns_empty_result():
    img, (x0, y0, x1, y1) = make_plot()
    res = extract_curve(img, RED, Rect(x0, y0, x1, y1))
    assert len(res) == 0
    assert res.stats["reason"] in {"no_match", "no_curve"}


def test_tiny_or_degenerate_area_does_not_crash():
    img, _ = make_plot()
    assert len(extract_curve(img, RED, Rect(10, 10, 10, 10))) == 0
    assert len(extract_curve(img, RED, Rect(-50, -50, 5, 5))) == 0


def test_keep_mask_returns_full_size_mask():
    img, (x0, y0, x1, y1) = make_plot()
    xs = np.linspace(x0 + 4, x1 - 4, 600)
    draw_curve(img, xs, 200 - 0.1 * (xs - x0), bgr(RED), 3)
    res = extract_curve(img, RED, Rect(x0, y0, x1, y1), keep_mask=True)
    assert res.mask.shape == img.shape[:2] and res.mask.dtype == bool and res.mask.any()


# --------------------------------------------------------------------------- models
def test_rect_pixel_slices_use_pixel_centres():
    r = Rect(2.5, 1.5, 6.5, 4.5)
    rs, cs = r.pixel_slices((10, 10))
    assert (cs.start, cs.stop) == (2, 6) and (rs.start, rs.stop) == (1, 4)
    assert Rect.from_points(5, 5, 1, 2) == Rect(1, 2, 5, 5)
    assert Rect(-5, -5, 3, 3).pixel_slices((10, 10))[0].start == 0


def test_curve_edit_operations():
    c = Curve("0.56A", RED)
    c.set_points([3, 1, 2], [30, 10, 20])
    assert list(c.x_px) == [1, 2, 3]
    c.add_point(2.5, 99)
    assert list(c.x_px) == [1, 2, 2.5, 3] and list(c.y_px) == [10, 20, 99, 30]
    assert c.remove_in_rect(Rect(2.2, 90, 2.8, 100)) == 1
    assert len(c) == 3
    assert c.remove_in_rect(Rect(100, 100, 200, 200)) == 0


def test_params_dict_roundtrip_ignores_unknown_keys():
    p = ExtractionParams(delta_e=33, remove_grid=False)
    d = p.to_dict()
    d["future_option"] = 1
    assert ExtractionParams.from_dict(d) == p


# --------------------------------------------------------------------------- seeds + dashed curves
def _two_red_curves(dashed_second=False):
    img, (x0, y0, x1, y1) = make_plot(grid=False)
    xs = np.linspace(x0 + 4, x1 - 4, 1200)
    f1 = lambda x: 130 + 0.15 * (x - x0)
    f2 = lambda x: 300 - 0.10 * (x - x0)
    draw_curve(img, xs, f1(xs), bgr(RED), 3)
    if dashed_second:
        for a in range(0, len(xs) - 14, 26):                       # 13-sample dash, 13-sample gap (~5 px each)
            draw_curve(img, xs[a:a + 13], f2(xs[a:a + 13]), bgr(RED), 3)
    else:
        draw_curve(img, xs, f2(xs), bgr(RED), 3)
    return img, Rect(x0, y0, x1, y1), xs, f1, f2


def test_seed_selects_which_of_two_same_coloured_curves_is_followed():
    img, area, xs, f1, f2 = _two_red_curves()
    x = 300.5
    on1 = extract_curve(img, RED, area, seed=(x, f1(x)))
    on2 = extract_curve(img, RED, area, seed=(x, f2(x)))
    assert _max_error(on1, xs, f1) < 0.4 and len(on1) > 500
    assert _max_error(on2, xs, f2) < 0.4 and len(on2) > 500
    assert np.all(np.abs(on1.y_px - f2(on1.x_px)) > 20)          # really two different traces


def test_click_slightly_off_the_curve_still_picks_the_nearest_one():
    img, area, xs, f1, f2 = _two_red_curves()
    x = 300.5
    res = extract_curve(img, RED, area, seed=(x + 2.0, f2(x) - 4.0))       # 4 px above the line, 2 px to the right
    assert _max_error(res, xs, f2) < 0.4


def test_seed_that_hits_nothing_falls_back_to_the_automatic_choice():
    img, area, xs, f1, f2 = _two_red_curves()
    auto = extract_curve(img, RED, area)
    far = extract_curve(img, RED, area, seed=(300.5, 60.0))                # empty white area, 70 px from any curve
    assert len(far) == len(auto) and np.allclose(far.y_px, auto.y_px)


def test_dashed_curve_is_followed_across_its_gaps_in_dashed_mode():
    img, area, xs, f1, f2 = _two_red_curves(dashed_second=True)
    x = 300.5
    # A dash under the seed: search a nearby column that is inside a dash.
    seed_x = next(px + 0.5 for px in range(300, 340) if img[int(f2(px + 0.5)), px, 2] > 200 and img[int(f2(px + 0.5)), px, 1] < 80)
    res = extract_curve(img, RED, area, params=ExtractionParams(dashed=True), seed=(seed_x, f2(seed_x)))
    assert _max_error(res, xs, f2) < 0.5
    assert len(res) > 0.5 * (xs[-1] - xs[0])                                   # about half the columns are dashes
    assert res.x_px.max() - res.x_px.min() > 0.9 * (xs[-1] - xs[0])          # ... but the whole extent is covered


def test_without_dashed_mode_short_dashes_are_dropped_next_to_a_solid_curve_of_the_same_colour():
    """Documents why the option exists: the dashes count as text-like specks and the solid curve wins."""
    img, area, xs, f1, f2 = _two_red_curves(dashed_second=True)
    seed_x = next(px + 0.5 for px in range(300, 340) if img[int(f2(px + 0.5)), px, 2] > 200 and img[int(f2(px + 0.5)), px, 1] < 80)
    res = extract_curve(img, RED, area, params=ExtractionParams(dashed=False), seed=(seed_x, f2(seed_x)))
    assert _max_error(res, xs, f1) < 0.4                                         # fell back to the solid curve


def test_dashed_curve_alone_needs_no_special_mode():
    img, area, xs, f1, f2 = _two_red_curves(dashed_second=True)
    img2, _ = make_plot(grid=False)
    for a in range(0, len(xs) - 14, 26):
        draw_curve(img2, xs[a:a + 13], f2(xs[a:a + 13]), bgr(RED), 3)
    res = extract_curve(img2, RED, area)
    assert _max_error(res, xs, f2) < 0.5 and res.x_px.max() - res.x_px.min() > 0.9 * (xs[-1] - xs[0])
