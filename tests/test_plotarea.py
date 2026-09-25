from pathlib import Path

import cv2
import numpy as np
import pytest

from core.imageio import load_image
from core.models import Rect
from core.plotarea import detect_plot_area
from tests.synth import dashed_line, draw_curve, make_plot

ROOT = Path(__file__).resolve().parents[1]
PNG = ROOT / "samples" / "discharge.png"
PDF = ROOT / "samples" / "INR18650P28A-V1-80093.pdf"


def _rect_close(r: Rect, expected: tuple, tol: float):
    assert (r.x0, r.y0, r.x1, r.y1) == pytest.approx(expected, abs=tol)


def test_synthetic_frame_is_found_along_its_centre_lines():
    img, (x0, y0, x1, y1) = make_plot()                      # 2 px black frame at (50,50)-(590,370)
    r = detect_plot_area(img)
    _rect_close(r, (x0 + 0.5, y0 + 0.5, x1 + 0.5, y1 + 0.5), 1.5)


def test_frame_is_found_with_curves_grid_and_text_inside():
    img, (x0, y0, x1, y1) = make_plot()
    xs = np.linspace(x0 + 4, x1 - 4, 800)
    draw_curve(img, xs, 200 - 0.2 * (xs - x0), (0, 0, 0), 3)            # black curve
    draw_curve(img, xs, np.full_like(xs, 300.0), (0, 0, 0), 3)          # flat black curve (an interior "band")
    cv2.putText(img, "Charge: CC-CV", (x0 + 30, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    _rect_close(detect_plot_area(img), (x0 + 0.5, y0 + 0.5, x1 + 0.5, y1 + 0.5), 1.5)


def test_no_frame_gives_none():
    img = np.full((300, 400, 3), 255, np.uint8)
    xs = np.linspace(10, 390, 300)
    draw_curve(img, xs, 150 + 30 * np.sin(xs / 40), (0, 0, 255), 3)
    assert detect_plot_area(img) is None
    assert detect_plot_area(np.full((10, 10, 3), 255, np.uint8)) is None      # tiny image


def test_only_two_axes_without_full_frame_gives_none():
    img = np.full((300, 400, 3), 255, np.uint8)
    cv2.line(img, (40, 20), (40, 260), (0, 0, 0), 2)                   # y axis
    cv2.line(img, (40, 260), (380, 260), (0, 0, 0), 2)                 # x axis
    assert detect_plot_area(img) is None


@pytest.mark.skipif(not PNG.exists(), reason="samples/discharge.png missing")
def test_sample_png_frame():
    r = detect_plot_area(load_image(PNG))
    _rect_close(r, (167.14, 24.11, 1591.67, 975.15), 1.0)              # frame centres measured on the image


@pytest.mark.skipif(not PDF.exists(), reason="sample PDF missing")
@pytest.mark.parametrize("dpi", [150, 400])
def test_matches_the_pdf_frames_of_all_four_charts(dpi):
    from core.pdf_source import PdfSource

    with PdfSource(PDF) as src:
        for chart in src.charts(0):
            r = src.render(0, chart.region, dpi=dpi)
            expected = r.rect_pt_to_px(chart.plot_rect)
            found = detect_plot_area(r.image)
            assert found is not None, chart.title
            # a heading underline just above a tight crop may add a line; the frame itself must agree
            tol = 2.0 * dpi / 300 + 1.0
            _rect_close(found, (expected.x0, expected.y0, expected.x1, expected.y1), tol)
