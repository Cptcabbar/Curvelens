"""Charts that are *pictures* inside a PDF (datasheets that paste a PNG/JPG of the chart).

Such a chart has no drawing commands to read, so it goes through the image path instead:

1. ``find_image_charts``: every embedded picture of a page that shows a plot frame is a chart; the heading
   above it (PDF text) names it.
2. ``detect_ticks``: tick marks and grid lines along the frame, so a click on an axis mark snaps onto it.
3. The user gives two marks per axis and their values (``calibrate``); nothing is guessed about the axes
   because there is no text to read (no OCR).  A second Y axis on the right is optional.
4. ``discover_curves``: coloured curves are found from the colour clusters inside the frame and traced
   with :mod:`core.extraction`; ``trace_curve_at`` adds a curve (e.g. a black one) from a click.

Everything downstream (lookup tables, value queries, hand corrections) is shared with vector charts:
the result is the same :class:`core.vector_charts.ChartData` / ``CurveResult``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
import pypdfium2.raw as raw

from .extraction import bgr_to_lab, extract_curve
from .models import ExtractionParams, Rect
from .pdf_source import ChartInfo, PdfSource, RenderedRegion
from .plotarea import detect_plot_area
from .postprocess import CurveData
from .vector_charts import AxisFit, ChartData, CurveResult

MIN_IMAGE_PX = (250, 150)          # smaller pictures (logos, icons) are not charts
MIN_IMAGE_PT = (90.0, 55.0)
MIN_CURVE_SPAN = 0.30              # a curve must cover this share of the frame width
MIN_EXTRA_SPAN = 0.55              # ... and a further track of the same colour this much
MIN_COVERAGE = 0.60                # share of its columns in which the trace actually found the curve
TICK_SNAP_PX = 6.0                 # a click this close to a tick mark snaps onto it
MAX_TRACKS_PER_COLOUR = 3          # e.g. a voltage and a temperature curve drawn in the same colour


# --------------------------------------------------------------------------- #
# Containers
# --------------------------------------------------------------------------- #
@dataclass
class Ticks:
    """Positions (continuous image pixels) of tick marks / grid lines along the frame."""

    x: list[float] = field(default_factory=list)            # along the bottom edge
    y_left: list[float] = field(default_factory=list)       # along the left edge
    y_right: list[float] = field(default_factory=list)      # along the right edge


@dataclass
class RasterChart:
    page_index: int
    index: int
    title: str
    notes: list[str]
    bbox: Rect                       # the picture on the page, points (y down)
    image: np.ndarray                # BGR, exactly as embedded in the PDF
    plot: Rect                       # frame centre-lines in image pixels
    ticks: Ticks
    traces: list = field(default_factory=list)      # RasterTrace of every curve read from the picture
    frame_found: bool = True         # False: no frame was recognised, the plot area is a guess the user should correct
    plot_manual: bool = False        # the user drew the plot area (it is remembered with the calibration)

    @property
    def rendered(self) -> RenderedRegion:
        """Page-point <-> picture-pixel mapping of the embedded picture itself (no resampling)."""
        h, w = self.image.shape[:2]
        sx, sy = w / self.bbox.width, h / self.bbox.height
        return RenderedRegion(self.image, sx, sy, self.bbox.x0 * sx, self.bbox.y0 * sy, 72.0 * sx)

    @property
    def pixel_pt(self) -> float:
        h, w = self.image.shape[:2]
        return max(self.bbox.width / w, self.bbox.height / h)

    @property
    def inner(self) -> Rect:
        """The plot interior without the frame strokes (a black curve would otherwise merge with the frame)."""
        return inner_area(self.image, self.plot)

    @property
    def inner_colour(self) -> Rect:
        """Interior for coloured curves: they may run right up to the axes, so keep more of the border."""
        return inner_area(self.image, self.plot, extra=0.5)


@dataclass
class RasterTrace:
    """One curve as read from the picture (pixels), before it is converted with the axes."""

    rgb: tuple[int, int, int]
    x_px: np.ndarray
    y_px: np.ndarray
    dashed: bool = False
    label: str = ""
    axis: int = 0                    # 0 = left Y axis, 1 = right Y axis
    thick: float = 3.0               # stroke thickness in pixels (median height of the mask run)


@dataclass
class AxisPoints:
    """Two marks of one axis: picture pixel position and the value written there."""

    px: tuple[float, float]
    value: tuple[float, float]
    name: str = ""
    log: bool = False


# --------------------------------------------------------------------------- #
# Images inside the PDF
# --------------------------------------------------------------------------- #
def _to_bgr(arr: np.ndarray) -> np.ndarray:
    """Bitmap of an embedded picture -> BGR uint8 (grey and alpha pictures included)."""
    a = np.asarray(arr)
    if a.ndim == 2:
        return cv2.cvtColor(np.ascontiguousarray(a), cv2.COLOR_GRAY2BGR)
    if a.shape[2] == 4:
        alpha = a[..., 3:4].astype(np.float32) / 255.0
        a = (a[..., :3].astype(np.float32) * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    return np.ascontiguousarray(a[..., :3])


def _upright(img: np.ndarray, matrix: tuple[float, ...]) -> np.ndarray:
    """Undo a mirrored placement: a PDF may store a picture upside down (negative ``d``) or back to front (negative ``a``)
    and let the placement matrix turn it right again; the raw bitmap is what we get, so flip it the same way."""
    a, b, c, d = (float(v) for v in matrix[:4])
    if abs(b) > 1e-6 * max(abs(a), abs(d), 1e-9) or abs(c) > 1e-6 * max(abs(a), abs(d), 1e-9):
        return img                                          # rotated placement: left as stored
    if a < 0:
        img = img[:, ::-1]
    if d < 0:
        img = img[::-1]
    return np.ascontiguousarray(img)


def inner_area(image: np.ndarray, plot: Rect, extra: float = 4.0) -> Rect:
    """Frame centre-lines shrunk past the (thick) frame strokes, plus ``extra`` pixels."""
    g = extra + 0.5 * _stroke_px(image, plot)
    return Rect(plot.x0 + g, plot.y0 + g, plot.x1 - g, plot.y1 - g)


def _stroke_px(image: np.ndarray, plot: Rect) -> float:
    """Thickness of the frame stroke (px), measured across the bottom edge."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    col = int(np.clip(0.5 * (plot.x0 + plot.x1), 0, w - 1))
    r = int(np.clip(round(plot.y1), 0, h - 1))
    lo = hi = r
    while lo > 0 and gray[lo - 1, col] < 130 and r - lo < 12:
        lo -= 1
    while hi < h - 1 and gray[hi + 1, col] < 130 and hi - r < 12:
        hi += 1
    return float(max(1, hi - lo + 1))


def _title_and_notes(src: PdfSource, page_index: int, bbox: Rect) -> tuple[str, list[str]]:
    """Heading above a picture (first line of the text block that starts at the picture's left edge)."""
    _, lines = src._text(page_index)
    cand = []
    for ln in lines:
        if ln.vertical or ln.is_number:
            continue
        gap = bbox.y0 - ln.box.y1
        if not (-1.0 <= gap < 70.0):
            continue
        if not (bbox.x0 - 8.0 <= ln.box.x0 <= bbox.x0 + 45.0) or ln.box.x0 > bbox.x1:
            continue
        cand.append(ln)
    cand.sort(key=lambda ln: -ln.box.y1)                   # nearest to the picture first
    block: list = []
    for ln in cand:
        if not block:
            block.append(ln)
            continue
        # the heading block: lines share the left edge of the nearest one and are set closely
        # (a table row above the heading is either indented differently or spaced wider)
        close = (block[-1].cy - ln.cy) <= 1.75 * max(block[-1].box.height, ln.box.height, 1.0)
        aligned = abs(ln.box.x0 - block[0].box.x0) <= 4.0
        if close and aligned:
            block.append(ln)
        else:
            break
    if not block:
        return "", []
    block.sort(key=lambda ln: ln.box.y0)                   # top line first
    return block[0].text, [ln.text for ln in block[1:]]


def find_image_charts(src: PdfSource, page_index: int, skip: list[Rect] | None = None) -> list[RasterChart]:
    """Embedded pictures of the page that show a plot frame; ``skip`` are page regions already taken by vector charts."""
    page = src._pdf[page_index]
    ph = float(page.get_size()[1])
    out: list[RasterChart] = []
    for obj in page.get_objects():
        if obj.type != raw.FPDF_PAGEOBJ_IMAGE:
            continue
        l, b, r, t = obj.get_bounds()
        bbox = Rect(float(l), ph - float(t), float(r), ph - float(b))
        if bbox.width < MIN_IMAGE_PT[0] or bbox.height < MIN_IMAGE_PT[1]:
            continue
        cx, cy = 0.5 * (bbox.x0 + bbox.x1), 0.5 * (bbox.y0 + bbox.y1)
        if any(s.x0 <= cx <= s.x1 and s.y0 <= cy <= s.y1 for s in (skip or [])):
            continue
        try:
            img = _to_bgr(obj.get_bitmap().to_numpy())
            img = _upright(img, obj.get_matrix().get())
        except Exception:                                   # unsupported filter / colour space
            continue
        if img.shape[1] < MIN_IMAGE_PX[0] or img.shape[0] < MIN_IMAGE_PX[1]:
            continue
        plot = detect_plot_area(img)
        if plot is None:
            continue
        title, notes = _title_and_notes(src, page_index, bbox)
        out.append(RasterChart(page_index, 0, title, notes, bbox, img, plot, detect_ticks(img, plot)))
    out.sort(key=lambda c: (round(c.bbox.y0 / 20.0), c.bbox.x0))
    for i, c in enumerate(out):
        c.index = i
    return out


# --------------------------------------------------------------------------- #
# Tick marks and grid lines (for click snapping)
# --------------------------------------------------------------------------- #
def _centres(flags: np.ndarray, max_width: int = 7, gap: int = 1) -> list[float]:
    """Centres (continuous coordinates) of runs of True no wider than ``max_width``."""
    idx = np.flatnonzero(flags)
    if idx.size == 0:
        return []
    runs, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > gap + 1:
            runs.append((start, prev))
            start = i
        prev = i
    runs.append((start, prev))
    return [0.5 * (a + b + 1) for a, b in runs if b - a + 1 <= max_width]


def _merge(values: list[float], tol: float = 3.0) -> list[float]:
    out: list[float] = []
    for v in sorted(values):
        if out and v - out[-1] <= tol:
            out[-1] = 0.5 * (out[-1] + v)
        else:
            out.append(v)
    return out


def _grid_flags(mask: np.ndarray, axis: int) -> np.ndarray:
    """Columns (axis=0) or rows (axis=1) of ``mask`` that look like a grid line: a dashed line (5+ separate
    runs covering a quarter to most of the length) or an almost unbroken one.  A curve is neither."""
    m = mask if axis == 1 else mask.T                      # one line per row
    frac = m.mean(axis=1)
    padded = np.zeros((m.shape[0], m.shape[1] + 2), np.int8)
    padded[:, 1:-1] = m
    runs = (np.diff(padded, axis=1) == 1).sum(axis=1)
    return ((runs >= 5) & (frac >= 0.25) & (frac <= 0.85)) | (frac >= 0.92)


def detect_ticks(image: np.ndarray, plot: Rect) -> Ticks:
    """Tick marks (outside or inside the frame) and dashed/solid grid lines, as pixel positions."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.int16)
    chroma = image.max(axis=2).astype(np.int16) - image.min(axis=2)
    h, w = gray.shape
    dark = gray < 125
    grid = (gray < 185) & (chroma < 40)                    # grey dashed lines are lighter than the frame
    t = int(round(0.5 * _stroke_px(image, plot))) + 1
    x0, x1, y0, y1 = (int(round(v)) for v in (plot.x0, plot.x1, plot.y0, plot.y1))

    def strip_cols(r0: int, r1: int) -> np.ndarray:
        r0, r1 = max(r0, 0), min(r1, h)
        return dark[r0:r1].mean(axis=0) >= 0.7 if r1 > r0 else np.zeros(w, bool)

    def strip_rows(c0: int, c1: int) -> np.ndarray:
        c0, c1 = max(c0, 0), min(c1, w)
        return dark[:, c0:c1].mean(axis=1) >= 0.7 if c1 > c0 else np.zeros(h, bool)

    # only *outside* the frame: inside, the curves themselves would look like tick marks
    xs = _centres(strip_cols(y1 + t + 1, y1 + t + 8))
    ys_l = _centres(strip_rows(x0 - t - 8, x0 - t - 1))
    ys_r = _centres(strip_rows(x1 + t + 1, x1 + t + 8))
    # grid lines run across the whole plot interior: dashed ones are many short runs, solid ones almost full
    inner_rows = slice(max(y0 + t + 2, 0), max(y1 - t - 2, 0))
    inner_cols = slice(max(x0 + t + 2, 0), max(x1 - t - 2, 0))
    if inner_rows.stop > inner_rows.start:
        xs += _centres(_grid_flags(grid[inner_rows, :], axis=0), max_width=4)
    if inner_cols.stop > inner_cols.start:
        ys_l += _centres(_grid_flags(grid[:, inner_cols], axis=1), max_width=4)
    # the frame edges are always offered: an axis very often starts or ends exactly there
    xs = [v for v in _merge(xs + [plot.x0, plot.x1]) if plot.x0 - 2 <= v <= plot.x1 + 2]
    ys_l = [v for v in _merge(ys_l + [plot.y0, plot.y1]) if plot.y0 - 2 <= v <= plot.y1 + 2]
    ys_r = [v for v in _merge(ys_r + [plot.y0, plot.y1]) if plot.y0 - 2 <= v <= plot.y1 + 2]
    return Ticks(xs, ys_l, ys_r)


def snap(value: float, candidates: list[float], tol: float = TICK_SNAP_PX) -> float:
    """``value`` moved onto the nearest candidate within ``tol``; unchanged when none is close."""
    if candidates:
        best = min(candidates, key=lambda c: abs(c - value))
        if abs(best - value) <= tol:
            return float(best)
    return float(value)


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def make_fit(orientation: str, pts: AxisPoints, rendered: RenderedRegion) -> AxisFit:
    """Axis mapping from two marks (picture pixels) with their values; linear or log10."""
    (p1, p2), (v1, v2) = pts.px, pts.value
    if abs(p2 - p1) < 2.0:
        raise ValueError("İki işaret birbirine çok yakın.")
    if v1 == v2:
        raise ValueError("İki işaretin değeri aynı olamaz.")
    if pts.log and (v1 <= 0 or v2 <= 0):
        raise ValueError("Logaritmik eksende değerler pozitif olmalı.")
    if orientation == "x":
        a, b = (float(rendered.px_to_pt(p, 0.0)[0]) for p in (p1, p2))
    else:
        a, b = (float(rendered.px_to_pt(0.0, p)[1]) for p in (p1, p2))
    u1, u2 = (np.log10(v1), np.log10(v2)) if pts.log else (v1, v2)
    slope = (u2 - u1) / (b - a)
    intercept = u1 - slope * a
    name = pts.name.strip() or ("X" if orientation == "x" else "Y")
    return AxisFit(orientation, name, "", pts.log, float(slope), float(intercept), 0.0,
                   side="bottom" if orientation == "x" else "left")


def _to_curve_data(name: str, x_px, y_px, xf: AxisFit, yf: AxisFit, rendered: RenderedRegion, pixel_pt: float) -> CurveData:
    xp, yp = rendered.px_to_pt(np.asarray(x_px, float), np.asarray(y_px, float))
    x, y = xf.to_value(xp), yf.to_value(yp)
    ux = xf.uncertainty(xp, pt=0.5 * pixel_pt)
    uy = yf.uncertainty(yp, pt=0.5 * pixel_pt)
    order = np.argsort(x, kind="stable")
    return CurveData(name, np.asarray(x)[order], np.asarray(y)[order], np.asarray(ux)[order], np.asarray(uy)[order])


def curve_from_trace(index: int, tr: RasterTrace, chart: ChartData, rc: RasterChart) -> CurveResult:
    """Convert a pixel trace with the chart's axes into a :class:`CurveResult`."""
    rend = rc.rendered
    yf = chart.y_fits[min(tr.axis, len(chart.y_fits) - 1)]
    data = _to_curve_data(tr.label, tr.x_px, tr.y_px, chart.x_fit, yf, rend, rc.pixel_pt)
    xp, yp = rend.px_to_pt(np.asarray(tr.x_px, float), np.asarray(tr.y_px, float))
    return CurveResult(index, "dashed" if tr.dashed else "solid", tr.rgb, tr.label, False, min(tr.axis, len(chart.y_fits) - 1),
                       np.asarray(xp), np.asarray(yp), data,
                       axis_note="" if len(chart.y_fits) < 2 else ("sağ eksen" if tr.axis else "sol eksen"))


def set_plot_area(rc: RasterChart, plot: Rect, manual: bool = True) -> None:
    """Use ``plot`` (picture pixels) as the plot area: tick marks are looked up again along its edges."""
    rc.plot = plot
    rc.ticks = detect_ticks(rc.image, plot)
    rc.plot_manual = manual
    rc.frame_found = True


def chart_from_image(path) -> "ChartData":
    """A chart from an image file (PNG, JPG, BMP, TIFF, WebP): the whole file is the picture.

    The frame is searched like in a PDF's picture; when none is recognised the plot area starts as almost the whole
    image (``frame_found`` is False) and the user is asked to outline it.  Picture pixels double as page points
    (1 px = 1 pt), so every later step is the one of a picture chart inside a PDF.
    """
    from pathlib import Path

    from .imageio import load_image

    img = load_image(path)
    h, w = img.shape[:2]
    if h < 60 or w < 60:
        raise ValueError("Görsel çok küçük.")
    plot = detect_plot_area(img)
    found = plot is not None
    if plot is None:
        plot = Rect(0.06 * w, 0.04 * h, 0.96 * w, 0.90 * h)
    rc = RasterChart(0, 0, Path(path).stem, [], Rect(0.0, 0.0, float(w), float(h)), img, plot, detect_ticks(img, plot),
                     frame_found=found)
    cd = build_chart(rc)
    cd.from_image = True
    return cd


def build_chart(rc: RasterChart) -> ChartData:
    """The gallery entry of a picture chart, before its axes are known."""
    b = rc.bbox
    plot_pt = Rect(*(float(v) for v in (*rc.rendered.px_to_pt(rc.plot.x0, rc.plot.y0), *rc.rendered.px_to_pt(rc.plot.x1, rc.plot.y1))))
    info = ChartInfo(rc.page_index, rc.index, rc.title, plot_pt, Rect(b.x0, b.y0, b.x1, b.y1))
    return ChartData(info, None, [], [], list(rc.notes), rc.image, kind="raster", calibrated=False,
                     pixel_pt=rc.pixel_pt, raster=rc)


def calibrate(chart: ChartData, x: AxisPoints, y: AxisPoints, y2: AxisPoints | None = None) -> None:
    """Set the axes of a picture chart (and forget the curves read with older axes)."""
    rc: RasterChart = chart.raster
    rend = rc.rendered
    fits = [make_fit("y", y, rend)]
    if y2 is not None:
        f2 = make_fit("y", y2, rend)
        f2.side = "right"
        fits.append(f2)
    chart.x_fit = make_fit("x", x, rend)
    chart.y_fits = fits
    chart.calibrated = True
    rebuild_curves(chart)


def rebuild_curves(chart: ChartData) -> None:
    """(Re)convert every stored pixel trace of the chart into curves with the current axes."""
    rc: RasterChart = chart.raster
    chart.curves = [curve_from_trace(i, tr, chart, rc) for i, tr in enumerate(rc.traces)]


# --------------------------------------------------------------------------- #
# Curves
# --------------------------------------------------------------------------- #
def _side_colour_counts(image: np.ndarray, plot: Rect, rgb, right: bool) -> int:
    """Pixels of colour ``rgb`` just outside the frame's left/right edge (coloured axis titles and labels)."""
    h, w = image.shape[:2]
    if right:
        c0, c1 = int(plot.x1 + 6), w
    else:
        c0, c1 = 0, int(plot.x0 - 6)
    r0, r1 = int(max(plot.y0 - 20, 0)), int(min(plot.y1 + 20, h))
    if c1 <= c0 or r1 <= r0:
        return 0
    lab = bgr_to_lab(image[r0:r1, c0:c1])
    ref = bgr_to_lab(np.array([[[rgb[2], rgb[1], rgb[0]]]], np.uint8))[0, 0]
    return int((np.linalg.norm(lab - ref, axis=2) <= 25.0).sum())


def guess_axis(rc: RasterChart, rgb, has_right_axis: bool) -> int:
    """0 = left, 1 = right: a curve whose colour is used by the right axis' title/labels belongs to the right axis."""
    if not has_right_axis:
        return 0
    right = _side_colour_counts(rc.image, rc.plot, rgb, True)
    left = _side_colour_counts(rc.image, rc.plot, rgb, False)
    return 1 if right >= 30 and right > 3 * left else 0


def _colour_clusters(image: np.ndarray, inner: Rect, min_count: int = 200) -> list[tuple[int, int, int]]:
    """Dominant saturated colours inside ``inner`` (RGB), most frequent first."""
    rs, cs = inner.pixel_slices(image.shape)
    crop = image[rs, cs]
    if crop.size == 0:
        return []
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    sel = (hsv[..., 1] > 55) & (hsv[..., 2] > 90)          # pale blues (S ~ 0.3) count as curves too
    if int(sel.sum()) < min_count:
        return []
    bgr = crop[sel]
    lab = bgr_to_lab(bgr.reshape(-1, 1, 3))[:, 0, :]
    ab = lab[:, 1:3].copy()
    keep = np.hypot(ab[:, 0], ab[:, 1]) >= 22.0
    bgr, ab = bgr[keep], ab[keep]
    if len(ab) < min_count:
        return []
    alive = np.ones(len(ab), bool)
    found: list[tuple[int, int, int]] = []
    cell = np.floor(ab / 6.0).astype(int)
    while alive.sum() >= min_count and len(found) < 14:
        keys, inv, counts = np.unique(cell[alive], axis=0, return_inverse=True, return_counts=True)
        top = int(np.argmax(counts))
        if counts[top] < 0.25 * min_count:
            break
        centre = ab[alive][inv == top].mean(axis=0)
        near = np.hypot(ab[:, 0] - centre[0], ab[:, 1] - centre[1]) <= 16.0
        members = near & alive
        alive &= ~near
        if members.sum() < min_count:
            continue
        b, g, r = (int(round(v)) for v in bgr[members].mean(axis=0))
        found.append((r, g, b))
    return found


def _overlaps(x1, y1, x2, y2, tol: float = 3.0) -> float:
    """Share of trace 1 that lies within ``tol`` px (vertically) of trace 2 at the same x."""
    if len(x1) == 0 or len(x2) == 0:
        return 0.0
    order = np.argsort(x2)
    y = np.interp(x1, np.asarray(x2)[order], np.asarray(y2)[order], left=np.nan, right=np.nan)
    ok = np.abs(y - np.asarray(y1)) <= tol
    return float(np.mean(ok))


def _make_trace(rgb, res, dashed: bool = False) -> RasterTrace:
    thick = float(np.median(res.thickness_px)) if len(res) else 3.0
    return RasterTrace(tuple(int(v) for v in rgb), np.asarray(res.x_px), np.asarray(res.y_px), dashed, thick=thick)


def _same_line(a: RasterTrace, b: RasterTrace) -> float:
    """Share of ``a`` that runs within the stroke width of ``b`` (an anti-aliasing / JPEG halo lies on its curve)."""
    return _overlaps(a.x_px, a.y_px, b.x_px, b.y_px, tol=0.5 * max(a.thick, b.thick) + 3.0)


def _prune(traces: list[RasterTrace]) -> list[RasterTrace]:
    """Drop traces that lie on another one: the shorter of two traces that follow each other is an edge halo."""
    keep = list(traces)
    for a in sorted(traces, key=lambda t: len(t.x_px)):          # shortest first
        for b in keep:
            if b is not a and len(b.x_px) >= len(a.x_px) and _same_line(a, b) >= 0.7:
                keep.remove(a)
                break
    return keep


def colour_name(rgb) -> str:
    """Turkish colour word from the hue (the palette-distance guess of the vector path mislabels pale blues)."""
    r, g, b = (int(v) for v in rgb)
    h, s, v = cv2.cvtColor(np.array([[[b, g, r]]], np.uint8), cv2.COLOR_BGR2HSV)[0, 0]
    hue, sat, val = float(h) * 2.0, s / 255.0, v / 255.0
    if val < 0.28:
        return "siyah"
    if sat < 0.22:
        return "gri"
    for upper, name in ((14, "kırmızı"), (42, "turuncu"), (68, "sarı"), (165, "yeşil"), (200, "camgöbeği"), (255, "mavi"),
                        (290, "mor"), (338, "macenta")):
        if hue < upper:
            return name
    return "kırmızı"


def _paint_out(image: np.ndarray, res, half: float) -> None:
    """White out the band around a traced curve so the next track of the same colour can be found."""
    xs = np.clip(np.floor(res.x_px).astype(int), 0, image.shape[1] - 1)
    for x, y in zip(xs, res.y_px):
        r0, r1 = int(max(np.floor(y - half), 0)), int(min(np.ceil(y + half) + 1, image.shape[0]))
        image[r0:r1, x] = 255


def discover_curves(rc: RasterChart, params: ExtractionParams | None = None) -> list[RasterTrace]:
    """Coloured curves inside the frame (plus black ones when a long dark trace exists), top curve first.

    One colour can carry several curves (a voltage and a temperature curve in the same colour): after
    a trace is found its pixels are blanked and the same colour is searched again.
    """
    width = rc.inner.width
    params = params or ExtractionParams()
    cands = _colour_clusters(rc.image, rc.inner)
    cands.append((0, 0, 0))                                  # black curve (grid/text are filtered by the tracker)
    traces: list[RasterTrace] = []
    for rgb in cands:
        work = rc.image.copy()
        inner = rc.inner if max(rgb) < 80 else rc.inner_colour
        for k in range(MAX_TRACKS_PER_COLOUR):
            res = extract_curve(work, rgb, inner, (), params)
            if len(res) < 5:
                break
            thick = float(np.median(res.thickness_px)) if len(res) else 3.0
            _paint_out(work, res, 0.5 * thick + 2.5)
            # a second track of a colour is only believed when it runs (almost) the whole width: shorter ones
            # are text, legend samples or the leftovers of a curve that other curves cover
            need = (MIN_CURVE_SPAN if k == 0 else MIN_EXTRA_SPAN) * width
            span = float(res.x_px[-1] - res.x_px[0])
            if span < need or len(res) < MIN_COVERAGE * span:
                continue
            new = _make_trace(rgb, res)
            if any(_same_line(new, t) > 0.7 for t in traces):
                continue
            traces.append(new)
    traces = _prune(traces)
    traces.sort(key=lambda t: float(np.mean(t.y_px)))
    for i, t in enumerate(traces):
        t.label = f"Eğri {i + 1} ({colour_name(t.rgb)})"
    return traces


def colour_at(image: np.ndarray, x: float, y: float, radius: int = 3) -> tuple[int, int, int]:
    """Colour of the curve at a click: the most saturated pixel (or the darkest one for black) near ``(x, y)``."""
    h, w = image.shape[:2]
    j, i = int(np.clip(np.floor(x), 0, w - 1)), int(np.clip(np.floor(y), 0, h - 1))
    win = image[max(i - radius, 0):i + radius + 1, max(j - radius, 0):j + radius + 1].reshape(-1, 3).astype(int)
    chroma = win.max(axis=1) - win.min(axis=1)
    k = int(np.argmax(chroma)) if chroma.max() >= 60 else int(np.argmin(win.sum(axis=1)))
    b, g, r = (int(v) for v in win[k])
    return r, g, b


def trace_curve_at(rc: RasterChart, x: float, y: float, dashed: bool = False,
                   params: ExtractionParams | None = None) -> RasterTrace | None:
    """Read the curve under the click ``(x, y)`` (picture pixels); ``None`` when nothing traceable is there."""
    rgb = colour_at(rc.image, x, y)
    if max(rgb) > 200 or (max(rgb) - min(rgb) < 30 and max(rgb) > 120):
        return None                                          # paper or a grey grid line, not a curve
    p = params or ExtractionParams()
    p = ExtractionParams.from_dict({**p.to_dict(), "dashed": dashed})
    res = extract_curve(rc.image, rgb, rc.inner if max(rgb) < 80 else rc.inner_colour, (), p, seed=(x, y))
    if len(res) < 5:
        return None
    return _make_trace(rgb, res, dashed)
