"""Exact curve data of the charts of a vector PDF, read straight from the drawing commands.

Every curve of a vector chart is stored as coordinates (polylines, short dash pieces or small
marker glyphs).  Reading those coordinates gives the plotted data with no pixel error at all, sees
curves that are hidden under other curves, and cannot mix up a solid and a dashed curve of the same
colour.  The pipeline per chart (frame + axes come from :mod:`core.pdf_source`):

1. take every path drawn inside the plot frame and drop what is not data: frame edges, dashed or
   solid grid lines, tick marks and the legend swatches;
2. join pieces that touch into solid curves, collect the remaining short dashes of one colour into a
   dashed curve, and take the centres of small closed glyphs of one colour as a marker series;
3. read the legend (swatch + text to its right) and the colour of the axis labels to say what every
   curve means and which Y axis its values belong to;
4. convert page coordinates to data values with a least-squares fit through *all* tick labels.

Coordinates are PDF points measured from the top-left corner of the page, as in ``pdf_source``.
"""
from __future__ import annotations

import ctypes
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pypdfium2 as pdfium
import pypdfium2.raw as raw
from scipy.spatial import cKDTree

from .curve_edit import CurveEditor
from .models import Rect
from .pdf_source import AxisHint, ChartInfo, PdfSource
from .postprocess import CurveData

# --------------------------------------------------------------------------- #
# Page objects
# --------------------------------------------------------------------------- #
RGB = tuple[int, int, int]


@dataclass
class VPath:
    pts: np.ndarray                  # (N, 2) page points, y down
    closed: bool
    stroke: RGB | None               # None when the path is not stroked (or fully transparent)
    fill: RGB | None
    width: float                     # stroke width in points
    dashed: bool = False             # the PDF strokes it with a dash pattern

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (float(self.pts[:, 0].min()), float(self.pts[:, 1].min()),
                float(self.pts[:, 0].max()), float(self.pts[:, 1].max()))


@dataclass
class VText:
    text: str
    box: Rect
    color: RGB

    @property
    def cx(self) -> float:
        return 0.5 * (self.box.x0 + self.box.x1)

    @property
    def cy(self) -> float:
        return 0.5 * (self.box.y0 + self.box.y1)


def _color(getter, obj) -> tuple[RGB, int]:
    r, g, b, a = (ctypes.c_uint() for _ in range(4))
    ok = getter(obj.raw, r, g, b, a)
    return ((r.value, g.value, b.value), a.value) if ok else ((0, 0, 0), 0)


def _compose(inner: tuple, outer: tuple) -> tuple:
    """Matrix that applies ``inner`` first and then ``outer`` (PDF ``(a b c d e f)`` convention)."""
    a1, b1, c1, d1, e1, f1 = inner
    a2, b2, c2, d2, e2, f2 = outer
    return (a2 * a1 + c2 * b1, b2 * a1 + d2 * b1, a2 * c1 + c2 * d1, b2 * c1 + d2 * d1,
            a2 * e1 + c2 * f1 + e2, b2 * e1 + d2 * f1 + f2)


def _bezier(p0, p1, p2, p3, n: int = 8) -> list[tuple[float, float]]:
    t = np.linspace(0, 1, n + 1)[1:, None]
    p0, p1, p2, p3 = (np.asarray(p, float) for p in (p0, p1, p2, p3))
    pts = (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3
    return [tuple(p) for p in pts]


def _subpaths(obj) -> list[tuple[list[tuple[float, float]], bool]]:
    """Segments of a path object as ``[(points in object space, closed), ...]`` (one per sub-path)."""
    n = raw.FPDFPath_CountSegments(obj.raw)
    subs: list[tuple[list[tuple[float, float]], bool]] = []
    cur: list[tuple[float, float]] = []
    closed = False
    i = 0
    while i < n:
        seg = raw.FPDFPath_GetPathSegment(obj.raw, i)
        typ = raw.FPDFPathSegment_GetType(seg)
        x, y = ctypes.c_float(), ctypes.c_float()
        raw.FPDFPathSegment_GetPoint(seg, x, y)
        pt = (x.value, y.value)
        if typ == raw.FPDF_SEGMENT_MOVETO:
            if len(cur) > 1:
                subs.append((cur, closed))
            cur, closed = [pt], False
        elif typ == raw.FPDF_SEGMENT_BEZIERTO and i + 2 < n and cur:
            pts = []
            for k in range(3):
                s2 = raw.FPDFPath_GetPathSegment(obj.raw, i + k)
                x2, y2 = ctypes.c_float(), ctypes.c_float()
                raw.FPDFPathSegment_GetPoint(s2, x2, y2)
                pts.append((x2.value, y2.value))
            cur.extend(_bezier(cur[-1], pts[0], pts[1], pts[2]))
            i += 2
        else:
            cur.append(pt)
        if raw.FPDFPathSegment_GetClose(seg):
            closed = True
        i += 1
    if len(cur) > 1:
        subs.append((cur, closed))
    return subs


class PageVector:
    """All paths and text runs of one page, in page points (y down)."""

    def text_in(self, box: Rect) -> str:
        """Text of the characters inside ``box`` (page points, y down), one line."""
        t = self._tp.get_text_bounded(box.x0 - 0.2, self._ph - box.y1 - 0.2, box.x1 + 0.2, self._ph - box.y0 + 0.2)
        return clean_text(t.replace("\r", " ").replace("\n", " "))

    def __init__(self, page, region: Rect | None = None):
        self.paths: list[VPath] = []
        self.texts: list[VText] = []
        ph = float(page.get_size()[1])
        tp = page.get_textpage()
        self._tp, self._ph = tp, ph
        for obj in page.get_objects(filter=[raw.FPDF_PAGEOBJ_PATH, raw.FPDF_PAGEOBJ_TEXT]):
            l, b, r, t = obj.get_bounds()
            y0, y1 = ph - t, ph - b
            if (region is not None and obj.level == 0 and
                    not (l >= region.x0 - 1 and r <= region.x1 + 1 and y0 >= region.y0 - 1 and y1 <= region.y1 + 1)):
                continue
            if obj.type == raw.FPDF_PAGEOBJ_TEXT:
                (rgb, alpha) = _color(raw.FPDFPageObj_GetFillColor, obj)
                text = tp.get_text_bounded(l - 0.3, b - 0.3, r + 0.3, t + 0.3).replace("\r", "").replace("\n", " ").strip()
                if text:
                    self.texts.append(VText(text, Rect(l, y0, r, y1), rgb))
                continue
            fill = ctypes.c_int()
            stroke = ctypes.c_int()
            raw.FPDFPath_GetDrawMode(obj.raw, fill, stroke)
            s_rgb, s_a = _color(raw.FPDFPageObj_GetStrokeColor, obj)
            f_rgb, f_a = _color(raw.FPDFPageObj_GetFillColor, obj)
            m = obj.get_matrix().get()
            form = obj.container
            while form is not None and hasattr(form, "get_matrix"):        # inside a form XObject
                m = _compose(m, form.get_matrix().get())
                form = getattr(form, "container", None)
            a, b_, c, d, e, f = m
            w = ctypes.c_float()
            raw.FPDFPageObj_GetStrokeWidth(obj.raw, w)
            width = abs(w.value * a)
            dashed = raw.FPDFPageObj_GetDashCount(obj.raw) > 0
            for pts, closed in _subpaths(obj):
                arr = np.asarray(pts, float)
                xs = a * arr[:, 0] + c * arr[:, 1] + e
                ys = ph - (b_ * arr[:, 0] + d * arr[:, 1] + f)
                self.paths.append(VPath(np.column_stack([xs, ys]), closed,
                                        s_rgb if stroke.value and s_a > 0 else None,
                                        f_rgb if fill.value and f_a > 0 else None, width, dashed))


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
STYLE_LABELS = {"solid": "sürekli çizgi", "dashed": "kesikli çizgi", "markers": "işaretçi"}


@dataclass
class AxisFit:
    """Least-squares mapping from a page coordinate (points) to a data value along one axis."""

    orientation: str                 # "x" or "y"
    title: str
    unit: str
    log: bool
    slope: float                     # (log10) value per point
    intercept: float
    rms_pt: float                    # RMS distance of the tick marks from the fit
    color: RGB = (0, 0, 0)
    side: str = ""

    def to_value(self, pos):
        u = self.slope * np.asarray(pos, dtype=float) + self.intercept
        return np.power(10.0, u) if self.log else u

    def to_pos(self, value):
        v = np.asarray(value, dtype=float)
        return ((np.log10(v) if self.log else v) - self.intercept) / self.slope

    def uncertainty(self, pos, pt: float = 0.06):
        """Value uncertainty for a position error of ``pt`` points (the drawing's own rounding)."""
        pos = np.asarray(pos, dtype=float)
        pt = max(pt, 2.0 * self.rms_pt)
        if not self.log:
            return np.full(pos.shape, abs(self.slope) * pt)
        return np.log(10.0) * np.abs(self.to_value(pos)) * abs(self.slope) * pt


def fit_axis(hint: AxisHint, color: RGB = (0, 0, 0)) -> AxisFit | None:
    """Fit ``value = f(position)`` through all tick labels of an axis (linear or log10).

    A log fit may ignore non-positive labels: some producers print the tick 0.1 of a log axis as "0",
    which must not force a linear fit.
    """
    if len(hint.ticks) < 2:
        return None
    pos = np.array([t.pos for t in hint.ticks], dtype=float)
    val = np.array([t.value for t in hint.ticks], dtype=float)
    best = None
    for log in (False, True):
        keep = val > 0 if log else np.ones(len(val), bool)
        if keep.sum() < (3 if log else 2):
            continue
        p, u = pos[keep], (np.log10(val[keep]) if log else val[keep])
        if np.ptp(p) == 0 or np.ptp(u) == 0:
            continue
        slope, icpt = np.polyfit(p, u, 1)
        rms = float(np.sqrt(np.mean(((u - (slope * p + icpt)) / slope) ** 2)))
        if best is None or rms < 0.5 * best[3]:          # prefer linear unless log is clearly better
            best = (log, float(slope), float(icpt), rms)
    if best is None:
        return None
    log, slope, icpt, rms = best
    return AxisFit(hint.orientation, hint.title, hint.unit, log, slope, icpt, rms, color, hint.side)


@dataclass
class CurveResult:
    index: int
    style: str                       # "solid" | "dashed" | "markers"
    color: RGB
    label: str                       # legend text (or a generated name)
    from_legend: bool
    y_axis: int                      # index into ChartData.y_fits
    x_pt: np.ndarray
    y_pt: np.ndarray
    data: CurveData                  # in data units, sorted by x (the *edited* data once the user corrected it)
    axis_note: str = ""
    editor: CurveEditor | None = field(default=None, repr=False)

    # ---- hand corrections (see core.curve_edit); ``data`` always mirrors the editor's current state
    def _ed(self) -> CurveEditor:
        if self.editor is None:
            self.editor = CurveEditor(self.data)
        return self.editor

    def _sync(self) -> None:
        self.data = self.editor.data

    @property
    def edited(self) -> bool:
        return self.editor is not None and self.editor.is_edited

    @property
    def n_changes(self) -> int:
        return self.editor.n_changes if self.edited else 0

    @property
    def can_undo(self) -> bool:
        return self.editor is not None and self.editor.can_undo

    @property
    def can_redo(self) -> bool:
        return self.editor is not None and self.editor.can_redo

    def add_point(self, x: float, y: float, x_unc: float | None = None, y_unc: float | None = None) -> int:
        i = self._ed().add_point(x, y, x_unc, y_unc)
        self._sync()
        return i

    def delete_points(self, indices) -> int:
        n = self._ed().delete(indices)
        self._sync()
        return n

    def delete_in_box(self, x0: float, x1: float, y0: float, y1: float) -> int:
        n = self._ed().delete_in_box(x0, x1, y0, y1)
        self._sync()
        return n

    def move_point(self, index: int, x: float, y: float) -> int:
        i = self._ed().move_point(index, x, y)
        self._sync()
        return i

    def undo(self) -> bool:
        ok = self.editor is not None and self.editor.undo()
        if ok:
            self._sync()
        return ok

    def redo(self) -> bool:
        ok = self.editor is not None and self.editor.redo()
        if ok:
            self._sync()
        return ok

    def reset_edits(self) -> bool:
        ok = self.editor is not None and self.editor.reset()
        if ok:
            self._sync()
        return ok

    @property
    def n_points(self) -> int:
        return len(self.data)

    @property
    def style_label(self) -> str:
        return STYLE_LABELS.get(self.style, self.style)


@dataclass
class ChartData:
    chart: ChartInfo
    x_fit: AxisFit | None
    y_fits: list[AxisFit]
    curves: list[CurveResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    thumbnail: np.ndarray | None = None
    kind: str = "vector"             # "vector" (drawn in the PDF) or "raster" (a picture inside the PDF)
    calibrated: bool = True          # raster charts start uncalibrated: the axes are set by the user
    pixel_pt: float = 0.06           # finest position the source can give, in points (vector rounding / picture pixel)
    raster: object | None = None     # core.raster_charts.RasterChart of a picture chart
    from_image: bool = False         # the chart is an image file of its own, not part of a PDF page

    @property
    def title(self) -> str:
        return self.chart.title or f"Grafik {self.chart.index + 1}"

    def y_fit_of(self, curve: CurveResult) -> AxisFit | None:
        return self.y_fits[curve.y_axis] if 0 <= curve.y_axis < len(self.y_fits) else None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
_DEG_RE = re.compile(r"(?<=\d)\s*[oº˚°]\s*(?=[CF]\b)")
_WHITE = (250, 250, 250)


def clean_text(text: str) -> str:
    """Datasheet artefacts: superscript 'o' typed as a letter, double spaces."""
    text = _DEG_RE.sub("°", text)
    return re.sub(r"\s+", " ", text).strip()


def _chroma(rgb: RGB) -> int:
    return max(rgb) - min(rgb)


def _is_white(rgb: RGB) -> bool:
    return min(rgb) >= _WHITE[0]


def _color_name(rgb: RGB) -> str:
    r, g, b = rgb
    if max(rgb) < 70:
        return "siyah"
    if _chroma(rgb) < 40:
        return "gri"
    names = {"kırmızı": (255, 0, 0), "yeşil": (0, 200, 0), "mavi": (0, 0, 255), "turuncu": (255, 128, 0),
             "macenta": (255, 0, 255), "camgöbeği": (0, 200, 255), "sarı": (230, 200, 0), "mor": (128, 0, 200)}
    return min(names, key=lambda n: sum((a - c) ** 2 for a, c in zip(rgb, names[n])))


def _merge_text_runs(texts: list[VText], text_in) -> list[VText]:
    """Join the text runs of one printed line (e.g. ``23`` + superscript ``o`` + ``C.``) into one line."""
    rows: list[list] = []                                   # [mean cy, height, runs]
    for t in sorted(texts, key=lambda t: t.cy):
        h = max(t.box.height, 1.0)
        if rows and abs(t.cy - rows[-1][0]) <= 0.6 * max(h, rows[-1][1]):
            rows[-1][2].append(t)
            rows[-1][0] = float(np.mean([r.cy for r in rows[-1][2]]))
            rows[-1][1] = max(rows[-1][1], h)
        else:
            rows.append([t.cy, h, [t]])
    lines: list[VText] = []
    for _, h, runs in rows:
        runs.sort(key=lambda t: t.box.x0)
        box, color = runs[0].box, runs[0].color
        for t in runs[1:] + [None]:
            if t is not None and -0.5 <= t.box.x0 - box.x1 <= 0.9 * h:
                box = Rect(box.x0, min(box.y0, t.box.y0), max(box.x1, t.box.x1), max(box.y1, t.box.y1))
                continue
            lines.append(VText(text_in(box), box, color))
            if t is not None:
                box, color = t.box, t.color
    return [ln for ln in lines if ln.text]


def _chains(pieces: list[VPath], eps: float = 0.3) -> list[list[VPath]]:
    """Group pieces whose end points touch (within ``eps`` points) into chains."""
    n = len(pieces)
    if n == 0:
        return []
    ends = np.array([p for pc in pieces for p in (pc.pts[0], pc.pts[-1])])
    owner = np.repeat(np.arange(n), 2)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in cKDTree(ends).query_pairs(eps):
        a, b = find(owner[i]), find(owner[j])
        if a != b:
            parent[a] = b
    groups: dict[int, list[VPath]] = {}
    for i, pc in enumerate(pieces):
        groups.setdefault(find(i), []).append(pc)
    return list(groups.values())


def _chain_points(chain: list[VPath]) -> np.ndarray:
    """One point list ordered by x: pieces oriented left-to-right, then concatenated."""
    oriented = []
    for pc in chain:
        pts = pc.pts if pc.pts[0, 0] <= pc.pts[-1, 0] else pc.pts[::-1]
        oriented.append(pts)
    oriented.sort(key=lambda p: (p[:, 0].min(), p[0, 1]))
    pts = np.vstack(oriented)
    order = np.argsort(pts[:, 0], kind="stable")
    pts = pts[order]
    keep = np.ones(len(pts), bool)
    keep[1:] = np.any(np.abs(np.diff(pts, axis=0)) > 1e-6, axis=1)
    return pts[keep]


# --------------------------------------------------------------------------- #
# Per chart
# --------------------------------------------------------------------------- #
def _text_color_near(box: Rect, texts: list[VText]) -> RGB:
    colours = [t.color for t in texts if box.x0 - 1 <= t.cx <= box.x1 + 1 and box.y0 - 1 <= t.cy <= box.y1 + 1]
    return Counter(colours).most_common(1)[0][0] if colours else (0, 0, 0)


def _grid_and_frame(paths: list[VPath], frame: Rect) -> set[int]:
    """Indices of paths that are frame edges, grid lines or tick marks (neutral straight lines)."""
    drop: set[int] = set()
    groups: dict[tuple[str, int], list[int]] = {}
    for i, p in enumerate(paths):
        if p.stroke is None or len(p.pts) != 2 or _chroma(p.stroke) >= 60:
            continue
        (x0, y0), (x1, y1) = p.pts
        if abs(x1 - x0) < 0.05:
            groups.setdefault(("v", int(round(x0 / 0.3))), []).append(i)
        elif abs(y1 - y0) < 0.05:
            groups.setdefault(("h", int(round(y0 / 0.3))), []).append(i)
    for (kind, _), idx in groups.items():
        pts = np.vstack([paths[i].pts for i in idx])
        span = float(np.ptp(pts[:, 1] if kind == "v" else pts[:, 0]))
        side = frame.height if kind == "v" else frame.width
        if span >= 0.8 * side:
            drop.update(idx)
    # Frame edges, coloured secondary-axis lines and tick marks: straight, axis-aligned lines lying on a
    # frame edge that either span it (axis) or are short (tick).  Any colour: a blue/red axis is not data.
    for i, p in enumerate(paths):
        if i in drop or p.stroke is None or len(p.pts) != 2:
            continue
        (x0, y0), (x1, y1) = p.pts
        vertical, horizontal = abs(x1 - x0) < 0.05, abs(y1 - y0) < 0.05
        if not (vertical or horizontal):
            continue
        length = float(np.hypot(x1 - x0, y1 - y0))
        if vertical:
            on_edge = min(abs(x0 - frame.x0), abs(x0 - frame.x1)) < 0.8
            spans = length >= 0.8 * frame.height
        else:
            on_edge = min(abs(y0 - frame.y0), abs(y0 - frame.y1)) < 0.8
            spans = length >= 0.8 * frame.width
        # ticks that start on an edge and point inwards
        starts_on_edge = any(min(abs(x - frame.x0), abs(x - frame.x1)) < 0.8 or min(abs(y - frame.y0), abs(y - frame.y1)) < 0.8
                             for x, y in p.pts)
        if (on_edge and spans) or (starts_on_edge and length <= 8.0 and _chroma(p.stroke) < 60) or (on_edge and length <= 8.0):
            drop.add(i)
    # Evenly repeated short lines that start on the same frame edge are tick marks whatever their colour
    # (a lone dash of a curve that happens to start on the axis is not repeated and stays).
    buckets: dict[tuple[str, int], list[int]] = {}
    for i, p in enumerate(paths):
        if i in drop or p.stroke is None or len(p.pts) != 2:
            continue
        (x0, y0), (x1, y1) = p.pts
        if not (abs(x1 - x0) < 0.05 or abs(y1 - y0) < 0.05):
            continue
        length = float(np.hypot(x1 - x0, y1 - y0))
        if length > 8.0:
            continue
        edge = None
        for x, y in p.pts:
            for name, dist in (("L", abs(x - frame.x0)), ("R", abs(x - frame.x1)),
                               ("T", abs(y - frame.y0)), ("B", abs(y - frame.y1))):
                if dist < 0.8:
                    edge = name
        if edge:
            buckets.setdefault((edge, int(round(length / 0.3))), []).append(i)
    for idx in buckets.values():
        if len(idx) >= 3:
            drop.update(idx)
    return drop


def _clip_to_frame(p: VPath, frame: Rect, pad: float) -> VPath | None:
    """The part of a path inside the frame (curves may run on outside it, clipped by the viewer)."""
    x0, y0, x1, y1 = p.bbox
    if x1 < frame.x0 - pad or x0 > frame.x1 + pad or y1 < frame.y0 - pad or y0 > frame.y1 + pad:
        return None
    ins = ((p.pts[:, 0] >= frame.x0 - pad) & (p.pts[:, 0] <= frame.x1 + pad)
           & (p.pts[:, 1] >= frame.y0 - pad) & (p.pts[:, 1] <= frame.y1 + pad))
    if ins.all():
        return p
    small = max(x1 - x0, y1 - y0) <= 8.0
    if small:                                              # a marker: keep when its centre is inside
        cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        return p if frame.x0 - pad <= cx <= frame.x1 + pad and frame.y0 - pad <= cy <= frame.y1 + pad else None
    if ins.sum() < 2:
        return None
    return VPath(p.pts[ins], False, p.stroke, p.fill, p.width, p.dashed)


def extract_chart(pv: PageVector, chart: ChartInfo) -> ChartData:
    """Curves, legend meanings, axis fits and notes of one chart."""
    frame = chart.plot_rect
    inside_pad = 1.0

    def inside(box):
        return (box[0] >= frame.x0 - inside_pad and box[2] <= frame.x1 + inside_pad
                and box[1] >= frame.y0 - inside_pad and box[3] <= frame.y1 + inside_pad)

    texts_all = pv.texts
    # axis fits (colour of an axis = colour of its labels)
    x_fit = None
    if chart.x_axes:
        x_fit = fit_axis(chart.x_axes[0], _text_color_near(chart.x_axes[0].box, texts_all) if chart.x_axes[0].box else (0, 0, 0))
    y_fits = []
    for h in chart.y_axes:
        f = fit_axis(h, _text_color_near(h.box, texts_all) if h.box else (0, 0, 0))
        if f is not None:
            y_fits.append(f)
    result = ChartData(chart, x_fit, y_fits)

    paths = []
    for p in pv.paths:
        if p.stroke is None and p.fill is None:
            continue
        clipped = _clip_to_frame(p, frame, inside_pad)
        if clipped is not None:
            paths.append(clipped)
    paths = [p for p in paths if not ((p.stroke is None or _is_white(p.stroke)) and (p.fill is None or _is_white(p.fill)))]
    drop = _grid_and_frame(paths, frame)
    paths = [p for i, p in enumerate(paths) if i not in drop]

    # --- text inside the frame: legend labels and notes ---------------------------------
    lines = _merge_text_runs([t for t in texts_all if frame.x0 < t.cx < frame.x1 and frame.y0 < t.cy < frame.y1],
                             pv.text_in)

    # --- classify paths: markers (small closed glyphs) vs strokes -------------------------
    def glyph_color(p: VPath) -> RGB | None:
        return p.fill or p.stroke

    marker_paths, stroke_paths = [], []
    for p in paths:
        x0, y0, x1, y1 = p.bbox
        small_closed = (p.closed or np.allclose(p.pts[0], p.pts[-1], atol=1e-3)) and len(p.pts) >= 4 \
            and max(x1 - x0, y1 - y0) <= 8.0 and min(x1 - x0, y1 - y0) >= 0.6
        if small_closed and glyph_color(p) is not None and not _is_white(glyph_color(p)):
            marker_paths.append(p)
        elif p.stroke is not None and not (p.closed and p.fill is not None and _is_white(p.fill)):
            stroke_paths.append(p)

    # --- legend: swatch (pieces / glyph) directly left of a text line ---------------------
    legend: list[tuple[str, RGB, str]] = []          # (text, colour, style)
    used: set[int] = set()
    note_lines: list[VText] = []
    # A closed outline around text is the legend box (or a caption frame), not data.
    def boxes_text(p: VPath) -> bool:
        x0, y0, x1, y1 = p.bbox
        return (p.closed or np.allclose(p.pts[0], p.pts[-1], atol=1e-3)) and max(x1 - x0, y1 - y0) > 8.0 and any(
            x0 <= ln.cx <= x1 and y0 <= ln.cy <= y1 for ln in lines)
    legend_boxes = [p.bbox for p in stroke_paths if boxes_text(p)]
    stroke_paths = [p for p in stroke_paths if not boxes_text(p)]
    all_pieces = [(i, p) for i, p in enumerate(stroke_paths) if len(p.pts) <= 40]
    marker_pieces = list(enumerate(marker_paths))
    for line in lines:
        h = max(line.box.height, 2.0)
        row_s = sorted(((i, p) for i, p in all_pieces
                        if p.bbox[2] - p.bbox[0] <= 45.0 and abs(0.5 * (p.bbox[1] + p.bbox[3]) - line.cy) <= 0.9 * h
                        and p.bbox[2] <= line.box.x0 + 1.0), key=lambda ip: -ip[1].bbox[2])
        near_s = []
        if row_s and line.box.x0 - row_s[0][1].bbox[2] <= 10.0:
            left = row_s[0][1].bbox[0]
            near_s = [row_s[0]]
            for i, p in row_s[1:]:
                if left - p.bbox[2] <= 6.0:                # a dash gap, not the end of the swatch
                    near_s.append((i, p))
                    left = min(left, p.bbox[0])
                else:
                    break
        near_m = []
        for i, p in marker_pieces:
            x0, y0, x1, y1 = p.bbox
            gap = line.box.x0 - x1
            cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
            in_box = any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in legend_boxes)
            # In a legend box the marker sits in the middle of a ~20 pt handle, i.e. further from the text.
            if abs(cy - line.cy) <= 0.9 * h and (-1.0 <= gap <= 10.0 or (in_box and 0.0 <= gap <= 40.0)):
                near_m.append((i, p))
        if near_s and not near_m:
            colour = Counter(p.stroke for _, p in near_s).most_common(1)[0][0]
            same = [(i, p) for i, p in near_s if p.stroke == colour]
            legend.append((clean_text(line.text), colour,
                           "dashed" if len(same) >= 2 or any(p.dashed for _, p in same) else "solid"))
            used.update(("s", i) for i, _ in near_s)
        elif near_m:
            colour = glyph_color(near_m[0][1])
            legend.append((clean_text(line.text), colour, "markers"))
            used.update(("m", i) for i, _ in near_m)
        else:
            note_lines.append(line)
    stroke_paths = [p for i, p in enumerate(stroke_paths) if ("s", i) not in used]
    marker_paths = [p for i, p in enumerate(marker_paths) if ("m", i) not in used]

    # --- notes: consecutive lines of free text -------------------------------------------
    result.notes = [clean_text(t.text) for t in sorted(note_lines, key=lambda t: (t.box.y0, t.box.x0))
                    if len(clean_text(t.text)) > 2]

    # --- curves ---------------------------------------------------------------------------
    raw_curves: list[tuple[str, RGB, np.ndarray]] = []
    by_color: dict[RGB, list[VPath]] = {}
    for p in stroke_paths:
        by_color.setdefault(p.stroke, []).append(p)
    long_min = 0.08 * frame.width
    for colour, pieces in by_color.items():
        dashes: list[VPath] = []
        for chain in _chains(pieces):
            pts = _chain_points(chain)
            extent = float(np.ptp(pts[:, 0]))
            if len(pts) >= 6 or extent >= long_min or (len(pts) >= 2 and extent >= long_min and chain[0].dashed):
                raw_curves.append(("dashed" if all(pc.dashed for pc in chain) else "solid", colour, pts))
            else:
                dashes.extend(chain)
        if len(dashes) >= 5:
            raw_curves.append(("dashed", colour, _chain_points(dashes)))
    mk_by_color: dict[RGB, dict[tuple[int, int], tuple[float, float]]] = {}
    for p in marker_paths:
        x0, y0, x1, y1 = p.bbox
        cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        mk_by_color.setdefault(glyph_color(p), {})[(round(cx / 0.05), round(cy / 0.05))] = (cx, cy)
    for colour, centres in mk_by_color.items():
        if len(centres) >= 5:
            pts = np.array(sorted(centres.values()))
            raw_curves.append(("markers", colour, pts))
    raw_curves.sort(key=lambda c: (c[2][0, 0], c[2][:, 1].mean()))

    # --- meaning of every curve ---------------------------------------------------------
    axis_colors = [f.color for f in y_fits]
    for k, (style, colour, pts) in enumerate(raw_curves):
        yi = 0
        if len(y_fits) > 1:
            same = [i for i, c in enumerate(axis_colors) if c == colour]
            yi = same[0] if same else 0
        label, from_legend = "", False
        exact = [t for t, c, s in legend if c == colour and s == style]
        by_style = [t for t, c, s in legend if s == style]
        if exact:
            label, from_legend = exact[0], True
        elif len(set(by_style)) == 1:
            label, from_legend = by_style[0], True             # e.g. "5.6A charge" applies to every dashed curve
        if not label and len(y_fits) > 1 and colour in axis_colors and _chroma(colour) >= 60:
            label = y_fits[yi].title or ""                 # a coloured axis names its curve
        if not label:
            label = f"Eğri {k + 1} ({_color_name(colour)})"
        xf = x_fit
        yf = y_fits[yi] if y_fits else None
        if xf is None or yf is None:
            continue
        xs, ys = pts[:, 0], pts[:, 1]
        data = CurveData(label, xf.to_value(xs), yf.to_value(ys), xf.uncertainty(xs), yf.uncertainty(ys))
        order = np.argsort(data.x, kind="stable")
        data = CurveData(label, data.x[order], data.y[order], data.x_unc[order], data.y_unc[order])
        result.curves.append(CurveResult(len(result.curves), style, colour, label, from_legend, yi, xs, ys, data))
    return result


# --------------------------------------------------------------------------- #
# Whole document
# --------------------------------------------------------------------------- #
def analyze_pdf(path, progress=None, thumbnail_dpi: float = 110.0) -> list[ChartData]:
    """Find every chart of every page and read its curves.  ``progress(done, total, text)`` is optional."""
    from .raster_charts import build_chart, find_image_charts       # local import: raster_charts builds on this module

    charts: list[ChartData] = []
    with PdfSource(path) as src:
        pdf = src._pdf
        n_pages = len(pdf)
        for pno in range(n_pages):
            if progress:
                progress(pno, n_pages, f"Sayfa {pno + 1}/{n_pages} taranıyor…")
            page_charts: list[ChartData] = []
            found = src.charts(pno)
            if found:
                pv = PageVector(pdf[pno])
                for ch in found:
                    data = extract_chart(pv, ch)
                    data.thumbnail = src.render(pno, ch.region, dpi=thumbnail_dpi).image
                    page_charts.append(data)
            # pictures of charts (PNG/JPG pasted into the PDF): no vector data, read through the image path
            pictures = find_image_charts(src, pno, skip=[c.chart.region for c in page_charts])
            if pictures:
                page_charts.extend(build_chart(rc) for rc in pictures)
                page_charts.sort(key=lambda c: (round(c.chart.region.y0 / 20.0), c.chart.region.x0))
                for i, c in enumerate(page_charts):
                    c.chart.index = i
            charts.extend(page_charts)
        if progress:
            progress(n_pages, n_pages, "Tamamlandı")
    return charts
