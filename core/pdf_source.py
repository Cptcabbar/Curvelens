"""PDF input: rasterise pages/regions and tell the charts on a page apart.

Vector PDFs carry more than pixels: the plot frame, grid lines, chart titles
and tick labels are all addressable.  This module uses them to

* find every chart on a page (frame drawn as four thin edges + nearest title),
* render one chart at any DPI with an exact point<->pixel transform, and
* propose axis calibration points from the tick labels (snapped to the grid
  lines, so the proposal is accurate to a fraction of a pixel).

Everything is *advisory*: the user can always calibrate by hand.  Page
coordinates used here are PDF points measured from the top-left corner of the
page (y grows downwards), like image coordinates.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_raw

from .calibration import AxisCalibration, Calibration
from .models import Rect

_NUM_RE = re.compile(r"^[-+]?\d+(?:[.,]\d+)?$")
_UNIT_RE = re.compile(r"\(([^)]*)\)")


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TickLabel:
    text: str
    value: float
    pos: float            # x (x-axis) or y (y-axis) in page points
    snapped: bool         # True when moved onto a real grid/frame line


@dataclass
class AxisHint:
    orientation: str      # "x" or "y"
    side: str             # "bottom", "left" or "right"
    title: str
    ticks: list[TickLabel] = field(default_factory=list)
    box: Rect | None = None         # union of the tick labels and the title, page points

    @property
    def unit(self) -> str:
        m = _UNIT_RE.search(self.title)
        return m.group(1) if m else ""


@dataclass
class ChartInfo:
    page_index: int
    index: int                      # order on the page (top-to-bottom, left-to-right)
    title: str
    plot_rect: Rect                 # frame edge centre-lines, page points
    region: Rect                    # plot + tick labels + axis titles, page points
    x_axes: list[AxisHint] = field(default_factory=list)
    y_axes: list[AxisHint] = field(default_factory=list)

    @property
    def label(self) -> str:
        name = self.title or f"Grafik {self.index + 1}"
        return f"Sayfa {self.page_index + 1} · {name}"


@dataclass
class RenderedRegion:
    """A raster of a page region plus the exact page-point <-> pixel mapping.

    The mapping is ``px = pt * scale - offset`` per axis (continuous pixel
    coordinates, see ``core.models``); it is exact because the page is drawn
    with an integer-sized target rectangle whose scale we know.
    """

    image: np.ndarray               # BGR uint8
    sx: float                       # pixels per point, horizontal
    sy: float                       # pixels per point, vertical
    off_x: float                    # pixel column of page x = 0 (negative when cropped)
    off_y: float
    dpi: float

    def pt_to_px(self, x_pt, y_pt):
        return np.asarray(x_pt) * self.sx - self.off_x, np.asarray(y_pt) * self.sy - self.off_y

    def px_to_pt(self, x_px, y_px):
        return (np.asarray(x_px) + self.off_x) / self.sx, (np.asarray(y_px) + self.off_y) / self.sy

    def rect_pt_to_px(self, r: Rect) -> Rect:
        x0, y0 = self.pt_to_px(r.x0, r.y0)
        x1, y1 = self.pt_to_px(r.x1, r.y1)
        return Rect(float(x0), float(y0), float(x1), float(y1))

    @property
    def region(self) -> Rect:
        """Page area (points) actually covered by the image."""
        h, w = self.image.shape[:2]
        x0, y0 = self.px_to_pt(0, 0)
        x1, y1 = self.px_to_pt(w, h)
        return Rect(float(x0), float(y0), float(x1), float(y1))


# --------------------------------------------------------------------------- #
# Internal text model (built from pdfium character boxes)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Word:
    text: str
    box: Rect                       # page points, y-down
    vertical: bool                  # text runs bottom-to-top (rotated 90 degrees)

    @property
    def cx(self) -> float:
        return 0.5 * (self.box.x0 + self.box.x1)

    @property
    def cy(self) -> float:
        return 0.5 * (self.box.y0 + self.box.y1)

    @property
    def is_number(self) -> bool:
        return bool(_NUM_RE.match(self.text))

    @property
    def value(self) -> float:
        return float(self.text.replace(",", "."))


def _union(a: Rect, b: Rect) -> Rect:
    return Rect(min(a.x0, b.x0), min(a.y0, b.y0), max(a.x1, b.x1), max(a.y1, b.y1))


def _clean_text(s: str) -> str:
    return "".join(ch for ch in s if ch.isprintable()).strip()


def _snap(pos: float, lines: list[float], tol: float) -> tuple[float, bool]:
    if lines:
        best = min(lines, key=lambda p: abs(p - pos))
        if abs(best - pos) <= tol:
            return best, True
    return pos, False


# --------------------------------------------------------------------------- #
# PDF document wrapper
# --------------------------------------------------------------------------- #
class PdfSource:
    """Read-only view of a PDF document (rasterisation + chart discovery)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._pdf = pdfium.PdfDocument(str(self.path))
        self._words_cache: dict[int, tuple[list[_Word], list[_Word]]] = {}
        self._charts_cache: dict[int, list[ChartInfo]] = {}

    def close(self) -> None:
        self._pdf.close()

    def __enter__(self) -> "PdfSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def page_count(self) -> int:
        return len(self._pdf)

    def page_size(self, page_index: int) -> tuple[float, float]:
        w, h = self._pdf[page_index].get_size()
        return float(w), float(h)

    # -- rendering ------------------------------------------------------------
    def render(self, page_index: int, region: Rect | None = None, dpi: float = 300.0) -> RenderedRegion:
        """Rasterise a page (or a region of it, in page points) as a BGR image.

        The whole page is mapped onto a ``round(page * dpi/72)`` sized rectangle
        and only the requested window of it is kept, so the resulting
        point -> pixel transform is exactly ``pt * (size/page) - offset``.
        """
        page = self._pdf[page_index]
        pw, ph = (float(v) for v in page.get_size())
        if region is None:
            region = Rect(0.0, 0.0, pw, ph)
        region = Rect(max(0.0, region.x0), max(0.0, region.y0), min(pw, region.x1), min(ph, region.y1))
        scale = dpi / 72.0
        size_x = max(1, int(round(pw * scale)))
        size_y = max(1, int(round(ph * scale)))
        sx, sy = size_x / pw, size_y / ph
        off_x, off_y = int(np.floor(region.x0 * sx)), int(np.floor(region.y0 * sy))
        width = int(np.ceil(region.x1 * sx)) - off_x
        height = int(np.ceil(region.y1 * sy)) - off_y
        if width < 1 or height < 1:
            raise ValueError("Boş görüntü bölgesi.")
        bitmap = pdfium.PdfBitmap.new_native(width, height, pdfium_raw.FPDFBitmap_BGR)
        bitmap.fill_rect((255, 255, 255, 255), 0, 0, width, height)
        pdfium_raw.FPDF_RenderPageBitmap(bitmap, page, -off_x, -off_y, size_x, size_y, 0, pdfium_raw.FPDF_ANNOT)
        img = np.ascontiguousarray(bitmap.to_numpy()[..., :3])
        return RenderedRegion(img, sx, sy, float(off_x), float(off_y), dpi)

    # -- text -------------------------------------------------------------------
    def _text(self, page_index: int) -> tuple[list[_Word], list[_Word]]:
        """Return ``(words, lines)`` of a page; ``lines`` join words of one text run."""
        if page_index in self._words_cache:
            return self._words_cache[page_index]
        page = self._pdf[page_index]
        ph = page.get_size()[1]
        tp = page.get_textpage()
        chars: list[tuple[str, tuple[float, float, float, float]]] = []
        for i in range(tp.count_chars()):
            ch = tp.get_text_range(i, 1)
            l, b, r, t = tp.get_charbox(i)
            chars.append((ch, (l, ph - t, r, ph - b)))       # y-down boxes

        words: list[_Word] = []
        cur: list[tuple[str, tuple[float, float, float, float]]] = []

        def flush() -> None:
            if not cur:
                return
            text = _clean_text("".join(c for c, _ in cur))
            if text:
                bx = [b for _, b in cur]
                box = Rect(min(b[0] for b in bx), min(b[1] for b in bx), max(b[2] for b in bx), max(b[3] for b in bx))
                dx = bx[-1][0] - bx[0][0]
                dy = bx[0][3] - bx[-1][3]                     # upwards on the page is positive
                vertical = len(cur) > 1 and abs(dy) > 2.0 * abs(dx)
                words.append(_Word(text, box, vertical))
            cur.clear()

        for ch, box in chars:
            if not ch.strip() or not ch.isprintable():
                flush()
                continue
            if cur:
                pb = cur[-1][1]
                # Break the word when the next glyph is not a plausible continuation.
                # Sizes use the larger of the two glyphs so tiny boxes ('.', ',') do not
                # fail the alignment test against their taller neighbours.
                row_size = max(pb[3] - pb[1], box[3] - box[1], 1e-3)     # glyph height
                col_size = max(pb[2] - pb[0], box[2] - box[0], 1e-3)     # glyph width
                same_row = abs(0.5 * (box[1] + box[3]) - 0.5 * (pb[1] + pb[3])) < 0.6 * row_size
                same_col = abs(0.5 * (box[0] + box[2]) - 0.5 * (pb[0] + pb[2])) < 0.6 * col_size
                step_x = box[0] - pb[0]
                step_y = pb[3] - box[3]
                horizontal_ok = same_row and 0 <= step_x < 2.5 * row_size
                vertical_ok = same_col and 0 < step_y < 2.5 * row_size
                if not (horizontal_ok or vertical_ok):
                    flush()
            cur.append((ch, box))
        flush()

        lines = self._join_lines(words)
        self._words_cache[page_index] = (words, lines)
        return words, lines

    @staticmethod
    def _join_lines(words: list[_Word]) -> list[_Word]:
        """Merge consecutive words of one text run into a line (titles, headings)."""
        lines: list[_Word] = []
        cur: _Word | None = None
        for w in words:
            if cur is None:
                cur = w
                continue
            size = max(min(cur.box.height, cur.box.width) if cur.vertical else cur.box.height, 1e-3)
            if cur.vertical and w.vertical:
                aligned = abs(w.cx - cur.cx) < 0.5 * size and 0 <= (cur.box.y0 - w.box.y1) < 1.5 * size
            elif not cur.vertical and not w.vertical:
                aligned = abs(w.cy - cur.cy) < 0.5 * size and 0 <= (w.box.x0 - cur.box.x1) < 1.5 * size
            else:
                aligned = False
            if aligned:
                cur = _Word(cur.text + " " + w.text, _union(cur.box, w.box), cur.vertical)
            else:
                lines.append(cur)
                cur = w
        if cur is not None:
            lines.append(cur)
        return lines

    # -- vector geometry ------------------------------------------------------
    def _thin_paths(self, page_index: int) -> list[Rect]:
        """Bounding boxes (y-down) of all path objects that look like line pieces."""
        page = self._pdf[page_index]
        ph = page.get_size()[1]
        out: list[Rect] = []
        for obj in page.get_objects():
            if obj.type != pdfium_raw.FPDF_PAGEOBJ_PATH:
                continue
            l, b, r, t = obj.get_bounds()
            if min(r - l, t - b) <= 3.0:
                out.append(Rect(l, ph - t, r, ph - b))
        return out

    @staticmethod
    def _find_frames(thin: list[Rect]) -> list[Rect]:
        """Plot frames = two long vertical + two long horizontal thin edges forming a box."""
        vert = [r for r in thin if r.width <= 3.0 and r.height >= 60]
        horz = [r for r in thin if r.height <= 3.0 and r.width >= 60]
        frames: list[Rect] = []
        for i, left in enumerate(vert):
            for right in vert[i + 1:]:
                a, b = (left, right) if left.x0 <= right.x0 else (right, left)
                if b.x0 - a.x1 < 40 or abs(a.y0 - b.y0) > 2 or abs(a.y1 - b.y1) > 2:
                    continue
                top = [h for h in horz if abs(h.x0 - a.x0) <= 2.5 and abs(h.x1 - b.x1) <= 2.5 and abs(h.y0 - a.y0) <= 2.5]
                bottom = [h for h in horz if abs(h.x0 - a.x0) <= 2.5 and abs(h.x1 - b.x1) <= 2.5 and abs(h.y1 - a.y1) <= 2.5]
                if top and bottom:
                    frames.append(Rect(0.5 * (a.x0 + a.x1), 0.5 * (top[0].y0 + top[0].y1),
                                       0.5 * (b.x0 + b.x1), 0.5 * (bottom[0].y0 + bottom[0].y1)))
        # Drop near-duplicates.
        unique: list[Rect] = []
        for f in frames:
            if not any(abs(f.x0 - u.x0) < 2 and abs(f.x1 - u.x1) < 2 and abs(f.y0 - u.y0) < 2 and abs(f.y1 - u.y1) < 2
                       for u in unique):
                unique.append(f)
        return unique

    @staticmethod
    def _line_positions(thin: list[Rect], frame: Rect, vertical: bool) -> list[float]:
        """Centre coordinates of vertical (or horizontal) lines spanning the frame.

        Dashed grid lines are drawn as many small pieces sharing one centre
        coordinate; they are recognised by the extent their pieces cover.
        """
        groups: dict[float, list[Rect]] = {}
        for r in thin:
            if vertical:
                if r.width > 1.6 or r.height < 0.3:
                    continue
                if not (frame.x0 - 2 <= r.x0 <= frame.x1 + 2 and r.y0 >= frame.y0 - 3 and r.y1 <= frame.y1 + 3):
                    continue
                key = round(0.5 * (r.x0 + r.x1) / 0.3)
            else:
                if r.height > 1.6 or r.width < 0.3:
                    continue
                if not (frame.y0 - 2 <= r.y0 <= frame.y1 + 2 and r.x0 >= frame.x0 - 3 and r.x1 <= frame.x1 + 3):
                    continue
                key = round(0.5 * (r.y0 + r.y1) / 0.3)
            groups.setdefault(key, []).append(r)
        span = frame.height if vertical else frame.width
        positions: list[float] = []
        for members in groups.values():
            if vertical:
                extent = max(m.y1 for m in members) - min(m.y0 for m in members)
                pos = float(np.mean([0.5 * (m.x0 + m.x1) for m in members]))
            else:
                extent = max(m.x1 for m in members) - min(m.x0 for m in members)
                pos = float(np.mean([0.5 * (m.y0 + m.y1) for m in members]))
            if extent >= 0.8 * span:
                positions.append(pos)
        return sorted(positions)

    # -- chart discovery --------------------------------------------------------
    def charts(self, page_index: int) -> list[ChartInfo]:
        """All charts found on a page (empty for pages without a drawn plot frame)."""
        if page_index in self._charts_cache:
            return self._charts_cache[page_index]
        thin = self._thin_paths(page_index)
        words, lines = self._text(page_index)
        pw, ph = self.page_size(page_index)
        frames = sorted(self._find_frames(thin), key=lambda f: (round(f.y0 / 20), f.x0))
        charts: list[ChartInfo] = []
        for idx, frame in enumerate(frames):
            title = self._title_for(frame, lines)
            v_lines = self._line_positions(thin, frame, vertical=True) + [frame.x0, frame.x1]
            h_lines = self._line_positions(thin, frame, vertical=False) + [frame.y0, frame.y1]
            x_axes = self._x_axes(frame, words, lines, sorted(set(v_lines)))
            y_axes = self._y_axes(frame, words, lines, sorted(set(h_lines)))
            region = self._chart_region(frame, x_axes + y_axes, pw, ph)
            charts.append(ChartInfo(page_index, idx, title, frame, region, x_axes, y_axes))
        self._charts_cache[page_index] = charts
        return charts

    @staticmethod
    def _title_for(frame: Rect, lines: list[_Word]) -> str:
        best, best_gap = "", 40.0
        for ln in lines:
            if ln.vertical or ln.is_number:
                continue
            gap = frame.y0 - ln.box.y1               # heading sits above the frame
            overlap = min(ln.box.x1, frame.x1) - max(ln.box.x0, frame.x0)
            if -1.0 <= gap < best_gap and overlap > 0.3 * ln.box.width:
                best, best_gap = ln.text, gap
        return best

    @staticmethod
    def _chart_region(frame: Rect, axes: list[AxisHint], pw: float, ph: float) -> Rect:
        """Frame grown to include the tick labels and titles of *its own* axes."""
        box = frame
        for ax in axes:
            if ax.box is not None:
                box = _union(box, ax.box)
        pad = 1.0        # small: neighbouring headings sit only ~1.5 pt away
        return Rect(max(0.0, box.x0 - pad), max(0.0, box.y0 - pad), min(pw, box.x1 + pad), min(ph, box.y1 + pad))

    @staticmethod
    def _make_ticks(labels: list[_Word], pos_of, lines: list[float]) -> list[TickLabel]:
        """Tick labels -> tick positions.

        Label centres are only good to ~1 pt, so positions are first regularised
        by a linear (or log10) fit over all labels of the axis and only then
        snapped onto a real grid/frame line - within a tight tolerance, so a
        label of one axis is never pulled onto another axis' grid.
        """
        if len(labels) < 2:
            return []
        val = np.array([w.value for w in labels], dtype=float)
        raw = np.array([pos_of(w) for w in labels], dtype=float)
        ref = raw
        if len(labels) >= 3:
            best: tuple[float, np.ndarray] | None = None
            for x in [val] + ([np.log10(val)] if np.all(val > 0) else []):
                fit = np.polyval(np.polyfit(x, raw, 1), x)
                err = float(np.max(np.abs(fit - raw)))
                if best is None or err < best[0]:
                    best = (err, fit)
            if best is not None and best[0] <= 2.0:
                ref = best[1]
        spacing = float(np.median(np.abs(np.diff(np.sort(ref)))))
        tol = min(1.5, 0.25 * spacing)
        ticks = []
        for w, r in zip(labels, ref):
            p, ok = _snap(float(r), lines, tol)
            ticks.append(TickLabel(w.text, w.value, p, ok))
        return sorted(ticks, key=lambda t: t.pos)

    def _x_axes(self, frame: Rect, words: list[_Word], lines: list[_Word], v_lines: list[float]) -> list[AxisHint]:
        cand = [w for w in words if w.is_number and not w.vertical
                and frame.x0 - 15 <= w.cx <= frame.x1 + 15 and frame.y1 + 0.5 <= w.cy <= frame.y1 + 25]
        if len(cand) < 2:
            return []
        rows: dict[int, list[_Word]] = {}
        for w in cand:
            rows.setdefault(round(w.cy / 1.5), []).append(w)
        row = min((r for r in rows.values() if len(r) >= 2), key=lambda r: np.mean([w.cy for w in r]), default=None)
        if row is None:
            return []
        row_bottom = max(w.box.y1 for w in row)
        titles = [ln for ln in lines if not ln.vertical and not ln.is_number
                  and 0 <= ln.box.y0 - row_bottom < 20 and frame.x0 <= ln.cx <= frame.x1]
        title_ln = min(titles, key=lambda ln: ln.box.y0) if titles else None
        box = row[0].box
        for w in row[1:] + ([title_ln] if title_ln else []):
            box = _union(box, w.box)
        return [AxisHint("x", "bottom", title_ln.text if title_ln else "",
                         self._make_ticks(row, lambda w: w.cx, v_lines), box)]

    def _y_axes(self, frame: Rect, words: list[_Word], lines: list[_Word], h_lines: list[float]) -> list[AxisHint]:
        hints: list[AxisHint] = []
        for side in ("left", "right"):
            if side == "left":
                cand = [w for w in words if w.is_number and not w.vertical and w.box.x1 <= frame.x0 + 0.5
                        and w.box.x0 >= frame.x0 - 70 and frame.y0 - 8 <= w.cy <= frame.y1 + 8]
                key = lambda w: round(w.box.x1)                  # right-aligned labels
            else:
                cand = [w for w in words if w.is_number and not w.vertical and w.box.x0 >= frame.x1 - 0.5
                        and w.box.x1 <= frame.x1 + 70 and frame.y0 - 8 <= w.cy <= frame.y1 + 8]
                key = lambda w: round(w.box.x0)                  # left-aligned labels
            cols: dict[int, list[_Word]] = {}
            for w in cand:
                cols.setdefault(key(w), []).append(w)
            # Merge neighbouring keys (rounding can split one column in two).
            merged: list[list[_Word]] = []
            for k in sorted(cols):
                if merged and k - key(merged[-1][0]) <= 1:
                    merged[-1].extend(cols[k])
                else:
                    merged.append(list(cols[k]))
            columns = [c for c in merged if len(c) >= 3]
            # nearest to the frame first
            columns.sort(key=lambda c: abs(np.mean([w.cx for w in c]) - (frame.x0 if side == "left" else frame.x1)))
            for col in columns:
                if side == "left":
                    edge = min(w.box.x0 for w in col)
                    titles = [ln for ln in lines if ln.vertical and 0 <= edge - ln.box.x1 < 22
                              and frame.y0 <= ln.cy <= frame.y1]
                else:
                    edge = max(w.box.x1 for w in col)
                    titles = [ln for ln in lines if ln.vertical and 0 <= ln.box.x0 - edge < 22
                              and frame.y0 <= ln.cy <= frame.y1]
                title_ln = min(titles, key=lambda ln: abs(ln.box.x0 - edge)) if titles else None
                ticks = self._make_ticks(col, lambda w: w.cy, h_lines)
                if ticks:
                    box = col[0].box
                    for w in col[1:] + ([title_ln] if title_ln else []):
                        box = _union(box, w.box)
                    hints.append(AxisHint("y", side, title_ln.text if title_ln else "", ticks, box))
        return hints

    # -- calibration proposal ---------------------------------------------------
    @staticmethod
    def axis_from_hint(hint: AxisHint, rendered: RenderedRegion) -> AxisCalibration | None:
        """Calibrate an axis from its outermost tick labels (linear or log)."""
        if len(hint.ticks) < 2:
            return None
        ticks = sorted(hint.ticks, key=lambda t: t.value)
        lo, hi = ticks[0], ticks[-1]
        if lo.value == hi.value:
            return None
        to_px = (lambda p: (p - rendered.region.x0) * rendered.sx) if hint.orientation == "x" \
            else (lambda p: (p - rendered.region.y0) * rendered.sy)
        pos = np.array([t.pos for t in ticks])
        val = np.array([t.value for t in ticks])
        log = False
        if len(ticks) >= 3 and np.all(val > 0):
            def resid(v):
                fit = np.polyfit(v, pos, 1)
                return float(np.max(np.abs(np.polyval(fit, v) - pos))) / max(np.ptp(pos), 1e-9)
            log = resid(np.log10(val)) < 0.5 * resid(val) and resid(np.log10(val)) < 0.02
        return AxisCalibration(to_px(lo.pos), lo.value, to_px(hi.pos), hi.value, log)

    def suggest_calibration(self, chart: ChartInfo, rendered: RenderedRegion,
                            x_index: int = 0, y_index: int = 0) -> Calibration | None:
        """Calibration from tick labels of the chart's ``x_index``-th / ``y_index``-th axis."""
        if x_index >= len(chart.x_axes) or y_index >= len(chart.y_axes):
            return None
        ax = self.axis_from_hint(chart.x_axes[x_index], rendered)
        ay = self.axis_from_hint(chart.y_axes[y_index], rendered)
        return Calibration(ax, ay) if ax and ay else None
