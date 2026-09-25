"""Reading a lookup table from a text file (CSV / TSV / TXT) and comparing it with a curve.

The application's own CSV export is accepted as is (``#`` comment lines, then a header and ``X,Y`` rows), and so is
what other tools and spreadsheets produce: ``;`` ``,`` tab, ``|`` or blanks between the columns, decimal comma or
point, an optional header line, several Y columns.  Cells that are not numbers (empty, ``—``, ``nan``) become NaN.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .postprocess import CurveData

_DELIMITERS = (";", "\t", "|", None, ",")            # None = runs of blanks
_EMPTY = {"", "-", "—", "–", "nan", "na", "n/a", "null", "none"}


@dataclass
class ImportedTable:
    name: str                                  # file name
    x_header: str
    y_headers: list[str]
    x: np.ndarray                              # ascending
    ys: list[np.ndarray]                       # one per Y column, NaN where a cell was empty
    skipped: int = 0                           # non-numeric lines inside the data (not counting the header)
    comments: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.x)

    def y_finite(self, i: int = 0) -> tuple[np.ndarray, np.ndarray]:
        y = self.ys[i]
        ok = np.isfinite(y)
        return self.x[ok], y[ok]


def parse_number(text: str) -> float:
    """``'1500'``, ``'3,65'``, ``'1.234,5'``, ``'2.5e3'`` -> float; anything else (or empty) -> NaN."""
    t = text.strip().replace(" ", "").replace(" ", "")
    if t.lower() in _EMPTY:
        return math.nan
    if "," in t and "." in t:
        t = t.replace(".", "").replace(",", ".") if t.rfind(",") > t.rfind(".") else t.replace(",", "")
    else:
        t = t.replace(",", ".")
    try:
        v = float(t)
    except ValueError:
        return math.nan
    return v if math.isfinite(v) else math.nan


def _is_number(cell: str) -> bool:
    return math.isfinite(parse_number(cell))


def _decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "cp1254", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1", errors="replace")


def _split(line: str, delim) -> list[str]:
    if delim is None:
        return line.split()
    return [c.strip() for c in line.split(delim)]


def _pick_delimiter(lines: list[str]) -> str | None:
    """First delimiter under which most lines have two or more cells and some line has numbers in two cells;
    failing that, the first under which most lines have two or more cells (the error is reported later)."""
    sample = lines[:60]
    fallback = "none"
    for delim in _DELIMITERS:
        cells = [_split(ln, delim) for ln in sample]
        multi = sum(len(c) >= 2 for c in cells)
        if multi < 0.6 * len(sample):
            continue
        if sum(len(c) >= 2 and sum(_is_number(x) for x in c) >= 2 for c in cells) >= 1:
            return delim
        if fallback == "none":
            fallback = delim
    if fallback != "none":
        return fallback
    raise ValueError("Sütunlar ayrılamadı: X ve Y değerlerini ; , sekme ya da boşlukla ayırın.")


def read_table(path: str | Path) -> ImportedTable:
    """Parse a lookup table file.  Raises ``ValueError`` (Turkish message) when it holds no usable X / Y numbers."""
    path = Path(path)
    try:
        text = _decode(path.read_bytes())
    except OSError as exc:
        raise ValueError(f"Dosya okunamadı: {exc}") from exc
    comments: list[str] = []
    lines: list[str] = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            continue
        if s.startswith("#"):
            comments.append(s.lstrip("#").strip())
            continue
        lines.append(s)
    if len(lines) < 2:
        raise ValueError("Dosyada tablo satırı yok.")
    delim = _pick_delimiter(lines)
    rows = [_split(ln, delim) for ln in lines]

    # header: the first line when it holds text cells and the next line is numeric
    header: list[str] | None = None
    first = rows[0]
    if any(c and not _is_number(c) and c.lower() not in _EMPTY for c in first) and len(rows) > 1 \
            and sum(_is_number(c) for c in rows[1]) >= 2:
        header, rows = first, rows[1:]

    numeric_rows, skipped = [], 0
    for r in rows:
        vals = [parse_number(c) for c in r]
        if len(vals) >= 2 and math.isfinite(vals[0]):
            numeric_rows.append(vals)
        else:
            skipped += 1
    if not numeric_rows:
        raise ValueError("Dosyada X ve Y için sayısal satır bulunamadı.")
    ncols = Counter(len(v) for v in numeric_rows).most_common(1)[0][0]
    numeric_rows = [v for v in numeric_rows if len(v) == ncols]
    if ncols < 2:
        raise ValueError("En az iki sütun (X ve Y) gerekli.")
    arr = np.array(numeric_rows, dtype=float)
    order = np.argsort(arr[:, 0], kind="stable")
    arr = arr[order]
    keep = [j for j in range(1, ncols) if np.isfinite(arr[:, j]).any()]
    if not keep:
        raise ValueError("Y sütununda sayı yok.")
    if len(arr) < 2:
        raise ValueError("En az iki satır gerekli.")
    names = header if header and len(header) == ncols else None
    xh = names[0] if names else "X"
    yh = [names[j] if names else ("Y" if len(keep) == 1 else f"Y{n + 1}") for n, j in enumerate(keep)]
    return ImportedTable(path.name, xh or "X", yh, arr[:, 0].copy(), [arr[:, j].copy() for j in keep], skipped, comments)


# --------------------------------------------------------------------------- #
# Comparison with a curve
# --------------------------------------------------------------------------- #
@dataclass
class Comparison:
    x: np.ndarray
    y: np.ndarray                              # the table's values
    y_curve: np.ndarray                        # the curve at the table's X (NaN outside the curve's X range)
    delta: np.ndarray                          # y - y_curve
    rel_pct: np.ndarray                        # |delta| / |y_curve| * 100
    n_total: int
    n_compared: int                            # points inside the curve's X range that have a Y
    mean: float
    mean_abs: float
    rms: float
    max_abs: float
    x_of_max: float
    max_rel_pct: float
    mean_rel_pct: float


def compare_to_curve(x, y, curve: CurveData) -> Comparison:
    """Table points against a curve: the curve is linearly interpolated at the table's X values."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    yc = np.full(len(x), np.nan)
    if len(curve) >= 2:
        ux, inv = np.unique(curve.x, return_inverse=True)
        uy = np.bincount(inv, weights=curve.y) / np.bincount(inv)
        inside = (x >= ux[0]) & (x <= ux[-1])
        yc[inside] = np.interp(x[inside], ux, uy)
    delta = y - yc
    ok = np.isfinite(delta)
    rel = np.where(ok, np.abs(delta) / np.where(np.abs(yc) > 0, np.abs(yc), np.nan) * 100.0, np.nan)
    n = int(ok.sum())
    if n:
        d = delta[ok]
        k = int(np.argmax(np.abs(d)))
        r = rel[ok]
        rel_ok = r[np.isfinite(r)]
        return Comparison(x, y, yc, delta, rel, len(x), n, float(d.mean()), float(np.abs(d).mean()),
                          float(np.sqrt((d ** 2).mean())), float(np.abs(d[k])), float(x[ok][k]),
                          float(rel_ok.max()) if rel_ok.size else math.nan,
                          float(rel_ok.mean()) if rel_ok.size else math.nan)
    nan = math.nan
    return Comparison(x, y, yc, delta, rel, len(x), 0, nan, nan, nan, nan, nan, nan, nan)


_UNIT = re.compile(r"\(([^()]*)\)\s*$")


def unit_in(header: str) -> str:
    """'Voltage (V)' -> 'V'."""
    m = _UNIT.search(header.strip())
    return m.group(1).strip() if m else ""


def units_conflict(a: str, b: str) -> bool:
    """True when both headers state a unit and the units differ (case-insensitively)."""
    ua, ub = unit_in(a), unit_in(b)
    return bool(ua and ub and ua.lower() != ub.lower())
