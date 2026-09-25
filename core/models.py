"""Plain data containers shared by the core algorithms and the UI.

All pixel coordinates in this project are *continuous image coordinates*:
pixel ``(i, j)`` covers ``[j, j+1) x [i, i+1)`` and its centre is at
``(j + 0.5, i + 0.5)``.  This matches Qt's scene coordinates for a pixmap
placed at the origin and PDF render transforms, so no half-pixel fudge is
needed anywhere else.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Rect:
    """Axis-aligned rectangle in continuous pixel coordinates (x0 <= x1, y0 <= y1)."""

    x0: float
    y0: float
    x1: float
    y1: float

    @staticmethod
    def from_points(xa: float, ya: float, xb: float, yb: float) -> "Rect":
        return Rect(min(xa, xb), min(ya, yb), max(xa, xb), max(ya, yb))

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    def contains(self, x, y):
        """Vectorised point-in-rect test (half-open on the far edges)."""
        x = np.asarray(x)
        y = np.asarray(y)
        return (x >= self.x0) & (x < self.x1) & (y >= self.y0) & (y < self.y1)

    def pixel_slices(self, shape: tuple[int, int]) -> tuple[slice, slice]:
        """Return ``(row_slice, col_slice)`` of the pixels whose centres lie inside.

        The result is clipped to an image of the given ``(height, width)``.
        """
        h, w = shape[:2]
        c0 = int(np.clip(np.ceil(self.x0 - 0.5), 0, w))
        c1 = int(np.clip(np.ceil(self.x1 - 0.5), 0, w))
        r0 = int(np.clip(np.ceil(self.y0 - 0.5), 0, h))
        r1 = int(np.clip(np.ceil(self.y1 - 0.5), 0, h))
        return slice(r0, max(r0, r1)), slice(c0, max(c0, c1))

    def to_list(self) -> list[float]:
        return [self.x0, self.y0, self.x1, self.y1]

    @staticmethod
    def from_list(v: Iterable[float]) -> "Rect":
        x0, y0, x1, y1 = (float(t) for t in v)
        return Rect(x0, y0, x1, y1)


@dataclass
class ExtractionParams:
    """Tunable knobs of the automatic trace extraction.

    Sizes are in image pixels unless stated otherwise.  A value of ``0`` for
    the ``auto`` marked entries lets the algorithm derive a sensible number
    from the plot-area size.
    """

    # Colour matching -------------------------------------------------------
    delta_e: float = 25.0            # CIE76 distance to the reference colour

    # Mask clean-up ----------------------------------------------------------
    open_size: int = 0               # auto: opening only when strokes are thick enough
    min_area: int = 3                # drop specks smaller than this (px)
    remove_grid: bool = True         # strip dashed / solid grid lines from the mask
    dashed: bool = False             # curve itself is dashed/dotted: keep short mask components
    grid_fill: float = 0.30          # row/col coverage that flags a grid line
    grid_max_width: int = 0          # auto: widest band still treated as a grid line
    min_component_frac: float = 0.05 # keep components >= this x longest component

    # Tracking ---------------------------------------------------------------
    merge_gap: int = 2               # bridge vertical gaps <= this inside a column
    max_gap: int = 0                 # auto: columns without a candidate before stopping
    slope_points: int = 8            # points used for the slope prediction
    jump_tol: float = 0.0            # auto: base tolerance for candidate selection (px)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "ExtractionParams":
        known = {k: v for k, v in d.items() if k in ExtractionParams.__dataclass_fields__}
        return ExtractionParams(**known)


@dataclass
class Curve:
    """A named data series whose points live in pixel space.

    Pixel space is the single source of truth: calibration is applied only when
    values are shown or exported, so changing the calibration never requires
    re-extracting a curve.
    """

    name: str
    color_rgb: tuple[int, int, int]
    params: ExtractionParams = field(default_factory=ExtractionParams)
    x_px: np.ndarray = field(default_factory=lambda: np.empty(0))
    y_px: np.ndarray = field(default_factory=lambda: np.empty(0))
    edited: bool = False             # True once the user changed points by hand
    seed: tuple[float, float] | None = None   # where the user clicked: tracking starts on that curve

    def __post_init__(self) -> None:
        self.x_px = np.asarray(self.x_px, dtype=float)
        self.y_px = np.asarray(self.y_px, dtype=float)

    def __len__(self) -> int:
        return len(self.x_px)

    def set_points(self, x_px, y_px) -> None:
        x_px = np.asarray(x_px, dtype=float)
        y_px = np.asarray(y_px, dtype=float)
        order = np.argsort(x_px, kind="stable")
        self.x_px, self.y_px = x_px[order], y_px[order]

    def add_point(self, x: float, y: float) -> None:
        """Insert one manual point, keeping the series sorted by x."""
        i = int(np.searchsorted(self.x_px, x))
        self.x_px = np.insert(self.x_px, i, x)
        self.y_px = np.insert(self.y_px, i, y)

    def remove_in_rect(self, rect: Rect) -> int:
        """Delete every point inside ``rect``; returns how many were removed."""
        if len(self) == 0:
            return 0
        inside = rect.contains(self.x_px, self.y_px)
        n = int(inside.sum())
        if n:
            self.x_px = self.x_px[~inside]
            self.y_px = self.y_px[~inside]
        return n
