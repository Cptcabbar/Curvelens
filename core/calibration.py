"""Pixel <-> data-value conversion for one or two axes (linear or log10)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_LN10 = float(np.log(10.0))


@dataclass(frozen=True)
class AxisCalibration:
    """Two reference points ``(pixel p1 -> value v1)`` and ``(p2 -> v2)`` on one axis.

    ``log=True`` means the axis is logarithmic: the mapping is linear in
    ``log10(value)``, so both reference values must be positive.
    """

    p1: float
    v1: float
    p2: float
    v2: float
    log: bool = False

    def __post_init__(self) -> None:
        if not all(np.isfinite([self.p1, self.v1, self.p2, self.v2])):
            raise ValueError("Kalibrasyon değerleri sonlu sayılar olmalı.")
        if self.p1 == self.p2:
            raise ValueError("İki kalibrasyon noktası aynı piksel konumunda olamaz.")
        if self.v1 == self.v2:
            raise ValueError("İki kalibrasyon değeri birbirinden farklı olmalı.")
        if self.log and (self.v1 <= 0 or self.v2 <= 0):
            raise ValueError("Logaritmik eksende değerler pozitif olmalı.")

    # -- internal: value space used for the linear interpolation --------------
    def _fwd(self, v):
        return np.log10(v) if self.log else np.asarray(v, dtype=float)

    def _inv(self, u):
        return np.power(10.0, u) if self.log else np.asarray(u, dtype=float)

    @property
    def _slope(self) -> float:
        """Change of the (possibly log10) value per pixel."""
        return float((self._fwd(self.v2) - self._fwd(self.v1)) / (self.p2 - self.p1))

    # -- public API -----------------------------------------------------------
    def to_value(self, px):
        """Pixel coordinate(s) -> data value(s)."""
        px = np.asarray(px, dtype=float)
        return self._inv(self._fwd(self.v1) + (px - self.p1) * self._slope)

    def to_pixel(self, value):
        """Data value(s) -> pixel coordinate(s)."""
        value = np.asarray(value, dtype=float)
        return self.p1 + (self._fwd(value) - self._fwd(self.v1)) / self._slope

    def resolution(self, px=None):
        """Absolute data change per pixel at pixel position(s) ``px``.

        Constant for a linear axis; proportional to the local value for a log
        axis (``dv/dpx = ln(10) * v * slope``).  With ``px=None`` the value at
        the first reference point is used (only relevant for log axes).
        """
        if not self.log:
            res = abs(self._slope)
            return res if px is None else np.full(np.shape(px), res, dtype=float)
        px = self.p1 if px is None else px
        v = self.to_value(px)
        return _LN10 * np.abs(v) * abs(self._slope)

    def uncertainty(self, px=None, pixel_error: float = 0.5):
        """Value uncertainty (+/-) caused by a position error of ``pixel_error`` px."""
        return pixel_error * self.resolution(px)

    def to_dict(self) -> dict:
        return {"p1": self.p1, "v1": self.v1, "p2": self.p2, "v2": self.v2, "log": self.log}

    @staticmethod
    def from_dict(d: dict) -> "AxisCalibration":
        return AxisCalibration(float(d["p1"]), float(d["v1"]), float(d["p2"]), float(d["v2"]),
                               bool(d.get("log", False)))


@dataclass(frozen=True)
class Calibration:
    """Independent X and Y axis calibrations.

    X uses only the x pixel coordinate and Y only the y pixel coordinate, which
    tolerates slightly rotated scans (perspective correction is out of scope).
    """

    x: AxisCalibration
    y: AxisCalibration

    def pixel_to_data(self, x_px, y_px):
        return self.x.to_value(x_px), self.y.to_value(y_px)

    def data_to_pixel(self, x_val, y_val):
        return self.x.to_pixel(x_val), self.y.to_pixel(y_val)

    def uncertainty(self, x_px=None, y_px=None, pixel_error: float = 0.5):
        """``(+/-X, +/-Y)`` for the given pixel position(s)."""
        return self.x.uncertainty(x_px, pixel_error), self.y.uncertainty(y_px, pixel_error)

    def to_dict(self) -> dict:
        return {"x": self.x.to_dict(), "y": self.y.to_dict()}

    @staticmethod
    def from_dict(d: dict) -> "Calibration":
        return Calibration(AxisCalibration.from_dict(d["x"]), AxisCalibration.from_dict(d["y"]))
