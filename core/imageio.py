"""Image file loading that survives non-ASCII paths and transparent PNGs."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def load_image(path: str | Path) -> np.ndarray:
    """Read an image as an ``H x W x 3`` uint8 BGR array.

    ``cv2.imread`` cannot open paths with non-ASCII characters on Windows, so the
    bytes are read with numpy and decoded from memory.  Transparent pixels are
    composited over white (a plot on a transparent background is meant to be
    seen on white).  Raises ``ValueError`` for unreadable files.
    """
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED) if data.size else None
    if img is None:
        raise ValueError(f"Görsel okunamadı: {path}")
    if img.dtype != np.uint8:
        img = cv2.convertScaleAbs(img, alpha=255.0 / (65535.0 if img.dtype == np.uint16 else 1.0))
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        alpha = img[..., 3:4].astype(np.float32) / 255.0
        rgb = img[..., :3].astype(np.float32) * alpha + 255.0 * (1.0 - alpha)
        return np.ascontiguousarray(np.round(rgb).astype(np.uint8))
    return np.ascontiguousarray(img[..., :3])
