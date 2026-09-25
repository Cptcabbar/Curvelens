"""Small numpy-backed table model shared by the windows."""
from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt


def fmt(v: float, digits: int = 6) -> str:
    return "—" if v is None or not math.isfinite(v) else f"{v:.{digits}g}"


class ArrayTableModel(QAbstractTableModel):
    """Read-only table over equally long columns; a column is either numbers or ready-made strings."""

    def __init__(self) -> None:
        super().__init__()
        self._headers: list[str] = []
        self._cols: list = []

    def set_data(self, headers: list[str], cols: list) -> None:
        self.beginResetModel()
        self._headers, self._cols = headers, cols
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() or not self._cols else len(self._cols[0])

    def columnCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._cols)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and index.isValid():
            v = self._cols[index.column()][index.row()]
            return v if isinstance(v, str) else fmt(float(v))
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            return self._headers[section] if orientation == Qt.Orientation.Horizontal else str(section + 1)
        return None
