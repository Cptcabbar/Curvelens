"""Runs in a subprocess with the *native* Qt platform (the offscreen platform has no fonts, so text
would be missing from the PDF).  Reads a JSON spec from argv[1], writes the report PDF and prints a
JSON summary (page count, page size, text of every page)."""
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.pop("QT_QPA_PLATFORM", None)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                                   # noqa: E402
import pypdfium2 as pdfium                                           # noqa: E402
from PySide6.QtWidgets import QApplication                           # noqa: E402

app = QApplication([])

from core.lookup import build_lookup, to_html                        # noqa: E402
from core.models import Rect                                         # noqa: E402
from core.pdf_source import ChartInfo                                # noqa: E402
from core.postprocess import CurveData                               # noqa: E402
from core.vector_charts import AxisFit, ChartData, CurveResult       # noqa: E402
from ui import printing                                              # noqa: E402

spec = json.loads(sys.argv[1])
x = np.array(spec["x"], float)
y = np.array(spec["y"], float)
fx = AxisFit("x", "Time (s)", "s", False, 1.0, 0.0, 0.01)
fy = AxisFit("y", "Voltage (V)", "V", False, 1.0, 0.0, 0.01)
data = CurveData(spec["label"], x, y, np.full(len(x), 0.05), np.full(len(x), 0.0026))
cv = CurveResult(0, "solid", (255, 0, 0), spec["label"], True, 0, np.zeros(len(x)), np.zeros(len(x)), data)
chart = ChartData(ChartInfo(0, 0, spec.get("title", "Demo chart"), Rect(0, 0, 1, 1), Rect(0, 0, 1, 1)), fx, [fy], [cv],
                  spec.get("notes", []))
table = build_lookup(chart, cv, step=spec["step"], source="demo.pdf")
out = Path(tempfile.mkdtemp()) / "t.pdf"
printing.save_pdf(to_html(table, columns=spec.get("columns", 1)), out)
pdf = pdfium.PdfDocument(str(out))
print(json.dumps({
    "pages": len(pdf),
    "size": [round(v) for v in pdf[0].get_size()],
    "text": [p.get_textpage().get_text_range() for p in pdf],
    "header_ok": out.read_bytes().startswith(b"%PDF"),
}))
