"""Lookup table -> PDF file, checked by reading the produced PDF back.

The PDFs are produced in a subprocess with Qt's native (windows) platform: the offscreen platform used
by the other GUI tests has no fonts, so its PDFs would contain no text.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")

WORKER = Path(__file__).with_name("print_worker.py")
pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="text checks need the native Windows platform")


def make_pdf(x, y, label="Motor A", step=10.0, columns=1, notes=(), title="Demo chart") -> dict:
    spec = {"x": list(map(float, x)), "y": list(map(float, y)), "label": label, "step": step, "columns": columns,
            "notes": list(notes), "title": title}
    env = {k: v for k, v in os.environ.items() if k != "QT_QPA_PLATFORM"}
    env["PYTHONUTF8"] = "1"
    res = subprocess.run([sys.executable, str(WORKER), json.dumps(spec)], capture_output=True, text=True,
                         encoding="utf-8", env=env, timeout=120)
    assert res.returncode == 0, res.stderr[-2000:]
    return json.loads(res.stdout.strip().splitlines()[-1])


def test_report_becomes_an_a4_pdf_with_the_table_text():
    out = make_pdf([0, 10, 20, 30], [1.0, 2.5, 3.75, 4.125], notes=["Temperature: 23°C."])
    assert out["header_ok"] and out["pages"] == 1 and out["size"] == [595, 842]
    text = out["text"][0]
    for needle in ("Lookup tablosu", "Motor A", "Demo chart", "Time (s)", "Voltage (V)", "3.750", "4.125", "demo.pdf",
                   "Temperature: 23°C."):
        assert needle in text, needle


def test_long_tables_continue_on_more_pages_and_repeat_the_header():
    x = np.arange(0, 400.0)
    out = make_pdf(x, np.sqrt(x), step=1.0)
    assert 6 <= out["pages"] <= 10                                   # ~50 rows per page
    compact = "".join("".join(out["text"]).split())                     # pdf text extraction may break numbers into glyph lines
    assert "399" in compact and "19.975" in compact                     # last row (sqrt(399)) is there
    assert all("Voltage(V)" in "".join(p.split()) for p in out["text"])  # header row repeats on every page


def test_column_blocks_use_fewer_pages():
    x = np.arange(0, 400.0)
    one = make_pdf(x, np.sqrt(x), step=1.0, columns=1)["pages"]
    two = make_pdf(x, np.sqrt(x), step=1.0, columns=2)["pages"]
    assert two < one


def test_turkish_and_special_characters_survive():
    out = make_pdf([0, 10], [1, 2], label="Şarj eğrisi 5.6 A ± 1 °C", title="Deşarj grafiği")
    text = out["text"][0]
    assert "Şarj eğrisi" in text and "°C" in text and "Deşarj grafiği" in text


def test_pdf_can_also_be_written_in_the_gui_test_environment(tmp_path):
    """In-process (offscreen) run: no text check, but the file must be a valid PDF."""
    from PySide6.QtWidgets import QApplication

    from ui import printing

    QApplication.instance() or QApplication([])
    out = printing.save_pdf("<html><body><h2>x</h2><table border='1'><tr><td>1</td></tr></table></body></html>",
                            tmp_path / "t.pdf")
    assert out.read_bytes().startswith(b"%PDF") and out.stat().st_size > 500
