from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.lookup import (auto_step, build_lookup, decimals_for, step_decimals, to_html, uncertainty_text, write_csv)
from core.pdf_source import ChartInfo
from core.postprocess import CurveData
from core.models import Rect
from core.vector_charts import AxisFit, ChartData, CurveResult

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "samples" / "INR18650P28A-V1-80093.pdf"


# ------------------------------------------------------------------------------- helpers
def test_auto_step_is_a_round_1_2_5_number_with_about_sixty_rows():
    assert auto_step(0, 3000) == 50
    assert auto_step(0, 1.5) == 0.05                                  # 30 rows: rounded up to a 1-2-5 step
    assert auto_step(0, 500) == 10
    assert auto_step(0.77, 2848.7) == 50
    assert auto_step(5, 5) == 1.0                                   # degenerate range
    assert auto_step(float("nan"), 3) == 1.0
    for span in (0.3, 7, 130, 9999):
        rows = span / auto_step(0, span)
        assert 20 <= rows <= 100


def test_decimals_follow_the_uncertainty_and_the_step():
    assert decimals_for(0.0026) == 3
    assert decimals_for(1.2) == 0
    assert decimals_for(0.00004) == 5
    assert decimals_for(0) == 3 and decimals_for(float("nan")) == 3
    assert step_decimals(50) == 0 and step_decimals(0.25) == 2 and step_decimals(0.02) == 2 and step_decimals(2.5) == 1


# ------------------------------------------------------------------------------- synthetic chart
def _chart(x, y, x_unc=0.05, y_unc=0.0026, style="solid", label="C1"):
    fit_x = AxisFit("x", "Time (s)", "s", False, 1.0, 0.0, 0.01)
    fit_y = AxisFit("y", "Voltage (V)", "V", False, 1.0, 0.0, 0.01)
    data = CurveData(label, np.asarray(x, float), np.asarray(y, float), np.full(len(x), x_unc), np.full(len(x), y_unc))
    cv = CurveResult(0, style, (255, 0, 0), label, True, 0, np.zeros(len(x)), np.zeros(len(x)), data)
    info = ChartInfo(0, 0, "Demo chart", Rect(0, 0, 1, 1), Rect(0, 0, 1, 1))
    return ChartData(info, fit_x, [fit_y], [cv], ["note one"]), cv


def test_build_lookup_on_a_regular_grid_with_linear_interpolation():
    chart, cv = _chart([3, 12, 27, 41], [0, 9, 24, 38])                # y = x - 3
    t = build_lookup(chart, cv, step=10.0, source="a.pdf")
    assert list(t.x) == [10, 20, 30, 40]
    assert t.y == pytest.approx([7, 17, 27, 37])
    assert t.step == 10.0 and t.x_decimals == 0 and t.y_decimals == 3
    assert (t.x_header, t.y_header) == ("Time (s)", "Voltage (V)")
    assert t.rows()[0] == ("10", "7.000") and t.page == 1 and t.source == "a.pdf" and t.notes == ["note one"]


def test_no_values_are_invented_outside_the_curve():
    chart, cv = _chart([15, 25, 35], [1, 2, 3])
    t = build_lookup(chart, cv, step=10.0)
    assert list(t.x) == [20, 30]                                       # 10 and 40 lie outside [15, 35]


def test_auto_and_raw_modes():
    x = np.linspace(0, 100, 401)
    chart, cv = _chart(x, np.sqrt(x))
    auto = build_lookup(chart, cv)                                     # step="auto"
    assert auto.step == 2.0 and len(auto) == 51
    raw = build_lookup(chart, cv, step=None)
    assert raw.step is None and len(raw) == 401 and raw.x[0] == 0
    assert raw.y_decimals == 3


def test_raw_mode_averages_duplicate_x():
    chart, cv = _chart([0, 1, 1, 2], [0, 1.0, 3.0, 4.0])
    t = build_lookup(chart, cv, step=None)
    assert list(t.x) == [0, 1, 2] and t.y == pytest.approx([0, 2.0, 4.0])


def test_uncertainty_text_and_html_report():
    chart, cv = _chart([0, 10, 20, 30], [1, 2, 3, 4], x_unc=0.5, y_unc=0.0026)
    t = build_lookup(chart, cv, step=10.0, source="demo.pdf")
    assert uncertainty_text(t) == "X ±0.5 s, Y ±0.0026 V"
    html = to_html(t)
    for needle in ("Lookup tablosu — C1", "Demo chart", "Time (s)", "Voltage (V)", "4.000", "demo.pdf", "sürekli çizgi",
                   "vektör verisinden", "note one", "±0.0026 V"):
        assert needle in html, needle
    assert "<thead>" in html and html.count("<tr>") >= 4 + len(t)      # header + rows


def test_html_can_split_into_column_blocks_and_escapes_text():
    x = np.arange(0, 100.0)
    chart, cv = _chart(x, x * 2, label="<b>&Q")
    t = build_lookup(chart, cv, step=1.0)
    two = to_html(t, columns=2)
    assert two.count("<th>Time (s)</th>") == 2
    assert "<b>&Q" not in two and "&lt;b&gt;&amp;Q" in two
    assert two.count("<tr>") < to_html(t, columns=1).count("<tr>")     # fewer rows: two blocks side by side


def test_write_csv_layout(tmp_path):
    chart, cv = _chart([15, 25, 35], [1.5, 2.5, 3.5])
    t = build_lookup(chart, cv, step=10.0, source="demo.pdf")
    p = write_csv(t, tmp_path / "out.csv")
    text = p.read_text(encoding="utf-8-sig")
    assert text.startswith("# Lookup tablosu: Demo chart / C1")
    df = pd.read_csv(p, comment="#", encoding="utf-8-sig")
    assert list(df.columns) == ["Time (s)", "Voltage (V)"] and df["Time (s)"].tolist() == [20, 30]
    assert df["Voltage (V)"].tolist() == [2.0, 3.0]
    assert p.read_bytes().startswith(b"\xef\xbb\xbf")                   # BOM so Excel reads the text correctly


# ------------------------------------------------------------------------------- the datasheet
@pytest.mark.skipif(not PDF.exists(), reason="sample datasheet PDF missing")
def test_lookup_of_a_real_curve_reproduces_the_control_values():
    from core.vector_charts import analyze_pdf

    charts = {c.title: c for c in analyze_pdf(PDF)}
    rate = charts["Discharge Rate Characteristics"]
    red = next(c for c in rate.curves if c.label == "0.56A")
    t = build_lookup(rate, red, step=50.0, source="INR18650P28A.pdf")
    values = dict(zip(t.x, t.y))
    assert values[1500.0] == pytest.approx(3.65, abs=0.01)             # the acceptance value of the original task
    assert t.x[0] == 50 and t.x[-1] == 2800 and t.x_decimals == 0 and t.y_decimals == 3
    assert values[50.0] > values[2800.0] > 2.5
    magenta = build_lookup(rate, next(c for c in rate.curves if c.label == "30A"), step=50.0)
    assert dict(zip(magenta.x, magenta.y))[1500.0] == pytest.approx(3.15, abs=0.01)

    dashed = next(c for c in charts["Charge Characteristics"].curves if c.style == "dashed")
    td = build_lookup(charts["Charge Characteristics"], dashed)
    assert np.all(np.isfinite(td.y)) and len(td) > 20                    # gaps between the dashes are interpolated
