"""Headless self-check of the whole processing stack (used to verify the packaged .exe).

``python app.py --selftest samples/INR18650P28A-V1-80093.pdf`` runs every library the application
depends on on the given PDF: chart detection and exact curve reading (pypdfium2 + NumPy/SciPy), the
lookup table (pandas CSV), the PDF report through Qt's print support, and the older pixel extraction
(OpenCV).  It exits with 0 when every step works.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np


def run(path: str | None, log=print) -> int:
    import cv2
    import pandas
    import pypdfium2
    import scipy

    pdfium_ver = getattr(pypdfium2, "__version__", None) or str(getattr(getattr(pypdfium2, "version", None), "PYPDFIUM_INFO", "?"))
    log(f"python {sys.version.split()[0]} numpy {np.__version__} scipy {scipy.__version__} "
        f"opencv {cv2.__version__} pandas {pandas.__version__} pypdfium2 {pdfium_ver}")
    if not path:
        log("SELFTEST: no file given, library imports only")
        return 0
    if not path.lower().endswith(".pdf"):
        log("SELFTEST FAILED: expected a PDF")
        return 1

    from core.lookup import build_lookup, to_html, write_csv
    from core.vector_charts import analyze_pdf

    # 1. charts + exact curves ------------------------------------------------------------
    charts = analyze_pdf(path)
    log(f"charts: {len(charts)} -> " + "; ".join(f"{c.title} ({len(c.curves)} curves)" for c in charts))
    with_curves = [c for c in charts if c.curves]
    if not with_curves:
        log("SELFTEST FAILED: no chart with curves found")
        return 1
    chart = next((c for c in with_curves if "Discharge Rate" in c.title), with_curves[0])
    curve = chart.curves[0]
    log(f"first curve of '{chart.title}': {curve.label!r} ({curve.style_label}, {curve.n_points} points, "
        f"x {curve.data.x[0]:.4g}..{curve.data.x[-1]:.4g})")

    # 2. lookup table -----------------------------------------------------------------------
    table = build_lookup(chart, curve, source=Path(path).name)
    log(f"lookup: {len(table)} rows, X step {table.step:g}, first rows {table.rows()[:2]}")
    ok = len(table) > 5 and np.isfinite(table.y).any()
    if "Discharge Rate" in chart.title and curve.label == "0.56A":
        v = float(np.interp(1500, curve.data.x, curve.data.y))
        log(f"V(1500 mAh) = {v:.4f} V")
        ok = ok and abs(v - 3.6452) < 0.005
    with tempfile.TemporaryDirectory() as tmp:
        csv = write_csv(table, Path(tmp) / "selftest.csv")
        df = pandas.read_csv(csv, comment="#", encoding="utf-8-sig")
        log(f"csv: {len(df)} rows, columns {list(df.columns)}")
        ok = ok and len(df) == len(table)

        # 3. PDF report through Qt print support (needs the GUI libraries, no window is shown) -------
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication(["selftest"])
        from ui import printing

        out = printing.save_pdf(to_html(table), Path(tmp) / "report.pdf")
        pdf = pypdfium2.PdfDocument(str(out))
        text = pdf[0].get_textpage().get_text_range()
        log(f"report pdf: {len(pdf)} page(s), {out.stat().st_size} bytes, text ok: {'Lookup tablosu' in text}")
        ok = ok and len(pdf) >= 1 and "Lookup tablosu" in text
        pdf.close()
        del app

    # 4. the older pixel extraction (scanned charts) ------------------------------------------------
    from core.extraction import extract_curve
    from core.pdf_source import PdfSource

    with PdfSource(path) as src:
        found = src.charts(0)
        if found:
            rendered = src.render(0, found[0].region, dpi=150)
            res = extract_curve(rendered.image, (255, 0, 0), rendered.rect_pt_to_px(found[0].plot_rect))
            log(f"pixel extraction (red curve, first chart): {len(res)} points")

            # 5. picture charts: the same chart, treated as if it were an image pasted into the PDF -------
            from core.plotarea import detect_plot_area
            from core.raster_charts import (AxisPoints, RasterChart, build_chart, calibrate, detect_ticks, discover_curves,
                                            rebuild_curves)

            plot = detect_plot_area(rendered.image)
            if plot is None:
                log("SELFTEST FAILED: plot frame of the rendered chart not found")
                ok = False
            else:
                rc = RasterChart(0, 0, "selftest", [], found[0].region, rendered.image, plot, detect_ticks(rendered.image, plot))
                pic = build_chart(rc)
                calibrate(pic, AxisPoints((plot.x0, plot.x1), (0.0, 1.0), "X"), AxisPoints((plot.y1, plot.y0), (0.0, 1.0), "Y"))
                rc.traces = discover_curves(rc)
                rebuild_curves(pic)
                log(f"picture-chart path: {len(rc.traces)} curves read, first with "
                    f"{pic.curves[0].n_points if pic.curves else 0} points")
                ok = ok and len(pic.curves) > 0

    # 6. hand corrections and value queries ----------------------------------------------------------
    from core.query import solve_x, value_at

    mid = 0.5 * (float(curve.data.x[0]) + float(curve.data.x[-1]))
    ans = value_at(curve.data, mid)
    hits = solve_x(curve.data, float(ans.value)) if ans else []
    n0 = curve.n_points
    curve.add_point(mid, float(ans.value) if ans else 0.0)
    edited_ok = curve.n_points == n0 + 1 and curve.undo() and curve.n_points == n0
    log(f"query: value_at({mid:.4g}) = {ans.value if ans else None}, {len(hits)} X answer(s); edit/undo ok: {edited_ok}")
    ok = ok and ans is not None and len(hits) >= 1 and edited_ok
    # 7. importing a lookup table and comparing it with the curve --------------------------------------
    from core.table_import import compare_to_curve, read_table

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = write_csv(table, Path(tmp) / "reimport.csv")
        back = read_table(csv_path)
        xs, ys = back.y_finite(0)
        cmp = compare_to_curve(xs, ys, curve.data)
        log(f"table import: {len(back)} rows re-read, {cmp.n_compared} compared with the curve, max |diff| {cmp.max_abs:.4g}")
        ok = ok and len(back) == len(table) and cmp.n_compared > 5 and cmp.max_abs < 0.01
    log("SELFTEST OK" if ok else "SELFTEST FAILED: unexpected values")
    return 0 if ok else 1
