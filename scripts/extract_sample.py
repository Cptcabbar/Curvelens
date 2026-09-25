"""Extract the curves of one chart of a PDF (or image) and write a debug overlay.

Examples (from the project root)::

    python -m scripts.extract_sample                       # datasheet chart "Discharge Rate"
    python -m scripts.extract_sample --list                # show the charts found in the PDF
    python -m scripts.extract_sample --make-sample-png     # (re)write samples/discharge.png
    python -m scripts.extract_sample --csv out/            # also export the curves
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.export import AxisInfo, export_combined                     # noqa: E402
from core.extraction import extract_curve, pick_reference_color        # noqa: E402
from core.models import ExtractionParams                               # noqa: E402
from core.pdf_source import PdfSource                                  # noqa: E402
from core.postprocess import resample, to_curve_data                   # noqa: E402

DEFAULT_PDF = ROOT / "samples" / "INR18650P28A-V1-80093.pdf"
DEFAULT_CURVES = {
    "0.56A": (255, 0, 0),
    "2.8A": (0, 255, 0),
    "10A": (0, 0, 255),
    "20A": (0, 0, 0),
    "30A": (255, 0, 255),
}


def parse_curve(spec: str) -> tuple[str, tuple[int, int, int]]:
    name, _, color = spec.partition("=")
    color = color.lstrip("#")
    if len(color) != 6:
        raise argparse.ArgumentTypeError("beklenen biçim: ISIM=RRGGBB")
    return name, tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))


def draw_overlay(image_bgr: np.ndarray, traces: dict[str, tuple[tuple[int, int, int], np.ndarray, np.ndarray]],
                 plot_area=None) -> np.ndarray:
    """Washed-out image with every trace drawn on top in its own colour."""
    out = cv2.addWeighted(image_bgr, 0.45, np.full_like(image_bgr, 255), 0.55, 0)
    if plot_area is not None:
        p0 = (int(round(plot_area.x0)), int(round(plot_area.y0)))
        p1 = (int(round(plot_area.x1)), int(round(plot_area.y1)))
        cv2.rectangle(out, p0, p1, (160, 160, 160), 1, cv2.LINE_AA)
    for name, (rgb, xs, ys) in traces.items():
        bgr = (rgb[2], rgb[1], rgb[0])
        for x, y in zip(xs, ys):
            cv2.circle(out, (int(round(x)), int(round(y))), 2, bgr, -1, cv2.LINE_AA)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    ap.add_argument("--page", type=int, default=1, help="1-based page number")
    ap.add_argument("--chart", default="Discharge Rate", help="chart index (0-based) or part of its title")
    ap.add_argument("--dpi", type=float, default=600)
    ap.add_argument("--delta-e", type=float, default=ExtractionParams().delta_e)
    ap.add_argument("--curve", action="append", type=parse_curve, help="NAME=RRGGBB (repeatable)")
    ap.add_argument("--out", type=Path, default=ROOT / "debug_overlay.png")
    ap.add_argument("--csv", type=Path, help="write one combined CSV to this directory")
    ap.add_argument("--step", type=float, default=10.0, help="x step for the CSV resampling")
    ap.add_argument("--list", action="store_true", help="only list the charts of the page")
    ap.add_argument("--make-sample-png", action="store_true", help="write samples/discharge.png and exit")
    args = ap.parse_args(argv)

    src = PdfSource(args.pdf)
    page = args.page - 1
    charts = src.charts(page)
    if args.list:
        for c in charts:
            axes = ", ".join(a.title or a.side for a in c.x_axes + c.y_axes)
            print(f"[{c.index}] {c.title!r}  axes: {axes}")
        return 0
    if args.chart.isdigit():
        chart = charts[int(args.chart)]
    else:
        matches = [c for c in charts if args.chart.lower() in c.title.lower()]
        if not matches:
            print(f"Grafik bulunamadı: {args.chart!r}", file=sys.stderr)
            return 2
        chart = matches[0]
    print(f"Grafik: {chart.label}")

    rendered = src.render(page, chart.region, dpi=args.dpi)
    print(f"Görüntü: {rendered.image.shape[1]}x{rendered.image.shape[0]} px @ {args.dpi:g} DPI")
    if args.make_sample_png:
        target = ROOT / "samples" / "discharge.png"
        cv2.imwrite(str(target), rendered.image)
        print(f"Yazıldı: {target}")
        return 0

    cal = src.suggest_calibration(chart, rendered)
    if cal is None:
        print("PDF metin katmanından kalibrasyon önerilemedi.", file=sys.stderr)
        return 2
    print(f"X ekseni: {cal.x.v1:g} -> {cal.x.v2:g}   Y ekseni: {cal.y.v1:g} -> {cal.y.v2:g}")
    plot = rendered.rect_pt_to_px(chart.plot_rect)

    curves = dict(args.curve) if args.curve else DEFAULT_CURVES
    params = ExtractionParams(delta_e=args.delta_e)
    traces, data = {}, []
    for name, rgb in curves.items():
        res = extract_curve(rendered.image, rgb, plot, params=params)
        if len(res) == 0:
            print(f"  {name:8s} çıkarılamadı ({res.stats.get('reason')})")
            continue
        traces[name] = (rgb, res.x_px, res.y_px)
        cd = to_curve_data(name, res.x_px, res.y_px, cal)
        data.append(cd)
        print(f"  {name:8s} {len(res):5d} nokta   x: {cd.x[0]:8.1f} .. {cd.x[-1]:8.1f}   "
              f"y: {cd.y.max():.3f} .. {cd.y.min():.3f}   ±x={cd.x_unc[0]:.3g} ±y={cd.y_unc[0]:.3g}")

    overlay = draw_overlay(rendered.image, traces, plot)
    cv2.imwrite(str(args.out), overlay)
    print(f"Yazıldı: {args.out}")

    if args.csv and data:
        xa = AxisInfo(chart.x_axes[0].title.split("(")[0].strip(), chart.x_axes[0].unit)
        ya = AxisInfo(chart.y_axes[0].title.split("(")[0].strip(), chart.y_axes[0].unit)
        grid = [resample(c, args.step, max_gap=5 * args.step) for c in data]
        path = export_combined(grid, args.csv / "curves.csv", xa, ya, source=f"{args.pdf.name} / {chart.title}",
                               step=args.step)
        print(f"Yazıldı: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
