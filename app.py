"""Curvelens entry point.

    python app.py [file.pdf]                     PDF -> gallery of charts -> curves -> lookup table
    python app.py file.png                       chart picture (PNG/JPG/BMP...): calibrate + read its curves
    python app.py --manual [file]                the manual image digitiser
    python app.py --selftest [file.pdf]          headless check of the processing stack, then exit
    python app.py --screenshot out.png [file.pdf] [--page gallery|detail] [--chart N] [--curve M]

``--log FILE`` sends the console output of the two check modes to a file (the packaged,
windowed .exe has no console).
"""
from __future__ import annotations

import argparse
import sys

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def make_logger(path: str | None):
    """``print`` replacement that also works without a console and can append to a file."""
    def log(*parts) -> None:
        text = " ".join(str(p) for p in parts)
        if sys.stdout is not None:
            try:
                print(text)
            except Exception:                                   # closed / unencodable console
                pass
        if path:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(text + "\n")
    return log


def _screenshot(app, args, log) -> int:
    """Open a window in the requested state, save a screenshot of it and exit."""
    size = (1500, 900)
    if args.manual:
        from ui.main_window import MainWindow

        win = MainWindow()
        win.resize(*size)
        win.show()
        if args.file:
            win.open_file(args.file)
        if args.demo:
            for name, rgb in (("0.56A", (255, 0, 0)), ("2.8A", (0, 255, 0)), ("10A", (0, 0, 255)),
                              ("20A", (0, 0, 0)), ("30A", (255, 0, 255))):
                win.project.add_curve_with_color(rgb, name)
            win.extract_all()
            win._selected = 0
            win._refresh_all()
        state = f"curves={[len(c) for c in win.project.curves]}"
    else:
        from ui.lookup_window import LookupWindow

        win = LookupWindow()
        win.resize(*size)
        win.show()
        if args.file:
            win.open_pdf(args.file, blocking=True)
            if args.page == "detail" and win.charts:
                win.show_chart(min(args.chart, len(win.charts) - 1))
                chart = win.charts[min(args.chart, len(win.charts) - 1)]
                if chart.curves:
                    win.curve_list.setCurrentRow(min(args.curve, len(chart.curves) - 1))
        state = f"charts={len(win.charts)} page={win.stack.currentIndex()} rows={len(win._table) if win._table else 0}"
    app.processEvents()
    ok = win.grab().save(args.screenshot)
    log(f"screenshot {'saved' if ok else 'FAILED'}: {args.screenshot}  {state}")
    win.close()
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    parser = argparse.ArgumentParser(prog="Curvelens", add_help=True)
    parser.add_argument("file", nargs="?", help="PDF or chart image (PNG, JPG, BMP, TIFF, WebP) to open")
    parser.add_argument("--manual", action="store_true", help="open the manual image digitiser instead")
    parser.add_argument("--selftest", action="store_true", help="headless check of the processing stack, then exit")
    parser.add_argument("--screenshot", metavar="OUT.png", help="save a window screenshot and exit")
    parser.add_argument("--page", choices=("gallery", "detail"), default="detail", help="with --screenshot")
    parser.add_argument("--chart", type=int, default=0, help="with --screenshot --page detail: chart index")
    parser.add_argument("--curve", type=int, default=0, help="with --screenshot --page detail: curve index")
    parser.add_argument("--demo", action="store_true", help="with --manual --screenshot: add the five sample curves")
    parser.add_argument("--log", metavar="FILE", help="append the output of --selftest/--screenshot to FILE")
    args, qt_args = parser.parse_known_args(argv[1:])
    log = make_logger(args.log)

    if args.selftest:
        from core.selftest import run

        try:
            return run(args.file, log)
        except Exception as exc:                                   # report import/runtime failures of a frozen build
            import traceback

            log("SELFTEST FAILED:", repr(exc))
            log(traceback.format_exc())
            return 1

    from PySide6.QtWidgets import QApplication

    from ui.appicon import make_icon
    from ui.lookup_window import APP_NAME

    app = QApplication([argv[0], *qt_args])
    app.setApplicationName(APP_NAME)
    app.setOrganizationName("Curvelens")
    app.setStyle("Fusion")
    app.setWindowIcon(make_icon())

    if args.screenshot:
        return _screenshot(app, args, log)

    manual = args.manual
    if manual:
        from ui.main_window import MainWindow

        win = MainWindow()
        win.show()
        if args.file:
            win.open_file(args.file)
    else:
        from ui.lookup_window import LookupWindow

        win = LookupWindow()
        win.show()
        if args.file:
            win.open_pdf(args.file)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
