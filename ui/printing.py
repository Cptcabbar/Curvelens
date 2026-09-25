"""Turn a lookup table (HTML report) into a PDF file or send it to a printer."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QMarginsF, QSizeF
from PySide6.QtGui import QPageLayout, QPageSize, QTextDocument
from PySide6.QtPrintSupport import QPrintDialog, QPrinter


def _configure(printer: QPrinter) -> None:
    printer.setPageSize(QPageSize(QPageSize.PageSizeId.A4))
    printer.setPageMargins(QMarginsF(15, 15, 15, 15), QPageLayout.Unit.Millimeter)


def _print_html(html_text: str, printer: QPrinter) -> None:
    """Lay the report out on the printer's page and print it.

    The page size of the document is set explicitly (in points): left to itself ``QTextDocument`` lays the
    text out in a much narrower area, which shows up as a ~35 mm left margin and half-empty pages.
    """
    doc = QTextDocument()
    doc.setHtml(html_text)
    rect = printer.pageRect(QPrinter.Unit.Point)
    # Qt lays the text out in 1/96 inch units but breaks pages by this height: 4/3 of the height in points fills the
    # page without running into the bottom margin (measured on a 400-row table; 1.0 wastes a quarter of every page).
    doc.setPageSize(QSizeF(rect.width(), rect.height() * 96.0 / 72.0))
    doc.print_(printer)


def save_pdf(html_text: str, path: str | Path) -> Path:
    """Render the report to a PDF file (A4, 15 mm margins; long tables continue on further pages)."""
    printer = QPrinter(QPrinter.PrinterMode.HighResolution)
    printer.setOutputFormat(QPrinter.OutputFormat.PdfFormat)
    printer.setOutputFileName(str(path))
    _configure(printer)
    _print_html(html_text, printer)
    return Path(path)


def print_with_dialog(html_text: str, parent=None) -> bool:
    """Show the system print dialog and print the report; ``False`` when the user cancels."""
    printer = QPrinter(QPrinter.PrinterMode.HighResolution)
    _configure(printer)
    dialog = QPrintDialog(printer, parent)
    dialog.setWindowTitle("Lookup tablosunu yazdır")
    if dialog.exec() != QPrintDialog.DialogCode.Accepted:
        return False
    _print_html(html_text, printer)
    return True
