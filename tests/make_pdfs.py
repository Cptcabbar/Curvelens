"""Vector PDFs with a *known* answer, produced by matplotlib (test helper).

Each builder returns ``(path, truth)`` where ``truth`` maps a label to the exact ``(x, y)`` arrays that
were plotted, so the digitiser can be checked against the real data, not just against itself.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42          # embedded TrueType: text stays extractable
matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["path.simplify"] = False      # keep every plotted vertex in the PDF
import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages               # noqa: E402
from matplotlib.ticker import FuncFormatter                        # noqa: E402


def _save(fig, path: Path) -> Path:
    fig.savefig(path, format="pdf")
    plt.close(fig)
    return path


def simple_lines(path: Path):
    """Solid + dashed + solid curves with legend, title and axis titles (linear axes)."""
    x = np.linspace(0, 100, 201)
    truth = {
        "Motor A": (x, 5.0 * np.exp(-x / 60.0)),
        "Motor B": (x, 4.0 - 0.02 * x),
        "Motor C": (x, 1.0 + 0.5 * np.sin(x / 12.0)),
    }
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(*truth["Motor A"], color="#d62728", lw=1.6, label="Motor A")
    ax.plot(*truth["Motor B"], color="#1f77b4", lw=1.6, ls="--", label="Motor B")
    ax.plot(*truth["Motor C"], color="#2ca02c", lw=1.6, label="Motor C")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 6)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title("Motor test")
    ax.grid(True, ls=":")
    ax.legend(loc="upper right")
    return _save(fig, path), truth


def log_axes(path: Path):
    """Both axes logarithmic with plain tick labels (1, 10, 100, ...)."""
    x = np.logspace(0, 3, 120)
    truth = {"Gain": (x, 100.0 / (1.0 + (x / 30.0) ** 2))}
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.loglog(*truth["Gain"], color="#9467bd", lw=1.8, label="Gain")
    ax.set_xlim(1, 1000)
    ax.set_ylim(0.1, 1000)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:g}"))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Gain (V/V)")
    ax.set_title("Bode")
    ax.grid(True, which="major")
    ax.legend(loc="lower left")
    return _save(fig, path), truth


def twin_axes(path: Path):
    """Left blue axis and right red axis (axis colour = curve colour)."""
    x = np.linspace(0, 10, 101)
    truth = {"Current": (x, 2.0 + 0.3 * x), "Temperature": (x, 20.0 + 6.0 * x ** 0.8)}
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax2 = ax.twinx()
    ax.plot(*truth["Current"], color="blue", lw=1.5)
    ax2.plot(*truth["Temperature"], color="red", lw=1.5)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax2.set_ylim(0, 100)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Current (A)", color="blue")
    ax2.set_ylabel("Temperature (C)", color="red")
    ax.tick_params(axis="y", colors="blue")
    ax2.tick_params(axis="y", colors="red")
    ax.set_title("Thermal")
    return _save(fig, path), truth


def markers(path: Path):
    """Two marker-only series (markers are drawn as form XObjects by matplotlib)."""
    n = np.arange(1, 41)
    truth = {"Cell 1": (n.astype(float), 100.0 - 0.4 * n), "Cell 2": (n.astype(float), 96.0 - 0.7 * n)}
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(*truth["Cell 1"], "o", color="black", ms=4, label="Cell 1")
    ax.plot(*truth["Cell 2"], "s", color="#ff7f0e", ms=4, label="Cell 2")
    ax.set_xlim(0, 45)
    ax.set_ylim(60, 105)
    ax.set_xlabel("Cycle (n)")
    ax.set_ylabel("Capacity (%)")
    ax.set_title("Ageing")
    ax.legend(loc="lower left")
    return _save(fig, path), truth


def two_charts(path: Path):
    """Two stacked charts on one page, each with its own title."""
    x = np.linspace(0, 10, 101)
    truth = {"P": (x, x ** 2), "Q": (x, 50.0 - 4.0 * x)}
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.4, 8.0))
    a1.plot(*truth["P"], color="tab:green", lw=1.5, label="P")
    a1.set_title("First chart")
    a1.set_xlabel("Distance (m)")
    a1.set_ylabel("Power (W)")
    a1.set_xlim(0, 10)
    a1.set_ylim(0, 100)
    a1.legend(loc="upper left")
    a2.plot(*truth["Q"], color="tab:red", lw=1.5, label="Q")
    a2.set_title("Second chart")
    a2.set_xlabel("Distance (m)")
    a2.set_ylabel("Level (L)")
    a2.set_xlim(0, 10)
    a2.set_ylim(0, 60)
    a2.legend(loc="upper right")
    fig.tight_layout()
    return _save(fig, path), truth


def two_pages(path: Path):
    """One chart on each of two pages."""
    x = np.linspace(0, 5, 51)
    truth = {"S1": (x, 3.0 * x), "S2": (x, 20.0 - 2.0 * x)}
    with PdfPages(path) as pdf:
        for name, title in (("S1", "Page one"), ("S2", "Page two")):
            fig, ax = plt.subplots(figsize=(6.4, 4.2))
            ax.plot(*truth[name], color="tab:blue", lw=1.5, label=name)
            ax.set_xlim(0, 5)
            ax.set_ylim(0, 25)
            ax.set_title(title)
            ax.set_xlabel("Length (cm)")
            ax.set_ylabel("Mass (g)")
            ax.legend()
            pdf.savefig(fig)
            plt.close(fig)
    return path, truth


GREEN, BLUE, RED = "#20a040", "#2040d0", "#e02020"


def _picture_bitmap(dual: bool = False, same_colour: bool = False):
    """The bitmap (1200x760 px RGB) of a chart plus its exact truth; see :func:`picture_chart`.

    A heading and a note line above it are real PDF text.  ``dual`` adds a right Y axis (0..100, green) with a
    green curve; ``same_colour`` draws that right-axis curve in the blue of a left-axis curve (a voltage and a
    temperature curve of one colour).  Returns ``(path, truth)``: the exact curves plus the exact pixel positions
    (continuous picture coordinates) of the tick marks.
    """
    x = np.linspace(0, 2000, 400)
    curves = {"red": (x, 4.2 - 0.0006 * x), "blue": (x, 3.9 - 0.0003 * x - 1.5e-7 * x ** 2),
              "black": (x, 3.4 - 0.0004 * x)}
    fig, ax = plt.subplots(figsize=(6.0, 3.8), dpi=200)
    fig.subplots_adjust(left=0.13, right=0.87 if dual else 0.96, top=0.95, bottom=0.16)
    for name, colour in (("red", RED), ("blue", BLUE), ("black", "black")):
        ax.plot(*curves[name], color=colour, lw=2.2)
    ax.set_xlim(0, 2000)
    ax.set_ylim(2.0, 4.5)
    ax.set_xticks([0, 500, 1000, 1500, 2000])
    ax.set_yticks([2.0, 2.5, 3.0, 3.5, 4.0, 4.5])
    ax.grid(True, ls="--", color="#909090", lw=0.8)
    ax.tick_params(direction="out", length=6, width=1.5)
    for s in ax.spines.values():
        s.set_linewidth(2.0)
    ax.set_xlabel("Capacity (mAh)")
    ax.set_ylabel("Voltage (V)")
    axes2 = None
    if dual:
        axes2 = ax.twinx()
        colour2 = BLUE if same_colour else GREEN
        curves["temp"] = (x, 20.0 + 0.03 * x)
        axes2.plot(*curves["temp"], color=colour2, lw=2.2)
        axes2.set_ylim(0, 100)
        axes2.set_yticks([0, 20, 40, 60, 80, 100])
        axes2.tick_params(direction="out", length=6, width=1.5, colors=colour2)
        axes2.set_ylabel("Temperature (C)", color=colour2)
        for s in axes2.spines.values():
            s.set_linewidth(2.0)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    h, w = img.shape[:2]

    def pos(a, x_val=None, y_val=None):
        px, py = a.transData.transform((0.0 if x_val is None else x_val, 0.0 if y_val is None else y_val))
        return float(px) if x_val is not None else float(h - py)

    truth = {
        "curves": curves, "shape": (h, w),
        "x_ticks": [(v, pos(ax, x_val=v)) for v in (0, 500, 1000, 1500, 2000)],
        "y_ticks": [(v, pos(ax, y_val=v)) for v in (2.0, 2.5, 3.0, 3.5, 4.0, 4.5)],
        "y2_ticks": [(v, pos(axes2, y_val=v)) for v in (0, 20, 40, 60, 80, 100)] if axes2 is not None else [],
        "frame": (pos(ax, x_val=0), pos(ax, y_val=4.5), pos(ax, x_val=2000), pos(ax, y_val=2.0)),
        "to_px": lambda name, xv: (pos(ax, x_val=xv),
                                   pos(axes2 if name == "temp" else ax, y_val=float(np.interp(xv, *curves[name])))),
    }
    plt.close(fig)
    return img, truth


def picture_chart(path: Path, dual: bool = False, same_colour: bool = False):
    """A page whose chart is only a *picture* (a 1200x760 px PNG pasted into the PDF), like many datasheets.

    A heading and a note line above it are real PDF text.  ``dual`` adds a right Y axis (0..100, green) with a
    green curve; ``same_colour`` draws that right-axis curve in the blue of a left-axis curve (a voltage and a
    temperature curve of one colour).  Returns ``(path, truth)``: the exact curves plus the exact pixel positions
    (continuous picture coordinates) of the tick marks.
    """
    img, truth = _picture_bitmap(dual, same_colour)
    h, w = img.shape[:2]
    page = plt.figure(figsize=(8.27, 11.69), dpi=72)          # A4, 1 unit = 1 point
    page.text(0.12, 0.918, "Picture chart", fontsize=11, weight="bold")
    page.text(0.12, 0.903, "Discharge at 25 C (2.5 V)", fontsize=9)
    frac_h = 0.76 * 595.0 / (w / h) / 842.0                   # keep the picture's aspect ratio
    ax_img = page.add_axes([0.12, 0.895 - frac_h, 0.76, frac_h])
    ax_img.axis("off")
    ax_img.imshow(img, interpolation="none", aspect="auto")
    return _save(page, path), truth


def picture_file(path: Path, dual: bool = False, same_colour: bool = False, frame: bool = True, quality: int = 92):
    """The chart as an image *file* (format from the suffix: .png, .jpg, .bmp, ...).  Returns ``(path, truth)``.

    ``frame=False`` blanks the frame lines (a chart without a box) so that the plot area has to be drawn by hand.
    ``quality`` is the JPEG quality.
    """
    import cv2

    img, truth = _picture_bitmap(dual, same_colour)
    if not frame:
        x0, y0, x1, y1 = (int(round(v)) for v in truth["frame"])
        pad = 4
        for a, b, c, d in ((x0 - pad, y0 - pad, x1 + pad, y0 + pad), (x0 - pad, y1 - pad, x1 + pad, y1 + pad),
                           (x0 - pad, y0 - pad, x0 + pad, y1 + pad), (x1 - pad, y0 - pad, x1 + pad, y1 + pad)):
            region = img[max(b, 0):d, max(a, 0):c]
            region[:] = 255
    bgr = np.ascontiguousarray(img[..., ::-1])
    ok = cv2.imencode(Path(path).suffix, bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])[1]
    Path(path).write_bytes(ok.tobytes())
    return path, truth
