"""Build a single-file Windows executable:  ``python build_exe.py``  ->  ``dist/Curvelens.exe``.

Needs PyInstaller (``pip install pyinstaller``).  The .exe unpacks itself to a temporary folder
on every start, so the first window takes a few seconds.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAME = "Curvelens"

# Nothing the application uses lives in these; keeping them out shrinks the file.
EXCLUDES = ["tkinter", "pytest", "IPython", "matplotlib", "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets",
            "PySide6.QtQml", "PySide6.QtQuick", "PySide6.Qt3DCore", "PySide6.QtMultimedia", "PySide6.QtSql",
            "PySide6.QtCharts", "PySide6.QtDataVisualization", "PySide6.QtPdf", "PySide6.QtPdfWidgets"]


def make_icon(target: Path) -> Path | None:
    """Write the application icon as .ico (drawn by ui/appicon.py)."""
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONPATH=str(ROOT))
    code = ("from PySide6.QtWidgets import QApplication;import sys;a=QApplication(sys.argv);"
            "from ui.appicon import make_pixmap;import sys as s;ok=make_pixmap(256).save(r'%s');s.exit(0 if ok else 1)" % target)
    res = subprocess.run([sys.executable, "-c", code], env=env, cwd=ROOT)
    return target if res.returncode == 0 and target.exists() else None


def main() -> int:
    build = ROOT / "build"
    build.mkdir(exist_ok=True)
    icon = make_icon(build / "icon.ico")
    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onefile", "--windowed",
           "--name", NAME, "--distpath", str(ROOT / "dist"), "--workpath", str(build / "pyinstaller"),
           "--specpath", str(build),
           "--paths", str(ROOT),
           "--collect-all", "pypdfium2", "--collect-all", "pypdfium2_raw", "--collect-all", "pypdfium2_cfg",
           "--hidden-import", "scipy.signal", "--hidden-import", "scipy.spatial"]
    for mod in EXCLUDES:
        cmd += ["--exclude-module", mod]
    if icon:
        cmd += ["--icon", str(icon)]
    cmd.append(str(ROOT / "app.py"))
    print(" ".join(cmd))
    return subprocess.call(cmd, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
