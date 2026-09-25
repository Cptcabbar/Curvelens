import os

# GUI tests run without a display; must be set before PySide6 creates the application.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
