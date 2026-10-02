import os
import tempfile

# GUI tests run without a display; must be set before PySide6 creates the application.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Remembered calibrations go to a throw-away folder: tests must never touch the user's real data.
os.environ["CURVELENS_DATA"] = tempfile.mkdtemp(prefix="curvelens-test-")
