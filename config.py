import os
import sys
import io
from pathlib import Path

# Windows 控制台强制 UTF-8，避免中文乱码
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).parent

ADB_PATH = os.environ.get("ADB_PATH", "adb")
DEVICE_SERIAL = os.environ.get("DEVICE_SERIAL", "")  # empty = first connected device

DEFAULT_TIMEOUT = float(os.environ.get("DEFAULT_TIMEOUT", "10"))
DEFAULT_INTERVAL = 0.5
DEFAULT_THRESHOLD = float(os.environ.get("DEFAULT_THRESHOLD", "0.8"))

REPORTS_DIR = BASE_DIR / "reports"
RECORDINGS_DIR = BASE_DIR / "recordings"
TEMPLATES_DIR = BASE_DIR / "assets" / "templates"
SCREENSHOTS_DIR = BASE_DIR / "screenshots"

REPORTS_DIR.mkdir(exist_ok=True)
RECORDINGS_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
SCREENSHOTS_DIR.mkdir(exist_ok=True)

OCR_LANGUAGES = ["ch_sim", "en"]

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
AI_MODEL = os.environ.get("AI_MODEL", "claude-sonnet-4-6")
