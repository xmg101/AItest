import subprocess
import threading
import io
import numpy as np
from pathlib import Path
from loguru import logger
from PIL import Image
import config


class ADBDevice:
    def __init__(self, serial: str = ""):
        self.serial = serial or config.DEVICE_SERIAL
        self._logcat_proc = None
        self._logcat_lines: list[str] = []
        self._logcat_lock = threading.Lock()

    def _adb(self, *args, capture=True) -> subprocess.CompletedProcess:
        cmd = [config.ADB_PATH]
        if self.serial:
            cmd += ["-s", self.serial]
        cmd += list(args)
        result = subprocess.run(cmd, capture_output=capture, timeout=30)
        if result.returncode != 0 and capture:
            logger.warning(f"ADB command failed: {' '.join(args)}\n{result.stderr.decode(errors='replace')}")
        return result

    def connect(self, serial: str = "") -> bool:
        if serial:
            self.serial = serial
        result = self._adb("get-state")
        ok = result.returncode == 0
        logger.info(f"Device {'connected' if ok else 'NOT connected'}: {self.serial or 'default'}")
        return ok

    def disconnect(self):
        self.stop_logcat()

    def screenshot(self) -> np.ndarray:
        result = self._adb("exec-out", "screencap", "-p")
        if result.returncode != 0:
            raise RuntimeError("Screenshot failed")
        img = Image.open(io.BytesIO(result.stdout))
        return np.array(img.convert("RGB"))

    def save_screenshot(self, path: str | Path = "screenshot.png") -> Path:
        arr = self.screenshot()
        img = Image.fromarray(arr)
        path = Path(path)
        if path.parent == Path("."):
            path = config.SCREENSHOTS_DIR / path
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(path)
        return path

    def start_logcat(self, tag_filter: str = ""):
        if self._logcat_proc:
            return
        cmd = [config.ADB_PATH]
        if self.serial:
            cmd += ["-s", self.serial]
        cmd += ["logcat", "-v", "threadtime"]
        if tag_filter:
            cmd += [tag_filter]
        self._logcat_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self._logcat_lines.clear()

        def _reader():
            for line in self._logcat_proc.stdout:
                with self._logcat_lock:
                    self._logcat_lines.append(line.decode(errors="replace").rstrip())

        threading.Thread(target=_reader, daemon=True).start()
        logger.info("logcat started")

    def stop_logcat(self):
        if self._logcat_proc:
            self._logcat_proc.terminate()
            self._logcat_proc = None
            logger.info("logcat stopped")

    def get_logs(self) -> list[str]:
        with self._logcat_lock:
            return list(self._logcat_lines)

    def clear_logcat(self):
        self._adb("logcat", "-c")
        with self._logcat_lock:
            self._logcat_lines.clear()

    def install_apk(self, apk_path: str | Path) -> bool:
        result = self._adb("install", "-r", str(apk_path))
        ok = result.returncode == 0
        logger.info(f"APK install {'OK' if ok else 'FAILED'}: {apk_path}")
        return ok

    def launch_app(self, package: str, activity: str = "") -> bool:
        if activity:
            component = f"{package}/{activity}"
        else:
            component = package
        result = self._adb("shell", "monkey", "-p", package, "-c", "android.intent.category.LAUNCHER", "1")
        ok = result.returncode == 0
        logger.info(f"Launch {'OK' if ok else 'FAILED'}: {component}")
        return ok

    def force_stop(self, package: str):
        self._adb("shell", "am", "force-stop", package)

    def get_device_info(self) -> dict:
        def prop(name):
            r = self._adb("shell", "getprop", name)
            return r.stdout.decode(errors="replace").strip() if r.returncode == 0 else ""

        return {
            "serial": self.serial,
            "model": prop("ro.product.model"),
            "android": prop("ro.build.version.release"),
            "sdk": prop("ro.build.version.sdk"),
            "resolution": self._get_resolution(),
        }

    def _get_resolution(self) -> str:
        r = self._adb("shell", "wm", "size")
        if r.returncode == 0:
            return r.stdout.decode(errors="replace").strip().split(":")[-1].strip()
        return "unknown"
