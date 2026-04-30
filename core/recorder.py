import json
import time
import threading
import subprocess
from pathlib import Path
from loguru import logger
import config


class ActionRecorder:
    """
    Records tap/swipe gestures by polling ADB getevent and converts to JSON.
    Playback replays via UIController with optional speed scaling.
    """

    def __init__(self, device, controller):
        self.device = device
        self.controller = controller
        self._actions: list[dict] = []
        self._recording = False
        self._start_time = 0.0
        self._thread: threading.Thread | None = None
        self._proc: subprocess.Popen | None = None

    def start(self):
        if self._recording:
            logger.warning("Already recording")
            return
        self._actions.clear()
        self._recording = True
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info("Recording started")

    def stop(self):
        self._recording = False
        if self._proc:
            self._proc.terminate()
            self._proc = None
        if self._thread:
            self._thread.join(timeout=2)
        logger.info(f"Recording stopped — {len(self._actions)} actions captured")

    def save(self, name: str) -> Path:
        path = config.RECORDINGS_DIR / f"{name}.json"
        path.write_text(json.dumps(self._actions, indent=2, ensure_ascii=False))
        logger.info(f"Recording saved: {path}")
        return path

    def load(self, name: str) -> list[dict]:
        path = config.RECORDINGS_DIR / f"{name}.json"
        actions = json.loads(path.read_text())
        logger.info(f"Loaded {len(actions)} actions from {path}")
        return actions

    def playback(self, actions: list[dict] | str, speed: float = 1.0):
        if isinstance(actions, str):
            actions = self.load(actions)
        logger.info(f"Playback: {len(actions)} actions at {speed}x speed")
        prev_t = 0.0
        for act in actions:
            delay = (act.get("t", 0) - prev_t) / speed
            if delay > 0:
                time.sleep(delay)
            prev_t = act.get("t", 0)
            atype = act.get("type")
            if atype == "tap":
                self.controller.tap(act["x"], act["y"])
            elif atype == "swipe":
                self.controller.swipe(act["x1"], act["y1"], act["x2"], act["y2"],
                                      int(act.get("duration_ms", 300) / speed))
            elif atype == "long_press":
                self.controller.long_press(act["x"], act["y"],
                                           int(act.get("duration_ms", 1000) / speed))
            elif atype == "key":
                self.controller.press_key(act["keycode"])

    def _capture_loop(self):
        cmd = [config.ADB_PATH]
        if self.device.serial:
            cmd += ["-s", self.device.serial]
        cmd += ["shell", "getevent", "-lt"]
        try:
            self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        except Exception as e:
            logger.error(f"getevent failed: {e}")
            return

        pending: dict = {}

        for raw_line in self._proc.stdout:
            if not self._recording:
                break
            line = raw_line.decode(errors="replace").strip()
            t = time.time() - self._start_time

            if "ABS_MT_POSITION_X" in line:
                try:
                    pending["x"] = int(line.split()[-1], 16)
                except ValueError:
                    pass
            elif "ABS_MT_POSITION_Y" in line:
                try:
                    pending["y"] = int(line.split()[-1], 16)
                except ValueError:
                    pass
            elif "BTN_TOUCH" in line and "DOWN" in line:
                pending["t_down"] = t
            elif "BTN_TOUCH" in line and "UP" in line:
                if "x" in pending and "y" in pending:
                    duration = (t - pending.get("t_down", t)) * 1000
                    if duration > 400:
                        self._actions.append({
                            "type": "long_press",
                            "x": pending["x"],
                            "y": pending["y"],
                            "duration_ms": int(duration),
                            "t": pending.get("t_down", t),
                        })
                    else:
                        self._actions.append({
                            "type": "tap",
                            "x": pending["x"],
                            "y": pending["y"],
                            "t": pending.get("t_down", t),
                        })
                    logger.debug(f"Recorded: {self._actions[-1]}")
                pending = {}
