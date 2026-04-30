import time
import subprocess
from loguru import logger
import config

# Touchscreen input device path (auto-detected on first tap)
_TOUCH_DEVICE: str = ""


def _detect_touch_device(device) -> str:
    """Find the /dev/input/eventX path that has ABS_MT_POSITION_X (touchscreen)."""
    result = device._adb("shell", "getevent", "-p")
    output = result.stdout.decode(errors="replace")
    current_dev = ""
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("add device"):
            current_dev = line.split(": ", 1)[-1].strip()
        elif "0035" in line and current_dev:  # ABS_MT_POSITION_X
            logger.info(f"Touch device detected: {current_dev}")
            return current_dev
    return "/dev/input/event4"  # fallback


class UIController:
    def __init__(self, device):
        self.device = device
        self._touch_dev = ""

    def _get_touch_dev(self) -> str:
        if not self._touch_dev:
            self._touch_dev = _detect_touch_device(self.device)
        return self._touch_dev

    def tap(self, x: int, y: int):
        logger.debug(f"tap ({x}, {y})")
        self.device._adb("shell", "input", "tap", str(x), str(y))

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300):
        logger.debug(f"swipe ({x1},{y1}) -> ({x2},{y2}) {duration_ms}ms")
        self.device._adb("shell", "input", "swipe",
                         str(x1), str(y1), str(x2), str(y2), str(duration_ms))

    # ── Raw sendevent methods (bypass adb input layer, works for Unity) ────────

    def raw_tap(self, x: int, y: int, hold_ms: int = 120):
        """Sendevent-based tap — directly injects kernel touch events."""
        dev = self._get_touch_dev()
        hold_us = hold_ms * 1000
        parts = [
            f"sendevent {dev} 3 47 0",      # ABS_MT_SLOT 0
            f"sendevent {dev} 3 57 1",      # ABS_MT_TRACKING_ID 1
            f"sendevent {dev} 3 53 {x}",    # ABS_MT_POSITION_X
            f"sendevent {dev} 3 54 {y}",    # ABS_MT_POSITION_Y
            f"sendevent {dev} 1 330 1",     # BTN_TOUCH=1
            f"sendevent {dev} 0 0 0",       # SYN_REPORT
            f"usleep {hold_us}",
            f"sendevent {dev} 3 47 0",
            f"sendevent {dev} 3 57 4294967295",  # ABS_MT_TRACKING_ID=-1 (lift)
            f"sendevent {dev} 1 330 0",     # BTN_TOUCH=0
            f"sendevent {dev} 0 0 0",
        ]
        logger.debug(f"raw_tap ({x},{y}) dev={dev}")
        self.device._adb("shell", "; ".join(parts))

    def raw_swipe(self, x1: int, y1: int, x2: int, y2: int,
                  duration_ms: int = 600, steps: int = 15):
        """Sendevent-based swipe — directly injects kernel touch events.

        Uses a long initial hold so Unity's drag threshold activates before movement.
        """
        dev = self._get_touch_dev()
        hold_us = 300_000   # 300ms press before moving (activates Unity drag)
        remaining_ms = max(100, duration_ms - 300)
        step_us = max(8_000, remaining_ms * 1000 // steps)

        parts = [
            # Finger down
            f"sendevent {dev} 3 47 0",
            f"sendevent {dev} 3 57 1",
            f"sendevent {dev} 3 53 {x1}",
            f"sendevent {dev} 3 54 {y1}",
            f"sendevent {dev} 1 330 1",
            f"sendevent {dev} 0 0 0",
            f"usleep {hold_us}",
        ]

        for i in range(1, steps + 1):
            t = i / steps
            xi = int(x1 + (x2 - x1) * t)
            yi = int(y1 + (y2 - y1) * t)
            parts += [
                f"sendevent {dev} 3 53 {xi}",
                f"sendevent {dev} 3 54 {yi}",
                f"sendevent {dev} 0 0 0",
                f"usleep {step_us}",
            ]

        parts += [
            # Finger up
            f"sendevent {dev} 3 47 0",
            f"sendevent {dev} 3 57 4294967295",
            f"sendevent {dev} 1 330 0",
            f"sendevent {dev} 0 0 0",
        ]

        logger.debug(f"raw_swipe ({x1},{y1})->({x2},{y2}) dev={dev} steps={steps}")
        self.device._adb("shell", "; ".join(parts))

    def long_press(self, x: int, y: int, duration_ms: int = 1000):
        logger.debug(f"long_press ({x},{y}) {duration_ms}ms")
        self.device._adb("shell", "input", "swipe",
                         str(x), str(y), str(x), str(y), str(duration_ms))

    def input_text(self, text: str):
        safe = text.replace(" ", "%s").replace("'", "\\'")
        logger.debug(f"input_text: {text!r}")
        self.device._adb("shell", "input", "text", safe)

    def press_key(self, keycode: int | str):
        logger.debug(f"press_key: {keycode}")
        self.device._adb("shell", "input", "keyevent", str(keycode))

    def press_back(self):
        self.press_key(4)

    def press_home(self):
        self.press_key(3)

    def press_enter(self):
        self.press_key(66)

    def wait(self, seconds: float):
        logger.debug(f"wait {seconds}s")
        time.sleep(seconds)

    def scroll_down(self, x: int = 540, start_y: int = 1600, end_y: int = 400, duration_ms: int = 500):
        self.swipe(x, start_y, x, end_y, duration_ms)

    def scroll_up(self, x: int = 540, start_y: int = 400, end_y: int = 1600, duration_ms: int = 500):
        self.swipe(x, start_y, x, end_y, duration_ms)
