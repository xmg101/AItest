import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.device import ADBDevice
from core.controller import UIController
from core.recognizer import ImageRecognizer
from core.recorder import ActionRecorder
from core.reporter import TestReporter
from core.ai_recognizer import AIRecognizer
import config


class BaseTest:
    """
    Base class for all Unity game automated tests.
    Inherit from this class and override test methods.
    """

    @pytest.fixture(autouse=True)
    def _setup(self, request):
        self.device = ADBDevice()
        self.controller = UIController(self.device)
        self.recognizer = ImageRecognizer(self.device)
        self.recorder = ActionRecorder(self.device, self.controller)
        self.reporter = TestReporter()
        self.ai = AIRecognizer(self.device) if config.ANTHROPIC_API_KEY else None

        suite_name = request.module.__name__.split(".")[-1]
        self.reporter.start_suite(suite_name)
        self.reporter.start_test(request.node.name)

        yield

        # Capture final screenshot on failure
        outcome = "pass"
        if hasattr(request.node, "rep_call") and request.node.rep_call.failed:
            outcome = "fail"
            try:
                ss = self.device.screenshot()
                self.reporter.add_step("Test failed — final screenshot", "fail", screenshot=ss)
            except Exception:
                pass

        self.reporter.end_test(outcome)
        self.reporter.end_suite()
        self.reporter.generate()

    def screenshot(self) -> np.ndarray:
        return self.device.screenshot()

    def take_screenshot(self, name: str = "screenshot"):
        ss = self.device.screenshot()
        self.reporter.add_step(name, "pass", screenshot=ss)
        return ss

    # ── Image assertions ─────────────────────────────────────────────────────

    def assert_image(self, template: str, timeout: float = 10.0, msg: str = ""):
        step_name = msg or f"Assert image present: {template}"
        try:
            pos = self.recognizer.wait_for_template(template, timeout=timeout)
            ss = self.device.screenshot()
            self.reporter.add_step(step_name, "pass", screenshot=ss)
            return pos
        except (TimeoutError, AssertionError) as e:
            ss = self.device.screenshot()
            self.reporter.add_step(step_name, "fail", screenshot=ss, error=str(e))
            raise AssertionError(str(e)) from e

    def assert_no_image(self, template: str, timeout: float = 5.0, msg: str = ""):
        step_name = msg or f"Assert image absent: {template}"
        try:
            self.recognizer.assert_template_absent(template, timeout=timeout)
            self.reporter.add_step(step_name, "pass")
        except AssertionError as e:
            ss = self.device.screenshot()
            self.reporter.add_step(step_name, "fail", screenshot=ss, error=str(e))
            raise

    # ── Text assertions ───────────────────────────────────────────────────────

    def assert_text(self, text: str, timeout: float = 10.0, msg: str = ""):
        step_name = msg or f"Assert text present: '{text}'"
        try:
            pos = self.recognizer.wait_for_text(text, timeout=timeout)
            ss = self.device.screenshot()
            self.reporter.add_step(step_name, "pass", screenshot=ss)
            return pos
        except (TimeoutError, AssertionError) as e:
            ss = self.device.screenshot()
            self.reporter.add_step(step_name, "fail", screenshot=ss, error=str(e))
            raise AssertionError(str(e)) from e

    # ── Combined tap helpers ──────────────────────────────────────────────────

    def tap_image(self, template: str, timeout: float = 10.0, msg: str = ""):
        step_name = msg or f"Tap image: {template}"
        pos = self.assert_image(template, timeout=timeout, msg=step_name)
        self.controller.tap(*pos)
        return pos

    def tap_text(self, text: str, timeout: float = 10.0, msg: str = ""):
        step_name = msg or f"Tap text: '{text}'"
        pos = self.assert_text(text, timeout=timeout, msg=step_name)
        self.controller.tap(*pos)
        return pos

    def tap_xy(self, x: int, y: int, msg: str = ""):
        step_name = msg or f"Tap ({x}, {y})"
        self.controller.tap(x, y)
        self.reporter.add_step(step_name, "pass")

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300, msg: str = ""):
        step_name = msg or f"Swipe ({x1},{y1})→({x2},{y2})"
        self.controller.swipe(x1, y1, x2, y2, duration_ms)
        self.reporter.add_step(step_name, "pass")

    def wait(self, seconds: float, msg: str = ""):
        step_name = msg or f"Wait {seconds}s"
        self.controller.wait(seconds)
        self.reporter.add_step(step_name, "pass")

    def log_step(self, name: str, status: str = "pass", error: str | None = None):
        ss = self.device.screenshot()
        self.reporter.add_step(name, status, screenshot=ss, error=error)

    # ── AI helpers (require ANTHROPIC_API_KEY) ────────────────────────────────

    def _require_ai(self):
        if self.ai is None:
            raise RuntimeError("AIRecognizer not available — set ANTHROPIC_API_KEY env var")

    def ai_tap(self, description: str, msg: str = ""):
        """Locate a UI element by natural language description and tap it."""
        self._require_ai()
        step_name = msg or f"AI tap: {description}"
        ss = self.device.screenshot()
        pos = self.ai.find_element(description, screenshot=ss)
        if pos is None:
            self.reporter.add_step(step_name, "fail", screenshot=ss,
                                   error=f"AI could not find: {description}")
            raise AssertionError(f"AI could not find element: {description}")
        self.controller.tap(*pos)
        ss2 = self.device.screenshot()
        self.reporter.add_step(step_name, "pass", screenshot=ss2)
        return pos

    def ai_assert(self, expectation: str, msg: str = ""):
        """Assert that the screen matches a natural-language expectation."""
        self._require_ai()
        step_name = msg or f"AI assert: {expectation}"
        ss = self.device.screenshot()
        matches, reason = self.ai.assert_screen(expectation, screenshot=ss)
        status = "pass" if matches else "fail"
        self.reporter.add_step(step_name, status, screenshot=ss,
                               error=None if matches else reason)
        if not matches:
            raise AssertionError(f"AI assertion failed: {expectation!r} — {reason}")
        return reason

    def ai_understand(self, msg: str = "") -> str:
        """Take a screenshot and return Claude's description of the game state."""
        self._require_ai()
        ss = self.device.screenshot()
        description = self.ai.understand_screen(screenshot=ss)
        step_name = msg or "AI screen understanding"
        self.reporter.add_step(step_name, "pass", screenshot=ss)
        return description

    def ai_generate_test(self, scenario: str) -> str:
        """Generate pytest test code for a given scenario description."""
        self._require_ai()
        return self.ai.generate_test_code(scenario)
