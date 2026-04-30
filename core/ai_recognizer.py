"""
AI-powered screen recognition using Claude Vision.

Provides four capabilities:
  1. find_element(description)  — locate a UI element by natural language description
  2. understand_screen()        — describe the current game state
  3. assert_screen(expectation) — check whether the screen matches an expectation
  4. generate_test_code(scenario) — produce pytest test code from a scenario description
"""

import base64
import io
import json
from typing import Optional

import numpy as np
from loguru import logger
from PIL import Image

import config

try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False
    logger.warning("anthropic package not installed — AIRecognizer unavailable. Run: pip install anthropic")

_SYSTEM_PROMPT = """\
You are an expert Android game UI automation assistant.
You receive screenshots of an Android Unity game and help automate testing by:
- Locating UI elements precisely by their pixel coordinates
- Describing game state and screen contents
- Verifying whether screens match expected conditions
- Generating pytest test code

COORDINATE RULES:
- Always return coordinates as integers in the device's native pixel space (matching the screenshot dimensions).
- (0,0) is the top-left corner; x increases right, y increases down.
- For element location, return the CENTER of the element.

OUTPUT FORMAT:
- For find_element: respond with ONLY a JSON object: {"x": <int>, "y": <int>, "confidence": <float 0-1>, "found": <bool>}
- For understand_screen: respond with concise plain text (2-5 sentences).
- For assert_screen: respond with ONLY a JSON object: {"matches": <bool>, "reason": "<short explanation>"}
- For generate_test_code: respond with ONLY a Python code block (no prose).
"""

_CODEGEN_SUFFIX = """\

Generate a complete pytest test method (inside a class inheriting BaseTest) for the following scenario.
Use these helper methods: tap_text(), tap_image(), assert_text(), assert_image(), wait(), swipe(), take_screenshot(), ai_tap(), ai_assert(), ai_understand().
"""


def _to_base64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode()


def _image_content(arr: np.ndarray) -> dict:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": _to_base64(arr),
        },
    }


class AIRecognizer:
    """Claude Vision-powered recognizer for Android Unity game screens."""

    def __init__(self, device):
        if not _ANTHROPIC_AVAILABLE:
            raise ImportError("Install anthropic: pip install anthropic>=0.25.0")
        api_key = config.ANTHROPIC_API_KEY
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set in environment or config.py")
        self._client = anthropic.Anthropic(api_key=api_key)
        self._device = device
        self._model = config.AI_MODEL

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _screenshot(self) -> np.ndarray:
        return self._device.screenshot()

    def _call(self, user_content: list, max_tokens: int = 512) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=[
                {
                    "type": "text",
                    "text": _SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_content}],
        )
        return response.content[0].text.strip()

    # ── Public API ────────────────────────────────────────────────────────────

    def find_element(
        self, description: str, screenshot: Optional[np.ndarray] = None
    ) -> Optional[tuple[int, int]]:
        """Locate a UI element by natural language description.

        Returns (x, y) center coordinates, or None if not found.
        """
        ss = screenshot if screenshot is not None else self._screenshot()
        h, w = ss.shape[:2]
        prompt = (
            f"The screenshot is {w}x{h} pixels.\n"
            f"Find this UI element: {description}\n"
            "Return JSON only."
        )
        raw = self._call([_image_content(ss), {"type": "text", "text": prompt}])
        try:
            data = json.loads(raw)
            if not data.get("found", True):
                logger.warning(f"AI: element not found — {description}")
                return None
            x, y = int(data["x"]), int(data["y"])
            logger.info(f"AI found '{description}' at ({x}, {y}) confidence={data.get('confidence', '?')}")
            return x, y
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.error(f"AI find_element parse error: {e!r}  raw={raw!r}")
            return None

    def understand_screen(self, screenshot: Optional[np.ndarray] = None) -> str:
        """Return a plain-text description of the current game screen."""
        ss = screenshot if screenshot is not None else self._screenshot()
        raw = self._call(
            [
                _image_content(ss),
                {"type": "text", "text": "Describe the current game screen state in 2-5 sentences."},
            ],
            max_tokens=256,
        )
        logger.info(f"AI screen understanding: {raw[:120]}...")
        return raw

    def assert_screen(
        self, expectation: str, screenshot: Optional[np.ndarray] = None
    ) -> tuple[bool, str]:
        """Check if the screen matches the given expectation.

        Returns (matches: bool, reason: str).
        """
        ss = screenshot if screenshot is not None else self._screenshot()
        prompt = (
            f"Does the screen match this expectation: {expectation}\n"
            "Return JSON only: {\"matches\": true/false, \"reason\": \"...\"}"
        )
        raw = self._call([_image_content(ss), {"type": "text", "text": prompt}])
        try:
            data = json.loads(raw)
            matches = bool(data["matches"])
            reason = data.get("reason", "")
            logger.log(
                "SUCCESS" if matches else "WARNING",
                f"AI assert '{expectation}': {'✓' if matches else '✗'} — {reason}",
            )
            return matches, reason
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.error(f"AI assert_screen parse error: {e!r}  raw={raw!r}")
            return False, f"parse error: {e}"

    def generate_test_code(self, scenario: str) -> str:
        """Generate a pytest test method for the given scenario description."""
        prompt = _CODEGEN_SUFFIX + f"\nScenario: {scenario}"
        code = self._call(
            [{"type": "text", "text": prompt}],
            max_tokens=1024,
        )
        # Strip markdown fences if present
        if code.startswith("```"):
            lines = code.splitlines()
            code = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
        logger.info(f"AI generated test code ({len(code)} chars) for: {scenario[:60]}")
        return code
