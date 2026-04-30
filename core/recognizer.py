import time
import numpy as np
import cv2
from pathlib import Path
from loguru import logger
import config


class ImageRecognizer:
    def __init__(self, device):
        self.device = device
        self._ocr = None  # lazy-loaded EasyOCR reader

    def _ocr_reader(self):
        if self._ocr is None:
            import easyocr
            logger.info("Loading EasyOCR model (first run may take a moment)...")
            self._ocr = easyocr.Reader(config.OCR_LANGUAGES, gpu=False)
        return self._ocr

    def _screenshot_gray(self) -> np.ndarray:
        frame = self.device.screenshot()
        return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    def find_template(self, template_path: str | Path, threshold: float = None) -> tuple[int, int] | None:
        threshold = threshold or config.DEFAULT_THRESHOLD
        template_path = Path(template_path)
        if not template_path.is_absolute():
            template_path = config.TEMPLATES_DIR / template_path

        tmpl = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
        if tmpl is None:
            raise FileNotFoundError(f"Template not found: {template_path}")

        screen = self._screenshot_gray()
        result = cv2.matchTemplate(screen, tmpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val >= threshold:
            h, w = tmpl.shape
            cx = max_loc[0] + w // 2
            cy = max_loc[1] + h // 2
            logger.debug(f"Template found: {template_path.name} @ ({cx},{cy}) conf={max_val:.3f}")
            return cx, cy

        logger.debug(f"Template NOT found: {template_path.name} best={max_val:.3f} < {threshold}")
        return None

    def wait_for_template(self, template_path: str | Path, timeout: float = None,
                          interval: float = None) -> tuple[int, int]:
        timeout = timeout or config.DEFAULT_TIMEOUT
        interval = interval or config.DEFAULT_INTERVAL
        deadline = time.time() + timeout
        while time.time() < deadline:
            pos = self.find_template(template_path)
            if pos:
                return pos
            time.sleep(interval)
        raise TimeoutError(f"Template not found within {timeout}s: {template_path}")

    def find_text(self, text: str) -> tuple[int, int] | None:
        frame = self.device.screenshot()
        reader = self._ocr_reader()
        results = reader.readtext(frame)
        text_lower = text.lower()
        for (bbox, detected, conf) in results:
            if text_lower in detected.lower():
                xs = [p[0] for p in bbox]
                ys = [p[1] for p in bbox]
                cx = int(sum(xs) / len(xs))
                cy = int(sum(ys) / len(ys))
                logger.debug(f"Text found: '{text}' @ ({cx},{cy}) conf={conf:.3f}")
                return cx, cy
        logger.debug(f"Text NOT found: '{text}'")
        return None

    def wait_for_text(self, text: str, timeout: float = None, interval: float = None) -> tuple[int, int]:
        timeout = timeout or config.DEFAULT_TIMEOUT
        interval = interval or config.DEFAULT_INTERVAL
        deadline = time.time() + timeout
        while time.time() < deadline:
            pos = self.find_text(text)
            if pos:
                return pos
            time.sleep(interval)
        raise TimeoutError(f"Text not found within {timeout}s: '{text}'")

    def assert_template_present(self, template_path: str | Path, timeout: float = 5.0):
        try:
            self.wait_for_template(template_path, timeout=timeout)
        except TimeoutError as e:
            raise AssertionError(str(e)) from e

    def assert_template_absent(self, template_path: str | Path, timeout: float = 5.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.find_template(template_path) is None:
                return
            time.sleep(config.DEFAULT_INTERVAL)
        raise AssertionError(f"Template still present after {timeout}s: {template_path}")

    def assert_text_present(self, text: str, timeout: float = 5.0):
        try:
            self.wait_for_text(text, timeout=timeout)
        except TimeoutError as e:
            raise AssertionError(str(e)) from e

    def get_region(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        frame = self.device.screenshot()
        return frame[y:y + h, x:x + w]

    def get_all_text(self) -> list[dict]:
        frame = self.device.screenshot()
        reader = self._ocr_reader()
        results = reader.readtext(frame)
        out = []
        for (bbox, text, conf) in results:
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            out.append({
                "text": text,
                "confidence": round(conf, 3),
                "x": int(sum(xs) / len(xs)),
                "y": int(sum(ys) / len(ys)),
            })
        return out
