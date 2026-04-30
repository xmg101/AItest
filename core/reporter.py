import base64
import io
import time
from datetime import datetime
from pathlib import Path
from loguru import logger
import numpy as np
from PIL import Image
from jinja2 import Template
import config

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ suite_name }} — Test Report</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; background: #f5f5f5; color: #333; }
  header { background: #1a1a2e; color: #eee; padding: 20px 32px; }
  header h1 { margin: 0 0 4px; font-size: 1.5rem; }
  header .meta { font-size: .85rem; opacity: .7; }
  .summary { display: flex; gap: 16px; padding: 16px 32px; background: #fff; border-bottom: 1px solid #ddd; flex-wrap: wrap; }
  .badge { padding: 6px 16px; border-radius: 20px; font-weight: 600; font-size: .9rem; }
  .pass  { background: #d4edda; color: #155724; }
  .fail  { background: #f8d7da; color: #721c24; }
  .skip  { background: #fff3cd; color: #856404; }
  .total { background: #e2e3e5; color: #383d41; }
  .tests { padding: 24px 32px; max-width: 1200px; margin: 0 auto; }
  .test-card { background: #fff; border-radius: 8px; margin-bottom: 16px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,.1); }
  .test-header { display: flex; align-items: center; gap: 12px; padding: 14px 20px; cursor: pointer; }
  .test-header:hover { background: #fafafa; }
  .test-header .status { width: 10px; height: 10px; border-radius: 50%; }
  .test-header .status.pass { background: #28a745; }
  .test-header .status.fail { background: #dc3545; }
  .test-header .status.skip { background: #ffc107; }
  .test-header .name { font-weight: 600; flex: 1; }
  .test-header .duration { font-size: .82rem; color: #888; }
  .steps { border-top: 1px solid #eee; display: none; }
  .steps.open { display: block; }
  .step { display: flex; align-items: flex-start; gap: 16px; padding: 12px 20px; border-bottom: 1px solid #f0f0f0; }
  .step:last-child { border-bottom: none; }
  .step .step-status { font-size: .75rem; font-weight: 700; min-width: 36px; padding-top: 2px; text-transform: uppercase; }
  .step .step-status.pass { color: #28a745; }
  .step .step-status.fail { color: #dc3545; }
  .step .step-status.skip { color: #ffc107; }
  .step .step-info { flex: 1; }
  .step .step-name { font-size: .9rem; }
  .step .step-error { margin-top: 4px; font-size: .82rem; color: #dc3545; font-family: monospace; white-space: pre-wrap; }
  .step img { max-width: 200px; max-height: 140px; border-radius: 4px; cursor: pointer; border: 1px solid #ddd; }
  .lightbox { display: none; position: fixed; inset: 0; background: rgba(0,0,0,.85); z-index: 9999; align-items: center; justify-content: center; }
  .lightbox.active { display: flex; }
  .lightbox img { max-width: 90vw; max-height: 90vh; border-radius: 4px; }
</style>
</head>
<body>
<header>
  <h1>{{ suite_name }}</h1>
  <div class="meta">Generated {{ generated_at }} &nbsp;|&nbsp; Duration {{ total_duration }}s</div>
</header>
<div class="summary">
  <span class="badge total">Total: {{ stats.total }}</span>
  <span class="badge pass">Passed: {{ stats.pass }}</span>
  <span class="badge fail">Failed: {{ stats.fail }}</span>
  <span class="badge skip">Skipped: {{ stats.skip }}</span>
</div>
<div class="tests">
{% for test in tests %}
  <div class="test-card">
    <div class="test-header" onclick="toggle(this)">
      <div class="status {{ test.status }}"></div>
      <span class="name">{{ test.name }}</span>
      <span class="duration">{{ test.duration }}s</span>
    </div>
    <div class="steps">
      {% for step in test.steps %}
      <div class="step">
        <div class="step-status {{ step.status }}">{{ step.status }}</div>
        <div class="step-info">
          <div class="step-name">{{ step.name }}</div>
          {% if step.error %}<div class="step-error">{{ step.error }}</div>{% endif %}
        </div>
        {% if step.screenshot %}
        <img src="{{ step.screenshot }}" onclick="showLightbox(this)" alt="screenshot">
        {% endif %}
      </div>
      {% endfor %}
    </div>
  </div>
{% endfor %}
</div>
<div class="lightbox" id="lb" onclick="this.classList.remove('active')">
  <img id="lb-img" src="">
</div>
<script>
function toggle(el){ el.nextElementSibling.classList.toggle('open'); }
function showLightbox(img){ document.getElementById('lb-img').src=img.src; document.getElementById('lb').classList.add('active'); }
</script>
</body>
</html>"""


def _img_to_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class TestReporter:
    def __init__(self):
        self._suite_name = "Test Suite"
        self._tests: list[dict] = []
        self._current_test: dict | None = None
        self._suite_start = 0.0

    def start_suite(self, name: str):
        self._suite_name = name
        self._tests.clear()
        self._suite_start = time.time()
        logger.info(f"Suite started: {name}")

    def start_test(self, name: str):
        self._current_test = {
            "name": name,
            "status": "pass",
            "steps": [],
            "start": time.time(),
            "duration": 0,
        }
        logger.info(f"Test started: {name}")

    def add_step(self, name: str, status: str = "pass",
                 screenshot: np.ndarray | None = None, error: str | None = None):
        if self._current_test is None:
            return
        step = {
            "name": name,
            "status": status,
            "screenshot": _img_to_b64(screenshot) if screenshot is not None else None,
            "error": error,
        }
        self._current_test["steps"].append(step)
        if status == "fail":
            self._current_test["status"] = "fail"
        logger.log("SUCCESS" if status == "pass" else "WARNING",
                   f"Step {'✓' if status == 'pass' else '✗'}: {name}")

    def end_test(self, status: str | None = None):
        if self._current_test is None:
            return
        if status:
            self._current_test["status"] = status
        self._current_test["duration"] = round(time.time() - self._current_test["start"], 2)
        self._tests.append(self._current_test)
        logger.info(f"Test ended: {self._current_test['name']} [{self._current_test['status']}]")
        self._current_test = None

    def end_suite(self):
        logger.info(f"Suite ended: {self._suite_name}")

    def generate(self) -> str:
        stats = {"total": len(self._tests), "pass": 0, "fail": 0, "skip": 0}
        for t in self._tests:
            stats[t["status"]] = stats.get(t["status"], 0) + 1

        total_dur = round(time.time() - self._suite_start, 1)
        html = Template(_HTML_TEMPLATE).render(
            suite_name=self._suite_name,
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            total_duration=total_dur,
            stats=stats,
            tests=self._tests,
        )
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = self._suite_name.replace(" ", "_").replace("/", "-")
        out_path = config.REPORTS_DIR / f"{safe_name}_{ts}.html"
        out_path.write_text(html, encoding="utf-8")
        logger.success(f"Report generated: {out_path}")
        return str(out_path)
