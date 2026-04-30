"""
快速控制脚本：启动游戏并点击开始关卡按钮
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from core.device import ADBDevice
from core.controller import UIController
from core.recognizer import ImageRecognizer
import config

PACKAGE_NAME = open(Path(__file__).parent / "tests/example_test.py", encoding="utf-8").read().split('PACKAGE_NAME = "')[1].split('"')[0]

dev = ADBDevice()
if not dev.connect():
    print("设备未连接，请检查 adb devices")
    sys.exit(1)

ctrl = UIController(dev)
rec = ImageRecognizer(dev)

print(f"包名: {PACKAGE_NAME}")
print("正在启动游戏...")
dev.force_stop(PACKAGE_NAME)
dev.launch_app(PACKAGE_NAME)
ctrl.wait(5.0)

print("截图并识别屏幕文字...")
ss = dev.screenshot()
dev.save_screenshot("screen.png")
print(f"截图已保存到 {config.SCREENSHOTS_DIR / 'screen.png'}")

texts = rec.get_all_text()
print("\n当前屏幕上识别到的文字：")
for item in texts:
    print(f"  [{item['x']}, {item['y']}] {item['text']}  (置信度: {item['confidence']})")

print("\n尝试查找包含 'level' 的按钮...")
target = None
for item in texts:
    if "level" in item["text"].lower():
        target = item
        print(f"  找到: {item['text']}  位置: ({item['x']}, {item['y']})")
        break

if target:
    ctrl.tap(target["x"], target["y"])
    ctrl.wait(2.0)
    dev.save_screenshot("after_tap.png")
    print(f"点击完成，截图已保存到 {config.SCREENSHOTS_DIR / 'after_tap.png'}")
else:
    print("未找到含 level 的文字，尝试点击屏幕底部中间...")
    ctrl.tap(540, 2000)
    ctrl.wait(2.0)
    dev.save_screenshot("after_tap.png")
    print(f"已点击 (540, 2000)，截图保存到 {config.SCREENSHOTS_DIR / 'after_tap.png'}")
