"""
Example test for a Unity Android game.

Replace PACKAGE_NAME and template file names with your actual values.
Place PNG screenshot templates in assets/templates/.
"""

import pytest
from tests.base_test import BaseTest

PACKAGE_NAME = "com.tapcolor.puzzle.sort.goods.match.triple"  # ← change to your game's package


class TestGameFlow(BaseTest):

    def test_launch_and_login(self):
        """Launch the game, wait for login screen, tap login, verify main scene."""
        # 1. Launch game
        self.device.force_stop(PACKAGE_NAME)
        self.device.launch_app(PACKAGE_NAME)
        self.reporter.add_step("Launch game", "pass")

        # 2. Wait for login screen to appear (match template or text)
        self.wait(3.0, "Wait for splash screen")
        self.assert_text("登录", timeout=15, msg="Login button visible")

        # 3. Tap login
        self.tap_text("登录", msg="Tap login button")
        self.wait(2.0, "Wait for main scene to load")

        # 4. Verify main scene
        self.assert_text("主界面", timeout=10, msg="Main scene loaded")
        self.take_screenshot("Main scene loaded")

    def test_tap_image_button(self):
        """Tap a UI button identified by image template."""
        # Place a cropped PNG of your button in assets/templates/btn_start.png
        self.tap_image("btn_start.png", timeout=10, msg="Tap Start button")
        self.wait(1.5)
        self.take_screenshot("After tapping Start")

    def test_scroll_inventory(self):
        """Swipe to scroll through in-game inventory."""
        self.assert_text("背包", timeout=10, msg="Inventory tab visible")
        self.tap_text("背包", msg="Open inventory")
        self.wait(1.0)

        # Scroll down three times
        for i in range(3):
            self.swipe(540, 1600, 540, 600, duration_ms=400, msg=f"Scroll inventory #{i+1}")
            self.wait(0.5)

        self.take_screenshot("Inventory after scroll")

    def test_record_and_replay(self):
        """Demonstrate recording a gesture sequence and replaying it."""
        # Record
        self.recorder.start()
        self.wait(5.0, "Manual: perform gestures on device during this wait")
        self.recorder.stop()
        self.recorder.save("my_gesture")
        self.reporter.add_step("Gestures recorded", "pass")

        # Replay
        self.wait(1.0)
        self.recorder.playback("my_gesture", speed=1.0)
        self.reporter.add_step("Gestures replayed", "pass")
        self.take_screenshot("After replay")
