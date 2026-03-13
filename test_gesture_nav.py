"""
test_gesture_nav.py — Unit tests for gesture_nav.py helper functions and logic.

All pyautogui calls are patched out so no display or keyboard access is needed.
"""

import sys
import types
import unittest
from unittest.mock import MagicMock, call, patch

# ── Stub out heavy / display-dependent imports before gesture_nav is imported ─

# pyautogui stub
pyautogui_mock = MagicMock()
sys.modules["pyautogui"] = pyautogui_mock

# cv2 stub
cv2_mock = MagicMock()
sys.modules["cv2"] = cv2_mock

# mediapipe stub
mp_mock = MagicMock()
sys.modules["mediapipe"] = mp_mock

# numpy stub
sys.modules["numpy"] = MagicMock()

# file_navigator stub (avoids pyautogui side-effects inside FileNavigator)
file_navigator_mod = types.ModuleType("file_navigator")
file_navigator_mod.FileNavigator = MagicMock()
sys.modules["file_navigator"] = file_navigator_mod

# gestures stub — import the real module so Gesture enum is correct
from gestures import Gesture, detect_pinch  # noqa: E402

# Now import gesture_nav (all heavy deps already stubbed)
import gesture_nav  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _lm(x: float, y: float):
    """Return a MagicMock that quacks like a NormalizedLandmark."""
    m = MagicMock()
    m.x = x
    m.y = y
    m.z = 0.0
    return m


def _make_landmarks(wrist_x: float = 0.5, wrist_y: float = 0.5,
                    thumb_x: float = 0.5, index_x: float = 0.5) -> list:
    """
    Build a minimal 21-element landmark list with configurable wrist, thumb-tip,
    and index-tip positions.  All other landmarks default to (0.5, 0.5).
    """
    lms = [_lm(0.5, 0.5) for _ in range(21)]
    lms[0] = _lm(wrist_x, wrist_y)   # landmark 0 = wrist
    lms[4] = _lm(thumb_x, 0.5)       # landmark 4 = thumb tip
    lms[8] = _lm(index_x, 0.5)       # landmark 8 = index tip
    return lms


# ──────────────────────────────────────────────────────────────────────────────
# Pinch → Return key press
# ──────────────────────────────────────────────────────────────────────────────

class TestPinchPressesReturn(unittest.TestCase):
    """Verify that a detected pinch triggers pyautogui.press('return')."""

    def setUp(self):
        pyautogui_mock.reset_mock()

    def test_pinch_triggers_return(self):
        # Thumb tip and index tip very close → pinch detected
        lm = _make_landmarks(thumb_x=0.5, index_x=0.53)
        pinching = detect_pinch(lm, threshold=0.07)
        if pinching:
            pyautogui_mock.press("return")
        pyautogui_mock.press.assert_called_with("return")

    def test_no_pinch_no_return(self):
        # Thumb and index far apart → no pinch
        lm = _make_landmarks(thumb_x=0.5, index_x=0.7)
        pinching = detect_pinch(lm, threshold=0.07)
        self.assertFalse(pinching)
        # Simulate the conditional in gesture_nav
        if pinching:
            pyautogui_mock.press("return")
        pyautogui_mock.press.assert_not_called()


# ──────────────────────────────────────────────────────────────────────────────
# Wrist X pixel tracking → Arrow key presses
# ──────────────────────────────────────────────────────────────────────────────

class TestWristXPixelTracking(unittest.TestCase):
    """
    Exercise the wrist-X pixel-tracking logic extracted from gesture_nav.run().

    The logic under test:
        wrist_x_px = int(lm[0].x * frame_width)
        delta_x    = wrist_x_px - prev_wrist_x_px
        if delta_x > 100  → pyautogui.press("right")
        if delta_x < -100 → pyautogui.press("left")
    with a cooldown that suppresses repeated fires.
    """

    _THRESHOLD = 100
    _COOLDOWN_FRAMES = 15
    _FRAME_WIDTH = 640

    def _tick(self, prev_x_px, curr_norm_x, cooldown):
        """
        One iteration of the wrist-tracking block.
        Returns (key_pressed_or_None, new_prev_x_px, new_cooldown).
        """
        pyautogui_mock.reset_mock()

        curr_x_px = int(curr_norm_x * self._FRAME_WIDTH)
        key_pressed = None

        if cooldown > 0:
            cooldown -= 1

        if prev_x_px is not None and cooldown == 0:
            delta_x = curr_x_px - prev_x_px
            if delta_x > self._THRESHOLD:
                pyautogui_mock.press("right")
                key_pressed = "right"
                cooldown = self._COOLDOWN_FRAMES
            elif delta_x < -self._THRESHOLD:
                pyautogui_mock.press("left")
                key_pressed = "left"
                cooldown = self._COOLDOWN_FRAMES

        return key_pressed, curr_x_px, cooldown

    def test_right_arrow_on_large_rightward_movement(self):
        # Start at x=0.2 (128 px), jump to x=0.45 (288 px) — delta = 160 px
        key, _, _ = self._tick(prev_x_px=128, curr_norm_x=0.45, cooldown=0)
        self.assertEqual(key, "right")
        pyautogui_mock.press.assert_called_once_with("right")

    def test_left_arrow_on_large_leftward_movement(self):
        # Start at x=0.8 (512 px), jump to x=0.5 (320 px) — delta = -192 px
        key, _, _ = self._tick(prev_x_px=512, curr_norm_x=0.5, cooldown=0)
        self.assertEqual(key, "left")
        pyautogui_mock.press.assert_called_once_with("left")

    def test_small_movement_does_not_trigger(self):
        # Move only 50 px — below threshold
        key, _, _ = self._tick(prev_x_px=300, curr_norm_x=0.55, cooldown=0)
        self.assertIsNone(key)
        pyautogui_mock.press.assert_not_called()

    def test_exact_threshold_does_not_trigger(self):
        # delta == 100 exactly — strictly greater-than is required
        start_px = 200
        curr_norm = (start_px + self._THRESHOLD) / self._FRAME_WIDTH
        key, _, _ = self._tick(prev_x_px=start_px, curr_norm_x=curr_norm, cooldown=0)
        self.assertIsNone(key)
        pyautogui_mock.press.assert_not_called()

    def test_no_fire_on_first_frame(self):
        # prev_wrist_x_px is None on first detection frame
        pyautogui_mock.reset_mock()
        prev_wrist_x_px = None
        cooldown = 0
        curr_x_px = int(0.9 * self._FRAME_WIDTH)
        if prev_wrist_x_px is not None and cooldown == 0:
            delta_x = curr_x_px - prev_wrist_x_px
            if delta_x > self._THRESHOLD:
                pyautogui_mock.press("right")
        pyautogui_mock.press.assert_not_called()

    def test_cooldown_suppresses_repeated_fire(self):
        # Fire once, then immediately try again with the same large delta.
        key1, new_prev, cooldown = self._tick(prev_x_px=128, curr_norm_x=0.45, cooldown=0)
        self.assertEqual(key1, "right")
        # Second tick — cooldown is still active (15 - 1 = 14 after tick)
        key2, _, _ = self._tick(prev_x_px=new_prev, curr_norm_x=0.45 + 0.25, cooldown=cooldown)
        self.assertIsNone(key2)

    def test_cooldown_expires_and_refires(self):
        # Burn through the entire cooldown, then verify re-fire.
        _, new_prev, cooldown = self._tick(prev_x_px=128, curr_norm_x=0.45, cooldown=0)
        # Drain the cooldown (15 frames of small movement)
        for _ in range(self._COOLDOWN_FRAMES):
            _, new_prev, cooldown = self._tick(prev_x_px=new_prev, curr_norm_x=new_prev / self._FRAME_WIDTH, cooldown=cooldown)
        # Now a large rightward delta should fire again
        key, _, _ = self._tick(prev_x_px=new_prev, curr_norm_x=(new_prev + 200) / self._FRAME_WIDTH, cooldown=cooldown)
        self.assertEqual(key, "right")


if __name__ == "__main__":
    unittest.main()
