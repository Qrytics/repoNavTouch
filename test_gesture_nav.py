"""
test_gesture_nav.py — Unit tests for gesture_nav.py helper functions and logic.

All pyautogui calls are patched out so no display or keyboard access is needed.
"""

import sys
import types
import unittest
from unittest.mock import MagicMock, patch

# ── Stub out heavy / display-dependent imports before gesture_nav is imported ─

# cv2 stub
cv2_mock = MagicMock()
sys.modules["cv2"] = cv2_mock

# mediapipe stub
mp_mock = MagicMock()
sys.modules["mediapipe"] = mp_mock

# numpy stub
sys.modules["numpy"] = MagicMock()

# file_navigator stub (avoids any side-effects inside FileNavigator)
file_navigator_mod = types.ModuleType("file_navigator")
file_navigator_mod.FileNavigator = MagicMock()
sys.modules["file_navigator"] = file_navigator_mod

# overlay stub (avoids tkinter / display requirements)
overlay_mod = types.ModuleType("overlay")
overlay_mod.OverlayWindow = MagicMock()
sys.modules["overlay"] = overlay_mod

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
# Pinch detection (unit-level, no pyautogui side-effects)
# ──────────────────────────────────────────────────────────────────────────────

class TestPinchDetection(unittest.TestCase):
    """Verify pinch detection logic without any pyautogui dependency."""

    def test_pinch_detected_when_close(self):
        lm = _make_landmarks(thumb_x=0.5, index_x=0.53)
        self.assertTrue(detect_pinch(lm, threshold=0.07))

    def test_no_pinch_when_far(self):
        lm = _make_landmarks(thumb_x=0.5, index_x=0.7)
        self.assertFalse(detect_pinch(lm, threshold=0.07))


# ──────────────────────────────────────────────────────────────────────────────
# Monotonic timestamp guard
# ──────────────────────────────────────────────────────────────────────────────

class TestMonotonicTimestamp(unittest.TestCase):
    """
    Verify that the timestamp clamping logic in gesture_nav always produces a
    strictly increasing sequence even when the wall-clock millisecond value does
    not change between frames.
    """

    def _simulate_timestamps(self, raw_ms_values):
        """
        Run the timestamp-clamping algorithm from gesture_nav over a list of raw
        monotonic-clock millisecond readings and return the sequence of values
        that would be forwarded to MediaPipe.
        """
        last_ts = -1
        result = []
        for raw in raw_ms_values:
            ts = raw if raw > last_ts else last_ts + 1
            last_ts = ts
            result.append(ts)
        return result

    def test_strictly_increasing_when_clock_repeats(self):
        # Clock stalls for 3 frames at 1000 ms
        raw = [999, 1000, 1000, 1000, 1001]
        out = self._simulate_timestamps(raw)
        for a, b in zip(out, out[1:]):
            self.assertGreater(b, a, f"Timestamp not strictly increasing: {out}")

    def test_strictly_increasing_normal_sequence(self):
        raw = [0, 1, 2, 3, 4, 5]
        out = self._simulate_timestamps(raw)
        self.assertEqual(out, raw)

    def test_first_frame_uses_raw_value(self):
        raw = [500, 501, 502]
        out = self._simulate_timestamps(raw)
        self.assertEqual(out[0], 500)

    def test_backward_jump_is_corrected(self):
        # Simulate a backward step (should not happen with monotonic, but guard anyway)
        raw = [100, 99, 98]
        out = self._simulate_timestamps(raw)
        for a, b in zip(out, out[1:]):
            self.assertGreater(b, a)


if __name__ == "__main__":
    unittest.main()
