"""
test_gestures.py — Unit tests for gesture classification logic.

These tests use synthetic landmark data so no webcam or MediaPipe runtime
is required; they exercise the pure geometry and state-machine logic.
"""

import math
import unittest
from unittest.mock import MagicMock

from gestures import (
    Gesture,
    GestureRecogniser,
    _distance,
    _is_finger_extended,
    _is_thumb_extended,
    detect_open_palm,
    detect_pinch,
    detect_two_fingers_up,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers to build fake landmark lists
# ──────────────────────────────────────────────────────────────────────────────

def _lm(x: float, y: float, z: float = 0.0):
    """Return a MagicMock that quacks like a NormalizedLandmark."""
    m = MagicMock()
    m.x = x
    m.y = y
    m.z = z
    return m


def _make_landmarks(positions: dict) -> list:
    """
    Build a 21-element list of fake landmarks.

    Parameters
    ----------
    positions : dict
        Mapping of landmark index → (x, y) tuple.  Missing indices are filled
        with a default (0.5, 0.5) placeholder.
    """
    return [_lm(*positions.get(i, (0.5, 0.5))) for i in range(21)]


# ──────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────────────────────────────────────

class TestDistance(unittest.TestCase):

    def test_zero_distance(self):
        a = _lm(0.3, 0.3)
        self.assertAlmostEqual(_distance(a, a), 0.0)

    def test_known_distance(self):
        a = _lm(0.0, 0.0)
        b = _lm(0.3, 0.4)
        self.assertAlmostEqual(_distance(a, b), 0.5, places=5)

    def test_tuple_input(self):
        self.assertAlmostEqual(_distance((0.0, 0.0), (1.0, 0.0)), 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# Pinch detection
# ──────────────────────────────────────────────────────────────────────────────

class TestDetectPinch(unittest.TestCase):

    def _make_pinch_landmarks(self, dist: float) -> list:
        """Place thumb tip and index tip exactly *dist* apart."""
        return _make_landmarks({
            4: (0.5, 0.5),                  # thumb tip
            8: (0.5 + dist, 0.5),           # index tip
        })

    def test_pinch_detected_when_close(self):
        lm = self._make_pinch_landmarks(0.04)
        self.assertTrue(detect_pinch(lm, threshold=0.07))

    def test_no_pinch_when_far(self):
        lm = self._make_pinch_landmarks(0.15)
        self.assertFalse(detect_pinch(lm, threshold=0.07))

    def test_pinch_at_exact_threshold_is_not_detected(self):
        lm = self._make_pinch_landmarks(0.07)
        self.assertFalse(detect_pinch(lm, threshold=0.07))

    def test_custom_threshold(self):
        lm = self._make_pinch_landmarks(0.10)
        self.assertTrue(detect_pinch(lm, threshold=0.15))


# ──────────────────────────────────────────────────────────────────────────────
# Finger-extension helpers
# ──────────────────────────────────────────────────────────────────────────────

class TestFingerExtended(unittest.TestCase):

    def test_finger_extended_when_tip_far_from_wrist(self):
        # Wrist at (0.5, 0.9), PIP at (0.5, 0.6), tip at (0.5, 0.2) — extended
        lm = _make_landmarks({
            0: (0.5, 0.9),   # wrist
            6: (0.5, 0.6),   # index PIP
            8: (0.5, 0.2),   # index tip
        })
        self.assertTrue(_is_finger_extended(lm, tip_idx=8, pip_idx=6))

    def test_finger_curled_when_tip_close_to_wrist(self):
        # Wrist at (0.5, 0.9), PIP at (0.5, 0.6), tip curls back to (0.5, 0.7)
        lm = _make_landmarks({
            0: (0.5, 0.9),
            6: (0.5, 0.6),
            8: (0.5, 0.7),
        })
        self.assertFalse(_is_finger_extended(lm, tip_idx=8, pip_idx=6))


# ──────────────────────────────────────────────────────────────────────────────
# Two-fingers-up
# ──────────────────────────────────────────────────────────────────────────────

class TestTwoFingersUp(unittest.TestCase):

    def _make_peace_sign(self) -> list:
        """
        Construct a rough peace-sign hand: wrist low, index + middle extended,
        ring + pinky + thumb curled.
        """
        wrist_y = 0.9
        return _make_landmarks({
            0:  (0.5, wrist_y),   # wrist
            # Thumb — curled (tip close to wrist)
            3:  (0.35, 0.75),     # thumb IP
            4:  (0.38, 0.78),     # thumb tip (closer to wrist than IP → curled)
            # Index — extended
            6:  (0.5, 0.65),      # index PIP
            8:  (0.5, 0.35),      # index tip (farther from wrist → extended)
            # Middle — extended
            10: (0.52, 0.65),     # middle PIP
            12: (0.52, 0.33),     # middle tip
            # Ring — curled
            14: (0.55, 0.65),     # ring PIP
            16: (0.55, 0.72),     # ring tip (closer to wrist → curled)
            # Pinky — curled
            18: (0.58, 0.68),     # pinky PIP
            20: (0.58, 0.75),     # pinky tip
        })

    def test_peace_sign_detected(self):
        lm = self._make_peace_sign()
        self.assertTrue(detect_two_fingers_up(lm))

    def test_open_palm_not_peace(self):
        """An open palm should NOT trigger two-fingers-up."""
        lm = _make_landmarks({
            0:  (0.5, 0.9),
            3:  (0.3, 0.75),  4:  (0.2, 0.6),   # thumb extended
            6:  (0.5, 0.65),  8:  (0.5, 0.35),   # index extended
            10: (0.52, 0.65), 12: (0.52, 0.33),  # middle extended
            14: (0.55, 0.65), 16: (0.55, 0.35),  # ring extended
            18: (0.58, 0.68), 20: (0.58, 0.38),  # pinky extended
        })
        self.assertFalse(detect_two_fingers_up(lm))


# ──────────────────────────────────────────────────────────────────────────────
# Open palm
# ──────────────────────────────────────────────────────────────────────────────

class TestDetectOpenPalm(unittest.TestCase):

    def _make_open_palm(self) -> list:
        return _make_landmarks({
            0:  (0.5, 0.9),
            3:  (0.3, 0.75),  4:  (0.2, 0.6),   # thumb extended
            6:  (0.5, 0.65),  8:  (0.5, 0.35),   # index extended
            10: (0.52, 0.65), 12: (0.52, 0.33),  # middle extended
            14: (0.55, 0.65), 16: (0.55, 0.35),  # ring extended
            18: (0.58, 0.68), 20: (0.58, 0.38),  # pinky extended
        })

    def test_open_palm_detected(self):
        lm = self._make_open_palm()
        self.assertTrue(detect_open_palm(lm))

    def test_fist_not_open_palm(self):
        lm = _make_landmarks({
            0:  (0.5, 0.9),
            3:  (0.45, 0.82), 4:  (0.48, 0.85),  # thumb curled
            6:  (0.5, 0.65),  8:  (0.5, 0.72),   # index curled
            10: (0.52, 0.65), 12: (0.52, 0.73),  # middle curled
            14: (0.55, 0.65), 16: (0.55, 0.74),  # ring curled
            18: (0.58, 0.68), 20: (0.58, 0.76),  # pinky curled
        })
        self.assertFalse(detect_open_palm(lm))


# ──────────────────────────────────────────────────────────────────────────────
# GestureRecogniser — stateful tests
# ──────────────────────────────────────────────────────────────────────────────

class TestGestureRecogniser(unittest.TestCase):

    def _peace_landmarks(self):
        """Peace-sign landmarks reused from TestTwoFingersUp."""
        wrist_y = 0.9
        return _make_landmarks({
            0:  (0.5, wrist_y),
            3:  (0.35, 0.75), 4:  (0.38, 0.78),
            6:  (0.5, 0.65),  8:  (0.5, 0.35),
            10: (0.52, 0.65), 12: (0.52, 0.33),
            14: (0.55, 0.65), 16: (0.55, 0.72),
            18: (0.58, 0.68), 20: (0.58, 0.75),
        })

    def _pinch_landmarks(self, dist=0.04):
        return _make_landmarks({
            4: (0.5, 0.5),
            8: (0.5 + dist, 0.5),
        })

    def test_pinch_fires_once_then_cooldown(self):
        rec = GestureRecogniser(pinch_threshold=0.07, cooldown_frames=5)
        lm = self._pinch_landmarks()
        first = rec.update(lm)
        self.assertEqual(first, Gesture.PINCH)
        # Subsequent frames within cooldown should NOT fire
        for _ in range(4):
            g = rec.update(lm)
            self.assertNotEqual(g, Gesture.PINCH)

    def test_pinch_fires_again_after_cooldown(self):
        # cooldown_frames=3: fire sets cooldown to 3; each update ticks it down
        # by 1 at the start, so the gesture re-fires on the (cooldown+1)-th
        # update after the initial fire: drain 2 frames then check the 3rd.
        rec = GestureRecogniser(pinch_threshold=0.07, cooldown_frames=3)
        lm = self._pinch_landmarks()
        rec.update(lm)  # fire → cooldown = 3
        for _ in range(2):
            rec.update(lm)  # drain: 3→2, 2→1
        g = rec.update(lm)  # tick 1→0, now ready → fires again
        self.assertEqual(g, Gesture.PINCH)

    def test_two_fingers_up_detected(self):
        rec = GestureRecogniser(cooldown_frames=5)
        lm = self._peace_landmarks()
        g = rec.update(lm)
        self.assertEqual(g, Gesture.TWO_FINGERS_UP)

    def test_swipe_left_detected(self):
        rec = GestureRecogniser(swipe_min_delta=0.10, swipe_history_frames=5, cooldown_frames=5)
        # Simulate wrist moving from x=0.8 to x=0.5 (rightward → left swipe in mirrored view)
        lm_base = _make_landmarks({0: (0.8, 0.5)})
        # Fill history partially
        for x in [0.8, 0.75, 0.70, 0.65]:
            lm = _make_landmarks({0: (x, 0.5)})
            rec.update(lm)
        # Final frame that completes the leftward delta
        lm_final = _make_landmarks({0: (0.5, 0.5)})
        g = rec.update(lm_final)
        self.assertEqual(g, Gesture.SWIPE_LEFT)

    def test_no_gesture_returns_none(self):
        rec = GestureRecogniser()
        # Fist-like landmarks (no gesture)
        lm = _make_landmarks({
            0:  (0.5, 0.9),
            4:  (0.48, 0.85),
            8:  (0.5, 0.72),
            12: (0.52, 0.73),
            16: (0.55, 0.74),
            20: (0.58, 0.76),
        })
        g = rec.update(lm)
        self.assertEqual(g, Gesture.NONE)


if __name__ == "__main__":
    unittest.main()
