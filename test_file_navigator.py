"""
test_file_navigator.py — Unit tests for FileNavigator.

No webcam, no pyautogui side-effects (patched out), no real keyboard events.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Patch pyautogui before importing file_navigator so no display is required
sys.modules["pyautogui"] = MagicMock()

from file_navigator import FileNavigator  # noqa: E402 (import after sys.modules patch)
from gestures import Gesture               # noqa: E402


class TestFileNavigatorInit(unittest.TestCase):

    def test_default_start_is_home(self):
        nav = FileNavigator()
        self.assertEqual(nav.cwd, Path.home())

    def test_custom_start_path(self):
        nav = FileNavigator(start_path="/tmp")
        self.assertEqual(nav.cwd, Path("/tmp"))

    def test_listing_is_populated(self):
        nav = FileNavigator(start_path="/tmp")
        self.assertIsInstance(nav.listing, list)


class TestGoUp(unittest.TestCase):

    def setUp(self):
        # Use a real nested temp path so go_up() has a parent to move to
        self.nav = FileNavigator(start_path="/tmp")

    def test_go_up_moves_to_parent(self):
        original = self.nav.cwd
        self.nav.go_up()
        self.assertEqual(self.nav.cwd, original.parent)

    def test_go_up_at_root_stays_at_root(self):
        self.nav._cwd = Path("/")
        self.nav.go_up()
        self.assertEqual(self.nav.cwd, Path("/"))


class TestGoBackForward(unittest.TestCase):

    def setUp(self):
        self.nav = FileNavigator(start_path="/tmp")

    def test_back_with_no_history_does_nothing(self):
        original = self.nav.cwd
        self.nav.go_back()
        self.assertEqual(self.nav.cwd, original)

    def test_forward_with_no_future_does_nothing(self):
        original = self.nav.cwd
        self.nav.go_forward()
        self.assertEqual(self.nav.cwd, original)

    def test_back_restores_previous(self):
        start = self.nav.cwd
        self.nav.go_up()            # navigate somewhere
        self.nav.go_back()          # should return to start
        self.assertEqual(self.nav.cwd, start)

    def test_forward_after_back(self):
        self.nav.go_up()
        after_up = self.nav.cwd
        self.nav.go_back()
        self.nav.go_forward()
        self.assertEqual(self.nav.cwd, after_up)


class TestScroll(unittest.TestCase):

    def setUp(self):
        self.nav = FileNavigator(start_path="/tmp")
        # Ensure there's something to scroll through
        self.nav._listing = [Path(f"/tmp/fake{i}") for i in range(10)]

    def test_scroll_down(self):
        self.nav.scroll(1)
        self.assertEqual(self.nav.scroll_offset, 1)

    def test_scroll_up_clamps_at_zero(self):
        self.nav.scroll(-1)
        self.assertEqual(self.nav.scroll_offset, 0)

    def test_scroll_down_clamps_at_end(self):
        self.nav._scroll_offset = 9
        self.nav.scroll(1)
        self.assertEqual(self.nav.scroll_offset, 9)


class TestHandleGesture(unittest.TestCase):

    def setUp(self):
        self.nav = FileNavigator(start_path="/tmp")

    def test_pinch_calls_select(self):
        with patch.object(self.nav, "select") as mock_select:
            self.nav.handle_gesture(Gesture.PINCH)
            mock_select.assert_called_once()

    def test_two_fingers_up_calls_go_up(self):
        with patch.object(self.nav, "go_up") as mock_up:
            self.nav.handle_gesture(Gesture.TWO_FINGERS_UP)
            mock_up.assert_called_once()

    def test_swipe_left_calls_go_back(self):
        with patch.object(self.nav, "go_back") as mock_back:
            self.nav.handle_gesture(Gesture.SWIPE_LEFT)
            mock_back.assert_called_once()

    def test_swipe_right_calls_go_forward(self):
        with patch.object(self.nav, "go_forward") as mock_fwd:
            self.nav.handle_gesture(Gesture.SWIPE_RIGHT)
            mock_fwd.assert_called_once()

    def test_open_palm_scroll_direction_upper_half(self):
        """Wrist in upper half (y < 0.5) should scroll up."""
        with patch.object(self.nav, "scroll") as mock_scroll:
            self.nav.handle_gesture(Gesture.OPEN_PALM_SCROLL, wrist_y=0.3)
            mock_scroll.assert_called_once_with(-1)

    def test_open_palm_scroll_direction_lower_half(self):
        """Wrist in lower half (y >= 0.5) should scroll down."""
        with patch.object(self.nav, "scroll") as mock_scroll:
            self.nav.handle_gesture(Gesture.OPEN_PALM_SCROLL, wrist_y=0.7)
            mock_scroll.assert_called_once_with(1)

    def test_none_gesture_does_nothing(self):
        with patch.object(self.nav, "select") as ms, \
             patch.object(self.nav, "go_up") as mu, \
             patch.object(self.nav, "go_back") as mb, \
             patch.object(self.nav, "go_forward") as mf, \
             patch.object(self.nav, "scroll") as msc:
            self.nav.handle_gesture(Gesture.NONE)
            ms.assert_not_called()
            mu.assert_not_called()
            mb.assert_not_called()
            mf.assert_not_called()
            msc.assert_not_called()


if __name__ == "__main__":
    unittest.main()
