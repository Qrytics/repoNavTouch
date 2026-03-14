"""
test_file_navigator.py — Unit tests for FileNavigator.

No webcam, no pyautogui side-effects (patched out), no real keyboard events.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Patch pyautogui before importing file_navigator so no display is required
sys.modules["pyautogui"] = MagicMock()

from file_navigator import FileNavigator, list_folders  # noqa: E402 (import after sys.modules patch)
from gestures import Gesture                             # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# list_folders()
# ──────────────────────────────────────────────────────────────────────────────

class TestListFolders(unittest.TestCase):

    def test_returns_only_directories(self):
        """list_folders must not include files — only directories."""
        folders = list_folders("/tmp")
        for p in folders:
            self.assertTrue(p.is_dir(), f"{p} should be a directory")

    def test_returns_list_of_paths(self):
        folders = list_folders("/tmp")
        self.assertIsInstance(folders, list)
        for p in folders:
            self.assertIsInstance(p, Path)

    def test_sorted_alphabetically(self):
        folders = list_folders("/tmp")
        names = [p.name.lower() for p in folders]
        self.assertEqual(names, sorted(names))

    def test_accepts_string_path(self):
        result = list_folders("/tmp")
        self.assertIsInstance(result, list)

    def test_accepts_path_object(self):
        result = list_folders(Path("/tmp"))
        self.assertIsInstance(result, list)

    def test_nonexistent_directory_returns_empty(self):
        result = list_folders("/tmp/this_path_does_not_exist_xyz123")
        self.assertEqual(result, [])


# ──────────────────────────────────────────────────────────────────────────────
# FileNavigator init & properties
# ──────────────────────────────────────────────────────────────────────────────

class TestFileNavigatorInit(unittest.TestCase):

    def test_default_start_is_home(self):
        nav = FileNavigator()
        self.assertEqual(nav.cwd, Path.home())

    def test_custom_start_path(self):
        nav = FileNavigator(start_path="/tmp")
        self.assertEqual(nav.cwd, Path("/tmp").resolve())

    def test_listing_is_populated(self):
        nav = FileNavigator(start_path="/tmp")
        self.assertIsInstance(nav.listing, list)

    def test_folders_property_contains_only_dirs(self):
        nav = FileNavigator(start_path="/tmp")
        for p in nav.folders:
            self.assertTrue(p.is_dir())

    def test_current_index_starts_at_zero(self):
        nav = FileNavigator(start_path="/tmp")
        self.assertEqual(nav.current_index, 0)


# ──────────────────────────────────────────────────────────────────────────────
# go_up
# ──────────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────────
# go_back / go_forward
# ──────────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────────
# Scroll
# ──────────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────────
# advance_folder_index
# ──────────────────────────────────────────────────────────────────────────────

class TestAdvanceFolderIndex(unittest.TestCase):

    def _nav_with_fake_folders(self, count: int) -> FileNavigator:
        """Create a navigator whose folder listing is replaced with fake dirs."""
        nav = FileNavigator(start_path="/tmp")
        nav._folder_listing = [Path(f"/tmp/dir{i}") for i in range(count)]
        nav._current_index = 0
        return nav

    def test_advances_by_one(self):
        nav = self._nav_with_fake_folders(3)
        nav.advance_folder_index()
        self.assertEqual(nav.current_index, 1)

    def test_wraps_around_at_end(self):
        nav = self._nav_with_fake_folders(3)
        nav._current_index = 2
        nav.advance_folder_index()
        self.assertEqual(nav.current_index, 0)

    def test_no_folders_does_not_change_index(self):
        nav = FileNavigator(start_path="/tmp")
        nav._folder_listing = []
        nav._current_index = 0
        nav.advance_folder_index()
        self.assertEqual(nav.current_index, 0)

    def test_single_folder_stays_at_zero_after_wrap(self):
        nav = self._nav_with_fake_folders(1)
        nav.advance_folder_index()
        self.assertEqual(nav.current_index, 0)


# ──────────────────────────────────────────────────────────────────────────────
# print_directory_structure
# ──────────────────────────────────────────────────────────────────────────────

class TestPrintDirectoryStructure(unittest.TestCase):

    def test_prints_cwd_header(self):
        nav = FileNavigator(start_path="/tmp")
        nav._listing = []
        with patch("builtins.print") as mock_print:
            nav.print_directory_structure()
        # The first print call should include the cwd path
        first_call_args = mock_print.call_args_list[0][0][0]
        self.assertIn(str(nav.cwd), first_call_args)

    def test_prints_each_entry(self):
        nav = FileNavigator(start_path="/tmp")
        fake_dir = MagicMock(spec=Path)
        fake_dir.is_dir.return_value = True
        fake_dir.name = "mydir"
        fake_file = MagicMock(spec=Path)
        fake_file.is_dir.return_value = False
        fake_file.name = "myfile.txt"
        nav._listing = [fake_dir, fake_file]

        output_lines = []
        with patch("builtins.print", side_effect=lambda *a, **kw: output_lines.append(a[0] if a else "")):
            nav.print_directory_structure()

        combined = "\n".join(output_lines)
        self.assertIn("mydir", combined)
        self.assertIn("myfile.txt", combined)


# ──────────────────────────────────────────────────────────────────────────────
# enter_current_folder
# ──────────────────────────────────────────────────────────────────────────────

class TestEnterCurrentFolder(unittest.TestCase):

    def setUp(self):
        self.nav = FileNavigator(start_path="/tmp")

    def test_navigates_into_folder_at_current_index(self):
        """enter_current_folder should call _navigate_to with the right path."""
        target = Path("/tmp")
        self.nav._folder_listing = [target]
        self.nav._current_index = 0
        with patch.object(self.nav, "_navigate_to") as mock_nav, \
             patch("os.chdir"), \
             patch.object(self.nav, "print_directory_structure"):
            self.nav.enter_current_folder()
            mock_nav.assert_called_once_with(target)

    def test_calls_os_chdir(self):
        """enter_current_folder must call os.chdir with the target path."""
        target = Path("/tmp")
        self.nav._folder_listing = [target]
        self.nav._current_index = 0
        with patch("os.chdir") as mock_chdir, \
             patch.object(self.nav, "_navigate_to"), \
             patch.object(self.nav, "print_directory_structure"):
            self.nav.enter_current_folder()
            mock_chdir.assert_called_once_with(target)

    def test_prints_directory_structure_after_move(self):
        target = Path("/tmp")
        self.nav._folder_listing = [target]
        self.nav._current_index = 0
        with patch("os.chdir"), \
             patch.object(self.nav, "_navigate_to"), \
             patch.object(self.nav, "print_directory_structure") as mock_print:
            self.nav.enter_current_folder()
            mock_print.assert_called_once()

    def test_no_folders_does_nothing(self):
        self.nav._folder_listing = []
        with patch.object(self.nav, "_navigate_to") as mock_nav, \
             patch("os.chdir") as mock_chdir:
            self.nav.enter_current_folder()
            mock_nav.assert_not_called()
            mock_chdir.assert_not_called()

    def test_oserror_aborts_navigation(self):
        target = Path("/tmp")
        self.nav._folder_listing = [target]
        self.nav._current_index = 0
        with patch("os.chdir", side_effect=OSError("permission denied")), \
             patch.object(self.nav, "_navigate_to") as mock_nav:
            self.nav.enter_current_folder()
            mock_nav.assert_not_called()


# ──────────────────────────────────────────────────────────────────────────────
# handle_gesture dispatcher
# ──────────────────────────────────────────────────────────────────────────────

class TestHandleGesture(unittest.TestCase):

    def setUp(self):
        self.nav = FileNavigator(start_path="/tmp")

    def test_pinch_calls_enter_current_folder(self):
        with patch.object(self.nav, "enter_current_folder") as mock_enter:
            self.nav.handle_gesture(Gesture.PINCH)
            mock_enter.assert_called_once()

    def test_two_fingers_up_calls_go_up(self):
        with patch.object(self.nav, "go_up") as mock_up:
            self.nav.handle_gesture(Gesture.TWO_FINGERS_UP)
            mock_up.assert_called_once()

    def test_swipe_left_calls_go_back(self):
        with patch.object(self.nav, "go_back") as mock_back:
            self.nav.handle_gesture(Gesture.SWIPE_LEFT)
            mock_back.assert_called_once()

    def test_swipe_right_calls_advance_folder_index(self):
        with patch.object(self.nav, "advance_folder_index") as mock_adv:
            self.nav.handle_gesture(Gesture.SWIPE_RIGHT)
            mock_adv.assert_called_once()

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
        with patch.object(self.nav, "enter_current_folder") as me, \
             patch.object(self.nav, "go_up") as mu, \
             patch.object(self.nav, "go_back") as mb, \
             patch.object(self.nav, "advance_folder_index") as ma, \
             patch.object(self.nav, "scroll") as msc:
            self.nav.handle_gesture(Gesture.NONE)
            me.assert_not_called()
            mu.assert_not_called()
            mb.assert_not_called()
            ma.assert_not_called()
            msc.assert_not_called()


if __name__ == "__main__":
    unittest.main()
