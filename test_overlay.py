"""
test_overlay.py — Unit tests for OverlayWindow.

The tkinter module (and its sub-modules) is fully stubbed via sys.modules so
that these tests run in headless CI environments without a display server.
No GUI is created; only the pure logic and thread-safe state are exercised.
"""

import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock

# ── Stub tkinter before importing overlay ─────────────────────────────────────
# overlay.py imports tkinter at the top level; we must pre-stub the module so
# no Tk instance or display connection is attempted during import.
_tk_stub = MagicMock()
sys.modules["tkinter"] = _tk_stub

from overlay import OverlayWindow  # noqa: E402 (import after sys.modules patch)


# ──────────────────────────────────────────────────────────────────────────────
# build_breadcrumb_segments — pure static helper
# ──────────────────────────────────────────────────────────────────────────────

class TestBuildBreadcrumbSegments(unittest.TestCase):
    """
    All tests use the pure static method so no GUI or thread is involved.
    """

    def test_root_only(self):
        segs = OverlayWindow.build_breadcrumb_segments(Path("/"))
        self.assertEqual(segs, [("active", "/")])

    def test_single_level(self):
        segs = OverlayWindow.build_breadcrumb_segments(Path("/home"))
        self.assertEqual(segs, [
            ("normal", "/"),
            ("sep",    " › "),
            ("active", "home"),
        ])

    def test_multi_level(self):
        segs = OverlayWindow.build_breadcrumb_segments(Path("/home/user/docs"))
        expected = [
            ("normal", "/"),
            ("sep",    " › "),
            ("normal", "home"),
            ("sep",    " › "),
            ("normal", "user"),
            ("sep",    " › "),
            ("active", "docs"),
        ]
        self.assertEqual(segs, expected)

    def test_last_segment_is_always_active(self):
        for depth in (1, 2, 5):
            parts = "/".join(["seg"] * depth)
            path = Path(f"/{parts}")
            segs = OverlayWindow.build_breadcrumb_segments(path)
            last_tag, last_text = segs[-1]
            self.assertEqual(
                last_tag, "active",
                f"depth={depth}, path={path}, last segment=({last_tag!r}, {last_text!r})"
            )

    def test_intermediate_segments_are_normal(self):
        segs = OverlayWindow.build_breadcrumb_segments(Path("/a/b/c"))
        normal_segs = [(t, v) for t, v in segs if t == "normal"]
        self.assertTrue(len(normal_segs) >= 1)
        for tag, _ in normal_segs:
            self.assertEqual(tag, "normal")

    def test_separators_between_each_pair(self):
        segs = OverlayWindow.build_breadcrumb_segments(Path("/a/b/c"))
        # Expected order: normal('/'), sep, normal('a'), sep, normal('b'), sep, active('c')
        tags = [t for t, _ in segs]
        # Separators should not be consecutive and should never be first or last
        self.assertNotEqual(tags[0], "sep")
        self.assertNotEqual(tags[-1], "sep")
        for i in range(len(tags) - 1):
            self.assertFalse(tags[i] == "sep" and tags[i + 1] == "sep")

    def test_separator_text_is_arrow(self):
        segs = OverlayWindow.build_breadcrumb_segments(Path("/a/b"))
        sep_texts = [txt for tag, txt in segs if tag == "sep"]
        self.assertTrue(len(sep_texts) >= 1)
        for txt in sep_texts:
            self.assertEqual(txt, " › ")

    def test_returns_list_of_tuples(self):
        segs = OverlayWindow.build_breadcrumb_segments(Path("/tmp"))
        self.assertIsInstance(segs, list)
        for item in segs:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)

    def test_single_component_path_has_no_separator(self):
        segs = OverlayWindow.build_breadcrumb_segments(Path("/"))
        tags = [t for t, _ in segs]
        self.assertNotIn("sep", tags)

    def test_three_component_path_has_two_separators(self):
        # /a/b → [normal, sep, normal, sep, active]  = 2 separators
        segs = OverlayWindow.build_breadcrumb_segments(Path("/a/b"))
        seps = [t for t, _ in segs if t == "sep"]
        self.assertEqual(len(seps), 2)


# ──────────────────────────────────────────────────────────────────────────────
# OverlayWindow — thread-safe state setters
# ──────────────────────────────────────────────────────────────────────────────

class TestOverlayWindowState(unittest.TestCase):
    """
    Exercises the public state setters without starting any GUI.  The daemon
    thread is never launched (start() is not called).
    """

    def _make(self) -> OverlayWindow:
        return OverlayWindow(width=1920, height=1080)

    # ── finger position ────────────────────────────────────────────────────────

    def test_initial_finger_pos_is_none(self):
        ov = self._make()
        self.assertIsNone(ov._finger_pos)

    def test_set_finger_pos_stores_float_tuple(self):
        ov = self._make()
        ov.set_finger_pos(0.4, 0.6)
        self.assertEqual(ov._finger_pos, (0.4, 0.6))

    def test_set_finger_pos_converts_to_float(self):
        ov = self._make()
        ov.set_finger_pos(1, 0)   # integers
        self.assertIsInstance(ov._finger_pos[0], float)
        self.assertIsInstance(ov._finger_pos[1], float)

    def test_set_finger_pos_both_none_clears(self):
        ov = self._make()
        ov.set_finger_pos(0.4, 0.6)
        ov.set_finger_pos(None, None)
        self.assertIsNone(ov._finger_pos)

    def test_set_finger_pos_x_none_clears(self):
        ov = self._make()
        ov.set_finger_pos(0.3, None)
        self.assertIsNone(ov._finger_pos)

    def test_set_finger_pos_y_none_clears(self):
        ov = self._make()
        ov.set_finger_pos(None, 0.7)
        self.assertIsNone(ov._finger_pos)

    def test_set_finger_pos_overwrites_previous(self):
        ov = self._make()
        ov.set_finger_pos(0.1, 0.2)
        ov.set_finger_pos(0.9, 0.8)
        self.assertEqual(ov._finger_pos, (0.9, 0.8))

    # ── current working directory ──────────────────────────────────────────────

    def test_initial_cwd_is_home(self):
        ov = self._make()
        self.assertEqual(ov._cwd, Path.home())

    def test_set_cwd_with_path_object(self):
        ov = self._make()
        ov.set_cwd(Path("/tmp"))
        self.assertEqual(ov._cwd, Path("/tmp"))

    def test_set_cwd_with_string(self):
        ov = self._make()
        ov.set_cwd("/tmp")
        self.assertEqual(ov._cwd, Path("/tmp"))

    def test_set_cwd_stores_path_instance(self):
        ov = self._make()
        ov.set_cwd("/some/dir")
        self.assertIsInstance(ov._cwd, Path)

    def test_set_cwd_overwrites_previous(self):
        ov = self._make()
        ov.set_cwd("/tmp")
        ov.set_cwd("/var")
        self.assertEqual(ov._cwd, Path("/var"))

    # ── thread safety ──────────────────────────────────────────────────────────

    def test_concurrent_set_finger_pos_does_not_raise(self):
        """Rapid concurrent writes from multiple threads must not corrupt state."""
        ov = self._make()
        errors = []

        def writer(x):
            try:
                for _ in range(200):
                    ov.set_finger_pos(x, x)
                    ov.set_finger_pos(None, None)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(i / 10,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [], f"Unexpected errors: {errors}")

    def test_concurrent_set_cwd_does_not_raise(self):
        ov = self._make()
        errors = []

        def writer(p):
            try:
                for _ in range(200):
                    ov.set_cwd(p)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(f"/tmp/dir{i}",)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [], f"Unexpected errors: {errors}")


# ──────────────────────────────────────────────────────────────────────────────
# OverlayWindow — constructor defaults
# ──────────────────────────────────────────────────────────────────────────────

class TestOverlayWindowDefaults(unittest.TestCase):

    def test_width_height_none_by_default(self):
        ov = OverlayWindow()
        self.assertIsNone(ov._width)
        self.assertIsNone(ov._height)

    def test_explicit_dimensions_stored(self):
        ov = OverlayWindow(width=800, height=600)
        self.assertEqual(ov._width, 800)
        self.assertEqual(ov._height, 600)

    def test_root_is_none_before_start(self):
        ov = OverlayWindow()
        self.assertIsNone(ov._root)

    def test_canvas_is_none_before_start(self):
        ov = OverlayWindow()
        self.assertIsNone(ov._canvas)

    def test_crumb_text_is_none_before_start(self):
        ov = OverlayWindow()
        self.assertIsNone(ov._crumb_text)

    def test_daemon_thread_created(self):
        ov = OverlayWindow()
        self.assertIsInstance(ov._thread, threading.Thread)
        self.assertTrue(ov._thread.daemon)

    def test_destroy_before_start_does_not_raise(self):
        """destroy() must be safe to call even before start() is called."""
        ov = OverlayWindow()
        try:
            ov.destroy()
        except Exception as exc:
            self.fail(f"destroy() raised unexpectedly: {exc}")


if __name__ == "__main__":
    unittest.main()
