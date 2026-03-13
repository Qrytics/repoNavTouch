"""
file_navigator.py — File-system navigation layer driven by gestures.

Maps Gesture events onto OS-level directory navigation and keyboard actions.
"""

import os
import subprocess
import sys
from pathlib import Path

import pyautogui

from gestures import Gesture


class FileNavigator:
    """
    Maintains a current working directory and translates Gesture events into
    file-system navigation commands.  All actions are also forwarded to the
    active terminal window via PyAutoGUI so the user sees the effect live.

    Parameters
    ----------
    start_path : str | Path | None
        Initial directory.  Defaults to the user's home directory.
    """

    def __init__(self, start_path=None):
        self._cwd = Path(start_path or Path.home()).resolve()
        self._history: list[Path] = []   # backward history stack
        self._forward: list[Path] = []   # forward history stack
        self._scroll_offset: int = 0     # index into directory listing

        self._listing: list[Path] = []
        self._refresh_listing()

    # ──────────────────────────────────────────────────────────────────────────
    # Properties
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def cwd(self) -> Path:
        return self._cwd

    @property
    def listing(self) -> list[Path]:
        return self._listing

    @property
    def scroll_offset(self) -> int:
        return self._scroll_offset

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _refresh_listing(self):
        try:
            entries = sorted(self._cwd.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except PermissionError:
            entries = []
        self._listing = entries
        self._scroll_offset = 0

    def _navigate_to(self, path: Path):
        self._history.append(self._cwd)
        self._forward.clear()
        self._cwd = path.resolve()
        self._refresh_listing()
        print(f"[nav] cd {self._cwd}")
        self._send_cd_to_terminal(self._cwd)

    def _send_cd_to_terminal(self, path: Path):
        """Type `cd <path> ↵` into whatever terminal is focused."""
        try:
            pyautogui.hotkey("ctrl", "c")          # cancel any running command
            pyautogui.typewrite(f"cd {path}", interval=0.02)
            pyautogui.press("enter")
        except Exception:
            pass  # pyautogui may fail in headless environments — ignore

    # ──────────────────────────────────────────────────────────────────────────
    # Public navigation API
    # ──────────────────────────────────────────────────────────────────────────

    def go_up(self):
        """Navigate to the parent directory (Two Fingers Up gesture)."""
        parent = self._cwd.parent
        if parent != self._cwd:
            self._navigate_to(parent)
        else:
            print("[nav] Already at root — cannot go higher.")

    def go_back(self):
        """Navigate to the previous directory (Swipe Left gesture)."""
        if self._history:
            self._forward.append(self._cwd)
            self._cwd = self._history.pop()
            self._refresh_listing()
            print(f"[nav] back → {self._cwd}")
            self._send_cd_to_terminal(self._cwd)
        else:
            print("[nav] No back history.")

    def go_forward(self):
        """Navigate forward after a back (Swipe Right gesture)."""
        if self._forward:
            self._history.append(self._cwd)
            self._cwd = self._forward.pop()
            self._refresh_listing()
            print(f"[nav] forward → {self._cwd}")
            self._send_cd_to_terminal(self._cwd)
        else:
            print("[nav] No forward history.")

    def select(self):
        """
        Enter the highlighted directory, or open a file (Pinch gesture).
        Uses the entry at the current scroll offset.
        """
        if not self._listing:
            print("[nav] Directory is empty.")
            return
        target = self._listing[self._scroll_offset]
        if target.is_dir():
            self._navigate_to(target)
        else:
            print(f"[nav] open file: {target}")
            try:
                if sys.platform == "darwin":
                    subprocess.Popen(["open", str(target)])
                elif sys.platform.startswith("linux"):
                    subprocess.Popen(["xdg-open", str(target)])
                else:
                    # On Windows, 'start' is a shell built-in and requires shell=True
                    subprocess.Popen(["start", str(target)], shell=True)
            except Exception as exc:
                print(f"[nav] Could not open file: {exc}")

    def scroll(self, direction: int):
        """
        Move the scroll cursor through the directory listing.

        Parameters
        ----------
        direction : int
            +1 to scroll down (next entry), -1 to scroll up (previous entry).
        """
        if not self._listing:
            return
        self._scroll_offset = max(0, min(len(self._listing) - 1, self._scroll_offset + direction))
        print(f"[nav] scroll → {self._listing[self._scroll_offset].name}")
        try:
            key = "down" if direction > 0 else "up"
            pyautogui.press(key)
        except Exception:
            pass

    # ──────────────────────────────────────────────────────────────────────────
    # Gesture → action dispatcher
    # ──────────────────────────────────────────────────────────────────────────

    def handle_gesture(self, gesture: Gesture, wrist_y: float = 0.0):
        """
        Translate a Gesture enum value into the appropriate navigation action.

        Parameters
        ----------
        gesture : Gesture
            The gesture detected in the current frame.
        wrist_y : float
            Normalised Y position of the wrist (0 = top, 1 = bottom of frame).
            Used to determine scroll direction for OPEN_PALM_SCROLL.
        """
        if gesture == Gesture.PINCH:
            print("[gesture] Pinch Detected → select")
            self.select()

        elif gesture == Gesture.TWO_FINGERS_UP:
            print("[gesture] Two Fingers Up → cd ..")
            self.go_up()

        elif gesture == Gesture.SWIPE_LEFT:
            print("[gesture] Swipe Left → go back")
            self.go_back()

        elif gesture == Gesture.SWIPE_RIGHT:
            print("[gesture] Swipe Right → go forward")
            self.go_forward()

        elif gesture == Gesture.OPEN_PALM_SCROLL:
            # Wrist in upper half of frame → scroll up, lower half → scroll down
            direction = -1 if wrist_y < 0.5 else 1
            print(f"[gesture] Open Palm → scroll {'up' if direction < 0 else 'down'}")
            self.scroll(direction)
