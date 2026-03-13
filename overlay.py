"""
overlay.py — Transparent always-on-top overlay for repoNavTouch.

Draws a glowing circle that tracks the index-finger tip (MediaPipe landmark 8)
across the screen and renders a breadcrumb trail showing the current folder
hierarchy being navigated.

The overlay is built with the Python standard-library ``tkinter`` module so no
extra package is required.  It runs in a background daemon thread, which keeps
the main OpenCV / MediaPipe capture loop unblocked.  All state updates from the
capture thread are serialised through a :class:`threading.Lock`.

Usage
-----
    from overlay import OverlayWindow

    overlay = OverlayWindow()
    overlay.start()               # launches daemon thread — returns immediately

    # inside capture loop:
    overlay.set_finger_pos(lm[8].x, lm[8].y)   # normalised 0–1 coords
    overlay.set_cwd(navigator.cwd)              # pathlib.Path or str

    # on shutdown:
    overlay.destroy()
"""

import threading
import tkinter as tk
from pathlib import Path


# ─── Transparent key colour ────────────────────────────────────────────────────
# This exact magenta is used as the transparent "chroma key" colour.
# It must not appear anywhere in the drawn content.
_TRANSPARENT_KEY = "#fe01fd"

# ─── Breadcrumb bar palette ────────────────────────────────────────────────────
_CRUMB_BG     = "#111122"   # dark panel behind the trail
_CRUMB_NORMAL = "#aaaacc"   # colour for intermediate path segments
_CRUMB_ACTIVE = "#00d4ff"   # accent colour for the current (last) segment
_CRUMB_SEP    = "#445566"   # separator › colour
_CRUMB_HEIGHT = 36          # breadcrumb strip height in pixels
_CRUMB_FONT   = ("Helvetica", 12, "bold")

# ─── Glow ring definitions (outermost → innermost) ────────────────────────────
# Each tuple is (radius_px, fill_colour).  Drawing outermost first lets each
# inner ring paint on top, producing a natural radial glow gradient.
_GLOW_RINGS = [
    (28, "#001828"),
    (21, "#003a52"),
    (15, "#006688"),
    (10, "#00aad4"),
    (6,  "#00eeff"),
    (3,  "#ffffff"),
]

# Overall window alpha used as a fallback when -transparentcolor is unsupported.
_OVERLAY_ALPHA = 0.78


class OverlayWindow:
    """
    Semi-transparent always-on-top tkinter window for repoNavTouch.

    Two visual elements are rendered every ~16 ms (~60 fps):

    * **Glow circle** — a cluster of concentric coloured ovals on a transparent
      canvas canvas, centred at the normalised screen position of the user's
      index-finger tip.
    * **Breadcrumb bar** — a dark horizontal strip across the bottom of the
      screen showing the current directory path as
      ``/ › home › user › projects``, with the last segment highlighted in the
      accent colour.

    The window is:

    * **Frameless** (``overrideredirect=True``) — no title bar or window chrome.
    * **Always-on-top** (``-topmost True``) — floats above the terminal.
    * **Background-transparent** via ``-transparentcolor`` (Windows / some
      Linux compositing WMs) or via ``-alpha`` as a fallback.

    All public methods are safe to call from any thread.

    Parameters
    ----------
    width, height : int | None
        Window dimensions in pixels.  Defaults to the full primary screen size
        when *None* (determined once the tkinter root is available).
    """

    def __init__(self, width: int | None = None, height: int | None = None):
        self._lock = threading.Lock()
        self._finger_pos: tuple[float, float] | None = None
        self._cwd: Path = Path.home()
        self._width = width
        self._height = height

        # Populated once _run() builds the widgets
        self._root: tk.Tk | None = None
        self._canvas: tk.Canvas | None = None
        self._crumb_text: tk.Text | None = None
        self._sw: int = 0
        self._sh: int = 0
        self._canvas_height: int = 0

        self._thread = threading.Thread(
            target=self._run, daemon=True, name="overlay-thread"
        )

    # ── Public thread-safe API ─────────────────────────────────────────────────

    def start(self) -> None:
        """Launch the overlay in a background daemon thread."""
        self._thread.start()

    def set_finger_pos(
        self, x_norm: float | None, y_norm: float | None
    ) -> None:
        """
        Update the index-finger tip position in normalised camera space (0–1).

        Pass *None* for both arguments when no hand is detected so the glow
        circle is hidden.
        """
        with self._lock:
            if x_norm is None or y_norm is None:
                self._finger_pos = None
            else:
                self._finger_pos = (float(x_norm), float(y_norm))

    def set_cwd(self, path) -> None:
        """Update the path displayed in the breadcrumb trail."""
        with self._lock:
            self._cwd = Path(path)

    def destroy(self) -> None:
        """Request clean shutdown of the overlay (safe to call from any thread)."""
        root = self._root
        if root is not None:
            try:
                root.after(0, root.destroy)
            except Exception:
                pass

    # ── Pure helper ────────────────────────────────────────────────────────────

    @staticmethod
    def build_breadcrumb_segments(cwd: Path) -> list:
        """
        Convert a directory path into an ordered list of ``(tag, text)`` pairs
        ready to insert into a tk.Text widget.

        Tags
        ----
        ``'normal'``
            An intermediate path component rendered in the muted foreground colour.
        ``'sep'``
            The ``›`` separator between components, rendered in a dimmed colour.
        ``'active'``
            The final (current) component, rendered in the accent colour.

        Parameters
        ----------
        cwd : Path
            The current working directory to represent as a breadcrumb.

        Returns
        -------
        list[tuple[str, str]]
            Ordered ``[(tag, text), …]`` pairs.

        Examples
        --------
        >>> OverlayWindow.build_breadcrumb_segments(Path('/home/user/docs'))
        [('normal', '/'), ('sep', ' › '), ('normal', 'home'), ('sep', ' › '),
         ('normal', 'user'), ('sep', ' › '), ('active', 'docs')]
        """
        parts = cwd.parts
        if not parts:
            return [("active", str(cwd))]

        segments: list[tuple[str, str]] = []
        for i, part in enumerate(parts):
            is_last = i == len(parts) - 1
            # Normalise display text: strip trailing separators from the root
            # component so POSIX '/' stays as '/' and a Windows drive such as
            # r'C:\' becomes 'C:'.
            display = part.rstrip("/\\") or "/"
            tag = "active" if is_last else "normal"
            segments.append((tag, display))
            if not is_last:
                segments.append(("sep", " › "))

        return segments

    # ── Internal (daemon thread) ───────────────────────────────────────────────

    def _run(self) -> None:
        """Build the tkinter window and enter its event loop (daemon thread)."""
        root = tk.Tk()
        self._root = root

        # ── Window chrome ──────────────────────────────────────────────────────
        root.overrideredirect(True)           # remove title bar / window border
        root.wm_attributes("-topmost", True)  # always above other windows

        sw = self._width  or root.winfo_screenwidth()
        sh = self._height or root.winfo_screenheight()
        root.geometry(f"{sw}x{sh}+0+0")
        root.configure(bg=_TRANSPARENT_KEY)
        self._sw = sw
        self._sh = sh

        # ── Transparency ───────────────────────────────────────────────────────
        # Prefer making the key colour fully transparent (pixels become see-through).
        # This works on Windows and Linux compositing WMs that honour the hint.
        # Fall back to a semi-transparent overall alpha on systems that don't.
        try:
            root.wm_attributes("-transparentcolor", _TRANSPARENT_KEY)
        except Exception:
            try:
                root.wm_attributes("-alpha", _OVERLAY_ALPHA)
            except Exception:
                pass

        # ── Canvas (transparent background for glow drawing) ──────────────────
        canvas_height = sh - _CRUMB_HEIGHT
        self._canvas_height = canvas_height
        canvas = tk.Canvas(
            root,
            width=sw,
            height=canvas_height,
            bg=_TRANSPARENT_KEY,
            highlightthickness=0,
        )
        canvas.pack(side="top", fill="both", expand=True)
        self._canvas = canvas

        # ── Breadcrumb bar (bottom strip) ──────────────────────────────────────
        crumb_frame = tk.Frame(root, bg=_CRUMB_BG, height=_CRUMB_HEIGHT)
        crumb_frame.pack(side="bottom", fill="x")
        crumb_frame.pack_propagate(False)

        crumb_text = tk.Text(
            crumb_frame,
            height=1,
            bg=_CRUMB_BG,
            fg=_CRUMB_NORMAL,
            font=_CRUMB_FONT,
            borderwidth=0,
            highlightthickness=0,
            state="disabled",
            wrap="none",
            cursor="",
        )
        crumb_text.tag_configure("normal", foreground=_CRUMB_NORMAL)
        crumb_text.tag_configure("active", foreground=_CRUMB_ACTIVE)
        crumb_text.tag_configure("sep",    foreground=_CRUMB_SEP)
        crumb_text.pack(fill="both", expand=True, padx=8, pady=4)
        self._crumb_text = crumb_text

        # ── Start periodic redraw ──────────────────────────────────────────────
        root.after(16, self._tick)
        root.mainloop()

    def _tick(self) -> None:
        """Periodic callback (~60 fps): snapshot shared state and repaint."""
        with self._lock:
            finger_pos = self._finger_pos
            cwd = self._cwd

        self._redraw_glow(finger_pos)
        self._redraw_breadcrumb(cwd)

        root = self._root
        if root is not None:
            root.after(16, self._tick)

    def _redraw_glow(self, finger_pos: tuple[float, float] | None) -> None:
        """Repaint the glowing circle on the transparent canvas."""
        canvas = self._canvas
        if canvas is None:
            return

        canvas.delete("glow")

        if finger_pos is None:
            return  # no hand detected — leave canvas blank

        x_norm, y_norm = finger_pos
        px = int(x_norm * self._sw)
        py = int(y_norm * self._canvas_height)

        # Paint from outermost ring to innermost so inner rings overwrite outer ones.
        for radius, colour in _GLOW_RINGS:
            canvas.create_oval(
                px - radius, py - radius,
                px + radius, py + radius,
                fill=colour,
                outline="",
                tags="glow",
            )

    def _redraw_breadcrumb(self, cwd: Path) -> None:
        """Update the breadcrumb text widget with the latest path."""
        crumb_text = self._crumb_text
        if crumb_text is None:
            return

        segments = self.build_breadcrumb_segments(cwd)
        crumb_text.config(state="normal")
        crumb_text.delete("1.0", "end")
        for tag, text in segments:
            crumb_text.insert("end", text, tag)
        crumb_text.config(state="disabled")
