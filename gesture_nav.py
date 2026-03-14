"""
gesture_nav.py — Main entry point for repoNavTouch.

See → Think → Act pipeline
──────────────────────────
See  : OpenCV grabs webcam frames and feeds them to MediaPipe Hand Landmarker.
Think: GestureRecogniser classifies the landmarks into a Gesture enum value.
Act  : FileNavigator translates the Gesture into OS-level navigation commands.

Usage
-----
    python gesture_nav.py [--camera INDEX] [--pinch-threshold FLOAT]
                          [--start-dir PATH]

Press  Q  or  Esc  to quit.
"""

import argparse
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from gestures import Gesture, GestureRecogniser, detect_pinch
from file_navigator import FileNavigator
from overlay import OverlayWindow

# ──────────────────────────────────────────────────────────────────────────────
# MediaPipe setup (Tasks API — mediapipe >= 0.10.x)
# ──────────────────────────────────────────────────────────────────────────────

_mp_vision = mp.tasks.vision
_mp_base = mp.tasks.BaseOptions

# Hand landmarker model (downloaded on first run and cached in ~/.cache)
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
_MODEL_PATH = Path.home() / ".cache" / "reponavtouch" / "hand_landmarker.task"


def _ensure_model() -> Path:
    """Download the hand landmarker model if it is not already cached."""
    if not _MODEL_PATH.exists():
        _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"[model] Downloading hand landmarker model to {_MODEL_PATH} …")
        try:
            urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        except Exception as exc:
            # Remove any partial download before re-raising
            if _MODEL_PATH.exists():
                _MODEL_PATH.unlink()
            raise RuntimeError(
                f"[model] Failed to download hand landmarker model from {_MODEL_URL}: {exc}"
            ) from exc
        print("[model] Download complete.")
    return _MODEL_PATH


# Colour used to highlight the pinch pair (BGR)
_PINCH_COLOUR = (0, 255, 0)   # green when pinching
_DEFAULT_COLOUR = (255, 0, 0)  # blue otherwise

# ──────────────────────────────────────────────────────────────────────────────
# Overlay helpers
# ──────────────────────────────────────────────────────────────────────────────

def _draw_landmarks(frame, lm):
    """Draw all 21 hand landmarks and their connections onto *frame*."""
    _mp_vision.drawing_utils.draw_landmarks(
        frame,
        lm,
        _mp_vision.HandLandmarksConnections.HAND_CONNECTIONS,
        _mp_vision.drawing_styles.get_default_hand_landmarks_style(),
        _mp_vision.drawing_styles.get_default_hand_connections_style(),
    )


def _draw_pinch_indicator(frame, landmarks, is_pinching: bool):
    """Draw a line between thumb tip and index finger tip; colour shows state."""
    h, w = frame.shape[:2]
    thumb = landmarks[4]
    index = landmarks[8]

    pt1 = (int(thumb.x * w), int(thumb.y * h))
    pt2 = (int(index.x * w), int(index.y * h))
    colour = _PINCH_COLOUR if is_pinching else _DEFAULT_COLOUR
    cv2.line(frame, pt1, pt2, colour, 2)
    cv2.circle(frame, pt1, 6, colour, -1)
    cv2.circle(frame, pt2, 6, colour, -1)


def _draw_hud(frame, gesture: Gesture, cwd: str, listing, scroll_offset: int):
    """Render the heads-up display overlay (current dir, listing, last gesture)."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 150), (0, 0, 0), -1)
    alpha = 0.45
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    # Current working directory
    cv2.putText(frame, f"Dir: {cwd}", (10, 22), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    # Last gesture
    gesture_label = gesture.name.replace("_", " ") if gesture != Gesture.NONE else ""
    if gesture_label:
        cv2.putText(frame, f"Gesture: {gesture_label}", (10, 46), font, 0.55, (0, 220, 255), 1, cv2.LINE_AA)

    # Directory listing (up to 4 visible entries)
    visible = 4
    start = max(0, scroll_offset - 1)
    end = min(len(listing), start + visible)
    for i, entry in enumerate(listing[start:end]):
        idx = start + i
        prefix = "▶ " if idx == scroll_offset else "  "
        label = ("📁 " if entry.is_dir() else "📄 ") + entry.name
        colour = (0, 255, 180) if idx == scroll_offset else (200, 200, 200)
        cv2.putText(frame, prefix + label, (10, 74 + i * 20), font, 0.48, colour, 1, cv2.LINE_AA)

    # Key legend
    cv2.putText(frame, "Q/Esc: quit", (w - 120, h - 10), font, 0.4, (150, 150, 150), 1, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────────────────

def run(camera_index: int = 0, pinch_threshold: float = 0.07, start_path=None):
    """Open the webcam and start the gesture navigation loop."""
    recogniser = GestureRecogniser(pinch_threshold=pinch_threshold)
    navigator = FileNavigator(start_path=start_path)

    # ── Transparent overlay (runs in background daemon thread) ────────────────
    overlay = OverlayWindow()
    overlay.start()
    overlay.set_cwd(navigator.cwd)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        overlay.destroy()
        print(f"[error] Cannot open camera {camera_index}.", file=sys.stderr)
        sys.exit(1)

    last_gesture = Gesture.NONE

    # Ensure the hand landmarker model is available (downloads on first run)
    model_path = _ensure_model()

    options = _mp_vision.HandLandmarkerOptions(
        base_options=_mp_base(model_asset_path=str(model_path)),
        running_mode=_mp_vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    # Monotonic clock reference so timestamps are always increasing
    _start_time = time.monotonic()
    # Track the last timestamp sent to MediaPipe so we can guarantee strict
    # monotonicity even when two frames are captured within the same millisecond.
    _last_timestamp_ms: int = -1

    with _mp_vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[warn] Empty frame — retrying…")
                continue

            # Flip horizontally for a mirror view, then convert colour space
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Build a MediaPipe Image and compute a monotonic timestamp (ms).
            # Guard against two frames landing in the same millisecond by
            # clamping the value to always be strictly greater than the last
            # one sent to MediaPipe (the library raises ValueError otherwise).
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.monotonic() - _start_time) * 1000)
            if timestamp_ms <= _last_timestamp_ms:
                timestamp_ms = _last_timestamp_ms + 1
            _last_timestamp_ms = timestamp_ms

            results = landmarker.detect_for_video(mp_image, timestamp_ms)

            gesture = Gesture.NONE

            if results.hand_landmarks:
                lm = results.hand_landmarks[0]

                # ── Draw landmarks ───────────────────────────────────────────
                _draw_landmarks(frame, lm)

                # ── Pinch detection (shown even outside cooldown) ────────────
                pinching = detect_pinch(lm, pinch_threshold)
                _draw_pinch_indicator(frame, lm, pinching)

                # ── Full gesture classification ──────────────────────────────
                gesture = recogniser.update(lm)
                if gesture != Gesture.NONE:
                    last_gesture = gesture
                    navigator.handle_gesture(gesture, wrist_y=lm[0].y)

                # ── Update overlay: index-finger tip (landmark 8) position ────
                overlay.set_finger_pos(lm[8].x, lm[8].y)
            else:
                # No hand detected — reset hold-state so next detection starts fresh.
                overlay.set_finger_pos(None, None)  # hide glow circle

            # ── Keep overlay breadcrumb current ───────────────────────────────
            overlay.set_cwd(navigator.cwd)

            # ── HUD overlay ──────────────────────────────────────────────────
            _draw_hud(
                frame,
                last_gesture,
                str(navigator.cwd),
                navigator.listing,
                navigator.scroll_offset,
            )

            cv2.imshow("repoNavTouch — Hand Gesture File Navigator", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):  # Q or Esc
                break

    cap.release()
    cv2.destroyAllWindows()
    overlay.destroy()


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def _ask_start_dir() -> Path:
    """
    Interactively prompt the user for the starting directory.

    Accepts a blank answer (uses the home directory), ``~``-prefixed paths,
    and relative or absolute paths.  Re-prompts if the path does not exist
    or is not a directory.
    """
    default = Path.home()
    print("\nrepoNavTouch -- Hand Gesture File Navigator")
    print("-----------------------------------------")
    while True:
        raw = input(f"Starting directory [{default}]: ").strip()
        if not raw:
            return default
        candidate = Path(raw).expanduser().resolve()
        if candidate.is_dir():
            return candidate
        print(f"  [!] '{raw}' is not a valid directory. Please try again.")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Navigate your file system with hand gestures via webcam."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        metavar="INDEX",
        help="Webcam device index (default: 0).",
    )
    parser.add_argument(
        "--pinch-threshold",
        type=float,
        default=0.07,
        metavar="FLOAT",
        help="Normalised distance threshold for pinch detection (default: 0.07).",
    )
    parser.add_argument(
        "--start-dir",
        type=str,
        default=None,
        metavar="PATH",
        help="Starting directory (default: prompt on startup).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.start_dir is not None:
        start = Path(args.start_dir).expanduser().resolve()
        if not start.is_dir():
            print(f"[error] --start-dir '{args.start_dir}' is not a valid directory.", file=sys.stderr)
            sys.exit(1)
    else:
        start = _ask_start_dir()
    run(camera_index=args.camera, pinch_threshold=args.pinch_threshold, start_path=start)
