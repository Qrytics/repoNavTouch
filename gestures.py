"""
gestures.py — Gesture classification from MediaPipe hand landmarks.

Landmark indices used throughout (MediaPipe convention):
  4  = Thumb tip
  8  = Index finger tip
  12 = Middle finger tip
  16 = Ring finger tip
  20 = Pinky tip

  3  = Thumb IP (inner joint)
  6  = Index PIP
  10 = Middle PIP
  14 = Ring PIP
  18 = Pinky PIP
  0  = Wrist
"""

import math
from collections import deque
from enum import Enum, auto


class Gesture(Enum):
    NONE = auto()
    PINCH = auto()          # Index tip + Thumb tip close together → select / click
    SWIPE_LEFT = auto()     # Hand moves right-to-left              → go back / previous
    SWIPE_RIGHT = auto()    # Hand moves left-to-right              → go forward / next
    OPEN_PALM_SCROLL = auto()  # Open palm moving up/down           → scroll file list
    TWO_FINGERS_UP = auto() # Index + middle extended, others curled → cd ..


# ──────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────────────────────────────────────

def _distance(a, b) -> float:
    """Euclidean distance between two landmark objects (or (x, y) tuples)."""
    ax, ay = (a.x, a.y) if hasattr(a, "x") else a
    bx, by = (b.x, b.y) if hasattr(b, "x") else b
    return math.hypot(ax - bx, ay - by)


def _is_finger_extended(landmarks, tip_idx: int, pip_idx: int) -> bool:
    """Return True when the fingertip is farther from the wrist than the PIP joint."""
    wrist = landmarks[0]
    tip = landmarks[tip_idx]
    pip = landmarks[pip_idx]
    return _distance(tip, wrist) > _distance(pip, wrist)


def _is_thumb_extended(landmarks) -> bool:
    """Thumb extension: tip (4) farther from wrist than thumb IP joint (3)."""
    return _distance(landmarks[4], landmarks[0]) > _distance(landmarks[3], landmarks[0])


# ──────────────────────────────────────────────────────────────────────────────
# Per-frame gesture classifiers
# ──────────────────────────────────────────────────────────────────────────────

def detect_pinch(landmarks, threshold: float = 0.07) -> bool:
    """
    Return True when the index-finger tip (8) and thumb tip (4) are within
    *threshold* of each other (normalised image coordinates).
    """
    dist = _distance(landmarks[4], landmarks[8])
    return dist < threshold


def detect_two_fingers_up(landmarks) -> bool:
    """
    Return True when index and middle fingers are extended but ring, pinky,
    and thumb are curled — the classic "peace / two-fingers-up" pose.
    """
    index_up = _is_finger_extended(landmarks, tip_idx=8, pip_idx=6)
    middle_up = _is_finger_extended(landmarks, tip_idx=12, pip_idx=10)
    ring_down = not _is_finger_extended(landmarks, tip_idx=16, pip_idx=14)
    pinky_down = not _is_finger_extended(landmarks, tip_idx=20, pip_idx=18)
    thumb_down = not _is_thumb_extended(landmarks)
    return index_up and middle_up and ring_down and pinky_down and thumb_down


def detect_open_palm(landmarks) -> bool:
    """Return True when all five fingers are extended (open flat hand)."""
    return (
        _is_thumb_extended(landmarks)
        and _is_finger_extended(landmarks, 8, 6)
        and _is_finger_extended(landmarks, 12, 10)
        and _is_finger_extended(landmarks, 16, 14)
        and _is_finger_extended(landmarks, 20, 18)
    )


# ──────────────────────────────────────────────────────────────────────────────
# Stateful gesture recogniser (tracks motion over time)
# ──────────────────────────────────────────────────────────────────────────────

class GestureRecogniser:
    """
    Wraps per-frame classifiers with temporal logic so that swipes and
    hold-based gestures are debounced and do not fire on every frame.

    Parameters
    ----------
    pinch_threshold : float
        Normalised distance below which a pinch is registered.
    swipe_min_delta : float
        Minimum normalised horizontal displacement to register a swipe.
    swipe_history_frames : int
        How many recent wrist positions to inspect for swipe detection.
    cooldown_frames : int
        Minimum frames between two consecutive gesture events of the same type.
    """

    def __init__(
        self,
        pinch_threshold: float = 0.07,
        swipe_min_delta: float = 0.15,
        swipe_history_frames: int = 20,
        cooldown_frames: int = 15,
    ):
        self.pinch_threshold = pinch_threshold
        self.swipe_min_delta = swipe_min_delta
        self.swipe_history_frames = swipe_history_frames
        self.cooldown_frames = cooldown_frames

        # Circular buffer of recent wrist X positions (normalised 0–1)
        self._wrist_x_history: deque = deque(maxlen=swipe_history_frames)
        # Cooldown counters keyed by Gesture member
        self._cooldowns: dict = {g: 0 for g in Gesture}

    def _tick_cooldowns(self):
        for key in self._cooldowns:
            if self._cooldowns[key] > 0:
                self._cooldowns[key] -= 1

    def _fire(self, gesture: Gesture) -> Gesture:
        """Register a gesture fire and reset its cooldown."""
        self._cooldowns[gesture] = self.cooldown_frames
        return gesture

    def _ready(self, gesture: Gesture) -> bool:
        return self._cooldowns[gesture] == 0

    def update(self, landmarks) -> Gesture:
        """
        Accept a list of 21 MediaPipe NormalizedLandmark objects for one frame
        and return the dominant Gesture (or Gesture.NONE).
        """
        self._tick_cooldowns()

        # Record wrist position for swipe tracking
        wrist_x = landmarks[0].x
        self._wrist_x_history.append(wrist_x)

        # ── Pinch ────────────────────────────────────────────────────────────
        if detect_pinch(landmarks, self.pinch_threshold) and self._ready(Gesture.PINCH):
            return self._fire(Gesture.PINCH)

        # ── Two fingers up (cd ..) ───────────────────────────────────────────
        if detect_two_fingers_up(landmarks) and self._ready(Gesture.TWO_FINGERS_UP):
            return self._fire(Gesture.TWO_FINGERS_UP)

        # ── Swipe (need enough history) ──────────────────────────────────────
        if len(self._wrist_x_history) == self.swipe_history_frames:
            delta = self._wrist_x_history[-1] - self._wrist_x_history[0]
            if delta < -self.swipe_min_delta and self._ready(Gesture.SWIPE_LEFT):
                self._wrist_x_history.clear()
                return self._fire(Gesture.SWIPE_LEFT)
            if delta > self.swipe_min_delta and self._ready(Gesture.SWIPE_RIGHT):
                self._wrist_x_history.clear()
                return self._fire(Gesture.SWIPE_RIGHT)

        # ── Open palm scroll ─────────────────────────────────────────────────
        if detect_open_palm(landmarks) and self._ready(Gesture.OPEN_PALM_SCROLL):
            return self._fire(Gesture.OPEN_PALM_SCROLL)

        return Gesture.NONE
