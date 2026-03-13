# repoNavTouch

Hook up your webcam and use hand gestures to navigate through your computer files. Available for file explorer and terminals.

---

## Overview

**repoNavTouch** implements a *See → Think → Act* pipeline:

| Stage | Technology | Responsibility |
|-------|-----------|----------------|
| **See** | OpenCV + MediaPipe | Capture webcam frames, detect 21 3-D hand landmarks |
| **Think** | `gestures.py` | Classify landmarks into a `Gesture` enum (pinch, swipe, palm, …) |
| **Act** | `file_navigator.py` | Translate gestures into OS-level directory navigation |

---

## Gestures

| Hand pose | Gesture | Action |
|-----------|---------|--------|
| Index tip + thumb tip close together | **Pinch** | Enter directory / open file |
| Wrist sweeps right → left | **Swipe Left** | Go back (directory history) |
| Wrist sweeps left → right | **Swipe Right** | Go forward (directory history) |
| All five fingers extended, hand in upper half | **Open Palm ↑** | Scroll file list up |
| All five fingers extended, hand in lower half | **Open Palm ↓** | Scroll file list down |
| Index + middle extended, others curled | **Two Fingers Up** | `cd ..` (parent directory) |

---

## Requirements

- Python 3.9 +
- A USB or built-in webcam
- A desktop environment (X11 / Wayland / macOS) — PyAutoGUI needs a display

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Quick Start

```bash
python gesture_nav.py
```

Optional flags:

```
--camera INDEX          Webcam device index (default: 0)
--pinch-threshold FLOAT Normalised pinch distance threshold (default: 0.07)
```

Press **Q** or **Esc** to quit.

---

## Project Structure

```
repoNavTouch/
├── gesture_nav.py       # Entry point — webcam loop + HUD overlay
├── gestures.py          # Geometry helpers + GestureRecogniser state machine
├── file_navigator.py    # FileNavigator — maps gestures to OS actions
├── test_gestures.py     # Unit tests for gesture classification
├── test_file_navigator.py  # Unit tests for file navigation logic
└── requirements.txt
```

---

## Running Tests

```bash
python -m pytest test_gestures.py test_file_navigator.py -v
```

---

## Tech Stack

- [MediaPipe](https://mediapipe.dev/) — real-time hand landmark detection (21 3-D points)
- [OpenCV](https://opencv.org/) — webcam capture and image rendering
- [PyAutoGUI](https://pyautogui.readthedocs.io/) — OS-level keyboard / mouse control
- [Pynput](https://pynput.readthedocs.io/) — low-level input event handling

