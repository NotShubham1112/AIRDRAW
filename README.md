# ✋ Hand Gesture Air Canvas — Complete Guide

> Draw in the air with your finger. Recognize shapes. Play the shape challenge game.
> Built with Python · OpenCV · MediaPipe · NumPy

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)  
2. [How It Works — Core Concepts](#2-how-it-works--core-concepts)  
3. [System Architecture](#3-system-architecture)  
4. [Hand Landmarks Reference](#4-hand-landmarks-reference)  
5. [Gesture Dictionary](#5-gesture-dictionary)  
6. [Project File Structure](#6-project-file-structure)  
7. [Installation](#7-installation)  
8. [Running Each App](#8-running-each-app)  
9. [Code Walkthrough](#9-code-walkthrough)  
   - [hand_draw.py](#91-hand_drawpy--basic-air-canvas)  
   - [gesture_detector.py](#92-gesture_detectorpy--gesture-recognition)  
   - [gesture_canvas.py](#93-gesture_canvaspy--advanced-canvas)  
   - [shape_matcher.py](#94-shape_matcherpy--shape-recognition)  
   - [shape_challenge.py](#95-shape_challengepy--the-game)  
10. [Shape Matching Algorithms Explained](#10-shape-matching-algorithms-explained)  
11. [Keyboard Controls](#11-keyboard-controls)  
12. [Extending the Project](#12-extending-the-project)  
13. [Troubleshooting](#13-troubleshooting)  

---

## 1. What This Project Does

This project lets you **draw on your screen using only your index finger** captured by a regular webcam.

| App | What it does |
|-----|-------------|
| `hand_draw.py` | Basic air canvas — move finger, draw lines |
| `gesture_canvas.py` | Advanced canvas — different gestures for draw, erase, move, clear |
| `shape_challenge.py` | Game — draw the target shape, get scored on accuracy |

---

## 2. How It Works — Core Concepts

### Step 1 — Capture Webcam Frame
OpenCV reads frames from your webcam (30fps+).

### Step 2 — Detect Hand Skeleton
MediaPipe's Hand model finds **21 landmark points** on your hand in real-time — fingertips, joints, palm center.

### Step 3 — Extract Fingertip Position
We read **landmark 8** (index fingertip) and convert its normalised `(0–1)` coordinates into pixel coordinates on screen.

### Step 4 — Detect Gesture
We check which fingers are raised to classify the gesture (DRAW, MOVE, ERASE, etc.).

### Step 5 — React on Canvas
Based on the gesture, we either draw a line, erase, clear, or submit the drawing for shape matching.

### Step 6 — Overlay and Display
The drawing canvas is merged with the live webcam feed using `cv2.add()`.

---

## 3. System Architecture

```
┌────────────────┐
│  Webcam Input  │   OpenCV VideoCapture
└───────┬────────┘
        │ raw frame (BGR)
        ▼
┌────────────────────┐
│ MediaPipe Hands    │   21 landmarks per hand
│ Hand Tracking      │   x, y, z (normalised)
└───────┬────────────┘
        │ landmark[8] = index fingertip
        ▼
┌────────────────────┐
│ Gesture Detector   │   Classify hand pose
│  - fingers_up()    │   → DRAW / MOVE / ERASE
│  - distance()      │   → CLEAR / PINCH
└───────┬────────────┘
        │ gesture + cursor position
        ▼
┌────────────────────────────────────────┐
│  Action Engine                         │
│  DRAW  → cv2.line() on canvas          │
│  ERASE → cv2.circle() black on canvas  │
│  CLEAR → reset canvas to zeros         │
│  MOVE  → lift pen / submit drawing     │
└───────┬────────────────────────────────┘
        │
        ▼  (optional)
┌────────────────────┐
│ Shape Matcher      │   Compare drawn path
│  - Hu Moments      │   to reference shapes
│  - DTW Distance    │   → "circle" 87%
└───────┬────────────┘
        │
        ▼
┌────────────────┐
│ Canvas Overlay │   cv2.add(frame, canvas)
│ + HUD Display  │   cv2.imshow()
└────────────────┘
```

---

## 4. Hand Landmarks Reference

MediaPipe detects **21 points** per hand. Key indices used in this project:

```
        8   ← INDEX TIP  (we draw with this)
        |
        7
        |
        6
        |
   4    5
   |    |
   3    4-pip
   |
  THUMB

Full map:
  0  = Wrist
  1  = Thumb CMC       4  = Thumb Tip
  5  = Index MCP       8  = Index Tip  ← PRIMARY CURSOR
  9  = Middle MCP     12  = Middle Tip
 13  = Ring MCP       16  = Ring Tip
 17  = Pinky MCP      20  = Pinky Tip
```

### How to read a landmark in code

```python
h, w, _ = frame.shape

lm = hand_landmarks.landmark[8]   # index fingertip

x = int(lm.x * w)   # normalised 0-1 → pixel column
y = int(lm.y * h)   # normalised 0-1 → pixel row
```

---

## 5. Gesture Dictionary

| Gesture | Fingers Up | Detection Logic | Action |
|---------|-----------|-----------------|--------|
| **DRAW** | ☝️ Index only | `up == [F,T,F,F,F]` | Draw line on canvas |
| **MOVE** | ✌️ Index + Middle | `up == [F,T,T,F,F]` | Lift pen (reposition) |
| **ERASE** | 🖐 All 5 | `total_up == 5` | Erase under fingertip |
| **CLEAR** | ✊ Fist | `total_up <= 1` | Wipe entire canvas |
| **PINCH** | Thumb≈Index | `dist(4,8) < 0.06` | Grab / submit |

---

## 6. Project File Structure

```
hand-gesture-project/
│
├── src/
│   ├── hand_draw.py          # Beginner: simple air canvas
│   ├── gesture_detector.py   # Module: classify gestures
│   ├── gesture_canvas.py     # Advanced: gesture-controlled canvas
│   ├── shape_matcher.py      # Module: compare shapes with DTW + Hu moments
│   └── shape_challenge.py    # Game: draw target shapes, get scored
│
├── docs/
│   └── README.md             # This file
│
└── requirements.txt
```

---

## 7. Installation

### Requirements

- Python 3.8+
- Webcam

### Install dependencies

```bash
pip install opencv-python mediapipe numpy
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

**`requirements.txt`:**
```
opencv-python>=4.8.0
mediapipe>=0.10.0
numpy>=1.24.0
```

### Verify install

```python
import cv2, mediapipe, numpy
print(cv2.__version__, mediapipe.__version__, numpy.__version__)
```

---

## 8. Running Each App

All scripts live in the `src/` directory. Run from that folder:

```bash
cd src/

# Beginner – just draw
python hand_draw.py

# Advanced – with gesture controls
python gesture_canvas.py

# Game – draw the target shape
python shape_challenge.py
```

> **Tip:** Make sure you're in good lighting. A plain background helps MediaPipe track hands more accurately.

---

## 9. Code Walkthrough

### 9.1 `hand_draw.py` — Basic Air Canvas

This is the simplest version. It does one thing: track landmark 8 and draw lines.

```python
import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands    = mp_hands.Hands(max_num_hands=1)
mp_draw  = mp.solutions.drawing_utils

cap    = cv2.VideoCapture(0)
canvas = None
prev_x, prev_y = 0, 0

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)              # mirror like a selfie

    if canvas is None:
        canvas = np.zeros_like(frame)       # blank black layer

    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)             # run MediaPipe

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            h, w, _ = frame.shape
            x = int(hand.landmark[8].x * w)   # index tip → pixels
            y = int(hand.landmark[8].y * h)

            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = x, y

            cv2.line(canvas, (prev_x, prev_y), (x, y), (0, 255, 255), 5)
            prev_x, prev_y = x, y

            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
    else:
        prev_x, prev_y = 0, 0   # lift pen when no hand

    frame = cv2.add(frame, canvas)          # merge drawing + camera
    cv2.imshow("Hand Draw", frame)

    if cv2.waitKey(1) & 0xFF == 27:         # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
```

**Key concept:** `cv2.add()` overlays two same-size images. Because the canvas is black everywhere except drawn lines, only the lines show up over the camera feed.

---

### 9.2 `gesture_detector.py` — Gesture Recognition

The module that turns landmark positions into named gestures.

```python
def fingers_up(lms) -> list[bool]:
    """Returns [thumb, index, middle, ring, pinky] True/False."""
    raised = []

    # Thumb: compare horizontal position (left/right)
    raised.append(lms.landmark[4].x < lms.landmark[3].x)

    # Other 4 fingers: tip y < pip y means finger is UP
    for tip, pip in zip([8,12,16,20], [6,10,14,18]):
        raised.append(lms.landmark[tip].y < lms.landmark[pip].y)

    return raised
```

Why compare `y` values? In screen coordinates, **y=0 is at the top**. A raised fingertip has a **smaller y** than the knuckle below it.

```python
def detect(self, lms) -> str:
    up    = self.fingers_up(lms)
    total = sum(up)

    if self.distance(lms, 4, 8) < 0.06:     # thumb and index touching
        return "PINCH"

    if up == [False, True, False, False, False]:
        return "DRAW"                         # only index up

    if up == [False, True, True, False, False]:
        return "MOVE"                         # peace sign

    if total == 5:
        return "ERASE"                        # open hand

    if total <= 1:
        return "CLEAR"                        # fist

    return "UNKNOWN"
```

---

### 9.3 `gesture_canvas.py` — Advanced Canvas

Combines `hand_draw.py` + `gesture_detector.py`. 

New features:
- Colour cycling with Tab key
- Eraser mode (open palm draws a black circle on canvas)
- Gesture status shown in HUD
- Colour swatch preview

The main loop pattern:

```python
gesture = detector.detect(lms)

if gesture == "DRAW":
    cv2.line(canvas, prev_pt, cursor_pt, colour, brush)

elif gesture == "ERASE":
    cv2.circle(canvas, cursor_pt, eraser_size, (0,0,0), -1)

elif gesture == "CLEAR":
    canvas = np.zeros_like(frame)
```

---

### 9.4 `shape_matcher.py` — Shape Recognition

Two algorithms work together to score how similar the user's path is to a reference shape.

#### Algorithm 1: Hu Moments (`cv2.matchShapes`)

Hu Moments are 7 numbers that describe the **shape of a contour** independent of size, rotation, and position.

```python
hu_dist = cv2.matchShapes(user_contour, ref_contour,
                          cv2.CONTOURS_MATCH_I2, 0)
# 0 = identical shapes, higher = more different
```

**Good for:** distinguishing totally different shapes (circle vs triangle).  
**Bad for:** order of drawing — doesn't care how you drew it.

#### Algorithm 2: DTW — Dynamic Time Warping

DTW compares two **sequences of points** while allowing one to be stretched or compressed in time.

```python
def dtw_distance(a, b):
    n, m = len(a), len(b)
    dtw  = np.full((n+1, m+1), np.inf)
    dtw[0,0] = 0.0

    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = np.linalg.norm(a[i-1] - b[j-1])   # Euclidean dist
            dtw[i,j] = cost + min(dtw[i-1,j],
                                  dtw[i,j-1],
                                  dtw[i-1,j-1])
    return dtw[n,m] / (n + m)   # normalise by path length
```

**Good for:** detecting drawing direction, start point, completeness.  
**Combined score:** `1 - (0.5 * hu_dist + 0.5 * dtw_dist)` → 0 to 1 scale.

---

### 9.5 `shape_challenge.py` — The Game

Combines everything:

1. A target shape name is shown (e.g. "Draw: CIRCLE")
2. Player draws with DRAW gesture
3. Player submits with MOVE gesture
4. `match_shape()` compares the drawn path to all reference shapes
5. If best match == target AND score > 60% → correct, score added
6. Next challenge shown

```python
elif gesture == "MOVE" and user_path:
    match  = match_shape(user_path)
    target = CHALLENGES[chall_idx % len(CHALLENGES)]

    if match["best_match"] == target and match["score"] > 0.60:
        score_total += int(match["score"] * 100)
        chall_idx   += 1
        result_text  = f"✓ Correct! Score: {int(match['score']*100)}%"
    else:
        result_text = f"✗ Got '{match['best_match']}', try again"
```

---

## 10. Shape Matching Algorithms Explained

### What is DTW?

Normal Euclidean distance between two sequences only works if they're the same length and aligned. DTW fixes this by finding the optimal alignment:

```
User path:  ● ─ ● ─ ● ─ ● ─ ● ─ ●
                ╲     ╲   ╲
Reference:  ● ─ ● ─ ● ─ ● ─ ● ─ ●
```

This means a slowly-drawn circle and a quickly-drawn circle still match well.

### What are Hu Moments?

Seven invariant moments computed from pixel intensities. The key property: they don't change when you scale, rotate, or translate the shape.

| Hu Moment | Sensitive To |
|-----------|-------------|
| h1–h7     | Shape only (not size/rotation/position) |

### Normalisation

Before any comparison, all point sequences are normalised to the `[0,1]×[0,1]` bounding box:

```python
def normalise(pts):
    arr = np.array(pts)
    mn  = arr.min(axis=0)
    rng = arr.max(axis=0) - mn
    rng[rng == 0] = 1           # avoid division by zero
    return (arr - mn) / rng
```

This makes the matcher size-independent — a small circle and a big circle match equally well.

---

## 11. Keyboard Controls

### hand_draw.py

| Key | Action |
|-----|--------|
| `C` | Clear canvas |
| `R` | Red brush |
| `G` | Green brush |
| `B` | Blue brush |
| `Y` | Yellow brush |
| `+` | Increase brush size |
| `-` | Decrease brush size |
| `ESC` | Quit |

### gesture_canvas.py

| Key | Action |
|-----|--------|
| `Tab` | Cycle through colours |
| `+` | Increase brush size |
| `-` | Decrease brush size |
| `ESC` | Quit |

### shape_challenge.py

| Gesture | Action |
|---------|--------|
| DRAW (index up) | Draw on canvas |
| MOVE (peace sign) | Submit drawing |
| CLEAR (fist) | Clear and restart |
| `ESC` | Quit |

---

## 12. Extending the Project

### Add a new gesture

In `gesture_detector.py`, add a new check in the `detect()` method:

```python
# Example: thumbs up
if up == [True, False, False, False, False]:
    return "THUMBS_UP"
```

### Add a new target shape

In `shape_matcher.py`, add a generator function and register it:

```python
def make_heart(n=80) -> list[tuple]:
    pts = []
    for t in np.linspace(0, 2*math.pi, n):
        x = 0.5 + 0.4 * (16 * math.sin(t)**3) / 16
        y = 0.5 - 0.4 * (13*math.cos(t) - 5*math.cos(2*t)
                          - 2*math.cos(3*t) - math.cos(4*t)) / 17
        pts.append((x, y))
    return pts

SHAPES["heart"] = make_heart()
```

### Gesture mouse control

```python
import pyautogui

if gesture == "DRAW":
    pyautogui.moveTo(x * screen_w / cam_w, y * screen_h / cam_h)

if gesture == "PINCH":
    pyautogui.click()
```

### Save canvas as image

```python
if key == ord('s'):
    cv2.imwrite("my_drawing.png", canvas)
    print("Saved!")
```

---

## 13. Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: mediapipe` | Run `pip install mediapipe` |
| Webcam not opening | Change `VideoCapture(0)` to `VideoCapture(1)` |
| Hand not detected | Improve lighting; use plain background |
| Gesture misclassified | Slow down; make gestures more deliberate |
| Lines lag behind finger | Lower resolution: `cap.set(CAP_PROP_FRAME_WIDTH, 640)` |
| Shape not matching correctly | Draw slowly and completely; close circles fully |
| `pip install` fails on some systems | Try `pip3` or `python -m pip install` |

---

## How a Markdown File Is Structured

A Markdown file (`.md`) uses simple text symbols to create formatted documents:

| Syntax | Result |
|--------|--------|
| `# Title` | Big heading |
| `## Section` | Medium heading |
| `### Subsection` | Small heading |
| `**bold**` | **bold** |
| `` `code` `` | `inline code` |
| ` ```python ... ``` ` | Code block with syntax highlighting |
| `- item` | Bullet list |
| `1. item` | Numbered list |
| `\| col \| col \|` | Table |
| `[text](url)` | Hyperlink |
| `> text` | Blockquote |
| `---` | Horizontal rule |

Markdown renders in: GitHub, VS Code Preview, Obsidian, Notion imports, and most documentation systems.

---

*Project complete. Happy drawing! ✋*
