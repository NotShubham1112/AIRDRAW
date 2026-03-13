import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from gesture_detector import GestureDetector

# ── Config ───────────────────────────────────────────────────────────────────
COLOURS = {
    "yellow": (0, 255, 255),
    "green":  (0, 255, 0),
    "red":    (0, 0, 255),
    "blue":   (255, 0, 0),
    "white":  (255, 255, 255),
}
COLOUR_NAMES = list(COLOURS.keys())

# ── MediaPipe Tasks API Setup ─────────────────────────────────────────────────
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7)

hand_landmarker = vision.HandLandmarker.create_from_options(options)
detector = GestureDetector()

# Hardcoded connections since mp.solutions is missing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),    # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),    # Index
    (5, 9), (9, 10), (10, 11), (11, 12),# Middle
    (9, 13), (13, 14), (14, 15), (15, 16),# Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17) # Palm bottom
]

def draw_custom_landmarks(image, landmarks, connections, color=(200, 200, 200)):
    """Custom landmark drawing without mp.solutions."""
    h, w, _ = image.shape
    pts = []
    for lm in landmarks:
        pts.append((int(lm.x * w), int(lm.y * h)))
    for start_idx, end_idx in connections:
        cv2.line(image, pts[start_idx], pts[end_idx], color, 1)
    for pt in pts:
        cv2.circle(image, pt, 2, (255, 255, 255), -1)

# ── Static State ─────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

canvas       = None
prev_pt      = None
colour_idx   = 0
brush        = 5
eraser_size  = 40

# ── Stroke storage (so drawings can be dragged) ───────────────────────────────
strokes = []  # each: {"pts":[(x,y),...], "col":(b,g,r), "th":int}
current_stroke = None

# Pinch-drag selection
selected_idx = None
pinch_prev = None  # previous cursor while pinching
drag_radius = 55

# Cursor smoothing (reduces jitter)
smoothed_cursor = None
CURSOR_ALPHA = 0.35

GESTURE_COLORS = {
    "DRAW":    (0, 255, 255),
    "MOVE":    (200, 200, 200),
    "ERASE":   (0, 0, 255),
    "PINCH":   (255, 165, 0),
    "UNKNOWN": (100, 100, 100),
}

print("ESC=quit | Tab=next colour | +/-=brush size")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # We still use the camera frame for hand detection,
    # but the visible canvas/background is pure black.
    if canvas is None:
        canvas = np.zeros_like(frame)
    display = np.zeros_like(frame)

    # Re-render canvas from strokes each frame (black background)
    canvas[:] = 0
    for s in strokes:
        pts = s["pts"]
        if len(pts) >= 2:
            for a, b in zip(pts[:-1], pts[1:]):
                cv2.line(canvas, a, b, s["col"], s["th"], lineType=cv2.LINE_AA)

    # ── MediaPipe Tasks Inference ─────────────────────────────────────────────
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    if timestamp_ms <= 0:
        import time
        timestamp_ms = int(time.time() * 1000)

    result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

    gesture   = "UNKNOWN"
    cursor_pt = None

    if result and result.hand_landmarks:
        lms = result.hand_landmarks[0]

        # Current fingertip position (index = landmark 8)
        cx = int(lms[8].x * w)
        cy = int(lms[8].y * h)
        cursor_pt = (cx, cy)

        # Smooth the cursor for a nicer UI + more stable interactions
        if smoothed_cursor is None:
            smoothed_cursor = cursor_pt
        else:
            sx, sy = smoothed_cursor
            smoothed_cursor = (
                int(sx + CURSOR_ALPHA * (cx - sx)),
                int(sy + CURSOR_ALPHA * (cy - sy)),
            )
        cursor_pt = smoothed_cursor

        # Get handedness (category_name is "Left" or "Right")
        handedness = result.handedness[0][0].category_name
        gesture = detector.detect(lms, handedness)

        # ── PINCH: hold to grab + drag a drawn stroke ───────────────────────
        if gesture == "PINCH":
            px, py = cursor_pt
            if selected_idx is None:
                # Pick the nearest stroke point within radius
                best = None
                best_d2 = drag_radius * drag_radius
                for i, s in enumerate(strokes):
                    for (x, y) in s["pts"]:
                        d2 = (px - x) ** 2 + (py - y) ** 2
                        if d2 < best_d2:
                            best_d2 = d2
                            best = i
                selected_idx = best
                pinch_prev = cursor_pt

            # Drag selected stroke by cursor delta
            if selected_idx is not None and pinch_prev is not None:
                dx = px - pinch_prev[0]
                dy = py - pinch_prev[1]
                if dx or dy:
                    pts = strokes[selected_idx]["pts"]
                    strokes[selected_idx]["pts"] = [(x + dx, y + dy) for (x, y) in pts]
                pinch_prev = cursor_pt
            prev_pt = None
            if current_stroke is not None:
                strokes.append(current_stroke)
                current_stroke = None

        elif gesture == "DRAW":
            # If we were pinching, release selection when leaving pinch
            selected_idx = None
            pinch_prev = None

            if current_stroke is None:
                current_stroke = {
                    "pts": [cursor_pt],
                    "col": COLOURS[COLOUR_NAMES[colour_idx]],
                    "th": brush,
                }
            else:
                current_stroke["pts"].append(cursor_pt)
            prev_pt = cursor_pt
        elif gesture == "MOVE":
            selected_idx = None
            pinch_prev = None
            prev_pt = None          # lift pen
            if current_stroke is not None:
                if len(current_stroke["pts"]) >= 2:
                    strokes.append(current_stroke)
                current_stroke = None
        elif gesture == "ERASE":
            selected_idx = None
            pinch_prev = None
            cv2.circle(canvas, cursor_pt, eraser_size, (0, 0, 0), -1)
            prev_pt = None
            # Erase strokes by removing points near cursor (simple + effective)
            ex, ey = cursor_pt
            new_strokes = []
            rr2 = (eraser_size * eraser_size)
            for s in strokes:
                kept = [(x, y) for (x, y) in s["pts"] if (x - ex) ** 2 + (y - ey) ** 2 > rr2]
                if len(kept) >= 2:
                    s2 = dict(s)
                    s2["pts"] = kept
                    new_strokes.append(s2)
            strokes = new_strokes
        else:
            prev_pt = None
            selected_idx = None
            pinch_prev = None
            if current_stroke is not None:
                if len(current_stroke["pts"]) >= 2:
                    strokes.append(current_stroke)
                current_stroke = None

        # Overlay Hand Skeleton on the black display canvas
        draw_custom_landmarks(display, lms, HAND_CONNECTIONS)
    else:
        prev_pt = None
        selected_idx = None
        pinch_prev = None
        smoothed_cursor = None
        if current_stroke is not None:
            if len(current_stroke["pts"]) >= 2:
                strokes.append(current_stroke)
            current_stroke = None

    # ── Compose output on black background ──────────────────────────────────
    output = cv2.add(display, canvas)

    # Render current in-progress stroke on top
    if current_stroke is not None and len(current_stroke["pts"]) >= 2:
        pts = current_stroke["pts"]
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(output, a, b, current_stroke["col"], current_stroke["th"], lineType=cv2.LINE_AA)

    # Highlight selected stroke during pinch
    if selected_idx is not None and 0 <= selected_idx < len(strokes):
        pts = strokes[selected_idx]["pts"]
        if len(pts) >= 2:
            for a, b in zip(pts[:-1], pts[1:]):
                cv2.line(output, a, b, (255, 165, 0), max(2, brush), lineType=cv2.LINE_AA)

    # Cursor ring
    if cursor_pt:
        ring_col = GESTURE_COLORS.get(gesture, (255, 255, 255))
        cv2.circle(output, cursor_pt, brush + 6, ring_col, 2)

    # HUD
    colour_name = COLOUR_NAMES[colour_idx]
    cv2.putText(output, f"Gesture: {gesture}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, GESTURE_COLORS.get(gesture, (255,255,255)), 2)
    cv2.putText(output, f"Colour: {colour_name}  Brush: {brush}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Compact on-screen help for better UI
    help_text = "DRAW=index | MOVE=peace | ERASE=palm | PINCH=grab+drag your drawing"
    cv2.putText(output, help_text, (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    # Colour swatch
    swatch_col = COLOURS[colour_name]
    cv2.rectangle(output, (w - 60, 10), (w - 10, 60), swatch_col, -1)

    cv2.imshow("Gesture Canvas", output)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == 9:          # Tab → cycle colour
        colour_idx = (colour_idx + 1) % len(COLOUR_NAMES)
    elif key == ord('+'):
        brush = min(brush + 2, 30)
    elif key == ord('-'):
        brush = max(brush - 2, 1)

cap.release()
cv2.destroyAllWindows()
