import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

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

# ── Webcam ───────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

canvas   = None
prev_x, prev_y = 0, 0
draw_color     = (0, 255, 255)   # default: yellow
brush_size     = 5

print("Controls:  ESC = quit | C = clear | R/G/B/Y = colour | +/- = brush size")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Use a pure black background for drawing.
    if canvas is None:
        canvas = np.zeros_like(frame)
    display = np.zeros_like(frame)

    # ── MediaPipe Tasks Inference ─────────────────────────────────────────────
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    if timestamp_ms <= 0:
        import time
        timestamp_ms = int(time.time() * 1000)

    result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

    if result and result.hand_landmarks:
        for lms in result.hand_landmarks:
            # Landmark 8 = index fingertip
            x = int(lms[8].x * w)
            y = int(lms[8].y * h)

            # Draw cursor dot on black display
            cv2.circle(display, (x, y), brush_size + 4, draw_color, -1)

            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = x, y

            cv2.line(canvas, (prev_x, prev_y), (x, y), draw_color, brush_size)
            prev_x, prev_y = x, y

            # Overlay hand skeleton on black display
            draw_custom_landmarks(display, lms, HAND_CONNECTIONS)
    else:
        prev_x, prev_y = 0, 0     # lift pen when hand disappears

    # ── Merge canvas onto black display ──────────────────────────────────────
    output = cv2.add(display, canvas)

    # ── HUD ──────────────────────────────────────────────────────────────────
    cv2.putText(output, f"Colour: {draw_color}  Brush: {brush_size}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.imshow("Air Canvas", output)

    # ── Keyboard controls ────────────────────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF
    if key == 27:                          # ESC → quit
        break
    elif key == ord('c'):                  # C → clear
        canvas = np.zeros_like(frame)
    elif key == ord('r'):
        draw_color = (0, 0, 255)
    elif key == ord('g'):
        draw_color = (0, 255, 0)
    elif key == ord('b'):
        draw_color = (255, 0, 0)
    elif key == ord('y'):
        draw_color = (0, 255, 255)
    elif key == ord('+'):
        brush_size = min(brush_size + 2, 30)
    elif key == ord('-'):
        brush_size = max(brush_size - 2, 1)

cap.release()
cv2.destroyAllWindows()
