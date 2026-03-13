import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
from gesture_detector import GestureDetector
from shape_matcher import match_shape

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

# ── Setup ────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

CHALLENGES = ["circle", "triangle", "square", "star"]
chall_idx   = 0
canvas      = None
user_path   = []            # raw (x,y) path for shape matching
drawing     = False
result_text = ""
result_time = 0.0
score_total = 0
rounds      = 0

prev_pt = None

def reset_round():
    global canvas, user_path, result_text, prev_pt
    canvas      = None
    user_path   = []
    result_text = ""
    prev_pt     = None

print("Draw the shape shown  |  DRAW gesture = draw  |  MOVE gesture = submit  |  ESC = quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Use camera only for detection; show a pure black canvas.
    if canvas is None:
        canvas = np.zeros_like(frame)
    display = np.zeros_like(frame)

    # ── MediaPipe Tasks Inference ─────────────────────────────────────────────
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    if timestamp_ms <= 0:
        timestamp_ms = int(time.time() * 1000)

    result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

    gesture   = "UNKNOWN"
    cursor_pt = None

    if result and result.hand_landmarks:
        lms = result.hand_landmarks[0]
        cx  = int(lms[8].x * w)
        cy  = int(lms[8].y * h)
        cursor_pt = (cx, cy)
        handedness = result.handedness[0][0].category_name
        gesture   = detector.detect(lms, handedness)

        if gesture == "DRAW":
            if prev_pt:
                cv2.line(canvas, prev_pt, cursor_pt, (0, 255, 255), 4)
            user_path.append((lms[8].x, lms[8].y))
            prev_pt = cursor_pt

        elif gesture == "MOVE" and user_path:
            # Submit drawing
            match = match_shape(user_path)
            target = CHALLENGES[chall_idx % len(CHALLENGES)]
            matched = match["best_match"] == target
            pct = int(match["score"] * 100)

            if matched and pct > 60:
                score_total += pct
                result_text = f"✓ {target.upper()}! Score: {pct}%"
                chall_idx  += 1
            else:
                result_text = (f"✗ Got '{match['best_match']}' ({pct}%). "
                               f"Try again: {target}")
            rounds     += 1
            result_time = time.time()
            reset_round()
            prev_pt = None

        elif gesture == "CLEAR":
            reset_round()

        # Overlay Hand Skeleton on black display
        draw_custom_landmarks(display, lms, HAND_CONNECTIONS)
    else:
        prev_pt = None

    # Composite drawing on black background
    output = cv2.add(display, canvas)

    # Target prompt
    target_name = CHALLENGES[chall_idx % len(CHALLENGES)].upper()
    cv2.putText(output, f"Draw: {target_name}", (10, 40),
                cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 200, 255), 2)

    # Gesture indicator
    cv2.putText(output, f"[{gesture}]", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Score
    cv2.putText(output, f"Total: {score_total}  Rounds: {rounds}",
                (w - 300, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    # Feedback overlay
    if result_text and time.time() - result_time < 2.5:
        col = (0, 255, 0) if "✓" in result_text else (0, 0, 255)
        cv2.putText(output, result_text, (10, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)

    cv2.imshow("Shape Challenge", output)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
print(f"\nFinal score: {score_total} over {rounds} rounds.")
