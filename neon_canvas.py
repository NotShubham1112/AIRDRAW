"""
neon_canvas.py — Black canvas with neon glow drawing, exactly like the reel.

HOW TO USE:
    python neon_canvas.py

GESTURES:
    ☝️  Index finger only  → DRAW (neon line follows fingertip)
    ✌️  Two fingers up     → LIFT PEN (move without drawing)
    ✊  Fist               → CLEAR canvas
    🖐  Open palm          → ERASE (wipes under fingertip)

KEYBOARD:
    1-5  → Change colour (red / cyan / yellow / green / magenta)
    +/-  → Brush size
    C    → Clear
    S    → Save screenshot
    ESC  → Quit
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from gesture_detector import GestureDetector
import math
import time

# ── MediaPipe Tasks API Setup ─────────────────────────────────────────────────
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.75,
    min_hand_presence_confidence=0.75,
    min_tracking_confidence=0.75)

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

detector = GestureDetector()

# ── Webcam ─────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ── Neon colour palette (BGR) ─────────────────────────────────────────────────
NEON = {
    "1_red":     (0,   50,  255),
    "2_cyan":    (255, 230, 0  ),
    "3_yellow":  (0,   255, 255),
    "4_green":   (0,   255, 80 ),
    "5_magenta": (255, 0,   200),
}
COLOUR_KEYS = list(NEON.keys())
colour_idx  = 1          # default = cyan
brush       = 3          
eraser_r    = 45
GLOW_LAYERS = 4          

# ── State ─────────────────────────────────────────────────────────────────────
canvas   = None
prev_pt  = None
# Start directly in pure black mode so only the neon drawing is visible.
pure_black_mode = True

def draw_neon_line(canvas, pt1, pt2, colour, thickness):
    """Draw a glowing neon line by drawing on a temp layer and adding it."""
    h, w, _ = canvas.shape
    # Draw the glow on a temporary black layer
    temp_layer = np.zeros((h, w, 3), dtype=np.uint8)
    
    for i in range(GLOW_LAYERS, 0, -1):
        alpha   = 1.0 / i
        t_layer = thickness + (GLOW_LAYERS - i) * 6
        col_faded = tuple(int(c * alpha) for c in colour)
        cv2.line(temp_layer, pt1, pt2, col_faded, t_layer, lineType=cv2.LINE_AA)
    
    # Blur the temp layer once to spread the glow
    temp_layer = cv2.GaussianBlur(temp_layer, (0, 0), sigmaX=5)
    
    # Draw the sharp core on top of the temp layer
    cv2.line(temp_layer, pt1, pt2, (255, 255, 255), max(1, thickness - 1), lineType=cv2.LINE_AA)
    cv2.line(temp_layer, pt1, pt2, colour, max(1, thickness - 2), lineType=cv2.LINE_AA)
    
    # Add the temp layer to the persistent canvas
    return cv2.add(canvas, temp_layer)

def draw_neon_dot(canvas, pt, colour, radius):
    """Glow dot for cursor center."""
    h, w, _ = canvas.shape
    temp_layer = np.zeros((h, w, 3), dtype=np.uint8)
    
    for i in range(GLOW_LAYERS, 0, -1):
        alpha    = 1.0 / i
        r_layer  = radius + (GLOW_LAYERS - i) * 4
        col_fade = tuple(int(c * alpha) for c in colour)
        cv2.circle(temp_layer, pt, r_layer, col_fade, -1)
    
    temp_layer = cv2.GaussianBlur(temp_layer, (0, 0), sigmaX=3)
    cv2.circle(temp_layer, pt, max(1, radius - 1), (255, 255, 255), -1)
    
    return cv2.add(canvas, temp_layer)

def draw_custom_landmarks(image, landmarks, connections, color=(80, 80, 80)):
    """Custom landmark drawing without mp.solutions."""
    h, w, _ = image.shape
    pts = []
    for lm in landmarks:
        pts.append((int(lm.x * w), int(lm.y * h)))
    
    # Draw connections
    for start_idx, end_idx in connections:
        cv2.line(image, pts[start_idx], pts[end_idx], color, 2)
    
    # Draw points
    for pt in pts:
        cv2.circle(image, pt, 2, (100, 100, 100), -1)

class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.prev_x = 0
        self.prev_y = 0
        self.dx = 0
        self.dy = 0
        self.last_time = 0

    def _alpha(self, cutoff, dt):
        r = 2 * math.pi * cutoff * dt
        return r / (r + 1)

    def filter(self, x, y, t):
        if not self.last_time:
            self.prev_x, self.prev_y = x, y
            self.last_time = t
            return x, y
        
        dt = (t - self.last_time)
        if dt <= 0: return int(self.prev_x), int(self.prev_y)

        ad = self._alpha(self.d_cutoff, dt)
        self.dx = (x - self.prev_x) / dt * ad + self.dx * (1 - ad)
        self.dy = (y - self.prev_y) / dt * ad + self.dy * (1 - ad)

        cutoff = self.min_cutoff + self.beta * math.hypot(self.dx, self.dy)
        a = self._alpha(cutoff, dt)
        self.prev_x = x * a + self.prev_x * (1 - a)
        self.prev_y = y * a + self.prev_y * (1 - a)
        self.last_time = t
        return int(self.prev_x), int(self.prev_y)

# ── State Management for Multiple Hands ──────────────────────────────────────
class HandState:
    def __init__(self):
        self.detector = GestureDetector()
        self.prev_pt = None
        self.filter = OneEuroFilter(min_cutoff=0.8, beta=0.015)
        self.smoothed_pt = None

    def get_smoothed_pt(self, pt, t):
        self.smoothed_pt = self.filter.filter(pt[0], pt[1], t)
        return self.smoothed_pt

# Track states for up to 2 hands
hand_states = [HandState(), HandState()]

screenshot_n = 0
print("Neon Canvas ready! Show one or two hands to the webcam.")
print("Keys: 1-5=colour | +/-=brush | C=clear | B=toggle backup (shadow) | S=save | ESC=quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    if canvas is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # ── MediaPipe Tasks Inference ─────────────────────────────────────────────
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    
    if timestamp_ms <= 0:
        import time
        timestamp_ms = int(time.time() * 1000)

    result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

    colour = NEON[COLOUR_KEYS[colour_idx]]
    current_gestures = []

    if result and result.hand_landmarks:
        for i, lms in enumerate(result.hand_landmarks):
            if i >= len(hand_states): break
            
            state = hand_states[i]
            
            # Tip Stabilization: Weighted average of index tip (8) and index PIP (7)
            # This reduces rotational jitter of the fingertip.
            tip_lm = lms[8]
            pip_lm = lms[7]
            cx_raw = int((tip_lm.x * 0.8 + pip_lm.x * 0.2) * w)
            cy_raw = int((tip_lm.y * 0.8 + pip_lm.y * 0.2) * h)
            
            # Use real-world timestamp for OneEuroFilter
            curr_t = time.time()
            cursor_pt = state.get_smoothed_pt((cx_raw, cy_raw), curr_t)

            handedness = result.handedness[i][0].category_name
            gesture = state.detector.detect(lms, handedness)
            current_gestures.append(gesture)

            if gesture == "DRAW":
                if state.prev_pt is not None:
                    canvas = draw_neon_line(canvas, state.prev_pt, cursor_pt, colour, brush)
                state.prev_pt = cursor_pt
            elif gesture == "ERASE":
                cv2.circle(canvas, cursor_pt, eraser_r, (0, 0, 0), -1)
                state.prev_pt = None
            elif gesture == "CLEAR":
                canvas = np.zeros((h, w, 3), dtype=np.uint8)
                state.prev_pt = None
            else:
                state.prev_pt = None
            
            # Cursor glow dot for each active hand
            output_glow = draw_neon_dot(np.zeros_like(canvas), cursor_pt, colour, brush + 3)
            # Add early to visualize per hand if needed, or composite later

        # Ensure unused hands reset
        for i in range(len(result.hand_landmarks), len(hand_states)):
            hand_states[i].prev_pt = None
            hand_states[i].smoothed_pt = None
    else:
        for state in hand_states:
            state.prev_pt = None
            state.smoothed_pt = None

    # ── Composite Output ──────────────────────────────────────────────────────
    if pure_black_mode:
        output = canvas.copy()
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, shadow = cv2.threshold(gray, 40, 50, cv2.THRESH_BINARY)
        shadow_bgr = cv2.cvtColor(shadow, cv2.GRAY2BGR)
        output = cv2.add(canvas, shadow_bgr)

    # ── Overlays ─────────────────────────────────────────────────────────────
    if result and result.hand_landmarks:
        for i, lms in enumerate(result.hand_landmarks):
            draw_custom_landmarks(output, lms, HAND_CONNECTIONS)
            # Cursor dot on final output
            st = hand_states[i]
            if st.smoothed_pt:
                handedness = result.handedness[i][0].category_name
                gesture = st.detector.detect(result.hand_landmarks[i], handedness)
                
                if gesture == "ERASE":
                    # Red translucent circle for eraser
                    overlay = output.copy()
                    cv2.circle(overlay, st.smoothed_pt, eraser_r, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.circle(overlay, st.smoothed_pt, eraser_r, (50, 50, 150), -1)
                    cv2.addWeighted(overlay, 0.25, output, 0.75, 0, output)
                elif gesture == "MOVE":
                    # Grayish circle for move
                    cv2.circle(output, st.smoothed_pt, brush + 6, (150, 150, 150), 2, cv2.LINE_AA)
                else:
                    output = draw_neon_dot(output, st.smoothed_pt, colour, brush + 3)

    # ── HUD ──────────────────────────────────────────────────────────────────
    col_name = COLOUR_KEYS[colour_idx].split("_")[1].upper()
    mode_str = "BLACK" if pure_black_mode else "SHADOW"
    g_str = " & ".join(current_gestures) if current_gestures else "---"
    hud = f"{col_name}  brush:{brush}  [{g_str}]  MODE:{mode_str}"
    cv2.putText(output, hud, (12, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA)

    cv2.imshow("Neon Canvas", output)

    # ── Keys ──────────────────────────────────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord('c') or key == ord('C'):
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
    elif key == ord('b') or key == ord('B'):
        pure_black_mode = not pure_black_mode
    elif key == ord('s') or key == ord('S'):
        fname = f"neon_drawing_{screenshot_n}.png"
        cv2.imwrite(fname, output)
        print(f"Saved: {fname}")
        screenshot_n += 1
    elif key == ord('+') or key == ord('='):
        brush = min(brush + 1, 15)
    elif key == ord('-'):
        brush = max(brush - 1, 1)
    elif key in [ord(str(i)) for i in range(1, 6)]:
        colour_idx = int(chr(key)) - 1

cap.release()
cv2.destroyAllWindows()
