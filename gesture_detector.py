"""
gesture_detector.py — Classifies hand gestures from MediaPipe landmarks.

Supported gestures:
    DRAW        – only index finger extended
    MOVE        – index + middle fingers extended (peace sign)
    ERASE       – open palm (all 5 fingers extended)
    PINCH       – thumb tip close to index tip
    UNKNOWN     – anything else
"""

import math

# ── Finger tip / base landmark indices ───────────────────────────────────────
FINGER_TIPS  = [4, 8, 12, 16, 20]          # thumb, index, middle, ring, pinky
FINGER_PIPS  = [3, 6, 10, 14, 18]          # knuckle below tip


class GestureDetector:
    """Detect gestures from a single MediaPipe hand landmark list."""

    # Pinch thresholds are relative to hand size (see _hand_scale).
    # Use hysteresis so pinch must be "held" to stay active (less flicker).
    PINCH_ON_THRESHOLD = 0.42
    PINCH_OFF_THRESHOLD = 0.55

    # Require a gesture to be seen for a few frames before committing.
    STABLE_FRAMES = 3

    def __init__(self) -> None:
        self._pinched = False
        self._last_gesture: str | None = None
        self._stable_count = 0
        self._stable_gesture: str = "UNKNOWN"

    @staticmethod
    def fingers_up(lms, handedness: str = "Right") -> list[bool]:
        """Return [thumb, index, middle, ring, pinky] up/down.
        Accepts `lms` as a list of NormalizedLandmark from MediaPipe Tasks API.
        `handedness` is "Left" or "Right" as reported by MediaPipe.
        """
        raised = []

        # Determine if palm is facing camera or back of hand is facing camera.
        # In mirrored view (default for webcam):
        # Right hand: palm facing if index_base.x < pinky_base.x
        if handedness == "Right":
            palm_facing = lms[5].x < lms[17].x
        else:
            palm_facing = lms[5].x > lms[17].x

        # Thumb: compare x-axis, adjusting for palm/back orientation
        if handedness == "Right":
            if palm_facing:
                raised.append(lms[4].x < lms[3].x)
            else:
                raised.append(lms[4].x > lms[3].x)
        else: # Left hand
            if palm_facing:
                raised.append(lms[4].x > lms[3].x)
            else:
                raised.append(lms[4].x < lms[3].x)

        # Other fingers: tip y < pip y  (higher on screen = smaller y)
        for tip, pip in zip(FINGER_TIPS[1:], FINGER_PIPS[1:]):
            raised.append(lms[tip].y < lms[pip].y)

        return raised

    @staticmethod
    def distance(lms, a: int, b: int) -> float:
        """Euclidean distance between landmarks a and b in normalised coords."""
        dx = lms[a].x - lms[b].x
        dy = lms[a].y - lms[b].y
        return math.hypot(dx, dy)

    def _hand_scale(self, lms) -> float:
        """Approximate hand size to make pinch distance scale‑invariant."""
        # Use a couple of reasonably stable bone lengths / spans.
        palm_span = self.distance(lms, 5, 17)      # across palm
        finger_len = self.distance(lms, 0, 9)      # wrist to middle MCP
        scale = max(palm_span, finger_len)
        return scale or 1.0

    def detect(self, lms, handedness: str = "Right") -> str:
        """Return gesture name string."""
        up = self.fingers_up(lms, handedness)
        total_up = sum(up)

        # Normalised pinch distance relative to hand size
        scale = self._hand_scale(lms)
        pinch_norm = self.distance(lms, 4, 8) / scale

        # Pinch: thumb + index close together, index raised, other fingers mostly down.
        # Hysteresis makes it "holdable".
        pinch_candidate = up[1] and sum(up[2:]) <= 1 and (
            (not self._pinched and pinch_norm < self.PINCH_ON_THRESHOLD)
            or (self._pinched and pinch_norm < self.PINCH_OFF_THRESHOLD)
        )
        self._pinched = bool(pinch_candidate)

        if self._pinched:
            gesture = "PINCH"
        # Draw: only index up
        elif up == [False, True, False, False, False]:
            gesture = "DRAW"
        # Move: index + middle up
        elif up == [False, True, True, False, False]:
            gesture = "MOVE"
        # Erase: all 5 up
        elif total_up == 5:
            gesture = "ERASE"
        else:
            gesture = "UNKNOWN"

        # Simple temporal stability filter to reduce noisy flipping.
        # While stabilizing, keep reporting the last stable gesture (reduces jitter
        # without "blanking out" the UI/interactions).
        if gesture == self._last_gesture:
            self._stable_count += 1
        else:
            self._last_gesture = gesture
            self._stable_count = 1

        if self._stable_count >= self.STABLE_FRAMES:
            self._stable_gesture = gesture
            return gesture

        # Pinch needs to feel responsive; if pinch is currently held, surface it.
        if self._pinched:
            return "PINCH"

        return self._stable_gesture

    def pinch_value(self, lms) -> float:
        """Return pinch tightness in [0..1] (higher = tighter pinch)."""
        scale = self._hand_scale(lms)
        pinch_norm = self.distance(lms, 4, 8) / scale
        # Map typical range to [0..1]
        # tight ~0.25, loose ~0.70
        tight, loose = 0.25, 0.70
        v = (loose - pinch_norm) / (loose - tight)
        return max(0.0, min(1.0, float(v)))
