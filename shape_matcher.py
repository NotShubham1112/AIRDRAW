"""
shape_matcher.py — Compare user-drawn paths to target shapes.

Uses:
    1. OpenCV matchShapes (Hu Moments) — fast
    2. DTW (Dynamic Time Warping)      — sequence-aware
"""

import cv2
import numpy as np
import math
from typing import Sequence

# ──────────────────────────────────────────────────────────────────────────────
# Reference shape generators (returns list of (x,y) tuples)
# ──────────────────────────────────────────────────────────────────────────────

def make_triangle(n=60) -> list[tuple]:
    pts = []
    verts = [(0.5, 0.0), (1.0, 1.0), (0.0, 1.0)]
    for i in range(3):
        x0, y0 = verts[i]
        x1, y1 = verts[(i + 1) % 3]
        for t in np.linspace(0, 1, n // 3, endpoint=False):
            pts.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
    return pts

def make_circle(n=120) -> list[tuple]:
    return [(0.5 + 0.5 * math.cos(2 * math.pi * i / n),
             0.5 + 0.5 * math.sin(2 * math.pi * i / n))
            for i in range(n)]

def make_star(n=10) -> list[tuple]:
    pts = []
    for i in range(n):
        angle = math.pi / 2 + i * 2 * math.pi / n
        r = 0.5 if i % 2 == 0 else 0.22
        pts.append((0.5 + r * math.cos(angle), 0.5 + r * math.sin(angle)))
    return pts

def make_square(n=80) -> list[tuple]:
    pts = []
    corners = [(0, 0), (1, 0), (1, 1), (0, 1)]
    for i in range(4):
        x0, y0 = corners[i]
        x1, y1 = corners[(i + 1) % 4]
        for t in np.linspace(0, 1, n // 4, endpoint=False):
            pts.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
    return pts

SHAPES = {
    "triangle": make_triangle(),
    "circle":   make_circle(),
    "star":     make_star(),
    "square":   make_square(),
}

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def normalise(pts: Sequence[tuple]) -> np.ndarray:
    """Normalise points to [0,1]x[0,1]."""
    arr = np.array(pts, dtype=np.float32)
    mn  = arr.min(axis=0)
    rng = arr.max(axis=0) - mn
    rng[rng == 0] = 1
    return (arr - mn) / rng

def pts_to_contour(pts: Sequence[tuple]) -> np.ndarray:
    """Convert (x,y) list to OpenCV contour format."""
    arr = (normalise(pts) * 255).astype(np.int32)
    return arr.reshape((-1, 1, 2))

def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Simple O(n*m) DTW between two 2-D point sequences."""
    n, m = len(a), len(b)
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = float(np.linalg.norm(a[i - 1] - b[j - 1]))
            dtw[i, j] = cost + min(dtw[i - 1, j],
                                   dtw[i, j - 1],
                                   dtw[i - 1, j - 1])
    return dtw[n, m] / (n + m)

# ──────────────────────────────────────────────────────────────────────────────
# Main matcher
# ──────────────────────────────────────────────────────────────────────────────

def match_shape(user_pts: list[tuple]) -> dict:
    """
    Compare user_pts against all reference shapes.

    Returns:
        {
          "best_match": "circle",
          "score":      0.87,       # 0–1, higher = better
          "all_scores": {"triangle": 0.4, ...}
        }
    """
    if len(user_pts) < 10:
        return {"best_match": None, "score": 0.0, "all_scores": {}}

    user_norm    = normalise(user_pts)
    user_contour = pts_to_contour(user_pts)

    scores = {}

    for name, ref_pts in SHAPES.items():
        ref_norm    = normalise(ref_pts)
        ref_contour = pts_to_contour(ref_pts)

        # Hu moment distance (0 = identical)
        hu_dist = cv2.matchShapes(user_contour, ref_contour,
                                  cv2.CONTOURS_MATCH_I2, 0)

        # DTW on downsampled points
        us = user_norm[::max(1, len(user_norm) // 60)]
        rs = ref_norm [::max(1, len(ref_norm)  // 60)]
        dtw_dist = dtw_distance(us, rs)

        # Combine (lower = more similar → invert to score)
        combined = 0.5 * min(hu_dist, 1.0) + 0.5 * min(dtw_dist, 1.0)
        scores[name] = round(1 - combined, 3)

    best = max(scores, key=scores.get)
    return {
        "best_match": best,
        "score":      scores[best],
        "all_scores": scores,
    }


# ── Quick demo ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Simulate drawing a circle
    test_pts = [(0.5 + 0.5 * math.cos(t), 0.5 + 0.5 * math.sin(t))
                for t in np.linspace(0, 2 * math.pi, 100)]
    result = match_shape(test_pts)
    print("Test shape (circle):", result)
