
import unittest
from dataclasses import dataclass

@dataclass
class MockLandmark:
    x: float
    y: float

class TestGestureDetector(unittest.TestCase):
    def setUp(self):
        from gesture_detector import GestureDetector
        self.detector = GestureDetector()

    def test_thumb_up_logic(self):
        # Mock landmarks: 
        # 3: thumb base, 4: thumb tip
        # 5: index base, 17: pinky base
        lms_base = [MockLandmark(0.5, 0.5)] * 21
        
        # --- Case 1: Right hand, PALM facing camera ---
        lms_right_palm = list(lms_base)
        lms_right_palm[5] = MockLandmark(0.4, 0.5)
        lms_right_palm[17] = MockLandmark(0.6, 0.5)
        lms_right_palm[3] = MockLandmark(0.4, 0.6)
        lms_right_palm[4] = MockLandmark(0.3, 0.6)
        
        res = self.detector.fingers_up(lms_right_palm, "Right")
        self.assertTrue(res[0], "Right hand thumb (PALM) should be UP when tip.x < base.x")
        
        # --- Case 2: Right hand, BACK facing camera ---
        lms_right_back = list(lms_base)
        lms_right_back[5] = MockLandmark(0.6, 0.5)
        lms_right_back[17] = MockLandmark(0.4, 0.5)
        lms_right_back[3] = MockLandmark(0.6, 0.6)
        lms_right_back[4] = MockLandmark(0.7, 0.6)
        
        res = self.detector.fingers_up(lms_right_back, "Right")
        self.assertTrue(res[0], "Right hand thumb (BACK) should be UP when tip.x > base.x")

        # --- Case 3: Left hand, PALM facing camera ---
        lms_left_palm = list(lms_base)
        lms_left_palm[5] = MockLandmark(0.6, 0.5)
        lms_left_palm[17] = MockLandmark(0.4, 0.5)
        lms_left_palm[3] = MockLandmark(0.6, 0.6)
        lms_left_palm[4] = MockLandmark(0.7, 0.6)
        
        res = self.detector.fingers_up(lms_left_palm, "Left")
        self.assertTrue(res[0], "Left hand thumb (PALM) should be UP when tip.x > base.x")
        
        # --- Case 4: Left hand, BACK facing camera ---
        lms_left_back = list(lms_base)
        lms_left_back[5] = MockLandmark(0.4, 0.5)
        lms_left_back[17] = MockLandmark(0.6, 0.5)
        lms_left_back[3] = MockLandmark(0.4, 0.6)
        lms_left_back[4] = MockLandmark(0.3, 0.6)
        
        res = self.detector.fingers_up(lms_left_back, "Left")
        self.assertTrue(res[0], "Left hand thumb (BACK) should be UP when tip.x < base.x")

    def test_draw_gesture_robust(self):
        # DRAW should have ONLY index finger up.
        # Test Left hand, palm facing (index_base.x > pinky_base.x)
        lms = [MockLandmark(0.5, 0.5)] * 21
        lms[5] = MockLandmark(0.6, 0.5)  # Index base
        lms[17] = MockLandmark(0.4, 0.5) # Pinky base
        
        # Index UP (tip.y < pip.y or base.y)
        lms[8] = MockLandmark(0.5, 0.3)
        lms[6] = MockLandmark(0.5, 0.4)
        
        # Middle, Ring, Pinky DOWN (tip.y > pip.y)
        for tip, pip in [(12, 10), (16, 14), (20, 18)]:
            lms[tip] = MockLandmark(0.5, 0.7)
            lms[pip] = MockLandmark(0.5, 0.6)
        
        # Thumb FOLDED (for Left hand palm facing, tip.x < base.x is folded)
        lms[3] = MockLandmark(0.6, 0.6)
        lms[4] = MockLandmark(0.5, 0.6)
        
        # Wrist and Middle MCP for scale
        lms[0] = MockLandmark(0.5, 0.8)
        lms[9] = MockLandmark(0.5, 0.5)
        
        # Call multiple times to bypass STABLE_FRAMES (3)
        self.detector.detect(lms, "Left")
        self.detector.detect(lms, "Left")
        res = self.detector.detect(lms, "Left")
        self.assertEqual(res, "DRAW")

if __name__ == "__main__":
    unittest.main()
