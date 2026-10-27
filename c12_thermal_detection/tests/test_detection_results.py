"""Unit tests for DetectionResult, BoundingBox geometry, and overlays."""

import unittest
import numpy as np

from c12_thermal_detection.detection.results import BoundingBox, DetectionResult


class TestDetectionResults(unittest.TestCase):

    def test_bounding_box_geometry(self):
        box = BoundingBox(x1=100.0, y1=50.0, x2=200.0, y2=250.0, confidence=0.88)
        self.assertEqual(box.width, 100.0)
        self.assertEqual(box.height, 200.0)
        self.assertEqual(box.area, 20000.0)

    def test_bounding_box_to_xywhn(self):
        box = BoundingBox(x1=200.0, y1=100.0, x2=400.0, y2=300.0, confidence=0.90)
        cx, cy, w, h = box.to_xywhn(img_w=1000, img_h=500)
        self.assertAlmostEqual(cx, 0.30, places=3)
        self.assertAlmostEqual(cy, 0.40, places=3)
        self.assertAlmostEqual(w, 0.20, places=3)
        self.assertAlmostEqual(h, 0.40, places=3)

    def test_detection_result_metrics(self):
        b1 = BoundingBox(x1=10, y1=10, x2=50, y2=80, confidence=0.82)
        b2 = BoundingBox(x1=100, y1=100, x2=150, y2=190, confidence=0.94)

        res = DetectionResult(
            boxes=[b1, b2],
            preprocess_time_ms=2.5,
            inference_time_ms=15.0,
            postprocess_time_ms=1.5,
            frame_width=384,
            frame_height=288,
        )

        self.assertEqual(res.person_count, 2)
        self.assertEqual(res.highest_confidence, 0.94)
        self.assertEqual(res.total_time_ms, 19.0)
        self.assertAlmostEqual(res.effective_fps, 1000.0 / 19.0, places=1)

    def test_overlay_rendering(self):
        frame = np.zeros((288, 384, 3), dtype=np.uint8)
        box = BoundingBox(x1=50.0, y1=50.0, x2=100.0, y2=120.0, confidence=0.91)
        res = DetectionResult(boxes=[box], frame_width=384, frame_height=288)

        rendered = res.draw_overlays(frame, show_boxes=True, show_labels=True)
        self.assertEqual(rendered.shape, frame.shape)
        self.assertTrue(np.any(rendered > 0))


if __name__ == "__main__":
    unittest.main()

