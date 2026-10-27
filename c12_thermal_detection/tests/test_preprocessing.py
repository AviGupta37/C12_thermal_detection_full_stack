"""Unit tests for thermal image preprocessing."""

import unittest
import numpy as np

from c12_thermal_detection.detection.preprocessing import ThermalPreprocessor


class TestPreprocessing(unittest.TestCase):

    def test_preprocessor_passthrough(self):
        prep = ThermalPreprocessor(enable_clahe=False, normalize=False)
        frame = np.ones((288, 384, 3), dtype=np.uint8) * 128
        out = prep.process(frame)
        self.assertEqual(out.shape, (288, 384, 3))
        self.assertTrue(np.array_equal(out, frame))

    def test_preprocessor_clahe(self):
        prep = ThermalPreprocessor(enable_clahe=True, clahe_clip_limit=3.0)
        frame = np.random.randint(50, 200, size=(288, 384, 3), dtype=np.uint8)
        out = prep.process(frame)
        self.assertEqual(out.shape, (288, 384, 3))
        self.assertEqual(out.dtype, np.uint8)

    def test_preprocessor_uint16_radiometric(self):
        prep = ThermalPreprocessor(enable_clahe=False, normalize=False)
        raw_16bit = np.linspace(2000, 8000, 288 * 384, dtype=np.uint16).reshape((288, 384))
        out = prep.process(raw_16bit)
        self.assertEqual(out.shape, (288, 384, 3))
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(out.max(), 255)
        self.assertEqual(out.min(), 0)

    def test_empty_frame(self):
        prep = ThermalPreprocessor()
        empty = np.array([])
        out = prep.process(empty)
        self.assertEqual(out.size, 0)


if __name__ == "__main__":
    unittest.main()

