"""Unit tests for LatestFrameBuffer zero-latency discard queue."""

import unittest
import numpy as np

from c12_thermal_detection.video.frame_buffer import LatestFrameBuffer


class TestVideoBuffer(unittest.TestCase):

    def test_buffer_put_and_get(self):
        buf = LatestFrameBuffer(capacity=1)
        frame1 = np.ones((10, 10, 3), dtype=np.uint8)
        buf.put(frame1, timestamp=100.0)

        got_frame, ts, idx = buf.get(timeout=0.1)
        self.assertIsNotNone(got_frame)
        self.assertEqual(ts, 100.0)
        self.assertEqual(idx, 1)
        self.assertEqual(buf.dropped_count, 0)

    def test_buffer_discards_stale_frames(self):
        buf = LatestFrameBuffer(capacity=1)
        frame1 = np.ones((10, 10, 3), dtype=np.uint8) * 1
        frame2 = np.ones((10, 10, 3), dtype=np.uint8) * 2
        frame3 = np.ones((10, 10, 3), dtype=np.uint8) * 3

        # Fast producer: writes 3 frames before consumer reads
        buf.put(frame1)
        buf.put(frame2)
        buf.put(frame3)

        # Consumer should immediately receive frame 3, having dropped 2 frames
        got_frame, ts, idx = buf.get(timeout=0.1)
        self.assertIsNotNone(got_frame)
        self.assertEqual(got_frame[0, 0, 0], 3)
        self.assertEqual(buf.dropped_count, 2)
        self.assertEqual(buf.total_written, 3)
        self.assertEqual(buf.total_read, 1)

    def test_buffer_timeout_on_empty(self):
        buf = LatestFrameBuffer(capacity=1)
        frame, ts, idx = buf.get(timeout=0.05)
        self.assertIsNone(frame)
        self.assertEqual(idx, -1)


if __name__ == "__main__":
    unittest.main()

