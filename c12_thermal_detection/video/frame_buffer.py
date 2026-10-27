"""
Single-slot bounded frame buffer for real-time video processing.
Ensures that slow consumers (inference/UI) always receive the most up-to-date frame,
dropping stale frames rather than allowing latency to accumulate.
"""

import threading
import time
from typing import Optional, Tuple
import numpy as np


class LatestFrameBuffer:
    """Thread-safe latest-frame buffer with zero latency drift and dropped frame tracking."""

    def __init__(self, capacity: int = 1):
        self._lock = threading.Lock()
        self._new_frame_event = threading.Event()
        self._latest_frame: Optional[np.ndarray] = None
        self._timestamp: float = 0.0
        self._frame_index: int = 0
        self._dropped_count: int = 0
        self._total_written: int = 0
        self._total_read: int = 0

    def put(self, frame: np.ndarray, timestamp: Optional[float] = None) -> None:
        """Store the newest frame, overwriting the previous one if unconsumed."""
        if frame is None:
            return
        
        now = timestamp if timestamp is not None else time.time()
        with self._lock:
            # If the previous frame was never read, increment dropped counter
            if self._new_frame_event.is_set():
                self._dropped_count += 1
            
            self._latest_frame = frame
            self._timestamp = now
            self._frame_index += 1
            self._total_written += 1
            self._new_frame_event.set()

    def get(self, timeout: Optional[float] = None) -> Tuple[Optional[np.ndarray], float, int]:
        """
        Retrieve the latest available frame.
        Returns (frame, timestamp, frame_index).
        Blocks up to timeout seconds if no new frame is ready.
        """
        ready = self._new_frame_event.wait(timeout=timeout)
        if not ready:
            with self._lock:
                return None, 0.0, -1

        with self._lock:
            self._new_frame_event.clear()
            self._total_read += 1
            return self._latest_frame, self._timestamp, self._frame_index

    def peek(self) -> Tuple[Optional[np.ndarray], float, int]:
        """Read the latest frame without consuming the new-frame event."""
        with self._lock:
            return self._latest_frame, self._timestamp, self._frame_index

    def clear(self) -> None:
        """Clear the buffer and reset counters."""
        with self._lock:
            self._latest_frame = None
            self._timestamp = 0.0
            self._new_frame_event.clear()
            self._dropped_count = 0
            self._total_written = 0
            self._total_read = 0

    @property
    def dropped_count(self) -> int:
        with self._lock:
            return self._dropped_count

    @property
    def total_written(self) -> int:
        with self._lock:
            return self._total_written

    @property
    def total_read(self) -> int:
        with self._lock:
            return self._total_read

