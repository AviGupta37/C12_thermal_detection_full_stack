"""
Performance and latency metrics logger.
Tracks input FPS, inference FPS, inference milliseconds, video frame age, and dropped frames.
"""

from collections import deque
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import threading
from typing import List, Optional


@dataclass
class PerformanceSample:
    """Metrics sample at a specific instant."""
    timestamp: float
    input_fps: float
    inference_fps: float
    inference_ms: float
    video_latency_ms: float
    dropped_frames: int


class PerformanceLogger:
    """Collects and computes running averages for pipeline diagnostics."""

    def __init__(self, log_dir: str = "logs", window_size: int = 120):
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._window = deque(maxlen=window_size)
        self._lock = threading.Lock()

    def record(
        self,
        input_fps: float,
        inference_fps: float,
        inference_ms: float,
        video_latency_ms: float,
        dropped_frames: int,
    ) -> None:
        """Add a performance metric point."""
        sample = PerformanceSample(
            timestamp=datetime.now().timestamp(),
            input_fps=round(input_fps, 1),
            inference_fps=round(inference_fps, 1),
            inference_ms=round(inference_ms, 2),
            video_latency_ms=round(video_latency_ms, 1),
            dropped_frames=dropped_frames,
        )
        with self._lock:
            self._window.append(sample)

    def get_averages(self) -> dict:
        """Compute average metrics over recent rolling window."""
        with self._lock:
            if not self._window:
                return {
                    "input_fps": 0.0,
                    "inference_fps": 0.0,
                    "inference_ms": 0.0,
                    "video_latency_ms": 0.0,
                    "dropped_frames": 0,
                }
            n = len(self._window)
            return {
                "input_fps": round(sum(s.input_fps for s in self._window) / n, 1),
                "inference_fps": round(sum(s.inference_fps for s in self._window) / n, 1),
                "inference_ms": round(sum(s.inference_ms for s in self._window) / n, 1),
                "video_latency_ms": round(sum(s.video_latency_ms for s in self._window) / n, 1),
                "dropped_frames": self._window[-1].dropped_frames,
            }

