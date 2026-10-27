"""
Local video file source with controlled playback pacing, looping, and resolution reporting.
Allows offline testing and validation without live camera hardware.
"""

import threading
import time
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np

from .camera_source import CameraSource
from .frame_buffer import LatestFrameBuffer


class VideoFileSource(CameraSource):
    """Reads recorded video files with real-time frame rate throttling and looping."""

    def __init__(self, file_path: str, loop: bool = True, target_fps: Optional[float] = None):
        self._path = Path(file_path)
        self._loop = loop
        self._buffer = LatestFrameBuffer(capacity=1)
        self._is_running = False
        self._is_connected = False
        self._thread: Optional[threading.Thread] = None
        self._cap: Optional[cv2.VideoCapture] = None
        
        self._width: int = 0
        self._height: int = 0
        self._source_fps: float = 25.0
        self._target_fps = target_fps
        self._measured_fps: float = 0.0

    def start(self) -> bool:
        if not self._path.exists():
            return False

        self._cap = cv2.VideoCapture(str(self._path))
        if not self._cap.isOpened():
            return False

        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_read = self._cap.get(cv2.CAP_PROP_FPS)
        if fps_read > 0 and not np.isnan(fps_read):
            self._source_fps = fps_read
        
        self._is_running = True
        self._is_connected = True
        self._thread = threading.Thread(target=self._worker_loop, daemon=True, name="VideoFileWorker")
        self._thread.start()
        return True

    def stop(self) -> None:
        self._is_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        
        if self._cap:
            self._cap.release()
            self._cap = None
        self._is_connected = False
        self._buffer.clear()

    def _worker_loop(self) -> None:
        pace_fps = self._target_fps if self._target_fps is not None and self._target_fps > 0 else self._source_fps
        frame_delay = 1.0 / max(pace_fps, 1.0)
        frame_count = 0
        t_start = time.time()

        while self._is_running and self._cap and self._cap.isOpened():
            loop_start = time.time()
            ret, frame = self._cap.read()
            if not ret:
                if self._loop:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    self._is_connected = False
                    break

            timestamp = time.time()
            self._buffer.put(frame, timestamp=timestamp)
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - t_start
                if elapsed > 0:
                    self._measured_fps = frame_count / elapsed

            # Maintain natural video playback speed
            elapsed_work = time.time() - loop_start
            sleep_time = frame_delay - elapsed_work
            if sleep_time > 0:
                time.sleep(sleep_time)

    def get_frame(self, timeout: Optional[float] = 0.5) -> Tuple[Optional[np.ndarray], float, int]:
        return self._buffer.get(timeout=timeout)

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def resolution(self) -> Tuple[int, int]:
        return self._width, self._height

    @property
    def fps(self) -> float:
        return self._measured_fps if self._measured_fps > 0 else self._source_fps

    @property
    def dropped_frames(self) -> int:
        return self._buffer.dropped_count

    @property
    def source_name(self) -> str:
        return f"File: {self._path.name}"

