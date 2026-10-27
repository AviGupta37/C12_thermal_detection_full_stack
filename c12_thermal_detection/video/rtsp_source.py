"""
RTSP Video Stream Source for Skydroid C12 and standard IP cameras.
Operates in an isolated background thread with robust reconnection logic,
transport protocol preferences (TCP), and stale frame dropping.
"""

import os
import threading
import time
from typing import Callable, Optional, Tuple
import cv2
import numpy as np

from .camera_source import CameraSource
from .frame_buffer import LatestFrameBuffer


class RTSPSource(CameraSource):
    """Acquires live thermal frames from an RTSP stream (e.g. C12 receiver endpoint)."""

    def __init__(
        self,
        rtsp_url: str,
        transport: str = "tcp",
        connection_timeout_ms: int = 5000,
        read_timeout_ms: int = 3000,
        max_reconnect_attempts: int = 10,
        reconnect_delay_sec: float = 2.0,
        on_status_change: Optional[Callable[[str], None]] = None,
    ):
        self._url = rtsp_url
        self._transport = transport.lower()
        self._conn_timeout_ms = connection_timeout_ms
        self._read_timeout_ms = read_timeout_ms
        self._max_reconnects = max_reconnect_attempts
        self._reconnect_delay = reconnect_delay_sec
        self._on_status_change = on_status_change

        self._buffer = LatestFrameBuffer(capacity=1)
        self._is_running = False
        self._is_connected = False
        self._thread: Optional[threading.Thread] = None
        self._cap: Optional[cv2.VideoCapture] = None

        self._width: int = 0
        self._height: int = 0
        self._fps: float = 25.0
        self._measured_fps: float = 0.0
        self._last_frame_time: float = 0.0

    def start(self) -> bool:
        if self._is_running:
            return True

        self._is_running = True
        self._thread = threading.Thread(target=self._worker_loop, daemon=True, name="RTSPWorker")
        self._thread.start()
        return True

    def stop(self) -> None:
        self._is_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        self._close_capture()
        self._is_connected = False
        self._buffer.clear()
        self._notify_status("Disconnected")

    def _notify_status(self, status: str) -> None:
        if self._on_status_change:
            try:
                self._on_status_change(status)
            except Exception:
                pass

    def _open_capture(self) -> bool:
        # Enforce TCP transport to prevent UDP packet drop and tearing in thermal video
        if self._transport == "tcp":
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000"
        else:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|stimeout;5000000"

        self._notify_status(f"Connecting to {self._url} ({self._transport.upper()})...")
        cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            self._notify_status("Connection failed")
            return False

        self._cap = cap
        self._width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_read = cap.get(cv2.CAP_PROP_FPS)
        if fps_read > 0 and not np.isnan(fps_read):
            self._fps = fps_read

        self._is_connected = True
        self._last_frame_time = time.time()
        self._notify_status("Connected")
        return True

    def _close_capture(self) -> None:
        if self._cap:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        self._is_connected = False

    def _worker_loop(self) -> None:
        reconnect_attempts = 0

        while self._is_running:
            if not self._is_connected or self._cap is None or not self._cap.isOpened():
                success = self._open_capture()
                if not success:
                    reconnect_attempts += 1
                    if self._max_reconnects > 0 and reconnect_attempts >= self._max_reconnects:
                        self._notify_status("Max reconnect attempts reached. Stream stopped.")
                        self._is_running = False
                        break
                    time.sleep(self._reconnect_delay)
                    continue
                else:
                    reconnect_attempts = 0

            # Active frame capture loop
            frame_count = 0
            t_start = time.time()

            while self._is_running and self._is_connected and self._cap:
                ret, frame = self._cap.read()
                now = time.time()

                if not ret or frame is None:
                    # Check timeout
                    if (now - self._last_frame_time) * 1000 > self._read_timeout_ms:
                        self._notify_status("Read timeout: No frames received")
                        self._close_capture()
                        break
                    time.sleep(0.005)
                    continue

                self._last_frame_time = now
                self._buffer.put(frame, timestamp=now)
                frame_count += 1

                if frame_count % 30 == 0:
                    elapsed = now - t_start
                    if elapsed > 0:
                        self._measured_fps = frame_count / elapsed

        self._close_capture()

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
        return self._measured_fps if self._measured_fps > 0 else self._fps

    @property
    def dropped_frames(self) -> int:
        return self._buffer.dropped_count

    @property
    def source_name(self) -> str:
        return f"RTSP: {self._url}"

