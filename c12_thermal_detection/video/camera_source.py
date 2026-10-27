"""
Abstract base class for all video sources (local file, RTSP, mock).
Provides standardized lifecycle controls and diagnostic metadata.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np


class CameraSource(ABC):
    """Abstract interface defining video input capabilities."""

    @abstractmethod
    def start(self) -> bool:
        """Start acquisition worker thread. Returns True if started successfully."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop worker thread and release video resources."""
        pass

    @abstractmethod
    def get_frame(self, timeout: Optional[float] = 0.5) -> Tuple[Optional[np.ndarray], float, int]:
        """
        Retrieve the latest frame.
        Returns: (frame_bgr, timestamp, frame_index)
        """
        pass

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Indicates if the input stream is actively connected and receiving frames."""
        pass

    @property
    @abstractmethod
    def resolution(self) -> Tuple[int, int]:
        """Returns (width, height) of current source."""
        pass

    @property
    @abstractmethod
    def fps(self) -> float:
        """Returns nominal or measured source frame rate."""
        pass

    @property
    @abstractmethod
    def dropped_frames(self) -> int:
        """Returns count of dropped stale frames."""
        pass

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Human-readable identifier of the video source."""
        pass

