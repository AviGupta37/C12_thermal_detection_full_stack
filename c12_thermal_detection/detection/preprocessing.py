"""
Thermal imagery preprocessing pipeline.
Provides configurable contrast enhancement (CLAHE, normalization) tailored
for Long-Wave Infrared (LWIR 8-14 um) aerial sensor streams.
"""

from typing import Tuple
import cv2
import numpy as np


class ThermalPreprocessor:
    """
    Applies controlled preprocessing to thermal imagery.
    Maintains exact parity between training pipeline and live inference.
    """

    def __init__(
        self,
        enable_clahe: bool = False,
        clahe_clip_limit: float = 2.0,
        clahe_grid_size: Tuple[int, int] = (8, 8),
        normalize: bool = False,
    ):
        self.enable_clahe = enable_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_grid_size = clahe_grid_size
        self.normalize = normalize

        if self.enable_clahe:
            self._clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip_limit,
                tileGridSize=self.clahe_grid_size,
            )
        else:
            self._clahe = None

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Process an input thermal frame.
        Always returns a 3-channel 8-bit BGR image suitable for YOLO inference.
        """
        if frame is None or frame.size == 0:
            return frame

        # Ensure 8-bit representation if 16-bit radiometric data is received
        if frame.dtype == np.uint16:
            # Scale 16-bit to 8-bit using min-max stretch
            f_min, f_max = frame.min(), frame.max()
            if f_max > f_min:
                frame = np.uint8(255.0 * (frame - f_min) / (f_max - f_min))
            else:
                frame = np.zeros_like(frame, dtype=np.uint8)

        # Ensure single channel grayscale for equalization if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Apply CLAHE if enabled
        if self.enable_clahe and self._clahe is not None:
            gray = self._clahe.apply(gray)

        # Apply Min-Max normalization if enabled
        if self.normalize:
            g_min, g_max = gray.min(), gray.max()
            if g_max > g_min:
                gray = np.uint8(255.0 * (gray.astype(np.float32) - g_min) / (g_max - g_min))

        # Convert back to 3-channel BGR for standard YOLO architectures
        if len(frame.shape) == 3 and not self.enable_clahe and not self.normalize:
            return frame  # Pass-through unmodified BGR thermal image
        
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

