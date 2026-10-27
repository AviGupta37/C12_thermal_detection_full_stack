"""
High-performance Thermal Video Viewport for PySide6.
Renders 8-bit LWIR frames, detection bounding boxes, and HUD metrics overlay.
Preserves aspect ratio on window resize and clearly flags placeholder states.
"""

from typing import Optional, Tuple
import cv2
import numpy as np

from PySide6.QtCore import Qt, QRect, QPoint
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QFont, QPen
from PySide6.QtWidgets import QWidget, QSizePolicy


class ThermalView(QWidget):
    """Viewport for rendering aerial thermal video with detection overlays."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(480, 360)
        
        self._current_pixmap: Optional[QPixmap] = None
        self._show_overlay: bool = True
        self._placeholder_text: str = "NO VIDEO FEED CONNECTED\n[Select 'Open Video File' or 'Connect Camera']"
        
        # HUD statistics
        self._resolution: Tuple[int, int] = (0, 0)
        self._stream_fps: float = 0.0
        self._latency_ms: float = 0.0
        self._person_count: int = 0
        self._source_name: str = "Disconnected"

    def set_placeholder(self, message: str) -> None:
        """Set placeholder message and clear active video."""
        self._placeholder_text = message
        self._current_pixmap = None
        self._resolution = (0, 0)
        self._stream_fps = 0.0
        self.update()

    def set_overlay_visible(self, visible: bool) -> None:
        """Toggle detection bounding box overlays."""
        self._show_overlay = visible
        self.update()

    def update_frame(
        self,
        frame_bgr: np.ndarray,
        resolution: Tuple[int, int],
        stream_fps: float,
        latency_ms: float = 0.0,
        person_count: int = 0,
        source_name: str = "",
    ) -> None:
        """Update viewport with new BGR frame and metadata."""
        if frame_bgr is None or frame_bgr.size == 0 or Qt is None:
            return

        h, w, ch = frame_bgr.shape
        bytes_per_line = ch * w
        
        # Convert BGR (OpenCV) to RGB for Qt rendering
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self._current_pixmap = QPixmap.fromImage(q_img)
        self._resolution = resolution
        self._stream_fps = stream_fps
        self._latency_ms = latency_ms
        self._person_count = person_count
        self._source_name = source_name

        self.update()

    def paintEvent(self, event) -> None:
        if Qt is None:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Background fill
        painter.fillRect(self.rect(), QColor("#0D0F12"))

        if self._current_pixmap is not None and not self._current_pixmap.isNull():
            # Scale pixmap preserving aspect ratio
            scaled = self._current_pixmap.scaled(
                self.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            # Center the video in viewport
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)

            # Draw HUD Overlays (top-left and top-right)
            self._draw_hud(painter, x, y, scaled.width(), scaled.height())
        else:
            # Draw placeholder state
            self._draw_placeholder(painter)

        painter.end()

    def _draw_placeholder(self, painter: QPainter) -> None:
        """Renders explicit disconnected placeholder."""
        painter.setPen(QColor("#4A5568"))
        painter.setFont(QFont("Segoe UI", 12, QFont.Bold))

        # Camera lens placeholder graphic
        cx, cy = self.width() // 2, self.height() // 2
        painter.drawRect(cx - 160, cy - 80, 320, 160)
        
        # Message text
        painter.setPen(QColor("#718096"))
        painter.drawText(
            self.rect(),
            Qt.AlignCenter,
            f"SIGNAL STATE: OFFLINE\n\n{self._placeholder_text}",
        )

    def _draw_hud(self, painter: QPainter, vx: int, vy: int, vw: int, vh: int) -> None:
        """Renders transparent HUD chips showing stream resolution, FPS, and latency."""
        # Top-left HUD badge: Source & Resolution
        hud_bg = QColor(18, 22, 28, 200)
        hud_border = QColor(45, 55, 72)
        
        painter.setPen(QPen(hud_border, 1))
        painter.setBrush(hud_bg)
        painter.drawRoundedRect(vx + 10, vy + 10, 240, 48, 4, 4)

        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
        painter.setPen(QColor("#63B3ED"))
        painter.drawText(vx + 18, vy + 28, f"SOURCE: {self._source_name}")

        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QColor("#A0AEC0"))
        res_str = f"RES: {self._resolution[0]}x{self._resolution[1]}"
        fps_str = f"STREAM: {self._stream_fps:.1f} FPS"
        lat_str = f"LATENCY: {self._latency_ms:.0f} ms"
        painter.drawText(vx + 18, vy + 46, f"{res_str}  |  {fps_str}  |  {lat_str}")

        # Top-right HUD badge: Real-time detection count
        if self._person_count > 0:
            badge_color = QColor(39, 103, 73, 230)  # Green alert
            badge_border = QColor(72, 187, 120)
            text_color = QColor("#FFFFFF")
            count_str = f"HUMAN DETECTED: {self._person_count}"
        else:
            badge_color = QColor(26, 32, 44, 180)
            badge_border = QColor(45, 55, 72)
            text_color = QColor("#718096")
            count_str = "HUMAN DETECTED: 0"

        badge_w = 170
        bx = vx + vw - badge_w - 10
        painter.setPen(QPen(badge_border, 1))
        painter.setBrush(badge_color)
        painter.drawRoundedRect(bx, vy + 10, badge_w, 32, 4, 4)

        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
        painter.setPen(text_color)
        painter.drawText(QRect(bx, vy + 10, badge_w, 32), Qt.AlignCenter, count_str)

