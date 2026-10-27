"""
Detection results data contracts, bounding box geometries, and overlay visualization.
Strictly scoped to person detection without tracking IDs or trajectory trails.
"""

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class BoundingBox:
    """Represents a single detected person bounding box."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int = 0
    class_name: str = "person"

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_xywhn(self, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
        """Convert pixel coordinates [x1, y1, x2, y2] to YOLO normalized [cx, cy, w, h]."""
        if img_w <= 0 or img_h <= 0:
            return 0.0, 0.0, 0.0, 0.0
        
        w_norm = self.width / img_w
        h_norm = self.height / img_h
        cx_norm = (self.x1 + self.width / 2.0) / img_w
        cy_norm = (self.y1 + self.height / 2.0) / img_h
        
        # Clamp between 0.0 and 1.0
        cx = max(0.0, min(1.0, cx_norm))
        cy = max(0.0, min(1.0, cy_norm))
        w = max(0.0, min(1.0, w_norm))
        h = max(0.0, min(1.0, h_norm))
        return cx, cy, w, h

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x1": round(self.x1, 1),
            "y1": round(self.y1, 1),
            "x2": round(self.x2, 1),
            "y2": round(self.y2, 1),
            "confidence": round(self.confidence, 4),
            "class_id": self.class_id,
            "class_name": self.class_name,
        }


@dataclass
class DetectionResult:
    """Encapsulates all detections and performance metrics for a single frame."""
    boxes: List[BoundingBox] = field(default_factory=list)
    preprocess_time_ms: float = 0.0
    inference_time_ms: float = 0.0
    postprocess_time_ms: float = 0.0
    frame_width: int = 0
    frame_height: int = 0
    timestamp: float = field(default_factory=time.time)
    source_frame_index: int = 0
    error_message: Optional[str] = None

    @property
    def total_time_ms(self) -> float:
        return self.preprocess_time_ms + self.inference_time_ms + self.postprocess_time_ms

    @property
    def person_count(self) -> int:
        return len(self.boxes)

    @property
    def highest_confidence(self) -> float:
        if not self.boxes:
            return 0.0
        return max(b.confidence for b in self.boxes)

    @property
    def effective_fps(self) -> float:
        total = self.total_time_ms
        return (1000.0 / total) if total > 0 else 0.0

    def draw_overlays(
        self,
        image: np.ndarray,
        show_boxes: bool = True,
        show_labels: bool = True,
        show_confidences: bool = True,
        box_color: Tuple[int, int, int] = (0, 255, 0),  # BGR Green
        box_thickness: int = 2,
    ) -> np.ndarray:
        """Render person bounding boxes and confidence tags directly onto frame copy."""
        if not show_boxes or not self.boxes or image is None:
            return image

        canvas = image.copy()
        for box in self.boxes:
            ix1, iy1 = int(round(box.x1)), int(round(box.y1))
            ix2, iy2 = int(round(box.x2)), int(round(box.y2))
            
            # Draw bounding box
            cv2.rectangle(canvas, (ix1, iy1), (ix2, iy2), box_color, box_thickness)
            
            # Format label text
            if show_labels or show_confidences:
                label_parts = []
                if show_labels:
                    label_parts.append(box.class_name)
                if show_confidences:
                    label_parts.append(f"{int(box.confidence * 100)}%")
                label_text = " ".join(label_parts)
                
                # Label background badge
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
                badge_y1 = max(0, iy1 - th - baseline - 4)
                badge_y2 = iy1
                badge_x2 = min(canvas.shape[1], ix1 + tw + 6)
                
                cv2.rectangle(canvas, (ix1, badge_y1), (badge_x2, badge_y2), box_color, -1)
                cv2.putText(
                    canvas,
                    label_text,
                    (ix1 + 3, badge_y2 - baseline - 2),
                    font,
                    font_scale,
                    (0, 0, 0),
                    thickness,
                    cv2.LINE_AA,
                )

        return canvas

    def to_dict(self) -> Dict[str, Any]:
        return {
            "person_count": self.person_count,
            "highest_confidence": round(self.highest_confidence, 4),
            "preprocess_time_ms": round(self.preprocess_time_ms, 2),
            "inference_time_ms": round(self.inference_time_ms, 2),
            "postprocess_time_ms": round(self.postprocess_time_ms, 2),
            "total_time_ms": round(self.total_time_ms, 2),
            "frame_resolution": [self.frame_width, self.frame_height],
            "timestamp": self.timestamp,
            "boxes": [b.to_dict() for b in self.boxes],
            "error_message": self.error_message,
        }

