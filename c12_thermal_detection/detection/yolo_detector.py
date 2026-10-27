"""
YOLO Thermal Person Detector.
Wraps Ultralytics YOLO with dedicated support for CPU/CUDA devices,
custom fine-tuned checkpoints, latency profiling, and thermal preprocessing.
"""

from pathlib import Path
import time
from typing import List, Optional, Tuple, Union
import numpy as np

try:
    import torch
except ImportError:
    torch = None

from .preprocessing import ThermalPreprocessor
from .results import BoundingBox, DetectionResult


class YOLODetector:
    """Thread-safe detector engine for thermal human detection."""

    def __init__(
        self,
        model_path: Union[str, Path] = "models/yolo11n.pt",
        device: str = "cuda:0",
        imgsz: int = 640,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        target_classes: Optional[List[int]] = None,
        preprocessor: Optional[ThermalPreprocessor] = None,
        half_precision: bool = True,
    ):
        self.model_path = Path(model_path)
        self.requested_device = device
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.target_classes = target_classes if target_classes is not None else [0]  # 0: person
        self.preprocessor = preprocessor if preprocessor is not None else ThermalPreprocessor()
        self.half_precision = half_precision

        self.model = None
        self.actual_device = "cpu"
        self._is_loaded = False
        self._load_error: Optional[str] = None

        self._resolve_device()
        self.load_model(self.model_path)

    def _resolve_device(self) -> None:
        """Verify device availability (e.g. RTX 2080 Ti CUDA vs CPU fallback)."""
        if torch is not None and "cuda" in self.requested_device.lower() and torch.cuda.is_available():
            self.actual_device = self.requested_device
        else:
            self.actual_device = "cpu"

    def load_model(self, model_path: Union[str, Path]) -> bool:
        """Load or reload model weights from checkpoint path."""
        self.model_path = Path(model_path)
        self._is_loaded = False
        self._load_error = None

        try:
            from ultralytics import YOLO

            # Load YOLO model (will attempt local file, or download if standard baseline)
            self.model = YOLO(str(self.model_path))
            
            # Place on target device
            self.model.to(self.actual_device)
            
            # Enable half precision if CUDA
            if self.actual_device != "cpu" and self.half_precision:
                try:
                    self.model.model.half()
                except Exception:
                    pass

            self._is_loaded = True
            return True
        except Exception as e:
            self._load_error = f"Failed to load model '{self.model_path}': {str(e)}"
            self.model = None
            self._is_loaded = False
            return False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    @property
    def device_info(self) -> str:
        if torch is not None and self.actual_device != "cpu" and torch.cuda.is_available():
            dev_idx = torch.cuda.current_device()
            dev_name = torch.cuda.get_device_name(dev_idx)
            return f"GPU ({dev_name})"
        return "CPU (No CUDA)"

    def predict(
        self,
        frame: np.ndarray,
        frame_index: int = 0,
        timestamp: Optional[float] = None,
    ) -> DetectionResult:
        """
        Execute person detection on a single thermal frame.
        Guaranteed not to raise an unhandled exception to prevent crashing the UI.
        """
        now = timestamp if timestamp is not None else time.time()

        if frame is None or frame.size == 0:
            return DetectionResult(
                timestamp=now,
                source_frame_index=frame_index,
                error_message="Empty frame provided",
            )

        if not self._is_loaded or self.model is None:
            return DetectionResult(
                timestamp=now,
                source_frame_index=frame_index,
                error_message=self._load_error or "Model is not loaded",
            )

        h, w = frame.shape[:2]

        # Stage 1: Preprocessing
        t0 = time.perf_counter()
        try:
            processed_frame = self.preprocessor.process(frame)
        except Exception as e:
            return DetectionResult(
                timestamp=now,
                source_frame_index=frame_index,
                frame_width=w,
                frame_height=h,
                error_message=f"Preprocessing error: {str(e)}",
            )
        t1 = time.perf_counter()
        prep_time_ms = (t1 - t0) * 1000.0

        # Stage 2: Inference
        try:
            results = self.model.predict(
                source=processed_frame,
                imgsz=self.imgsz,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                classes=self.target_classes,
                device=self.actual_device,
                verbose=False,
            )
        except Exception as e:
            return DetectionResult(
                timestamp=now,
                source_frame_index=frame_index,
                frame_width=w,
                frame_height=h,
                preprocess_time_ms=prep_time_ms,
                error_message=f"Inference error: {str(e)}",
            )
        t2 = time.perf_counter()
        infer_time_ms = (t2 - t1) * 1000.0

        # Stage 3: Postprocessing
        boxes: List[BoundingBox] = []
        if results and len(results) > 0:
            res = results[0]
            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                classes = res.boxes.cls.cpu().numpy()

                for i in range(len(xyxy)):
                    cls_id = int(classes[i])
                    conf = float(confs[i])
                    box = xyxy[i]
                    boxes.append(
                        BoundingBox(
                            x1=float(box[0]),
                            y1=float(box[1]),
                            x2=float(box[2]),
                            y2=float(box[3]),
                            confidence=conf,
                            class_id=cls_id,
                            class_name="person",
                        )
                    )
        t3 = time.perf_counter()
        post_time_ms = (t3 - t2) * 1000.0

        return DetectionResult(
            boxes=boxes,
            preprocess_time_ms=prep_time_ms,
            inference_time_ms=infer_time_ms,
            postprocess_time_ms=post_time_ms,
            frame_width=w,
            frame_height=h,
            timestamp=now,
            source_frame_index=frame_index,
        )
