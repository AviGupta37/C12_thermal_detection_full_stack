"""
Main Operator Window for Skydroid C12 Aerial Thermal Human Detection.
Orchestrates high-framerate thermal video rendering, threaded YOLO inference,
PX4 telemetry updates, video recording, snapshot capture, and audit logging.
"""

from datetime import datetime
from pathlib import Path
import sys
import threading
import time
from typing import Optional, Tuple
import cv2
import numpy as np

from PySide6.QtCore import Qt, QTimer, Signal, QObject
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QFileDialog, QMessageBox, QStatusBar
)

from .dashboard_panels import (
    HeaderBar, DetectionPanel, TelemetryPanel,
    PerformancePanel, ControlsPanel, EventLogPanel
)
from .settings_dialog import SettingsDialog
from .styles import DARK_THEME_QSS
from .thermal_view import ThermalView

from ..detection.preprocessing import ThermalPreprocessor
from ..detection.results import DetectionResult
from ..detection.yolo_detector import YOLODetector
from ..flight.mock_telemetry import MockTelemetryProvider
from ..flight.telemetry_interface import TelemetryData
from ..logging_tools.event_logger import EventLogger
from ..logging_tools.performance_logger import PerformanceLogger
from ..video.camera_source import CameraSource
from ..video.rtsp_source import RTSPSource
from ..video.video_file_source import VideoFileSource


class MainWindow(QMainWindow):
    """Primary Ground Station Dashboard."""

    def __init__(self, app_config: Optional[dict] = None):
        super().__init__()
        self.setWindowTitle("C12 Aerial Thermal Human Detection Ground Station")
        self.resize(1600, 900)
        self.setStyleSheet(DARK_THEME_QSS)

        # Output Directories
        self.recordings_dir = Path("recordings")
        self.snapshots_dir = Path("snapshots")
        self.logs_dir = Path("logs")
        self.recordings_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Loggers
        self.event_logger = EventLogger(log_dir=str(self.logs_dir))
        self.perf_logger = PerformanceLogger(log_dir=str(self.logs_dir))

        # Runtime Settings
        self.settings = {
            "rtsp_url": "rtsp://192.168.144.108:555/stream=2",
            "rtsp_transport": "tcp",
            "model_path": "models/yolo11n.pt",
            "conf_threshold": 0.35,
            "imgsz": 640,
            "enable_clahe": False,
            "use_mock_telemetry": True,
            "mavlink_conn": "udp:127.0.0.1:14550",
        }

        # Subsystems State
        self.video_source: Optional[CameraSource] = None
        self.detector: Optional[YOLODetector] = None
        self.detection_enabled: bool = False
        self.overlay_enabled: bool = True
        
        # Telemetry
        self.telemetry_provider: Optional[MockTelemetryProvider] = None
        
        # Video Recording
        self.is_recording: bool = False
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.recording_path: Optional[Path] = None
        self.recording_frames_count: int = 0

        # Latest cache
        self.latest_raw_frame: Optional[np.ndarray] = None
        self.latest_rendered_frame: Optional[np.ndarray] = None
        self.latest_detection: Optional[DetectionResult] = None
        self.last_infer_time_ms: float = 0.0
        self.total_session_detections: int = 0

        # Build UI layout
        self._init_ui()
        
        # Initialize Detection Engine
        self._init_detector()

        # Initialize Telemetry
        self._init_telemetry()

        # UI Refresh Timer (~30 FPS)
        self.render_timer = QTimer(self)
        self.render_timer.timeout.connect(self._on_ui_tick)
        self.render_timer.start(33)

        # Log session startup
        self.event_logger.info("SYSTEM", "Ground Station initialized. Ready for operations.")
        self._sync_event_log()

    def _init_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 1. Top Header Bar
        self.header_bar = HeaderBar(self)
        main_layout.addWidget(self.header_bar)

        # 2. Main Body Splitter (Left: Thermal Viewport | Right: Telemetry & Status Panels)
        body_splitter = QSplitter(Qt.Horizontal, self)
        
        # Left Container: Viewport + Event Log
        left_container = QWidget(self)
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(10)

        self.thermal_view = ThermalView(self)
        left_layout.addWidget(self.thermal_view, stretch=3)

        self.event_log_panel = EventLogPanel(on_export_csv=self._on_export_csv, parent=self)
        left_layout.addWidget(self.event_log_panel, stretch=1)

        body_splitter.addWidget(left_container)

        # Right Container: Side Panels
        right_container = QWidget(self)
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(10)

        self.detection_panel = DetectionPanel(self)
        right_layout.addWidget(self.detection_panel)

        self.telemetry_panel = TelemetryPanel(self)
        right_layout.addWidget(self.telemetry_panel)

        self.perf_panel = PerformancePanel(self)
        right_layout.addWidget(self.perf_panel)

        self.controls_panel = ControlsPanel(
            on_start_det=self._on_start_detection,
            on_stop_det=self._on_stop_detection,
            on_open_file=self._on_open_video_file,
            on_connect_cam=self._on_connect_camera,
            on_disconnect_cam=self._on_disconnect_camera,
            on_record_toggle=self._on_toggle_recording,
            on_snapshot=self._on_save_snapshot,
            on_toggle_overlay=self._on_toggle_overlay,
            on_open_settings=self._on_open_settings,
            parent=self,
        )
        right_layout.addWidget(self.controls_panel)
        right_layout.addStretch()

        body_splitter.addWidget(right_container)
        body_splitter.setStretchFactor(0, 7)
        body_splitter.setStretchFactor(1, 3)

        main_layout.addWidget(body_splitter)

    def _init_detector(self):
        prep = ThermalPreprocessor(enable_clahe=self.settings["enable_clahe"])
        self.detector = YOLODetector(
            model_path=self.settings["model_path"],
            conf_threshold=self.settings["conf_threshold"],
            imgsz=self.settings["imgsz"],
            preprocessor=prep,
        )
        has_cuda = "GPU" in self.detector.device_info
        self.header_bar.set_gpu_status(has_cuda, self.detector.device_info)
        
        if self.detector.is_loaded:
            self.event_logger.info("DETECTOR", f"Loaded model: {Path(self.settings['model_path']).name} on {self.detector.device_info}")
        else:
            self.event_logger.warning("DETECTOR", f"Baseline weights not found. You can select a model in Settings.")

    def _init_telemetry(self):
        if self.settings["use_mock_telemetry"]:
            self.telemetry_provider = MockTelemetryProvider(
                update_rate_hz=4.0,
                on_update=self._on_telemetry_update,
            )
            self.telemetry_provider.start()
            self.header_bar.set_telemetry_status(is_connected=True, is_simulated=True)
            self.event_logger.info("TELEMETRY", "Started simulated PX4 telemetry feed.")

    def _on_telemetry_update(self, data: TelemetryData):
        self.telemetry_panel.update_telemetry(
            mode=data.flight_mode,
            alt_str=data.altitude_str,
            batt_str=data.battery_str,
            gps_str=data.gps_status,
        )

    def _on_ui_tick(self):
        """Render loop invoked every ~33ms."""
        if not self.video_source or not self.video_source.is_connected:
            return

        frame, ts, idx = self.video_source.get_frame(timeout=0.01)
        if frame is None:
            return

        self.latest_raw_frame = frame
        h, w = frame.shape[:2]
        now = time.time()
        video_latency = max(0.0, (now - ts) * 1000.0) if ts > 0 else 0.0

        # Perform YOLO inference if detection active
        person_count = 0
        peak_conf = 0.0

        if self.detection_enabled and self.detector and self.detector.is_loaded:
            det_result = self.detector.predict(frame, frame_index=idx, timestamp=ts)
            self.latest_detection = det_result
            self.last_infer_time_ms = det_result.total_time_ms
            person_count = det_result.person_count
            peak_conf = det_result.highest_confidence

            if person_count > 0:
                self.total_session_detections += 1
                if self.total_session_detections % 30 == 1:
                    self.event_logger.alert("DETECTION", f"Human detected ({person_count} in frame, max conf {int(peak_conf*100)}%)")
                    self._sync_event_log()

            # Render overlay if enabled
            if self.overlay_enabled:
                rendered = det_result.draw_overlays(frame)
            else:
                rendered = frame
        else:
            rendered = frame

        self.latest_rendered_frame = rendered

        # Video recording stream capture
        if self.is_recording and self.video_writer:
            try:
                self.video_writer.write(rendered)
                self.recording_frames_count += 1
            except Exception as e:
                self.event_logger.error("RECORDING", f"Frame write error: {e}")

        # Update viewport
        self.thermal_view.update_frame(
            frame_bgr=rendered,
            resolution=(w, h),
            stream_fps=self.video_source.fps,
            latency_ms=video_latency,
            person_count=person_count,
            source_name=self.video_source.source_name,
        )

        # Update detection panel
        self.detection_panel.update_metrics(
            count=person_count,
            highest_conf=peak_conf,
            infer_ms=self.last_infer_time_ms,
        )

        # Update performance diagnostics
        self.perf_panel.update_performance(
            input_fps=self.video_source.fps,
            latency_ms=video_latency,
            dropped=self.video_source.dropped_frames,
            device_str=self.detector.device_info if self.detector else "N/A",
        )

    def _sync_event_log(self):
        recent = self.event_logger.get_recent_entries(limit=100)
        self.event_log_panel.table.setRowCount(0)
        for e in recent:
            self.event_log_panel.add_entry(e.timestamp, e.level, e.category, e.message)

    # Operator Action Callbacks
    def _on_start_detection(self):
        if not self.detector or not self.detector.is_loaded:
            QMessageBox.warning(self, "Detection Unavailable", "YOLO model weights are not loaded. Please select a model in Settings.")
            return
        self.detection_enabled = True
        self.header_bar.set_detector_status(True, Path(self.settings["model_path"]).stem)
        self.event_logger.info("DETECTOR", "Thermal human detection engine started.")
        self._sync_event_log()

    def _on_stop_detection(self):
        self.detection_enabled = False
        self.header_bar.set_detector_status(False)
        self.event_logger.info("DETECTOR", "Detection engine stopped.")
        self._sync_event_log()

    def _on_open_video_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Prerecorded Thermal Video",
            "",
            "Video Files (*.mp4 *.avi *.mkv *.mov);;All Files (*)",
        )
        if not file_path:
            return

        self._on_disconnect_camera()
        source = VideoFileSource(file_path=file_path, loop=True)
        if source.start():
            self.video_source = source
            self.header_bar.set_camera_status(True, Path(file_path).name)
            self.event_logger.info("CAMERA", f"Opened local video source: {Path(file_path).name}")
            self._sync_event_log()
        else:
            QMessageBox.critical(self, "Error", f"Could not open video file: {file_path}")

    def _on_connect_camera(self):
        self._on_disconnect_camera()
        url = self.settings["rtsp_url"]
        transport = self.settings["rtsp_transport"]

        source = RTSPSource(
            rtsp_url=url,
            transport=transport,
            on_status_change=self._on_rtsp_status_change,
        )
        source.start()
        self.video_source = source
        self.header_bar.set_camera_status(True, "RTSP")
        self.event_logger.info("CAMERA", f"Attempting RTSP connection to {url} ({transport.upper()})...")
        self._sync_event_log()

    def _on_rtsp_status_change(self, status: str):
        self.event_logger.info("CAMERA", f"Stream status: {status}")

    def _on_disconnect_camera(self):
        if self.is_recording:
            self._on_toggle_recording()

        if self.video_source:
            self.video_source.stop()
            self.video_source = None
            self.header_bar.set_camera_status(False)
            self.thermal_view.set_placeholder("VIDEO SOURCE DISCONNECTED\n[Select 'Open Video File' or 'Connect Camera']")
            self.event_logger.info("CAMERA", "Video source disconnected.")
            self._sync_event_log()

    def _on_toggle_overlay(self):
        self.overlay_enabled = not self.overlay_enabled
        self.thermal_view.set_overlay_visible(self.overlay_enabled)
        state_str = "ENABLED" if self.overlay_enabled else "DISABLED"
        self.event_logger.info("VIEW", f"Detection overlays {state_str}.")
        self._sync_event_log()

    def _on_toggle_recording(self):
        if not self.is_recording:
            if self.latest_raw_frame is None:
                QMessageBox.warning(self, "Recording Failed", "No active video stream to record.")
                return
            h, w = self.latest_raw_frame.shape[:2]
            ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.recording_path = self.recordings_dir / f"c12_rec_{ts_str}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = self.video_source.fps if (self.video_source and self.video_source.fps > 0) else 25.0
            
            self.video_writer = cv2.VideoWriter(str(self.recording_path), fourcc, fps, (w, h))
            if not self.video_writer.isOpened():
                QMessageBox.critical(self, "Recording Error", "Failed to initialize video encoder.")
                return

            self.is_recording = True
            self.recording_frames_count = 0
            self.controls_panel.btn_record.setText("Stop Recording")
            self.controls_panel.btn_record.setProperty("class", "DangerButton")
            self.controls_panel.btn_record.style().unpolish(self.controls_panel.btn_record)
            self.controls_panel.btn_record.style().polish(self.controls_panel.btn_record)
            self.event_logger.info("RECORDING", f"Recording started: {self.recording_path.name}")
            self._sync_event_log()
        else:
            self.is_recording = False
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            self.controls_panel.btn_record.setText("Record Video")
            self.controls_panel.btn_record.setProperty("class", "")
            self.controls_panel.btn_record.style().unpolish(self.controls_panel.btn_record)
            self.controls_panel.btn_record.style().polish(self.controls_panel.btn_record)
            self.event_logger.info("RECORDING", f"Recording saved ({self.recording_frames_count} frames): {self.recording_path}")
            self._sync_event_log()

    def _on_save_snapshot(self):
        target = self.latest_rendered_frame if self.latest_rendered_frame is not None else self.latest_raw_frame
        if target is None:
            QMessageBox.warning(self, "Snapshot Error", "No frame available to capture.")
            return

        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        snap_path = self.snapshots_dir / f"snapshot_{ts_str}.png"
        cv2.imwrite(str(snap_path), target)
        self.event_logger.info("SNAPSHOT", f"Saved snapshot: {snap_path.name}")
        self._sync_event_log()
        QMessageBox.information(self, "Snapshot Saved", f"Saved snapshot to:\n{snap_path.resolve()}")

    def _on_open_settings(self):
        dlg = SettingsDialog(self.settings, self)
        if dlg.exec():
            self.settings.update(dlg.settings)
            self.event_logger.info("SYSTEM", "Updated configuration settings.")
            
            # Apply updated model if changed
            if self.detector:
                self.detector.conf_threshold = self.settings["conf_threshold"]
                self.detector.imgsz = self.settings["imgsz"]
                self.detector.preprocessor.enable_clahe = self.settings["enable_clahe"]
                if Path(self.settings["model_path"]) != self.detector.model_path:
                    self.detector.load_model(self.settings["model_path"])
                    self.header_bar.set_gpu_status("GPU" in self.detector.device_info, self.detector.device_info)
            self._sync_event_log()

    def _on_export_csv(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Audit Log to CSV",
            str(self.logs_dir / f"audit_export_{int(time.time())}.csv"),
            "CSV Files (*.csv);;All Files (*)",
        )
        if file_path:
            success = self.event_logger.export_csv(Path(file_path))
            if success:
                QMessageBox.information(self, "Export Success", f"Log successfully exported to:\n{file_path}")
            else:
                QMessageBox.critical(self, "Export Error", "Failed to write CSV file.")

    def closeEvent(self, event):
        """Clean shutdown of all worker threads and background streams."""
        self._on_disconnect_camera()
        if self.telemetry_provider:
            self.telemetry_provider.stop()
        self.event_logger.info("SYSTEM", "Ground Station application terminated.")
        event.accept()

