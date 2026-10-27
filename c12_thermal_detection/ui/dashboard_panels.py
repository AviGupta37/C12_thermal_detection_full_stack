"""
Modular dashboard panels for the C12 Thermal Ground Station UI.
Includes HeaderBar, DetectionPanel, TelemetryPanel, PerformancePanel, ControlsPanel, and EventLogPanel.
"""

from datetime import datetime
from typing import Callable, Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QPushButton, QFrame, QGroupBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView
)


class HeaderBar(QFrame):
    """Top status bar displaying system clock, camera link, detector, GPU, and telemetry status."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("HeaderBar")
        self._init_ui()

    def _init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 0, 12, 0)
        layout.setSpacing(14)

        # Title & Subtitle
        title_box = QVBoxLayout()
        title_box.setSpacing(1)
        self.title_label = QLabel("C12 THERMAL DETECTION")
        self.title_label.setObjectName("AppTitle")
        self.subtitle_label = QLabel("AERIAL HUMAN DETECTION GROUND STATION | PX4 OFFBOARD MONITOR")
        self.subtitle_label.setObjectName("AppSubtitle")
        title_box.addWidget(self.title_label)
        title_box.addWidget(self.subtitle_label)
        layout.addLayout(title_box)

        layout.addStretch()

        # Status Chips
        self.camera_badge = QLabel("CAMERA: DISCONNECTED")
        self.camera_badge.setProperty("class", "StatusBadge StatusInactive")
        layout.addWidget(self.camera_badge)

        self.detector_badge = QLabel("DETECTOR: IDLE")
        self.detector_badge.setProperty("class", "StatusBadge StatusInactive")
        layout.addWidget(self.detector_badge)

        self.gpu_badge = QLabel("GPU: CHECKING...")
        self.gpu_badge.setProperty("class", "StatusBadge StatusInactive")
        layout.addWidget(self.gpu_badge)

        self.telemetry_badge = QLabel("PX4: SIMULATED")
        self.telemetry_badge.setProperty("class", "StatusBadge StatusWarning")
        layout.addWidget(self.telemetry_badge)

        # Clock
        self.clock_label = QLabel("00:00:00")
        self.clock_label.setStyleSheet("font-family: monospace; font-size: 13px; font-weight: bold; color: #CBD5E0;")
        layout.addWidget(self.clock_label)

        # Timer for clock update
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_clock)
        self._timer.start(1000)
        self._update_clock()

    def _update_clock(self):
        self.clock_label.setText(datetime.now().strftime("%H:%M:%S UTC"))

    def set_camera_status(self, is_connected: bool, info: str = ""):
        if is_connected:
            self.camera_badge.setText(f"CAMERA: ONLINE ({info})" if info else "CAMERA: ONLINE")
            self.camera_badge.setProperty("class", "StatusBadge StatusActive")
        else:
            self.camera_badge.setText("CAMERA: DISCONNECTED")
            self.camera_badge.setProperty("class", "StatusBadge StatusInactive")
        self.camera_badge.style().unpolish(self.camera_badge)
        self.camera_badge.style().polish(self.camera_badge)

    def set_detector_status(self, is_running: bool, info: str = "YOLO"):
        if is_running:
            self.detector_badge.setText(f"DETECTOR: ACTIVE ({info})")
            self.detector_badge.setProperty("class", "StatusBadge StatusActive")
        else:
            self.detector_badge.setText("DETECTOR: IDLE")
            self.detector_badge.setProperty("class", "StatusBadge StatusInactive")
        self.detector_badge.style().unpolish(self.detector_badge)
        self.detector_badge.style().polish(self.detector_badge)

    def set_gpu_status(self, is_available: bool, gpu_name: str = ""):
        if is_available:
            self.gpu_badge.setText(f"GPU: {gpu_name}")
            self.gpu_badge.setProperty("class", "StatusBadge StatusActive")
        else:
            self.gpu_badge.setText("GPU: CPU FALLBACK")
            self.gpu_badge.setProperty("class", "StatusBadge StatusWarning")
        self.gpu_badge.style().unpolish(self.gpu_badge)
        self.gpu_badge.style().polish(self.gpu_badge)

    def set_telemetry_status(self, is_connected: bool, is_simulated: bool):
        if not is_connected:
            self.telemetry_badge.setText("PX4: DISCONNECTED")
            self.telemetry_badge.setProperty("class", "StatusBadge StatusError")
        elif is_simulated:
            self.telemetry_badge.setText("PX4: SIMULATED")
            self.telemetry_badge.setProperty("class", "StatusBadge StatusWarning")
        else:
            self.telemetry_badge.setText("PX4: CONNECTED")
            self.telemetry_badge.setProperty("class", "StatusBadge StatusActive")
        self.telemetry_badge.style().unpolish(self.telemetry_badge)
        self.telemetry_badge.style().polish(self.telemetry_badge)


class DetectionPanel(QGroupBox):
    """Displays human detection count, top confidence, and timing stats."""

    def __init__(self, parent=None):
        super().__init__("DETECTION STATUS (FRAME-LEVEL)", parent)
        self._init_ui()

    def _init_ui(self):
        grid = QGridLayout(self)
        grid.setSpacing(10)
        grid.setContentsMargins(12, 16, 12, 12)

        # People Count
        lbl1 = QLabel("PEOPLE IN FRAME")
        lbl1.setProperty("class", "MetricLabel")
        self.count_val = QLabel("0")
        self.count_val.setProperty("class", "MetricValueHighlight")
        grid.addWidget(lbl1, 0, 0)
        grid.addWidget(self.count_val, 1, 0)

        # Peak Confidence
        lbl2 = QLabel("TOP CONFIDENCE")
        lbl2.setProperty("class", "MetricLabel")
        self.conf_val = QLabel("0%")
        self.conf_val.setProperty("class", "MetricValue")
        grid.addWidget(lbl2, 0, 1)
        grid.addWidget(self.conf_val, 1, 1)

        # Inference Time
        lbl3 = QLabel("INFERENCE TIME")
        lbl3.setProperty("class", "MetricLabel")
        self.time_val = QLabel("0.0 ms")
        self.time_val.setProperty("class", "MetricValue")
        grid.addWidget(lbl3, 2, 0)
        grid.addWidget(self.time_val, 3, 0)

        # Inference Rate
        lbl4 = QLabel("INFERENCE SPEED")
        lbl4.setProperty("class", "MetricLabel")
        self.rate_val = QLabel("0.0 FPS")
        self.rate_val.setProperty("class", "MetricValue")
        grid.addWidget(lbl4, 2, 1)
        grid.addWidget(self.rate_val, 3, 1)

    def update_metrics(self, count: int, highest_conf: float, infer_ms: float):
        self.count_val.setText(str(count))
        if count > 0:
            self.count_val.setProperty("class", "MetricValueHighlight")
        else:
            self.count_val.setProperty("class", "MetricValue")
        self.count_val.style().unpolish(self.count_val)
        self.count_val.style().polish(self.count_val)

        self.conf_val.setText(f"{int(round(highest_conf * 100))}%")
        self.time_val.setText(f"{infer_ms:.1f} ms")
        fps = (1000.0 / infer_ms) if infer_ms > 0 else 0.0
        self.rate_val.setText(f"{fps:.1f} FPS")


class TelemetryPanel(QGroupBox):
    """Displays read-only PX4 vehicle state."""

    def __init__(self, parent=None):
        super().__init__("FLIGHT TELEMETRY (PX4 READ-ONLY)", parent)
        self._init_ui()

    def _init_ui(self):
        grid = QGridLayout(self)
        grid.setSpacing(10)
        grid.setContentsMargins(12, 16, 12, 12)

        # Flight Mode
        lbl1 = QLabel("FLIGHT MODE")
        lbl1.setProperty("class", "MetricLabel")
        self.mode_val = QLabel("N/A")
        self.mode_val.setProperty("class", "MetricValue")
        grid.addWidget(lbl1, 0, 0)
        grid.addWidget(self.mode_val, 1, 0)

        # Altitude
        lbl2 = QLabel("ALTITUDE (AGL)")
        lbl2.setProperty("class", "MetricLabel")
        self.alt_val = QLabel("N/A")
        self.alt_val.setProperty("class", "MetricValue")
        grid.addWidget(lbl2, 0, 1)
        grid.addWidget(self.alt_val, 1, 1)

        # Battery
        lbl3 = QLabel("BATTERY")
        lbl3.setProperty("class", "MetricLabel")
        self.batt_val = QLabel("N/A")
        self.batt_val.setProperty("class", "MetricValue")
        grid.addWidget(lbl3, 2, 0)
        grid.addWidget(self.batt_val, 3, 0)

        # GPS Fix
        lbl4 = QLabel("GPS POSITION")
        lbl4.setProperty("class", "MetricLabel")
        self.gps_val = QLabel("N/A")
        self.gps_val.setProperty("class", "MetricValue")
        grid.addWidget(lbl4, 2, 1)
        grid.addWidget(self.gps_val, 3, 1)

    def update_telemetry(self, mode: str, alt_str: str, batt_str: str, gps_str: str):
        self.mode_val.setText(mode)
        self.alt_val.setText(alt_str)
        self.batt_val.setText(batt_str)
        self.gps_val.setText(gps_str)


class PerformancePanel(QGroupBox):
    """Displays pipeline diagnostics: stream FPS, inference FPS, latency, dropped frames."""

    def __init__(self, parent=None):
        super().__init__("PIPELINE PERFORMANCE", parent)
        self._init_ui()

    def _init_ui(self):
        grid = QGridLayout(self)
        grid.setSpacing(10)
        grid.setContentsMargins(12, 16, 12, 12)

        lbl1 = QLabel("INPUT FPS")
        lbl1.setProperty("class", "MetricLabel")
        self.input_fps_val = QLabel("0.0")
        self.input_fps_val.setProperty("class", "MetricValue")
        grid.addWidget(lbl1, 0, 0)
        grid.addWidget(self.input_fps_val, 1, 0)

        lbl2 = QLabel("EST. LATENCY")
        lbl2.setProperty("class", "MetricLabel")
        self.latency_val = QLabel("0 ms")
        self.latency_val.setProperty("class", "MetricValue")
        grid.addWidget(lbl2, 0, 1)
        grid.addWidget(self.latency_val, 1, 1)

        lbl3 = QLabel("DROPPED FRAMES")
        lbl3.setProperty("class", "MetricLabel")
        self.dropped_val = QLabel("0")
        self.dropped_val.setProperty("class", "MetricValue")
        grid.addWidget(lbl3, 2, 0)
        grid.addWidget(self.dropped_val, 3, 0)

        lbl4 = QLabel("HARDWARE ACCEL")
        lbl4.setProperty("class", "MetricLabel")
        self.device_val = QLabel("N/A")
        self.device_val.setStyleSheet("font-size: 13px; font-weight: bold; color: #63B3ED;")
        grid.addWidget(lbl4, 2, 1)
        grid.addWidget(self.device_val, 3, 1)

    def update_performance(self, input_fps: float, latency_ms: float, dropped: int, device_str: str):
        self.input_fps_val.setText(f"{input_fps:.1f}")
        self.latency_val.setText(f"{latency_ms:.0f} ms")
        self.dropped_val.setText(str(dropped))
        self.device_val.setText(device_str)


class ControlsPanel(QGroupBox):
    """Operator buttons: detection, video source, recording, snapshots, settings."""

    def __init__(
        self,
        on_start_det: Callable[[], None],
        on_stop_det: Callable[[], None],
        on_open_file: Callable[[], None],
        on_connect_cam: Callable[[], None],
        on_disconnect_cam: Callable[[], None],
        on_record_toggle: Callable[[], None],
        on_snapshot: Callable[[], None],
        on_toggle_overlay: Callable[[], None],
        on_open_settings: Callable[[], None],
        parent=None,
    ):
        super().__init__("OPERATOR CONTROLS", parent)
        self._init_ui(
            on_start_det, on_stop_det, on_open_file, on_connect_cam,
            on_disconnect_cam, on_record_toggle, on_snapshot,
            on_toggle_overlay, on_open_settings
        )

    def _init_ui(self, on_start, on_stop, on_open, on_conn, on_disconn, on_rec, on_snap, on_over, on_sett):
        layout = QGridLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(10, 14, 10, 10)

        # Video source controls
        self.btn_open_file = QPushButton("Open Video File")
        self.btn_open_file.clicked.connect(on_open)
        layout.addWidget(self.btn_open_file, 0, 0)

        self.btn_connect = QPushButton("Connect RTSP")
        self.btn_connect.clicked.connect(on_conn)
        layout.addWidget(self.btn_connect, 0, 1)

        self.btn_disconnect = QPushButton("Disconnect")
        self.btn_disconnect.clicked.connect(on_disconn)
        layout.addWidget(self.btn_disconnect, 0, 2)

        # Detection controls
        self.btn_start_det = QPushButton("Start Detection")
        self.btn_start_det.setProperty("class", "SuccessButton")
        self.btn_start_det.clicked.connect(on_start)
        layout.addWidget(self.btn_start_det, 1, 0)

        self.btn_stop_det = QPushButton("Stop Detection")
        self.btn_stop_det.setProperty("class", "DangerButton")
        self.btn_stop_det.clicked.connect(on_stop)
        layout.addWidget(self.btn_stop_det, 1, 1)

        self.btn_overlay = QPushButton("Toggle Overlay")
        self.btn_overlay.clicked.connect(on_over)
        layout.addWidget(self.btn_overlay, 1, 2)

        # Recording & Snapshots
        self.btn_record = QPushButton("Record Video")
        self.btn_record.clicked.connect(on_rec)
        layout.addWidget(self.btn_record, 2, 0)

        self.btn_snapshot = QPushButton("Save Snapshot")
        self.btn_snapshot.clicked.connect(on_snap)
        layout.addWidget(self.btn_snapshot, 2, 1)

        self.btn_settings = QPushButton("Settings")
        self.btn_settings.clicked.connect(on_sett)
        layout.addWidget(self.btn_settings, 2, 2)


class EventLogPanel(QGroupBox):
    """Audit event log table with CSV export."""

    def __init__(self, on_export_csv: Callable[[], None], parent=None):
        super().__init__("SYSTEM & DETECTION AUDIT LOG", parent)
        self._init_ui(on_export_csv)

    def _init_ui(self, on_export_csv):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 14, 8, 8)
        layout.setSpacing(6)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Timestamp", "Level", "Category", "Message"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        layout.addWidget(self.table)

        btn_box = QHBoxLayout()
        btn_box.addStretch()
        self.btn_export = QPushButton("Export Log (CSV)")
        self.btn_export.clicked.connect(on_export_csv)
        btn_box.addWidget(self.btn_export)
        layout.addLayout(btn_box)

    def add_entry(self, timestamp: str, level: str, category: str, message: str):
        row = self.table.rowCount()
        self.table.insertRow(row)

        item_ts = QTableWidgetItem(timestamp)
        item_lvl = QTableWidgetItem(level)
        item_cat = QTableWidgetItem(category)
        item_msg = QTableWidgetItem(message)

        # Level coloring
        if level == "ALERT":
            item_lvl.setForeground(QColor("#48BB78"))
        elif level == "ERROR":
            item_lvl.setForeground(QColor("#F56565"))
        elif level == "WARNING":
            item_lvl.setForeground(QColor("#ECC94B"))

        self.table.setItem(row, 0, item_ts)
        self.table.setItem(row, 1, item_lvl)
        self.table.setItem(row, 2, item_cat)
        self.table.setItem(row, 3, item_msg)
        self.table.scrollToBottom()

