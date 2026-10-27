"""
Operator Settings Dialog for C12 Thermal Human Detection Ground Station.
Allows updating stream URLs, YOLO checkpoints, confidence thresholds, and telemetry options.
"""

from pathlib import Path
from typing import Any, Dict

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLineEdit,
    QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QPushButton,
    QFileDialog, QGroupBox, QLabel
)


class SettingsDialog(QDialog):
    """Configuration modal for runtime parameters."""

    def __init__(self, current_settings: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ground Station Settings & Configuration")
        self.setMinimumWidth(520)
        self.settings = dict(current_settings)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(14)

        # 1. Camera / Stream Group
        grp_stream = QGroupBox("Camera Stream Configuration")
        form_stream = QFormLayout(grp_stream)

        self.rtsp_url_edit = QLineEdit(self.settings.get("rtsp_url", "rtsp://192.168.144.108:555/stream=2"))
        form_stream.addRow("RTSP Stream URI:", self.rtsp_url_edit)

        self.transport_combo = QComboBox()
        self.transport_combo.addItems(["TCP", "UDP"])
        self.transport_combo.setCurrentText(self.settings.get("rtsp_transport", "TCP").upper())
        form_stream.addRow("Transport Protocol:", self.transport_combo)
        layout.addWidget(grp_stream)

        # 2. YOLO Model Group
        grp_yolo = QGroupBox("YOLO Thermal Model Configuration")
        form_yolo = QFormLayout(grp_yolo)

        model_row = QHBoxLayout()
        self.model_path_edit = QLineEdit(self.settings.get("model_path", "models/yolo11n.pt"))
        btn_browse_model = QPushButton("Browse...")
        btn_browse_model.clicked.connect(self._browse_model)
        model_row.addWidget(self.model_path_edit)
        model_row.addWidget(btn_browse_model)
        form_yolo.addRow("Model Weights (.pt):", model_row)

        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.05, 0.99)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(float(self.settings.get("conf_threshold", 0.35)))
        form_yolo.addRow("Confidence Threshold:", self.conf_spin)

        self.imgsz_combo = QComboBox()
        self.imgsz_combo.addItems(["640", "384"])
        self.imgsz_combo.setCurrentText(str(self.settings.get("imgsz", 640)))
        form_yolo.addRow("Inference ImgSz:", self.imgsz_combo)

        self.clahe_check = QCheckBox("Enable CLAHE Contrast Enhancement (LWIR)")
        self.clahe_check.setChecked(bool(self.settings.get("enable_clahe", False)))
        form_yolo.addRow("", self.clahe_check)
        layout.addWidget(grp_yolo)

        # 3. Telemetry Group
        grp_telem = QGroupBox("PX4 Telemetry Configuration")
        form_telem = QFormLayout(grp_telem)

        self.telem_type_combo = QComboBox()
        self.telem_type_combo.addItems(["Simulated (Development Mock)", "MAVLink UDP"])
        if not self.settings.get("use_mock_telemetry", True):
            self.telem_type_combo.setCurrentIndex(1)
        form_telem.addRow("Telemetry Provider:", self.telem_type_combo)

        self.mavlink_edit = QLineEdit(self.settings.get("mavlink_conn", "udp:127.0.0.1:14550"))
        form_telem.addRow("MAVLink Endpoint:", self.mavlink_edit)
        layout.addWidget(grp_telem)

        # Bottom buttons
        btn_box = QHBoxLayout()
        btn_box.addStretch()

        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_box.addWidget(btn_cancel)

        btn_save = QPushButton("Save Settings")
        btn_save.setProperty("class", "PrimaryButton")
        btn_save.clicked.connect(self._save_settings)
        btn_box.addWidget(btn_save)

        layout.addLayout(btn_box)

    def _browse_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select YOLO Model Checkpoint",
            "models",
            "YOLO Models (*.pt *.onnx);;All Files (*)",
        )
        if file_path:
            self.model_path_edit.setText(file_path)

    def _save_settings(self):
        self.settings["rtsp_url"] = self.rtsp_url_edit.text().strip()
        self.settings["rtsp_transport"] = self.transport_combo.currentText().lower()
        self.settings["model_path"] = self.model_path_edit.text().strip()
        self.settings["conf_threshold"] = self.conf_spin.value()
        self.settings["imgsz"] = int(self.imgsz_combo.currentText())
        self.settings["enable_clahe"] = self.clahe_check.isChecked()
        self.settings["use_mock_telemetry"] = (self.telem_type_combo.currentIndex() == 0)
        self.settings["mavlink_conn"] = self.mavlink_edit.text().strip()
        self.accept()

