# Implementation Plan: C12 Aerial Thermal Human Detection System

## 1. System Overview & Context

This system is a modular ground-station desktop application and machine learning pipeline for aerial thermal human detection using a Skydroid C12 camera (384x288 LWIR, 8–14 µm, 7mm lens, ~18.7° vFOV) mounted on a quadrotor running PX4 Autopilot. 
Inference runs on a Ground PC (equipped with Ubuntu 22.04 and an NVIDIA RTX 2080 Ti with 11 GB VRAM).

### Strict Project Scope
- **Included**: Real-time LWIR thermal frame acquisition (RTSP stream or recorded video playback), YOLO-based human detection, PySide6 dark-themed operator dashboard, detection overlays, performance monitoring (FPS, inference ms, latency, dropped frames), optional PX4 telemetry display (read-only), dataset preparation tools, fine-tuning scripts, evaluation metrics, video recording, snapshot capture, and event logging.
- **Strictly Excluded**:
  - Target tracking / persistent target IDs.
  - Target trajectory trails.
  - Target geolocation / geo-estimation.
  - Autonomous following or targeting.
  - Detection-driven flight control or modifying PX4 flight setpoints/firmware.

---

## 2. Proposed Architecture

```
                  +-----------------------------------+
                  |    AERIAL VEHICLE & SENSORS       |
                  |  - Skydroid C12 Thermal Camera    |
                  |  - PX4 Autopilot (MAVLink Telemetry|
                  +-----------------+-----------------+
                                    |
          +-------------------------+-------------------------+
          | Thermal Video (RTSP/IP)                           | MAVLink Radio / UDP
          v                                                   v
+-----------------------------+             +----------------------------------+
|      VIDEO INPUT ENGINE     |             |      PX4 TELEMETRY INTERFACE     |
| - RTSP Stream Decoder       |             | - Read-only MAVLink / UDP        |
| - Prerecorded File Source   |             | - Mock Telemetry Provider        |
| - Stale Frame Dropper       |             | - State: Mode, Alt, Batt, Fix    |
+--------------+--------------+             +-----------------+----------------+
               |                                              |
               v (Latest Frame)                               |
+-----------------------------+                               |
|   SINGLE-SLOT FRAME BUFFER  |                               |
+--------------+--------------+                               |
               |                                              |
               v                                              |
+-----------------------------+                               |
|   YOLO INFERENCE WORKER     |                               |
| - Preprocessing (LWIR/CLAHE)|                               |
| - PyTorch CUDA / CPU        |                               |
| - Confidence Thresholding   |                               |
| - Metrics & Timing (ms)     |                               |
+--------------+--------------+                               |
               |                                              |
               +----------------------+-----------------------+
                                      |
                                      v (Signals / Thread-safe Events)
                      +-------------------------------+
                      |   OPERATOR DESKTOP UI         |
                      |   (PySide6 / Qt 6)            |
                      | - Main Thermal Video Display  |
                      | - Detection Bounding Boxes    |
                      | - Detection Statistics Panel  |
                      | - Flight Telemetry Panel      |
                      | - Performance & Latency Panel |
                      | - Event & Audit Log View      |
                      | - Controls & Settings Dialog  |
                      +---------------+---------------+
                                      |
                      +---------------+---------------+
                      | RECORDING & AUDIT LOGGING     |
                      | - Video Recorder (.mp4/.avi)  |
                      | - Snapshots (.png / .jpg)     |
                      | - Event Logger (CSV / JSON)   |
                      | - Performance Logger (CSV)    |
                      +-------------------------------+
```

---

## 3. Detailed Component Breakdown

### 3.1 Video Subsystem (`video/`)
- `CameraSource` (Abstract Base Class): Generic thread interface for video sources.
- `RTSPSource`: Threaded OpenCV/FFmpeg stream consumer with connection timeouts, reconnect logic, TCP transport flag, and error notifications.
- `VideoFileSource`: Local prerecorded video reader with looping, seek, and native playback pacing.
- `FrameBuffer`: Single-slot, non-blocking / bounded buffer that replaces older frames with newly arrived frames, completely preventing unbounded queue memory growth and frame latency drift.

### 3.2 Detection & Inference Subsystem (`detection/`)
- `YOLODetector`: Wraps Ultralytics YOLO (`yolo11n.pt` baseline or custom checkpoint). Handles CUDA GPU allocation (with graceful CPU fallback), batch-1 real-time inference, confidence filtering, and non-crashing exception handling.
- `ThermalPreprocessor`: Optional LWIR contrast enhancement (adaptive histogram equalization, normalization) with pass-through mode to ensure consistency across training and live inference.
- `DetectionResult`: Dataclass encapsulating detected bounding boxes (`xyxy`, `xywhn`), confidences, class labels (`person`), inference latency (preprocess, inference, postprocess), timestamp, and frame resolution.

### 3.3 Training & Dataset Pipeline (`training/` and `dataset_tools/`)
- `train.py`: Fine-tunes YOLO models on custom thermal datasets with configurable hyper-parameters (epochs, batch size, image size 640/384, workers, patience, device).
- `evaluate.py`: Computes test-set validation metrics (Precision, Recall, mAP50, mAP50-95, latency breakdown). Supports environment/altitude metadata analysis if available.
- `export_model.py`: Exports trained weights to ONNX format.
- `extract_frames.py`: Samples frames from thermal flight videos at configurable frame rates or stride intervals.
- `validate_dataset.py`: Verifies YOLO annotation format, class IDs (0 = person), normalized bounds [0, 1], non-zero sizes, and missing/corrupt image-label pairs.
- `split_dataset.py`: Splits datasets by video recording / flight session to prevent data leakage from consecutive frames.
- `visualize_labels.py`: Generates annotated image previews for manual inspection.
- `dataset_statistics.py`: Analyzes label counts, bounding box sizes, empty frame ratios, and split distributions.

### 3.4 Telemetry Interface (`flight/`)
- `TelemetryInterface`: Read-only telemetry abstraction providing flight mode (e.g. `OFFBOARD`, `POSCTL`, `MANUAL`), altitude (m AGL), battery percentage, GPS fix, and timestamp.
- `MockTelemetryProvider`: Simulates realistic quadrotor telemetry (altitude 20–50 m, varying battery, GPS lock) for UI testing without a physical aircraft.
- `MAVLinkTelemetryProvider`: Optional UDP/Serial listener using `pymavlink` or MAVSDK when hardware is present, isolated from video processing.

### 3.5 Desktop Ground-Station UI (`ui/`)
- `MainWindow`: Main PySide6 dashboard (1920x1080 target, responsive layout).
- `HeaderBar`: Connection status chips (Camera, Detector, GPU, PX4, Clock).
- `ThermalView`: High-performance viewport rendering thermal frames with aspect-ratio preservation, scalable vector bounding box overlays, confidence labels, and error placeholders.
- `DashboardPanels`:
  - `DetectionPanel`: Person count, highest confidence, inference time, detector status.
  - `TelemetryPanel`: PX4 flight mode, altitude, battery gauge, GPS fix status.
  - `PerformancePanel`: Stream FPS, Inference FPS, end-to-end latency, dropped frames.
  - `ControlsPanel`: Start/Stop detection, Open file, Connect/Disconnect camera, Snapshot, Record, Model selector, Overlay toggle, Settings.
  - `EventLogView`: Real-time timestamped audit log with CSV export.
- `SettingsDialog`: Configures RTSP URL, confidence threshold, IOU threshold, model path, recording directory, and telemetry options.

### 3.6 Logging & Storage (`logging_tools/`)
- `EventLogger`: Structured JSON/CSV logging for system events and detection alerts.
- `PerformanceLogger`: Time-series logging of FPS, latencies, and dropped frames.
- `VideoRecorder`: Background video writer (OpenCV `VideoWriter`) saving MP4 footage.

---

## 4. Dependencies & Operating Environment

- **Host OS**: Ubuntu 22.04 LTS (Ground PC) / Cross-platform support for local development.
- **Python**: 3.10+ (Current environment: Python 3.13)
- **Core Libraries**:
  - `PySide6`: Modern desktop GUI framework.
  - `ultralytics` (>= 8.0.0): YOLO object detection and training.
  - `torch`, `torchvision`: Deep learning runtime with CUDA acceleration.
  - `opencv-python` (>= 4.8.0): Frame decoding, rendering, and video capture.
  - `pyyaml`: Configuration management.
  - `numpy`: Numerical processing and tensor transformations.
  - `pytest`: Comprehensive unit test framework.

---

## 5. Directory Layout & Expected Files

```
c12_thermal_detection/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── .env.example
├── main.py
│
├── config/
│   ├── app.yaml
│   ├── detection.yaml
│   ├── training.yaml
│   └── camera.yaml
│
├── ui/
│   ├── __init__.py
│   ├── main_window.py
│   ├── thermal_view.py
│   ├── dashboard_panels.py
│   ├── settings_dialog.py
│   └── styles.py
│
├── video/
│   ├── __init__.py
│   ├── camera_source.py
│   ├── rtsp_source.py
│   ├── video_file_source.py
│   └── frame_buffer.py
│
├── detection/
│   ├── __init__.py
│   ├── yolo_detector.py
│   ├── preprocessing.py
│   └── results.py
│
├── dataset_tools/
│   ├── __init__.py
│   ├── extract_frames.py
│   ├── validate_dataset.py
│   ├── split_dataset.py
│   ├── visualize_labels.py
│   └── dataset_statistics.py
│
├── training/
│   ├── __init__.py
│   ├── train.py
│   ├── evaluate.py
│   ├── export_model.py
│   └── training_utils.py
│
├── flight/
│   ├── __init__.py
│   ├── telemetry_interface.py
│   └── mock_telemetry.py
│
├── logging_tools/
│   ├── __init__.py
│   ├── event_logger.py
│   └── performance_logger.py
│
├── scripts/
│   ├── setup_ubuntu.sh
│   ├── check_gpu.py
│   ├── launch_ui.sh
│   ├── train_model.sh
│   └── validate_model.sh
│
├── tests/
│   ├── __init__.py
│   ├── test_dataset_validation.py
│   ├── test_preprocessing.py
│   ├── test_detection_results.py
│   ├── test_video_buffer.py
│   └── test_config.py
│
├── models/
│   └── README.md
│
├── datasets/
│   ├── README.md
│   ├── raw/
│   ├── extracted_frames/
│   └── yolo_dataset/
│       ├── images/ {train, val, test}
│       └── labels/ {train, val, test}
│
├── recordings/
├── snapshots/
├── logs/
├── runs/
└── outputs/
```

---

## 6. Testing Strategy

1. **Unit Testing (`pytest`)**:
   - `test_dataset_validation.py`: Tests YOLO label syntax validation, invalid coordinates (> 1.0 or < 0.0), corrupted files, empty files, and mismatched pairs.
   - `test_preprocessing.py`: Tests LWIR image normalization, CLAHE application, shape preservation, and color mapping.
   - `test_detection_results.py`: Tests `DetectionResult` parsing, filtering by confidence, bounding box coordinate transformations, and serialization.
   - `test_video_buffer.py`: Tests single-slot queue semantics under rapid writes and slower reads to verify zero-lag frame replacement.
   - `test_config.py`: Tests YAML configuration loading, fallback defaults, and validation.
2. **Integration Testing**:
   - Synthetic video generation and detector loop verification.
   - UI worker thread lifecycle and graceful shutdown tests.
   - Mock telemetry data feed rate and status chip updates.

---

## 7. Hardware-Dependent Tasks (Requiring Physical C12 & Drone)

The following tasks cannot be completely verified without physical hardware on site:
1. **Actual C12 RTSP Endpoint & Authentication**: Confirming whether the unit streams over `rtsp://192.168.144.108:555/stream=2` or a different URI/port exposed by the Skydroid ground receiver / datalink.
2. **Stream Encoding & Dynamic Range**: Validating whether the camera output is 8-bit display-enhanced LWIR (e.g. White-Hot / Black-Hot) or 14-bit raw radiometric data.
3. **Physical Altitude Detection Baseline (20–50 m)**: Collecting actual flight footage over test fields to establish empirical recall and precision at varying nadir / gimbal angles.
4. **Live MAVLink Radio Serial / UDP Baudrate**: Matching the ground station telemetry serial port or UDP endpoint to the vehicle's telemetry transceiver.

