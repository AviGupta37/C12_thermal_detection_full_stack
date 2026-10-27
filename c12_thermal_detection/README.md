# Skydroid C12 Aerial Thermal Human Detection Ground Station

A modular, real-time desktop application and computer-vision training pipeline for aerial thermal human detection using the **Skydroid C12** dual thermal/visible gimbal camera mounted on a **PX4-powered quadrotor**, with AI inference executed on an **Ubuntu 22.04 Ground PC** powered by an **NVIDIA GeForce RTX 2080 Ti (11 GB VRAM)**.

---

## 1. System Overview & Scope

The application acquires live LWIR thermal video streams from the C12 camera (via RTSP over an Ethernet/IP datalink) or local prerecorded flight footage, executes YOLO-based human detection, and presents an operator dashboard built with PySide6/Qt. Flight telemetry (flight mode, altitude, battery percentage, GPS fix) is displayed passively via a read-only telemetry interface.

### Strict Scope Boundary
- **Included**: Real-time LWIR video decoding, YOLO human detection, high-contrast bounding box overlays, performance monitoring (stream FPS, inference ms, latency, dropped frames), passive PX4 telemetry monitoring, dataset preparation utilities, fine-tuning scripts, evaluation metrics, video recording, snapshot capture, and structured audit logging.
- **Strictly Excluded**:
  - Person tracking or persistent tracking IDs.
  - Trajectory trails.
  - Person GPS geolocation.
  - Autonomous following or targeting.
  - Modifying PX4 flight setpoints, firmware, or flight control modes.

---

## 2. End-to-End Architecture

```
                 +---------------------------------------------+
                 |            QUADROTOR SENSORS                |
                 | - Skydroid C12 Thermal Camera (384x288 LWIR)|
                 | - PX4 Flight Controller (MAVLink Telemetry) |
                 +----------------------+----------------------+
                                        |
                 +----------------------+----------------------+
                 |                                             |
           Thermal Video                                 MAVLink Telemetry
        (RTSP over IP Link)                            (Telemetry Radio / UDP)
                 |                                             |
                 v                                             v
     +-----------------------+                    +-------------------------+
     |   Video Input Engine  |                    | PX4 Telemetry Interface |
     | - RTSP Stream Decoder |                    | - Read-only listener    |
     | - Video File Source   |                    | - Mock Provider (Dev)   |
     +-----------+-----------+                    +------------+------------+
                 |                                             |
                 v                                             |
     +-----------------------+                                 |
     | Single-Slot Buffer    |                                 |
     | (Drops Stale Frames)  |                                 |
     +-----------+-----------+                                 |
                 |                                             |
                 v                                             |
     +-----------------------+                                 |
     | YOLO Inference Engine |                                 |
     | - RTX 2080 Ti (CUDA)  |                                 |
     | - Thermal Preprocess  |                                 |
     +-----------+-----------+                                 |
                 |                                             |
                 +----------------------+----------------------+
                                        |
                                        v
                    +---------------------------------------+
                    |    PySide6 Desktop Operator UI        |
                    | - Thermal Video Viewport & Overlays   |
                    | - Detection Counts & Confidences      |
                    | - Read-Only PX4 Telemetry Display     |
                    | - Performance & Latency Monitor       |
                    | - Controls & Structured Audit Log     |
                    +-------------------+-------------------+
                                        |
                    +-------------------+-------------------+
                    | Video Recording & Snapshot Storage    |
                    | - recordings/ (.mp4)                  |
                    | - snapshots/ (.png)                   |
                    | - logs/ (.log, .csv)                  |
                    +---------------------------------------+
```

---

## 3. Supported Operating Environment

- **Operating System**: Ubuntu 22.04 LTS (x86_64) / Cross-platform support for Windows development.
- **Compute**: Ground PC with NVIDIA GeForce RTX 2080 Ti (11 GB VRAM).
- **Python**: 3.10+ (tested on Python 3.10, 3.11, 3.12, 3.13).
- **Camera**: Skydroid C12 Thermal & Visible Gimbal (384 × 288 LWIR, 8–14 µm, 7 mm lens, ~18.7° vFOV).
- **Target Flight Altitude**: 20–50 metres AGL.

---

## 4. Installation & Environment Setup

### 4.1 System Prerequisites (Ubuntu 22.04)
Install required system packages for video decoding and OpenGL rendering:
```bash
sudo apt update
sudo apt install -y python3-venv python3-pip ffmpeg libgl1-mesa-glx libglib2.0-0
```

### 4.2 Automated Setup
Run the automated installation script:
```bash
chmod +x scripts/setup_ubuntu.sh
./scripts/setup_ubuntu.sh
```

### 4.3 Manual Virtual Environment Setup
```bash
cd c12_thermal_detection
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 4.4 GPU Verification
Verify that PyTorch detects the NVIDIA RTX 2080 Ti and CUDA build:
```bash
python scripts/check_gpu.py
```
Expected output:
```
[*] Compute Device : NVIDIA GeForce RTX 2080 Ti (11.0 GB VRAM)
[*] CUDA Available : True
[OK] GPU Environment verification completed.
```

---

## 5. Launching the Desktop UI

### Normal Launch
```bash
bash scripts/launch_ui.sh
# or directly:
python main.py
```

### Launch with Immediate Video File
```bash
python main.py --video /path/to/flight_test.mp4
```

### Launch with Live RTSP Stream
```bash
python main.py --rtsp rtsp://192.168.144.108:555/stream=2
```

### Headless Verification Mode (CI / Bench Testing)
```bash
python main.py --headless
```

---

## 6. Video Input Modes

### Mode A: Recorded Video File
1. Click **Open Video File** in the operator dashboard.
2. Select any local thermal video (`.mp4`, `.avi`, `.mkv`).
3. The video loops continuously at native frame rate, allowing offline model validation without live aircraft or camera hardware.

### Mode B: RTSP Camera Stream
1. Verify the ground PC network connection to the C12 receiver / IP bridge.
2. Example endpoint referenced in Skydroid C12 documentation:
   ```
   rtsp://192.168.144.108:555/stream=2
   ```
   *(Confirm endpoint and IP on your specific unit; stream 2 typically designates the thermal sensor while stream 1 is visible light).*
3. Click **Connect RTSP** to begin live streaming.

---

## 7. Dataset Preparation & Annotation

To fine-tune YOLO on your flight footage:

### 1. Extract Frames from Flight Videos
Sample frames at a defined rate or stride:
```bash
python dataset_tools/extract_frames.py \
  --video datasets/raw/flight_01.mp4 \
  --output-dir datasets/extracted_frames \
  --stride 15
```

### 2. Annotate Imagery
Use an annotation tool such as **CVAT** (https://www.cvat.ai/) or **Label Studio** (https://labelstud.io/):
- **Class ID**: `0: person`
- Standard YOLO format: `<class_id> <cx> <cy> <w> <h>` (normalized [0, 1]).
- Images with no people should have an empty `.txt` file to train false-positive resistance.

### 3. Split Dataset Without Data Leakage
Split by flight session prefix to ensure frames from the same video do not leak across training and test sets:
```bash
python dataset_tools/split_dataset.py \
  --source-images datasets/extracted_frames \
  --source-labels datasets/extracted_labels \
  --target-dataset datasets/yolo_dataset \
  --train-ratio 0.70 --val-ratio 0.15 --test-ratio 0.15 \
  --copy
```

### 4. Validate Dataset Integrity
Verify coordinates, missing files, corrupted images, and classes before training:
```bash
python dataset_tools/validate_dataset.py --data datasets/yolo_dataset/data.yaml
```

### 5. Inspect Annotations & Dataset Statistics
```bash
# Render visual bounding box previews
python dataset_tools/visualize_labels.py \
  --images datasets/yolo_dataset/images/train \
  --labels datasets/yolo_dataset/labels/train \
  --output-dir outputs/annotated_previews

# Analyze target size distributions across altitudes
python dataset_tools/dataset_statistics.py --data datasets/yolo_dataset/data.yaml
```

---

## 8. YOLO Fine-Tuning & Evaluation

### 8.1 Fine-Tuning on RTX 2080 Ti (11 GB)
Launch training with:
```bash
bash scripts/train_model.sh
# or customize parameters via CLI:
python training/train.py \
  --model models/yolo11n.pt \
  --data datasets/yolo_dataset/data.yaml \
  --epochs 100 \
  --batch 16 \
  --imgsz 640 \
  --device 0
```
*Note: If GPU out-of-memory occurs, lower `--batch` to 8 or 4.*

### 8.2 Evaluation on Held-Out Test Set
Calculate true Precision, Recall, mAP50, and mAP50-95:
```bash
python training/evaluate.py \
  --model runs/training/c12_thermal_baseline/weights/best.pt \
  --data datasets/yolo_dataset/data.yaml \
  --split test \
  --imgsz 640
```
Reports and metrics are saved to `outputs/metrics/`.

### 8.3 Model Export (ONNX)
Export the trained model for optimized deployment:
```bash
python training/export_model.py \
  --model runs/training/c12_thermal_baseline/weights/best.pt \
  --format onnx \
  --imgsz 640
```

---

## 9. Running Unit Tests

Run the comprehensive unit test suite:
```bash
python -m pytest c12_thermal_detection/tests/ -v
```
All unit tests execute without requiring physical camera hardware, flight controllers, or GPU weights.

---

## 10. Troubleshooting & Known Limitations

| Symptom | Cause | Solution |
| :--- | :--- | :--- |
| **RTSP stream does not open** | Network subnet mismatch or incorrect camera IP | Check ping to camera IP (e.g. `192.168.144.108`), verify RTSP port and stream number (`stream=2` for thermal). |
| **Visible stream opens instead of thermal** | Selected wrong stream index | Switch stream URI to the thermal endpoint. |
| **CUDA out of memory (OOM)** | Batch size too large for 11 GB VRAM | Reduce batch size to 8 or 4 in `config/training.yaml` (`--batch 8`). |
| **Small target misses at 50m** | Person occupies only ~9-11 pixels vertically | Use higher input resolution (`imgsz=640`), adjust confidence threshold to 0.25-0.30, and fine-tune on high-altitude samples. |
| **False positives on sun-heated ground** | Thermal clutter in daytime open fields | Add hard negative samples (unannotated background images) to training set; toggle CLAHE off if contrast is excessive. |
| **Telemetry disconnected** | MAVLink radio link inactive | The application defaults to simulated telemetry for UI testing. Telemetry does not affect video detection. |

### Fundamental Limitations
- Thermal imaging cannot penetrate dense forest canopy or opaque solid obstacles.
- Pixels-on-target decrease sharply from ~22 px at 20 m to ~9 px at 50 m. Empirical test flights at each target altitude are mandatory.

