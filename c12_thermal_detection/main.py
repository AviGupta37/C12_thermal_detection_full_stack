"""
Main Application Entry Point for C12 Aerial Thermal Human Detection.
Initializes PySide6 application, parses configuration arguments, and launches the operator dashboard.
Supports --headless smoke test mode for headless CI/CD environments.
"""

import argparse
from pathlib import Path
import sys
import yaml

# Ensure project root is in python search path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="C12 Aerial Thermal Human Detection Ground Station")
    parser.add_argument("--config", type=str, default="config/app.yaml", help="Path to app configuration YAML")
    parser.add_argument("--video", type=str, default=None, help="Prerecorded video file to load immediately")
    parser.add_argument("--rtsp", type=str, default=None, help="RTSP stream URL to connect immediately")
    parser.add_argument("--model", type=str, default=None, help="YOLO checkpoint path")
    parser.add_argument("--headless", action="store_true", help="Run headless smoke test without launching GUI")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load configuration
    cfg_path = PROJECT_ROOT / args.config
    cfg = {}
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    print("=" * 60)
    print(" C12 THERMAL HUMAN DETECTION GROUND STATION")
    print("=" * 60)
    print(f"[*] Configuration : {cfg_path}")
    print(f"[*] Headless Mode : {args.headless}")
    print("=" * 60)

    if args.headless:
        print("[+] Executing headless verification smoke test...")
        from detection.yolo_detector import YOLODetector
        from flight.mock_telemetry import MockTelemetryProvider
        from logging_tools.event_logger import EventLogger
        import numpy as np

        logger = EventLogger(log_dir=str(PROJECT_ROOT / "logs"))
        logger.info("HEADLESS", "Running headless verification test.")
        
        telem = MockTelemetryProvider()
        telem.start()
        data = telem.get_latest_telemetry()
        print(f"[*] Telemetry test: Mode={data.flight_mode}, Alt={data.altitude_str}, Batt={data.battery_str}")
        telem.stop()

        dummy_frame = np.zeros((288, 384, 3), dtype=np.uint8)
        detector = YOLODetector(model_path=args.model or "models/yolo11n.pt")
        res = detector.predict(dummy_frame)
        print(f"[*] Detector test : Device={detector.device_info}, Loaded={detector.is_loaded}, Detections={res.person_count}")
        print("[OK] Headless verification passed successfully!")
        return 0

    # Launch GUI
    try:
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import Qt
    except ImportError:
        print("[ERROR] PySide6 is required to launch the Ground Station UI.", file=sys.stderr)
        print("Please install via: pip install PySide6", file=sys.stderr)
        sys.exit(1)

    app = QApplication(sys.argv)
    app.setApplicationName("C12 Thermal Detection")
    app.setOrganizationName("Defense Aerial CV")

    from ui.main_window import MainWindow

    window = MainWindow(app_config=cfg)

    # Apply direct CLI source overrides
    if args.model:
        window.detector.load_model(args.model)
        window.settings["model_path"] = args.model
    if args.video:
        from video.video_file_source import VideoFileSource
        source = VideoFileSource(file_path=args.video, loop=True)
        if source.start():
            window.video_source = source
            window.header_bar.set_camera_status(True, Path(args.video).name)
    elif args.rtsp:
        from video.rtsp_source import RTSPSource
        source = RTSPSource(rtsp_url=args.rtsp)
        source.start()
        window.video_source = source
        window.header_bar.set_camera_status(True, "RTSP")

    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
