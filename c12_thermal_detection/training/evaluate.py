"""
Evaluation script for YOLO Thermal Human Detection models.
Calculates mAP50, mAP50-95, precision, recall, and inference speed.
Saves evaluated metrics and confusion matrix data to outputs/metrics.
"""

import argparse
import json
from pathlib import Path
import sys
import time
import yaml
import torch

# Ensure package root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.training_utils import check_cuda_environment, load_yaml_config


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained YOLO model on thermal test set")
    parser.add_argument("--model", type=str, required=True, help="Path to checkpoint (e.g. runs/training/c12_thermal/weights/best.pt)")
    parser.add_argument("--data", type=str, default="datasets/yolo_dataset/data.yaml", help="Path to data.yaml")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--device", type=str, default=None, help="Device index ('0' or 'cpu')")
    parser.add_argument("--batch", type=int, default=16, help="Batch size for validation")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for NMS")
    parser.add_argument("--output-dir", type=str, default="outputs/metrics", help="Directory to save evaluation reports")
    return parser.parse_args()


def run_evaluation():
    args = parse_args()
    model_path = Path(args.model)
    data_path = Path(args.data)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(" C12 THERMAL HUMAN DETECTION - MODEL EVALUATION")
    print("=" * 60)

    # 1. Check Model Checkpoint
    if not model_path.exists():
        print(f"[ERROR] Checkpoint does not exist: {model_path}", file=sys.stderr)
        sys.exit(1)

    # 2. Check Data YAML
    if not data_path.exists():
        print(f"[ERROR] Dataset configuration does not exist: {data_path}", file=sys.stderr)
        sys.exit(1)

    # 3. Resolve Device
    has_cuda, cuda_desc, _ = check_cuda_environment()
    device = args.device
    if device is None:
        device = "0" if has_cuda else "cpu"
    elif "cpu" not in device.lower() and not has_cuda:
        print("[!] Warning: CUDA requested but unavailable. Falling back to CPU.")
        device = "cpu"

    print(f"[*] Checkpoint     : {model_path.resolve()}")
    print(f"[*] Dataset Config : {data_path.resolve()} (Split: {args.split})")
    print(f"[*] Compute Device : {device} ({cuda_desc})")
    print(f"[*] Input ImgSz    : {args.imgsz}")
    print("=" * 60)

    # 4. Load YOLO Model and Evaluate
    from ultralytics import YOLO
    try:
        model = YOLO(str(model_path))
    except Exception as e:
        print(f"[ERROR] Failed to instantiate YOLO model: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\n[+] Executing validation on '{args.split}' split...")
    t_start = time.time()
    try:
        metrics = model.val(
            data=str(data_path.resolve()),
            split=args.split,
            imgsz=args.imgsz,
            device=device,
            batch=args.batch,
            conf=args.conf,
            iou=args.iou,
            save_json=True,
            plots=True,
        )
    except Exception as e:
        print(f"\n[EVALUATION ERROR] Model evaluation failed: {e}", file=sys.stderr)
        sys.exit(1)
        
    duration = time.time() - t_start

    # 5. Extract and Report Ground-Truth Metrics
    # Ultralytics metrics objects provide box.mp (mean precision), box.mr (mean recall), box.map50, box.map
    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)
    speed_dict = metrics.speed  # preprocess, inference, loss, postprocess ms

    print("\n" + "=" * 60)
    print(" EVALUATION RESULTS (GROUND-TRUTH)")
    print("=" * 60)
    print(f"  Precision (P)   : {precision:.4f} ({precision*100:.1f}%)")
    print(f"  Recall (R)      : {recall:.4f} ({recall*100:.1f}%)")
    print(f"  mAP @ 0.50      : {map50:.4f}")
    print(f"  mAP @ 0.50-0.95 : {map50_95:.4f}")
    if speed_dict:
        inf_ms = speed_dict.get("inference", 0.0)
        fps = (1000.0 / inf_ms) if inf_ms > 0 else 0.0
        print(f"  Inference Latency: {inf_ms:.2f} ms ({fps:.1f} FPS)")
        print(f"  Speed Breakdown : Preprocess: {speed_dict.get('preprocess', 0):.2f}ms | Inference: {inf_ms:.2f}ms | Postprocess: {speed_dict.get('postprocess', 0):.2f}ms")
    print("=" * 60)

    # 6. Save JSON report
    report_filename = f"eval_{model_path.stem}_{args.split}_{int(time.time())}.json"
    report_path = out_dir / report_filename
    report_data = {
        "model": str(model_path),
        "split": args.split,
        "dataset_yaml": str(data_path),
        "imgsz": args.imgsz,
        "device": device,
        "precision": precision,
        "recall": recall,
        "map50": map50,
        "map50_95": map50_95,
        "speed": speed_dict,
        "evaluation_duration_sec": round(duration, 2),
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"[*] Evaluation report saved to: {report_path.resolve()}")


if __name__ == "__main__":
    run_evaluation()

