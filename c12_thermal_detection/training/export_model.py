"""
Model export utility for Skydroid C12 Ground Station.
Exports trained PyTorch YOLO checkpoints to ONNX format for deployment.
"""

import argparse
from pathlib import Path
import sys
import shutil

# Ensure package root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(description="Export YOLO model to ONNX format")
    parser.add_argument("--model", type=str, required=True, help="Path to PyTorch .pt model")
    parser.add_argument("--format", type=str, default="onnx", choices=["onnx", "torchscript", "engine"], help="Export target format")
    parser.add_argument("--imgsz", type=int, default=640, help="Export input resolution")
    parser.add_argument("--half", action="store_true", help="Enable FP16 half precision")
    parser.add_argument("--output-dir", type=str, default="outputs/exported_models", help="Export target directory")
    return parser.parse_args()


def run_export():
    args = parse_args()
    model_path = Path(args.model)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        print(f"[ERROR] Model file not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print(" C12 THERMAL HUMAN DETECTION - MODEL EXPORT")
    print("=" * 60)
    print(f"[*] Input Model    : {model_path.resolve()}")
    print(f"[*] Target Format  : {args.format}")
    print(f"[*] Target ImgSz   : {args.imgsz}")
    print(f"[*] FP16 Precision : {args.half}")
    print("=" * 60)

    from ultralytics import YOLO

    try:
        model = YOLO(str(model_path))
        exported_path_str = model.export(
            format=args.format,
            imgsz=args.imgsz,
            half=args.half,
            dynamic=False,
            simplify=True if args.format == "onnx" else False,
        )
        
        exported_path = Path(exported_path_str)
        if exported_path.exists():
            target_dest = out_dir / exported_path.name
            shutil.copy2(exported_path, target_dest)
            print(f"\n[OK] Export complete!")
            print(f"[*] Exported Artifact: {target_dest.resolve()}")
        else:
            print(f"[ERROR] Expected exported file at {exported_path} but not found.", file=sys.stderr)
            sys.exit(1)
            
    except Exception as e:
        print(f"\n[EXPORT ERROR] Export failed: {e}", file=sys.stderr)
        if args.format == "onnx":
            print("Ensure 'onnx' and 'onnxslim' are installed via 'pip install onnx onnxslim'.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    run_export()
