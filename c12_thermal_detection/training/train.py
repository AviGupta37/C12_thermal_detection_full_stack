"""
Fine-tuning script for YOLO on Skydroid C12 Thermal Human Detection Dataset.
Features:
- Configurable via YAML or CLI arguments
- CUDA device check & graceful fallback
- Out-of-memory exception handling
- Pre-training validation (prevents empty dataset training or test leakage)
- Checkpoint, log, and configuration persistence
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time
import yaml
import torch

# Ensure package root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.training_utils import check_cuda_environment, load_yaml_config, validate_training_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune YOLO model on thermal human dataset")
    parser.add_argument("--config", type=str, default="config/training.yaml", help="Path to training YAML config")
    parser.add_argument("--model", type=str, default=None, help="Base pretrained weights (e.g. yolo11n.pt)")
    parser.add_argument("--data", type=str, default=None, help="Path to dataset data.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Total training epochs")
    parser.add_argument("--batch", type=int, default=None, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=None, help="Image resolution (640 or 384)")
    parser.add_argument("--device", type=str, default=None, help="Device index (0 or 'cpu')")
    parser.add_argument("--workers", type=int, default=None, help="DataLoader workers")
    parser.add_argument("--name", type=str, default=None, help="Experiment run name")
    return parser.parse_args()


def run_training():
    args = parse_args()
    config_path = Path(args.config)
    
    # 1. Load YAML configuration
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)
        
    cfg = load_yaml_config(config_path)
    train_cfg = cfg.get("training", {})
    hyp_cfg = cfg.get("hyperparameters", {})
    aug_cfg = cfg.get("augmentation", {})

    # 2. Apply CLI overrides
    model_checkpoint = args.model or train_cfg.get("model_checkpoint", "models/yolo11n.pt")
    data_yaml = args.data or train_cfg.get("data_yaml", "datasets/yolo_dataset/data.yaml")
    epochs = args.epochs or train_cfg.get("epochs", 100)
    batch_size = args.batch or train_cfg.get("batch_size", 16)
    imgsz = args.imgsz or train_cfg.get("imgsz", 640)
    device = args.device or str(train_cfg.get("device", "0"))
    workers = args.workers or train_cfg.get("workers", 8)
    patience = train_cfg.get("patience", 20)
    project_dir = train_cfg.get("project", "runs/training")
    run_name = args.name or train_cfg.get("name", f"c12_thermal_{int(time.time())}")
    seed = train_cfg.get("seed", 42)

    print("=" * 60)
    print(" C12 THERMAL HUMAN DETECTION - YOLO FINE-TUNING")
    print("=" * 60)

    # 3. Verify Hardware and Environment
    has_cuda, cuda_desc, vram = check_cuda_environment()
    print(f"[*] Compute Device : {cuda_desc}")
    if "cpu" not in device.lower() and not has_cuda:
        print("[!] Warning: CUDA requested but not available. Switching to CPU.")
        device = "cpu"
        workers = min(workers, 2)
    elif has_cuda and vram and vram < 10.0 and batch_size > 8:
        print(f"[!] Note: Detected {vram:.1f} GB VRAM. If OOM occurs, reduce batch size to 8 or 4.")

    # 4. Framework Versions
    try:
        import ultralytics
        ultralytics_ver = ultralytics.__version__
    except ImportError:
        print("[ERROR] Ultralytics is not installed. Install with 'pip install ultralytics'.", file=sys.stderr)
        sys.exit(1)

    print(f"[*] Frameworks     : PyTorch {torch.__version__} | Ultralytics {ultralytics_ver}")
    print(f"[*] Model Weights  : {model_checkpoint}")
    print(f"[*] Dataset Config : {data_yaml}")
    print(f"[*] Hyperparams    : Epochs={epochs}, Batch={batch_size}, ImgSz={imgsz}, Device={device}")
    print("=" * 60)

    # 5. Validate Dataset before starting training
    data_yaml_path = Path(data_yaml)
    is_valid, validation_msg = validate_training_dataset(data_yaml_path)
    if not is_valid:
        print(f"[DATASET ERROR] {validation_msg}", file=sys.stderr)
        print("Please prepare and label dataset frames first using 'dataset_tools/'.", file=sys.stderr)
        sys.exit(2)
    else:
        print(f"[*] Dataset Status : {validation_msg}")

    # 6. Initialize YOLO Model
    from ultralytics import YOLO
    try:
        model = YOLO(model_checkpoint)
    except Exception as e:
        print(f"[ERROR] Failed to load checkpoint '{model_checkpoint}': {e}", file=sys.stderr)
        sys.exit(1)

    # Combine training parameters
    train_args = {
        "data": str(data_yaml_path.resolve()),
        "epochs": epochs,
        "batch": batch_size,
        "imgsz": imgsz,
        "device": device,
        "workers": workers,
        "patience": patience,
        "project": project_dir,
        "name": run_name,
        "seed": seed,
        "exist_ok": False,
        "verbose": True,
    }
    # Append custom hyperparameters and augmentations
    train_args.update(hyp_cfg)
    train_args.update(aug_cfg)

    # 7. Execute Training Loop with OOM Protection
    print("\n[+] Starting fine-tuning session...")
    t_start = time.time()
    try:
        results = model.train(**train_args)
    except torch.cuda.OutOfMemoryError as oom_err:
        print(f"\n[CUDA OOM ERROR] {oom_err}", file=sys.stderr)
        print("The GPU ran out of memory. Suggested actions:", file=sys.stderr)
        print(f"  1. Lower batch size from {batch_size} to {max(1, batch_size // 2)} (--batch {max(1, batch_size // 2)})", file=sys.stderr)
        print("  2. Lower input resolution from 640 to 384 (--imgsz 384)", file=sys.stderr)
        print("  3. Reduce DataLoader workers (--workers 2)", file=sys.stderr)
        sys.exit(3)
    except Exception as e:
        print(f"\n[TRAINING ERROR] Unexpected training failure: {e}", file=sys.stderr)
        sys.exit(1)

    elapsed_min = (time.time() - t_start) / 60.0
    print("\n" + "=" * 60)
    print(f"[OK] Training Completed Successfully in {elapsed_min:.1f} minutes!")
    
    # Save effective run configuration metadata
    run_dir = Path(project_dir) / run_name
    best_weights = run_dir / "weights" / "best.pt"
    print(f"[*] Run Directory  : {run_dir.resolve()}")
    if best_weights.exists():
        print(f"[*] Best Checkpoint: {best_weights.resolve()}")
    
    meta_info = {
        "timestamp": time.time(),
        "elapsed_minutes": round(elapsed_min, 2),
        "pytorch_version": torch.__version__,
        "ultralytics_version": ultralytics_ver,
        "device": cuda_desc,
        "train_args": train_args,
    }
    with open(run_dir / "training_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_info, f, indent=2)
    print(f"[*] Metadata Saved : {run_dir / 'training_meta.json'}")
    print("=" * 60)


if __name__ == "__main__":
    run_training()
