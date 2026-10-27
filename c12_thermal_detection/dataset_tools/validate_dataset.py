"""
YOLO Dataset Validator for C12 Thermal Human Detection.
Checks dataset structure, image readability, label syntax, coordinate bounds,
empty label files (negatives), and orphan files.
Returns non-zero exit code if validation errors are detected.
"""

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Set, Tuple
import cv2
import yaml


def validate_yolo_label_file(
    label_path: Path,
    expected_class_ids: Set[int],
) -> Tuple[bool, int, List[str]]:
    """
    Validates a single YOLO format label file.
    Format per line: <class_id> <cx> <cy> <w> <h> (normalized [0, 1])
    Returns: (is_valid, box_count, error_messages)
    """
    errors: List[str] = []
    box_count = 0

    if not label_path.exists():
        return False, 0, [f"Label file missing: {label_path}"]

    try:
        with open(label_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
    except Exception as e:
        return False, 0, [f"Could not read {label_path.name}: {e}"]

    # Empty file is valid (represents a negative image with no people)
    if not lines:
        return True, 0, []

    for line_idx, line in enumerate(lines, start=1):
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"Line {line_idx}: expected 5 elements (<cls> <cx> <cy> <w> <h>), got {len(parts)}: '{line}'")
            continue

        try:
            cls_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])
        except ValueError:
            errors.append(f"Line {line_idx}: non-numeric values found in '{line}'")
            continue

        if cls_id not in expected_class_ids:
            errors.append(f"Line {line_idx}: class ID {cls_id} not in expected classes {list(expected_class_ids)}")

        if not (0.0 <= cx <= 1.0):
            errors.append(f"Line {line_idx}: center X {cx} is outside [0.0, 1.0]")
        if not (0.0 <= cy <= 1.0):
            errors.append(f"Line {line_idx}: center Y {cy} is outside [0.0, 1.0]")
        if w <= 0.0 or w > 1.0:
            errors.append(f"Line {line_idx}: width {w} must be positive and <= 1.0")
        if h <= 0.0 or h > 1.0:
            errors.append(f"Line {line_idx}: height {h} must be positive and <= 1.0")

        box_count += 1

    return len(errors) == 0, box_count, errors


def validate_split(
    split_name: str,
    images_dir: Path,
    labels_dir: Path,
    expected_classes: Set[int],
    check_images_readable: bool = True,
) -> Dict[str, Any]:
    """Validate all images and labels in a specific split."""
    results = {
        "split": split_name,
        "images_count": 0,
        "labels_count": 0,
        "valid_images": 0,
        "corrupted_images": 0,
        "empty_label_images": 0,  # Negative samples
        "annotated_images": 0,
        "total_annotations": 0,
        "missing_label_files": 0,
        "orphan_label_files": 0,
        "syntax_errors": [],
        "is_valid": True,
    }

    if not images_dir.exists():
        results["is_valid"] = False
        results["syntax_errors"].append(f"Images directory missing: {images_dir}")
        return results

    if not labels_dir.exists():
        results["is_valid"] = False
        results["syntax_errors"].append(f"Labels directory missing: {labels_dir}")
        return results

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    image_files = {p.stem: p for p in images_dir.iterdir() if p.suffix.lower() in valid_exts}
    label_files = {p.stem: p for p in labels_dir.iterdir() if p.suffix.lower() == ".txt"}

    results["images_count"] = len(image_files)
    results["labels_count"] = len(label_files)

    # Check images
    for stem, img_path in image_files.items():
        if check_images_readable:
            img = cv2.imread(str(img_path))
            if img is None:
                results["corrupted_images"] += 1
                results["syntax_errors"].append(f"Unreadable image: {img_path.name}")
                continue
            else:
                results["valid_images"] += 1
        else:
            results["valid_images"] += 1

        label_path = label_files.get(stem)
        if label_path is None:
            results["missing_label_files"] += 1
            results["syntax_errors"].append(f"Missing label file for: {img_path.name}")
            continue

        is_ok, count, errs = validate_yolo_label_file(label_path, expected_classes)
        if not is_ok:
            results["syntax_errors"].extend([f"{label_path.name}: {e}" for e in errs])
        else:
            if count == 0:
                results["empty_label_images"] += 1
            else:
                results["annotated_images"] += 1
                results["total_annotations"] += count

    # Check for orphan labels
    for stem, lbl_path in label_files.items():
        if stem not in image_files:
            results["orphan_label_files"] += 1
            results["syntax_errors"].append(f"Orphan label without matching image: {lbl_path.name}")

    if results["corrupted_images"] > 0 or results["missing_label_files"] > 0 or len(results["syntax_errors"]) > 0:
        results["is_valid"] = False

    return results


def run_validation():
    parser = argparse.ArgumentParser(description="Validate YOLO dataset formatting and integrity")
    parser.add_argument("--data", type=str, default="datasets/yolo_dataset/data.yaml", help="Path to data.yaml")
    parser.add_argument("--skip-image-decode", action="store_true", help="Skip decoding every image with OpenCV for faster check")
    args = parser.parse_args()

    data_yaml_path = Path(args.data)
    if not data_yaml_path.exists():
        print(f"[ERROR] data.yaml not found at: {data_yaml_path}", file=sys.stderr)
        sys.exit(1)

    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    base_dir = data_yaml_path.parent
    if "path" in data_cfg and data_cfg["path"]:
        cfg_path = Path(data_cfg["path"])
        if cfg_path.is_absolute():
            base_dir = cfg_path
        else:
            base_dir = (data_yaml_path.parent / cfg_path).resolve()
    expected_classes = {0}
    if "names" in data_cfg and isinstance(data_cfg["names"], dict):
        expected_classes = set(int(k) for k in data_cfg["names"].keys())
    elif "names" in data_cfg and isinstance(data_cfg["names"], list):
        expected_classes = set(range(len(data_cfg["names"])))

    print("=" * 65)
    print(" C12 THERMAL HUMAN DETECTION - DATASET INTEGRITY VALIDATOR")
    print("=" * 65)
    print(f"[*] Dataset Config : {data_yaml_path.resolve()}")
    print(f"[*] Classes        : {expected_classes}")
    print("=" * 65)

    all_passed = True

    for split in ["train", "val", "test"]:
        if split not in data_cfg:
            continue

        split_img_path = Path(data_cfg[split])
        if not split_img_path.is_absolute():
            split_img_path = (base_dir / split_img_path).resolve()
        
        # Derive label directory (YOLO standard replaces /images/ with /labels/)
        split_lbl_path = Path(str(split_img_path).replace("images", "labels"))

        res = validate_split(
            split_name=split,
            images_dir=split_img_path,
            labels_dir=split_lbl_path,
            expected_classes=expected_classes,
            check_images_readable=not args.skip_image_decode,
        )

        print(f"\n--- Split: {split.upper()} ---")
        print(f"  Images Total      : {res['images_count']}")
        print(f"  Valid Decodable   : {res['valid_images']}")
        print(f"  Corrupted Images  : {res['corrupted_images']}")
        print(f"  Annotated Images  : {res['annotated_images']}")
        print(f"  Empty Negatives   : {res['empty_label_images']}")
        print(f"  Total People BBoxes: {res['total_annotations']}")
        print(f"  Missing Labels    : {res['missing_label_files']}")
        print(f"  Orphan Labels     : {res['orphan_label_files']}")

        if not res["is_valid"]:
            all_passed = False
            print(f"  [STATUS: FAILED] {len(res['syntax_errors'])} errors found:")
            for err in res["syntax_errors"][:10]:
                print(f"    - {err}")
            if len(res["syntax_errors"]) > 10:
                print(f"    ... and {len(res['syntax_errors']) - 10} more errors.")
        else:
            print("  [STATUS: PASSED]")

    print("\n" + "=" * 65)
    if all_passed:
        print("[OK] All dataset splits passed validation successfully!")
        sys.exit(0)
    else:
        print("[FAIL] Dataset validation failed. Please fix reported errors before training.")
        sys.exit(1)


if __name__ == "__main__":
    from typing import Any
    run_validation()
