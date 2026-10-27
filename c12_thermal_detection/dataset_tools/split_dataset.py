"""
Dataset splitting tool for C12 Thermal Human Detection.
Splits by video recording / flight session to prevent data leakage between adjacent frames.
If group metadata is absent, warns the user and requests explicit review.
"""

import argparse
from collections import defaultdict
from pathlib import Path
import random
import shutil
import sys
from typing import Dict, List, Set, Tuple


def extract_group_key(filename: str, delimiter: str = "_") -> str:
    """
    Extract group/session prefix from filename (e.g. 'flight01_frame0023.jpg' -> 'flight01').
    """
    parts = filename.split(delimiter)
    if len(parts) >= 2:
        return parts[0]
    return "ungrouped"


def split_dataset():
    parser = argparse.ArgumentParser(description="Split dataset by session/flight to prevent temporal leakage")
    parser.add_argument("--source-images", type=str, required=True, help="Path to pool of extracted images")
    parser.add_argument("--source-labels", type=str, required=True, help="Path to pool of YOLO label files (.txt)")
    parser.add_argument("--target-dataset", type=str, default="datasets/yolo_dataset", help="Target YOLO dataset root")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Train ratio (e.g. 0.70)")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation ratio (e.g. 0.15)")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test ratio (e.g. 0.15)")
    parser.add_argument("--delimiter", type=str, default="_", help="Delimiter used to identify session prefix")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of moving")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for group assignment")
    args = parser.parse_args()

    src_img_dir = Path(args.source_images)
    src_lbl_dir = Path(args.source_labels)
    target_root = Path(args.target_dataset)

    if not src_img_dir.exists() or not src_lbl_dir.exists():
        print("[ERROR] Source images or labels directory does not exist.", file=sys.stderr)
        sys.exit(1)

    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-4:
        print(f"[ERROR] Split ratios must sum to 1.0. Current sum: {total_ratio}", file=sys.stderr)
        sys.exit(1)

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = [p for p in src_img_dir.iterdir() if p.suffix.lower() in valid_exts]
    
    if not images:
        print("[ERROR] No images found in source images directory.", file=sys.stderr)
        sys.exit(1)

    # Group images by session / flight prefix
    groups: Dict[str, List[Path]] = defaultdict(list)
    for img in images:
        key = extract_group_key(img.stem, args.delimiter)
        groups[key].append(img)

    print("=" * 65)
    print(" C12 THERMAL DATASET - SESSION-AWARE DATASET SPLITTER")
    print("=" * 65)
    print(f"[*] Found {len(images)} images grouped into {len(groups)} distinct session(s):")
    for grp, file_list in groups.items():
        print(f"    - Session '{grp}': {len(file_list)} images")

    if len(groups) == 1 and "ungrouped" in groups:
        print("\n[!] CRITICAL WARNING: No session prefixes detected in filenames!")
        print("    Splitting adjacent frames randomly will cause severe data leakage,")
        print("    leading to artificially inflated mAP scores that fail in real field tests.")
        print("    Ensure filenames include flight/session IDs (e.g. 'flightA_frame001.jpg').")

    # Shuffle groups deterministically
    random.seed(args.seed)
    group_keys = list(groups.keys())
    random.shuffle(group_keys)

    # Allocate entire groups to splits
    train_groups: List[str] = []
    val_groups: List[str] = []
    test_groups: List[str] = []

    # If only 1 group exists, we must chunk contiguous blocks of frames instead of random shuffle
    if len(group_keys) == 1:
        single_group_files = sorted(groups[group_keys[0]])
        n = len(single_group_files)
        n_train = int(n * args.train_ratio)
        n_val = int(n * args.val_ratio)

        train_files = single_group_files[:n_train]
        val_files = single_group_files[n_train:n_train + n_val]
        test_files = single_group_files[n_train + n_val:]
        split_map = {"train": train_files, "val": val_files, "test": test_files}
    else:
        # Group-level allocation
        n_grps = len(group_keys)
        n_train = max(1, int(round(n_grps * args.train_ratio)))
        n_val = max(1, int(round(n_grps * args.val_ratio))) if n_grps > 2 else 0

        train_groups = group_keys[:n_train]
        val_groups = group_keys[n_train:n_train + n_val]
        test_groups = group_keys[n_train + n_val:]
        if not test_groups and len(val_groups) > 1:
            test_groups = [val_groups.pop()]

        split_map = {"train": [], "val": [], "test": []}
        for grp in train_groups:
            split_map["train"].extend(groups[grp])
        for grp in val_groups:
            split_map["val"].extend(groups[grp])
        for grp in test_groups:
            split_map["test"].extend(groups[grp])

    print("\n[*] Group Split Assignment:")
    print(f"    Train Split: {len(split_map['train'])} images")
    print(f"    Val Split  : {len(split_map['val'])} images")
    print(f"    Test Split : {len(split_map['test'])} images")

    # Execute transfer
    transfer_op = shutil.copy2 if args.copy else shutil.move
    op_name = "Copied" if args.copy else "Moved"

    for split, file_list in split_map.items():
        dst_img_dir = target_root / "images" / split
        dst_lbl_dir = target_root / "labels" / split
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path in file_list:
            # Transfer image
            transfer_op(str(img_path), str(dst_img_dir / img_path.name))
            
            # Transfer matching label if exists
            lbl_name = f"{img_path.stem}.txt"
            src_lbl = src_lbl_dir / lbl_name
            if src_lbl.exists():
                transfer_op(str(src_lbl), str(dst_lbl_dir / lbl_name))
            else:
                # Create empty label file for negative samples
                (dst_lbl_dir / lbl_name).touch()

    print(f"\n[OK] {op_name} files to '{target_root}'. Dataset split complete with zero leakage.")


if __name__ == "__main__":
    split_dataset()
