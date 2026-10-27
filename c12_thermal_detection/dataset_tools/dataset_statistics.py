"""
Dataset statistics analyzer for C12 Thermal Human Detection.
Calculates distribution of person bounding box sizes, aspect ratios,
negative image proportions, and split comparisons.
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List
import cv2
import yaml


def analyze_dataset():
    parser = argparse.ArgumentParser(description="Analyze YOLO dataset statistics")
    parser.add_argument("--data", type=str, default="datasets/yolo_dataset/data.yaml", help="Path to data.yaml")
    parser.add_argument("--output-json", type=str, default="outputs/dataset_reports/stats.json", help="Output JSON path")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[ERROR] data.yaml not found at: {data_path}", file=sys.stderr)
        sys.exit(1)

    with open(data_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base_dir = data_path.parent
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}

    overall_stats: Dict[str, Any] = {
        "dataset_yaml": str(data_path),
        "splits": {},
        "summary": {
            "total_images": 0,
            "total_annotations": 0,
            "total_negative_images": 0,
            "size_distribution": {
                "small_under_32px": 0,    # Typically small targets at 40-50m altitude
                "medium_32_to_96px": 0,   # Typical targets at 20-35m altitude
                "large_above_96px": 0,    # Close targets (<20m)
            },
        },
    }

    print("=" * 65)
    print(" C12 THERMAL DATASET - METRICS & STATISTICS ANALYZER")
    print("=" * 65)

    for split in ["train", "val", "test"]:
        if split not in cfg:
            continue

        split_img_dir = Path(cfg[split])
        if not split_img_dir.is_absolute():
            split_img_dir = (base_dir / split_img_dir).resolve()
        split_lbl_dir = Path(str(split_img_dir).replace("images", "labels"))

        if not split_img_dir.exists():
            continue

        images = [p for p in split_img_dir.iterdir() if p.suffix.lower() in valid_exts]
        split_stat = {
            "images": len(images),
            "annotated_images": 0,
            "negative_images": 0,
            "total_people": 0,
            "bbox_widths_norm": [],
            "bbox_heights_norm": [],
        }

        for img_path in images:
            lbl_path = split_lbl_dir / f"{img_path.stem}.txt"
            has_person = False
            if lbl_path.exists():
                with open(lbl_path, "r", encoding="utf-8") as f:
                    lines = [l.strip() for l in f if l.strip()]
                if lines:
                    has_person = True
                    for line in lines:
                        parts = line.split()
                        if len(parts) == 5:
                            w_norm, h_norm = float(parts[3]), float(parts[4])
                            split_stat["bbox_widths_norm"].append(w_norm)
                            split_stat["bbox_heights_norm"].append(h_norm)
                            split_stat["total_people"] += 1

                            # Approximate pixel scale assuming 640x512 standard or 384x288 C12
                            max_dim_px = max(w_norm * 384, h_norm * 288)
                            if max_dim_px < 32:
                                overall_stats["summary"]["size_distribution"]["small_under_32px"] += 1
                            elif max_dim_px <= 96:
                                overall_stats["summary"]["size_distribution"]["medium_32_to_96px"] += 1
                            else:
                                overall_stats["summary"]["size_distribution"]["large_above_96px"] += 1

            if has_person:
                split_stat["annotated_images"] += 1
            else:
                split_stat["negative_images"] += 1

        overall_stats["splits"][split] = {
            "images": split_stat["images"],
            "annotated_images": split_stat["annotated_images"],
            "negative_images": split_stat["negative_images"],
            "total_people": split_stat["total_people"],
            "avg_people_per_image": round(split_stat["total_people"] / max(1, split_stat["images"]), 2),
        }
        overall_stats["summary"]["total_images"] += split_stat["images"]
        overall_stats["summary"]["total_annotations"] += split_stat["total_people"]
        overall_stats["summary"]["total_negative_images"] += split_stat["negative_images"]

        print(f"\n--- Split: {split.upper()} ---")
        print(f"  Total Images     : {split_stat['images']}")
        print(f"  Images w/ People : {split_stat['annotated_images']}")
        print(f"  Negative Images  : {split_stat['negative_images']} ({split_stat['negative_images'] / max(1, split_stat['images']) * 100:.1f}%)")
        print(f"  Total People     : {split_stat['total_people']}")
        print(f"  Density          : {overall_stats['splits'][split]['avg_people_per_image']} people/frame")

    print("\n--- OBJECT SCALE DISTRIBUTION (AERIAL ALTITUDE PROXY) ---")
    sizes = overall_stats["summary"]["size_distribution"]
    tot_boxes = max(1, overall_stats["summary"]["total_annotations"])
    print(f"  Small (<32px / high alt 40-50m) : {sizes['small_under_32px']} ({sizes['small_under_32px']/tot_boxes*100:.1f}%)")
    print(f"  Medium (32-96px / mid alt 20-35m): {sizes['medium_32_to_96px']} ({sizes['medium_32_to_96px']/tot_boxes*100:.1f}%)")
    print(f"  Large (>96px / low alt <20m)    : {sizes['large_above_96px']} ({sizes['large_above_96px']/tot_boxes*100:.1f}%)")
    print("=" * 65)

    # Save to JSON
    out_json_path = Path(args.output_json)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(overall_stats, f, indent=2)
    print(f"[*] Analysis report saved to: {out_json_path.resolve()}")


if __name__ == "__main__":
    analyze_dataset()

