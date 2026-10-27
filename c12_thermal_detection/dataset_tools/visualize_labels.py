"""
YOLO Label Visualizer for C12 Thermal Human Detection.
Draws ground-truth bounding boxes and classes onto images and saves preview copies.
Useful for quality inspection before model training.
"""

import argparse
from pathlib import Path
import sys
import cv2
import yaml


def visualize_labels():
    parser = argparse.ArgumentParser(description="Render YOLO ground truth annotations onto images")
    parser.add_argument("--images", type=str, required=True, help="Directory containing images")
    parser.add_argument("--labels", type=str, required=True, help="Directory containing YOLO .txt labels")
    parser.add_argument("--output-dir", type=str, default="outputs/annotated_previews", help="Target preview directory")
    parser.add_argument("--data-yaml", type=str, default=None, help="Optional data.yaml for class name mapping")
    parser.add_argument("--max-images", type=int, default=50, help="Maximum number of images to render")
    args = parser.parse_args()

    img_dir = Path(args.images)
    lbl_dir = Path(args.labels)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not img_dir.exists() or not lbl_dir.exists():
        print("[ERROR] Input image or label directory does not exist.", file=sys.stderr)
        sys.exit(1)

    class_names = {0: "person"}
    if args.data_yaml and Path(args.data_yaml).exists():
        with open(args.data_yaml, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            if "names" in cfg:
                if isinstance(cfg["names"], dict):
                    class_names = {int(k): v for k, v in cfg["names"].items()}
                elif isinstance(cfg["names"], list):
                    class_names = {i: v for i, v in enumerate(cfg["names"])}

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    img_files = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in valid_exts])[:args.max_images]

    print("=" * 60)
    print(" C12 THERMAL DATASET - LABEL VISUALIZER")
    print("=" * 60)
    print(f"[*] Images Dir : {img_dir}")
    print(f"[*] Labels Dir : {lbl_dir}")
    print(f"[*] Output Dir : {out_dir}")
    print(f"[*] Rendering up to {len(img_files)} images...")
    print("=" * 60)

    rendered_count = 0

    for img_path in img_files:
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]

        if lbl_path.exists():
            with open(lbl_path, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]

            for line in lines:
                parts = line.split()
                if len(parts) == 5:
                    cls_id = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:])
                    
                    # Convert normalized center-width to pixel coordinates
                    x1 = int(round((cx - bw / 2.0) * w))
                    y1 = int(round((cy - bh / 2.0) * h))
                    x2 = int(round((cx + bw / 2.0) * w))
                    y2 = int(round((cy + bh / 2.0) * h))

                    label_text = class_names.get(cls_id, f"class_{cls_id}")
                    
                    # Draw green bounding box
                    color = (0, 255, 0)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw text badge
                    cv2.putText(
                        img,
                        label_text,
                        (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1,
                        cv2.LINE_AA,
                    )

        # Save annotated preview
        out_path = out_dir / f"preview_{img_path.name}"
        cv2.imwrite(str(out_path), img)
        rendered_count += 1

    print(f"[OK] Rendered and saved {rendered_count} annotated previews to {out_dir.resolve()}.")


if __name__ == "__main__":
    visualize_labels()
