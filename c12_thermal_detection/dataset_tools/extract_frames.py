"""
Frame extraction tool for raw C12 flight videos.
Samples frames at a configurable interval (stride or target FPS).
Note: Extracted frames are raw unannotated imagery and require labeling.
"""

import argparse
from pathlib import Path
import sys
import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Extract video frames for dataset preparation")
    parser.add_argument("--video", type=str, required=True, help="Path to input flight video (.mp4, .avi, etc.)")
    parser.add_argument("--output-dir", type=str, default="datasets/extracted_frames", help="Output directory for frames")
    parser.add_argument("--prefix", type=str, default=None, help="Frame filename prefix (defaults to video name)")
    parser.add_argument("--stride", type=int, default=10, help="Extract every Nth frame (stride)")
    parser.add_argument("--target-fps", type=float, default=None, help="Target extraction FPS (overrides stride if set)")
    parser.add_argument("--format", type=str, default="jpg", choices=["jpg", "png"], help="Output image format")
    return parser.parse_args()


def extract_frames():
    args = parse_args()
    video_path = Path(args.video)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        print(f"[ERROR] Video file not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Could not open video file: {video_path}", file=sys.stderr)
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    prefix = args.prefix if args.prefix else video_path.stem

    # Determine stride
    stride = args.stride
    if args.target_fps is not None and args.target_fps > 0:
        stride = max(1, int(round(video_fps / args.target_fps)))

    print("=" * 60)
    print(" C12 THERMAL DATASET - FRAME EXTRACTION")
    print("=" * 60)
    print(f"[*] Input Video    : {video_path.name}")
    print(f"[*] Resolution     : {width} x {height}")
    print(f"[*] Native FPS     : {video_fps:.1f}")
    print(f"[*] Total Frames   : {total_frames}")
    print(f"[*] Sampling Stride: Every {stride} frame(s)")
    print(f"[*] Output Dir     : {out_dir.resolve()}")
    print("=" * 60)

    saved_count = 0
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % stride == 0:
            out_filename = f"{prefix}_frame_{frame_idx:06d}.{args.format}"
            out_path = out_dir / out_filename
            cv2.imwrite(str(out_path), frame)
            saved_count += 1

        frame_idx += 1

    cap.release()
    print(f"\n[OK] Finished extraction. Processed {frame_idx} frames; saved {saved_count} frames to {out_dir}.")
    print("[!] Reminder: These extracted frames are raw imagery. They must be annotated using an annotation tool (e.g. CVAT or Label Studio) before training.")


if __name__ == "__main__":
    extract_frames()
