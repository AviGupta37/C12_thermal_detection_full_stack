# Datasets Directory

This directory contains data files, raw aerial recordings, extracted frames, and YOLO formatted datasets.

## Directory Structure
- `raw/`: Unedited flight video recordings (.mp4, .avi, .mkv) captured from the C12 camera.
- `extracted_frames/`: Frames sampled from raw footage via `dataset_tools/extract_frames.py`.
- `yolo_dataset/`: Formatted YOLO dataset with train, val, and test splits.
  - `images/`: Contains `train/`, `val/`, `test/` image folders (.jpg, .png).
  - `labels/`: Contains `train/`, `val/`, `test/` YOLO annotation files (.txt).
  - `data.yaml`: Dataset specification for YOLO training.

## Annotation Guidelines
- Single class: `0: person`.
- Format: `<class_id> <x_center> <y_center> <width> <height>` (normalized between 0.0 and 1.0).
- Empty frames (no persons) must have an empty `.txt` file to represent negative samples.
- Prevent data leakage: Split consecutive frames from the same flight recording together into either train or val/test, not across both.

