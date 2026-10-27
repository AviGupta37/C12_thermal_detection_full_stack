# Models Directory

This directory stores trained and pretrained YOLO model checkpoints.

## Recommended Weights
- `yolo11n.pt`: Nano baseline model. Suitable for initial testing and rapid training cycles.
- `yolo11s.pt`: Small model. Offers a good balance of accuracy and inference latency for aerial LWIR imagery.
- `best.pt`: The fine-tuned checkpoint exported from `runs/training/<run_name>/weights/best.pt`.

## Download Pretrained Baseline
You can download official Ultralytics weights using the Python CLI:
```bash
yolo detect predict model=yolo11n.pt
```
Or place your custom checkpoint directly in this directory and reference it in `config/detection.yaml` or through the operator UI.

> Note: Do not commit large binary `.pt` files to git.

