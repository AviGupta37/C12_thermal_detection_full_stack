#!/usr/bin/env bash
# ==============================================================================
# Train / Fine-Tune YOLO Model on Skydroid C12 Thermal Human Dataset
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

CONFIG_FILE="config/training.yaml"

echo "=================================================================="
echo " C12 Thermal Human Detection - Model Training"
echo "=================================================================="
echo "[*] Using configuration: $CONFIG_FILE"

# Launch Python fine-tuning script
python training/train.py --config "$CONFIG_FILE" "$@"

