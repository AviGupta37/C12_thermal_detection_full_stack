#!/usr/bin/env bash
# ==============================================================================
# Evaluate Trained YOLO Model on Held-Out Thermal Validation/Test Set
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Default to the latest best.pt if not supplied
MODEL_CHECKPOINT="${1:-runs/training/c12_thermal_baseline/weights/best.pt}"

if [ ! -f "$MODEL_CHECKPOINT" ]; then
    echo "[!] Model checkpoint not found at: $MODEL_CHECKPOINT"
    echo "    Usage: bash scripts/validate_model.sh <path_to_model.pt> [--split test]"
    exit 1
fi

echo "=================================================================="
echo " C12 Thermal Detection - Model Validation"
echo "=================================================================="
echo "[*] Evaluating: $MODEL_CHECKPOINT"

python training/evaluate.py --model "$MODEL_CHECKPOINT" "${@:2}"

