#!/usr/bin/env bash
# ==============================================================================
# Launch C12 Thermal Detection Desktop Ground Station UI
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "=================================================================="
echo " Starting C12 Thermal Detection Ground Station..."
echo "=================================================================="

python main.py "$@"

