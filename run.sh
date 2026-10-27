#!/usr/bin/env bash
# ==============================================================================
# C12 Aerial Thermal Human Detection Ground Station Launcher
# Usage:
#   ./run.sh                           # Launch standard UI
#   ./run.sh --video path/to/file.mp4   # Open directly with recorded video
#   ./run.sh --rtsp rtsp://ip:port/...  # Open directly with live RTSP stream
#   ./run.sh --headless                 # Run headless verification smoke test
# ==============================================================================
set -e

# Resolve project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. Activate Python Virtual Environment (handles both Linux & Windows Git-Bash / WSL)
if [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
elif [ -f "$SCRIPT_DIR/c12_thermal_detection/.venv/bin/activate" ]; then
    source "$SCRIPT_DIR/c12_thermal_detection/.venv/bin/activate"
elif [ -f "$SCRIPT_DIR/.venv/Scripts/activate" ]; then
    source "$SCRIPT_DIR/.venv/Scripts/activate"
elif [ -f "$SCRIPT_DIR/c12_thermal_detection/.venv/Scripts/activate" ]; then
    source "$SCRIPT_DIR/c12_thermal_detection/.venv/Scripts/activate"
fi

# 2. Pick Python binary
if command -v python3 &>/dev/null; then
    PY_BIN="python3"
elif command -v python &>/dev/null; then
    PY_BIN="python"
else
    echo "[ERROR] Python was not found in PATH or virtual environment." >&2
    exit 1
fi

# 3. Add to PYTHONPATH
export PYTHONPATH="$SCRIPT_DIR:$SCRIPT_DIR/c12_thermal_detection:${PYTHONPATH:-}"

echo "=================================================================="
echo " Starting C12 Aerial Thermal Human Detection Ground Station"
echo "=================================================================="
echo "[*] Python interpreter: $($PY_BIN -c 'import sys; print(sys.executable)')"
echo "[*] Working directory : $SCRIPT_DIR"
echo "=================================================================="

# 4. Launch Application with any provided arguments
exec "$PY_BIN" main.py "$@"

