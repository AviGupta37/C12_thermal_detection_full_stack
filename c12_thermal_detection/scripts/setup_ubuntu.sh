#!/usr/bin/env bash
# ==============================================================================
# Setup Script for C12 Thermal Detection Ground Station on Ubuntu 22.04
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=================================================================="
echo " C12 Thermal Human Detection - Ubuntu 22.04 Setup"
echo "=================================================================="
echo "[*] Project Directory: $PROJECT_ROOT"

# 1. Check Ubuntu Version
if [ -f /etc/os-release ]; then
    . /etc/os-release
    echo "[*] Operating System : $NAME $VERSION_ID"
fi

# 2. Inform and check for system dependencies
echo ""
echo "[1/4] Checking required system packages (apt)..."
echo "Note: The following packages may require sudo privileges if not already installed:"
echo "      python3-venv python3-pip ffmpeg libgl1-mesa-glx libglib2.0-0"

MISSING_PKGS=""
for pkg in python3-venv python3-pip ffmpeg; do
    if ! dpkg -s "$pkg" >/dev/null 2>&1; then
        MISSING_PKGS="$MISSING_PKGS $pkg"
    fi
done

if [ -n "$MISSING_PKGS" ]; then
    echo "[!] Missing system packages:$MISSING_PKGS"
    echo "    Run: sudo apt update && sudo apt install -y $MISSING_PKGS"
else
    echo "[OK] Core system packages installed."
fi

# 3. Create or reuse isolated virtual environment
echo ""
echo "[2/4] Setting up isolated Python virtual environment..."
cd "$PROJECT_ROOT"
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "[OK] Created virtual environment at .venv"
else
    echo "[OK] Existing virtual environment found at .venv"
fi

# Activate venv
source .venv/bin/activate
echo "[*] Activated virtual environment: $(which python)"

# 4. Upgrade pip and install project dependencies
echo ""
echo "[3/4] Installing project dependencies into virtual environment..."
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# 5. Check GPU acceleration (NVIDIA RTX 2080 Ti)
echo ""
echo "[4/4] Verifying GPU and PyTorch CUDA build..."
python scripts/check_gpu.py

echo ""
echo "=================================================================="
echo "[OK] Environment setup complete!"
echo "     Launch UI with:      bash scripts/launch_ui.sh"
echo "     Train baseline with: bash scripts/train_model.sh"
echo "=================================================================="

