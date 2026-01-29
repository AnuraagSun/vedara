#!/bin/bash

echo ""
echo "====================================================="
echo " VEDARA AR System Launcher (Linux/RPi)"
echo "====================================================="
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -f "venv/bin/activate" ]; then
    echo "[ERROR] Virtual environment not found!"
    echo "Run: python3 -m venv venv"
    echo "Then: source venv/bin/activate"
    echo "Then: pip install -r requirements.txt"
    exit 1
fi

source venv/bin/activate

if [ -f /proc/device-tree/model ]; then
    MODEL=$(cat /proc/device-tree/model)
    echo "Detected: $MODEL"

    export OPENBLAS_CORETYPE=ARMV8

fi

echo "Starting VEDARA..."
echo ""
python3 main.py "$@"

deactivate

echo ""
echo "VEDARA session ended."
