#!/bin/bash
# Install all dependencies including flash-attn with proper compilation flags

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MAIN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "Installing dependencies from requirements.txt..."
pip install -r "${MAIN_DIR}/requirements.txt"

echo ""
echo "Installing flash-attn with compilation flags..."
echo "This may take several minutes as it needs to compile from source..."
pip install flash-attn --no-build-isolation --no-cache-dir --no-binary flash-attn

echo ""
echo "Verifying flash-attn installation..."
python3 -c "import flash_attn; print('✓ flash-attn imported successfully')" || {
    echo "✗ flash-attn import failed"
    exit 1
}

echo ""
echo "✓ All dependencies installed successfully!"

