#!/bin/bash
# Install all dependencies including flash-attn with proper compilation flags

# Don't exit on error - we want to handle flash-attn installation failures gracefully
set +e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MAIN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "Installing dependencies from requirements.txt..."
pip install -r "${MAIN_DIR}/requirements.txt" || {
    echo "✗ Failed to install dependencies from requirements.txt"
    exit 1
}

echo ""
echo "Installing flash-attn..."
echo "First attempting to install pre-built wheel (faster, no compilation needed)..."
echo ""

# Try pre-built wheel first (faster, no memory issues)
if pip install flash-attn --no-build-isolation; then
    echo ""
    echo "Verifying flash-attn installation..."
    if python3 -c "import flash_attn; print('✓ flash-attn imported successfully')" 2>/dev/null; then
        echo "✓ flash-attn installed from pre-built wheel and verified successfully!"
    else
        echo "⚠ flash-attn installed but import failed. Trying compilation from source..."
        # Fall through to compilation
    fi
else
    echo "Pre-built wheel not available or failed. Trying compilation from source..."
fi

# If pre-built wheel didn't work, try compilation from source
if ! python3 -c "import flash_attn" 2>/dev/null; then
    echo ""
    echo "Compiling flash-attn from source..."
    echo "This may take several minutes and requires significant memory."
    echo ""
    
    # Try to install flash-attn with reduced parallel jobs to save memory
    # MAX_JOBS=2 limits parallel compilation to reduce memory usage
    MAX_JOBS=${MAX_JOBS:-2}
    export MAX_JOBS
    
    # Set environment variables to reduce memory usage during compilation
    export FLASH_ATTENTION_SKIP_CUDA_BUILD=0
    export FLASH_ATTENTION_FORCE_BUILD=1
    
    # Try installation with compilation from source
    if pip install flash-attn --no-build-isolation --no-cache-dir --no-binary flash-attn; then
        echo ""
        echo "Verifying flash-attn installation..."
        if python3 -c "import flash_attn; print('✓ flash-attn imported successfully')" 2>/dev/null; then
            echo "✓ flash-attn compiled and verified successfully!"
        else
            echo "⚠ flash-attn installed but import failed. This may indicate a compilation issue."
            echo "Try running: python3 -c 'import flash_attn' to see the error."
            exit 1
        fi
    else
        echo ""
        echo "✗ flash-attn installation failed. This is often due to:"
        echo "  1. Insufficient memory during compilation (OOM killer)"
        echo "  2. Missing CUDA development tools"
        echo "  3. Incompatible CUDA/PyTorch versions"
        echo ""
        echo "Troubleshooting:"
        echo "  - Check available memory: free -h"
        echo "  - Try with fewer parallel jobs: MAX_JOBS=1 bash script/install_dependencies.sh"
        echo "  - Check CUDA installation: nvcc --version"
        exit 1
    fi
fi

echo ""
echo "✓ All dependencies installed successfully!"

