#!/bin/bash
# Diagnostic script to check Lambda image compatibility

echo "=== Environment Diagnostics ==="
echo

echo "1. System Information:"
echo "   OS: $(uname -a)"
echo "   Available RAM: $(free -h | grep Mem | awk '{print $2}')"
echo "   Available Disk: $(df -h / | tail -1 | awk '{print $4}')"
echo

echo "2. Python/PyTorch Versions:"
python3 --version 2>/dev/null || echo "   Python not found"
python3 -c "import torch; print(f'   PyTorch: {torch.__version__}')" 2>/dev/null || echo "   PyTorch not installed"
python3 -c "import torch; print(f'   CUDA available: {torch.cuda.is_available()}')" 2>/dev/null || echo "   CUDA check failed"
if python3 -c "import torch" 2>/dev/null; then
    python3 -c "import torch; print(f'   CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')" 2>/dev/null
fi
echo

echo "3. Flash Attention Status:"
if python3 -c "import flash_attn" 2>/dev/null; then
    echo "   ✓ flash-attn imported successfully"
    python3 -c "import flash_attn; print(f'   Version: {flash_attn.__version__ if hasattr(flash_attn, \"__version__\") else \"unknown\"}')" 2>/dev/null
else
    echo "   ✗ flash-attn import failed"
    python3 -c "import flash_attn" 2>&1 | head -3
fi
echo

echo "4. GloVe Compilation Check:"
if [ -d "../GloVe" ]; then
    echo "   GloVe directory found"
    if [ -f "../GloVe/build/shuffle" ]; then
        echo "   ✓ shuffle binary exists"
        file "../GloVe/build/shuffle" 2>/dev/null || echo "   (file command not available)"
    else
        echo "   ✗ shuffle binary not found - needs compilation"
    fi
else
    echo "   ✗ GloVe directory not found at ../GloVe"
fi
echo

echo "5. Transformers Library:"
python3 -c "import transformers; print(f'   Version: {transformers.__version__}')" 2>/dev/null || echo "   transformers not installed"
echo

echo "=== Recommendations ==="
echo "If flash-attn fails:"
echo "  1. Reinstall: pip uninstall flash-attn && pip install flash-attn --no-build-isolation"
echo "  2. Or recompile: pip install flash-attn --no-build-isolation --no-binary flash-attn"
echo
echo "If GloVe memory issues persist:"
echo "  1. Ensure GloVe is freshly compiled: cd ../GloVe && make clean && make"
echo "  2. Check if memory parameter is being passed correctly (see train_glove.sh output)"
echo

