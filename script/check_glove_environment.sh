#!/bin/bash
# Check environment differences that might cause GloVe segfaults

echo "=== Environment Check for GloVe Segfault Debugging ==="
echo ""

echo "1. Stack size limit:"
ulimit -s
echo ""

echo "2. Memory limits:"
ulimit -a | grep -E "(virtual|data|stack)"
echo ""

echo "3. Available memory:"
free -h
echo ""

echo "4. GCC version:"
gcc --version 2>/dev/null || echo "GCC not found"
echo ""

echo "5. GloVe Makefile CFLAGS:"
if [ -f "Makefile" ]; then
    grep -E "CFLAGS\s*[+=]" Makefile | head -3
else
    echo "Makefile not found in current directory"
fi
echo ""

echo "6. Check if GloVe was compiled with Makefile fix:"
if [ -f "build/glove" ]; then
    echo "Checking compilation flags used:"
    strings build/glove | grep -E "(-O[0-3]|-funroll|-march)" | head -5 || echo "Could not extract flags"
else
    echo "GloVe binary not found"
fi
echo ""

echo "7. System architecture:"
uname -m
lscpu | grep -E "(Model name|Architecture|CPU op-mode)" | head -3
echo ""

echo "=== Recommendations ==="
STACK_SIZE=$(ulimit -s)
if [ "$STACK_SIZE" != "unlimited" ] && [ "$STACK_SIZE" -lt 8192 ]; then
    echo "⚠ Stack size is ${STACK_SIZE}KB - may be too small for large datasets"
    echo "  Try: ulimit -s unlimited"
fi

