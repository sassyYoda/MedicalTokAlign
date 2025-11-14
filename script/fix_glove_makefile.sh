#!/bin/bash
# Fix GloVe Makefile to use safer compilation flags that prevent segfaults with large datasets

GLOVE_DIR="${1:-../GloVe}"
MAKEFILE="${GLOVE_DIR}/Makefile"

if [ ! -f "$MAKEFILE" ]; then
    echo "Error: Makefile not found at $MAKEFILE"
    echo "Usage: $0 [GLOVE_DIR]"
    exit 1
fi

echo "Fixing GloVe Makefile compilation flags..."

# Create backup
cp "$MAKEFILE" "${MAKEFILE}.backup"
echo "Created backup: ${MAKEFILE}.backup"

# Fix 1: Change -O3 to -O2 (safer optimization, less likely to cause bugs)
# Fix 2: Remove -funroll-loops (can cause issues with large loops)
# Fix 3: Keep -march=native but ensure it's safe

python3 << PYTHON_FIX "$MAKEFILE"
import re
import sys

makefile_path = sys.argv[1]

with open(makefile_path, 'r') as f:
    content = f.read()

# Pattern to find CFLAGS line
# Common patterns:
# CFLAGS = -O3 -march=native -funroll-loops ...
# or
# CFLAGS += -O3 ...

changes_made = []

# Replace -O3 with -O2 (safer optimization)
# Also handle -Ofast which is even more aggressive
# Need to be careful with variable expansion like $(CPU_ARCHITECTURE_FLAGS)
if re.search(r'-O3', content):
    # Replace -O3 anywhere in CFLAGS line
    content = re.sub(r'(\s|^)-O3(\s|$)', r'\1-O2\2', content)
    changes_made.append("Changed -O3 to -O2")
if re.search(r'-Ofast', content):
    content = re.sub(r'(\s|^)-Ofast(\s|$)', r'\1-O2\2', content)
    changes_made.append("Changed -Ofast to -O2")

# Remove -funroll-loops (can cause issues with large loops)
if re.search(r'-funroll-loops', content):
    # Handle various patterns: -funroll-loops, $(VAR)-funroll-loops, etc.
    content = re.sub(r'\s+-funroll-loops\s+', ' ', content)
    content = re.sub(r'-funroll-loops\s+', '', content)
    content = re.sub(r'\s+-funroll-loops', '', content)
    changes_made.append("Removed -funroll-loops")

# Ensure we're using safe flags
# Add -fno-strict-aliasing if not present (helps with memory safety)
if not re.search(r'-fno-strict-aliasing', content):
    # Find CFLAGS line and add the flag after -pthread or at the start
    if re.search(r'CFLAGS\s*=', content):
        # Add after -pthread if present, otherwise at start
        content = re.sub(r'(CFLAGS\s*=\s*-pthread\s+)', r'\1-fno-strict-aliasing ', content)
        if not re.search(r'-fno-strict-aliasing', content):
            # If -pthread not found, add at start
            content = re.sub(r'(CFLAGS\s*=\s*)', r'\1-fno-strict-aliasing ', content)
        changes_made.append("Added -fno-strict-aliasing for memory safety")

if changes_made:
    with open(makefile_path, 'w') as f:
        f.write(content)
    print("✓ Makefile fixed:")
    for change in changes_made:
        print(f"  - {change}")
    sys.exit(0)
else:
    print("⚠ No changes needed - Makefile may already be optimized")
    sys.exit(0)
PYTHON_FIX

if [ $? -eq 0 ]; then
    echo ""
    echo "Verifying Makefile changes..."
    if grep -q "-O2" "$MAKEFILE" && ! grep -qE "\-O3|\-Ofast" "$MAKEFILE"; then
        echo "✓ Makefile verified: using -O2 (not -O3)"
    else
        echo "⚠ Warning: Makefile may still have -O3. Checking..."
        grep "CFLAGS" "$MAKEFILE" | head -1
    fi
    
    if ! grep -q "-funroll-loops" "$MAKEFILE"; then
        echo "✓ Makefile verified: -funroll-loops removed"
    else
        echo "⚠ Warning: Makefile may still have -funroll-loops"
    fi
    
    echo ""
    echo "Recompiling GloVe with safer flags..."
    cd "$GLOVE_DIR"
    make clean
    make
    if [ $? -eq 0 ]; then
        echo "✓ GloVe recompiled successfully with safer flags!"
    else
        echo "✗ Compilation failed. Restoring backup..."
        cp "${MAKEFILE}.backup" "$MAKEFILE"
        exit 1
    fi
else
    echo "✗ Failed to fix Makefile"
    exit 1
fi

