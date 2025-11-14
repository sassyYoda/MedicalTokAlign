#!/bin/bash
# Patch GloVe shuffle.c to fix integer overflow bug with large datasets

GLOVE_DIR="${1:-../GloVe}"
SHUFFLE_C="${GLOVE_DIR}/src/shuffle.c"

if [ ! -f "$SHUFFLE_C" ]; then
    echo "Error: shuffle.c not found at $SHUFFLE_C"
    echo "Usage: $0 [GLOVE_DIR]"
    exit 1
fi

echo "Patching shuffle.c to fix integer overflow bug..."

# Create backup
cp "$SHUFFLE_C" "${SHUFFLE_C}.backup"
echo "Created backup: ${SHUFFLE_C}.backup"

# Patch the array_size calculation to prevent overflow
python3 << PYTHON_PATCH "$SHUFFLE_C"
import re
import sys

shuffle_c_path = sys.argv[1]

with open(shuffle_c_path, 'r') as f:
    content = f.read()

# Find the exact problematic line from the code provided
# array_size = (long long) (0.95 * (real)memory_limit * 1073741824/(sizeof(CREC)));
pattern = r'array_size\s*=\s*\(long\s+long\)\s*\(\s*0\.95\s*\*\s*\(real\)memory_limit\s*\*\s*1073741824/\(sizeof\(CREC\)\)\s*\);'

replacement = '''    {
        // Fix: Use proper long long casting to prevent overflow
        // Calculate in steps to avoid integer overflow with large memory_limit values
        long long bytes_per_gb = 1073741824LL;
        real bytes_available_real = 0.95 * memory_limit * (real)bytes_per_gb;
        long long bytes_available = (long long)bytes_available_real;
        long long sizeof_crec = (long long)sizeof(CREC);
        long long calculated_size = bytes_available / sizeof_crec;
        
        // Cap at 500M records (~6GB) to prevent segfault on very large datasets
        // This prevents the integer overflow bug that causes array_size to be ~973GB
        if (calculated_size > 500000000LL) {
            array_size = 500000000LL;
            if (verbose > 0) fprintf(stderr, "Warning: array_size capped at 500M records (%.2f GB) to prevent overflow\\n", 
                    (double)(array_size * sizeof_crec) / (double)bytes_per_gb);
        } else if (calculated_size < 1000000LL) {
            array_size = 1000000LL; // Minimum 1 million records
        } else {
            array_size = calculated_size;
        }
    }'''

if re.search(pattern, content):
    content = re.sub(pattern, replacement, content)
    with open(shuffle_c_path, 'w') as f:
        f.write(content)
    print("✓ Patched array_size calculation in shuffle.c")
    sys.exit(0)
else:
    # Try a more flexible pattern that handles whitespace variations
    pattern2 = r'array_size\s*=\s*\(long\s+long\)\s*\([^)]*0\.95[^)]*1073741824[^)]*sizeof\(CREC\)[^)]*\);'
    if re.search(pattern2, content):
        content = re.sub(pattern2, replacement, content)
        with open(shuffle_c_path, 'w') as f:
            f.write(content)
        print("✓ Patched array_size calculation in shuffle.c (flexible match)")
        sys.exit(0)
    else:
        print("⚠ Could not find array_size calculation line")
        print("Searching for the line...")
        # Show context around potential matches
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'array_size' in line and ('1073741824' in line or 'memory_limit' in line):
                start = max(0, i-2)
                end = min(len(lines), i+3)
                print(f"Found potential match around line {i+1}:")
                for j in range(start, end):
                    marker = ">>>" if j == i else "   "
                    print(f"{marker} {j+1}: {lines[j]}")
        sys.exit(1)
PYTHON_PATCH

if [ $? -eq 0 ]; then
    echo "Patch applied successfully!"
    echo "Recompiling GloVe..."
    cd "$GLOVE_DIR"
    make clean
    make
    if [ $? -eq 0 ]; then
        echo "✓ GloVe recompiled successfully with shuffle.c fix!"
    else
        echo "✗ Compilation failed. Restoring backup..."
        cp "${SHUFFLE_C}.backup" "$SHUFFLE_C"
        exit 1
    fi
else
    echo "✗ Patch failed. File unchanged."
    exit 1
fi

