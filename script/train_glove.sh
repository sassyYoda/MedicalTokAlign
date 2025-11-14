#!/bin/bash
set -e
# args:
# corpus save_file

# Makes programs, downloads sample data, trains a GloVe model, and then evaluates it.
# One optional argument can specify the language used for eval script: matlab, octave or [default] python

# Patch shuffle.c to fix integer overflow bug (if not already patched)
# This is the only essential fix needed for large datasets
CURRENT_DIR=$(pwd)
SHUFFLE_C="${CURRENT_DIR}/src/shuffle.c"
if [ -f "$SHUFFLE_C" ] && ! grep -q "capped at 500M records" "$SHUFFLE_C" 2>/dev/null; then
    echo "Patching shuffle.c to fix integer overflow bug..."
    python3 << PYTHON_PATCH
import re
import sys

shuffle_c_path = "$SHUFFLE_C"
with open(shuffle_c_path, 'r') as f:
    content = f.read()

# Find: array_size = (long long) (0.95 * (real)memory_limit * 1073741824/(sizeof(CREC)));
patterns = [
    r'array_size\s*=\s*\(long\s+long\)\s*\(\s*0\.95\s*\*\s*\(real\)memory_limit\s*\*\s*1073741824/\(sizeof\(CREC\)\)\s*\);',
    r'array_size\s*=\s*\(long\s+long\)\s*\(0\.95\s*\*\s*\(real\)memory_limit\s*\*\s*1073741824/\(sizeof\(CREC\)\)\);',
    r'array_size\s*=\s*\(long\s+long\)\s*\([^)]*0\.95[^)]*memory_limit[^)]*1073741824[^)]*sizeof\(CREC\)[^)]*\);',
]

replacement = '''    {
        // Fix: Use proper long long casting to prevent overflow
        long long bytes_per_gb = 1073741824LL;
        real bytes_available_real = 0.95 * memory_limit * (real)bytes_per_gb;
        long long bytes_available = (long long)bytes_available_real;
        long long sizeof_crec = (long long)sizeof(CREC);
        long long calculated_size = bytes_available / sizeof_crec;
        
        // Cap at 500M records (~6GB) to prevent segfault on very large datasets
        if (calculated_size > 500000000LL) {
            array_size = 500000000LL;
            if (verbose > 0) fprintf(stderr, "Warning: array_size capped at 500M records (%.2f GB) to prevent overflow\\n", (double)(array_size * sizeof_crec) / (double)bytes_per_gb);
        } else if (calculated_size < 1000000LL) {
            array_size = 1000000LL;
        } else {
            array_size = calculated_size;
        }
    }'''

patched = False
for pattern in patterns:
    if re.search(pattern, content):
        content = re.sub(pattern, replacement, content)
        with open(shuffle_c_path, 'w') as f:
            f.write(content)
        print("✓ Patched shuffle.c")
        patched = True
        break

if not patched:
    # Manual line-by-line search
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'array_size' in line and '1073741824' in line:
            indent = len(line) - len(line.lstrip())
            indent_str = ' ' * indent
            replacement_lines = replacement.strip().split('\n')
            indented_replacement = '\n'.join([indent_str + l if l.strip() else l for l in replacement_lines])
            lines[i] = indented_replacement
            with open(shuffle_c_path, 'w') as f:
                f.write('\n'.join(lines))
            print("✓ Manually patched shuffle.c")
            patched = True
            break
PYTHON_PATCH
fi

make

CORPUS=$1
SAVE_FILE=$2
VOCAB_FILE=vocab.src.txt
COOCCURRENCE_FILE=cooccurrence.src.bin
COOCCURRENCE_SHUF_FILE=cooccurrence.shuf.src.bin
BUILDDIR=build
VERBOSE=2
MEMORY=1536.0
VOCAB_MIN_COUNT=5
VECTOR_SIZE=300
MAX_ITER=15
WINDOW_SIZE=15
BINARY=2
NUM_THREADS=64
X_MAX=10

echo
echo "$ $BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE"
$BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE

echo "$ $BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE"
$BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE

echo "$ $BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE"
$BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE

echo "$ $BUILDDIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE"
$BUILDDIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE
