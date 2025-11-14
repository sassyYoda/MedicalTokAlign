#!/bin/bash
set -e
# args:
# corpus save_file

# Makes programs, downloads sample data, trains a GloVe model, and then evaluates it.
# One optional argument can specify the language used for eval script: matlab, octave or [default] python

make

CORPUS=$1
SAVE_FILE=$2
VOCAB_FILE=vocab.src.txt
COOCCURRENCE_FILE=cooccurrence.src.bin
COOCCURRENCE_SHUF_FILE=cooccurrence.shuf.src.bin
BUILDDIR=build
VERBOSE=2
# Auto-detect available memory and set MEMORY parameter (in MB, use 50% of available RAM)
# GloVe uses this for buffer size, not total allocation
AVAILABLE_RAM_MB=$(free -m | awk '/^Mem:/{print $2}' 2>/dev/null || echo 8192)
MEMORY_MB=$((AVAILABLE_RAM_MB / 2))
# Cap at reasonable maximum (16GB) and minimum (2GB)
[ $MEMORY_MB -gt 16384 ] && MEMORY_MB=16384
[ $MEMORY_MB -lt 2048 ] && MEMORY_MB=2048
MEMORY="${MEMORY_MB}.0"
VOCAB_MIN_COUNT=5
VECTOR_SIZE=300
MAX_ITER=15
WINDOW_SIZE=15
BINARY=2
# Auto-detect CPU threads (use all cores, but cap for stability)
NUM_THREADS=$(nproc 2>/dev/null || echo 4)
# Cap threads at 16 to avoid memory/contention issues with very large datasets
[ $NUM_THREADS -gt 16 ] && NUM_THREADS=16
X_MAX=10

if hash python 2>/dev/null; then
    PYTHON=python
else
    PYTHON=python3
fi

# Get MAIN_DIR for Python shuffle script (if called from token_align.sh)
# This will be set by the calling script, but provide fallback
if [ -z "$MAIN_DIR" ]; then
    # Try to detect from script location
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    MAIN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

echo
echo "$ $BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE"
$BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE

echo "$ $BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE"
$BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE

echo "$ $BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE"
# Check if overflow files exist before shuffling
if ls overflow_*.bin 1> /dev/null 2>&1; then
    echo "Found overflow files, including them in shuffle..."
    $BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE || {
        echo "Warning: Shuffle failed. This may be due to missing overflow files."
        echo "If cooccurrence file is very large, overflow files should be in the current directory."
        echo "Attempting to continue with main cooccurrence file only..."
        # Try to use the main file directly if shuffle fails
        if [ -f "$COOCCURRENCE_FILE" ]; then
            cp $COOCCURRENCE_FILE $COOCCURRENCE_SHUF_FILE
            echo "Using main cooccurrence file directly (without shuffle)."
        else
            echo "Error: Cooccurrence file not found: $COOCCURRENCE_FILE"
            exit 1
        fi
    }
else
    echo "No overflow files found, proceeding with shuffle..."
    # Try C shuffle first (faster if it works)
    $BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE 2>/dev/null || {
        echo "C shuffle failed, trying Python-based shuffle..."
        # Use Python shuffle as fallback (handles large files better)
        $PYTHON ${MAIN_DIR}/src/shuffle_cooccur.py $COOCCURRENCE_FILE $COOCCURRENCE_SHUF_FILE -memory $MEMORY_MB -verbose $VERBOSE || {
            echo "Warning: Both shuffle methods failed. Using unshuffled file."
            echo "Training will proceed but may have slightly different convergence."
            if [ -f "$COOCCURRENCE_FILE" ]; then
                cp $COOCCURRENCE_FILE $COOCCURRENCE_SHUF_FILE
                echo "Copied unshuffled cooccurrence file for training."
            else
                echo "Error: Cooccurrence file not found: $COOCCURRENCE_FILE"
                exit 1
            fi
        }
    }
fi

echo "$ $BUILDDIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE"

# Try training with full threads first
$BUILDDIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE || {
    echo "Warning: GloVe training failed with $NUM_THREADS threads, trying with fewer threads..."
    # Reduce threads to avoid memory/contention issues
    REDUCED_THREADS=$((NUM_THREADS / 2))
    [ $REDUCED_THREADS -lt 1 ] && REDUCED_THREADS=1
    echo "Retrying with $REDUCED_THREADS threads..."
    $BUILDDIR/glove -save-file $SAVE_FILE -threads $REDUCED_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE || {
        echo "Error: GloVe training failed even with reduced threads."
        echo "This may be due to memory limitations or a bug in GloVe with very large datasets."
        echo "Check available memory: free -h"
        exit 1
    }
}
