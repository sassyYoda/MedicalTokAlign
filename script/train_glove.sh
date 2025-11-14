#!/bin/bash
set -e
# args:
# corpus save_file

# Makes programs, downloads sample data, trains a GloVe model, and then evaluates it.
# One optional argument can specify the language used for eval script: matlab, octave or [default] python

# Clean and rebuild to ensure binaries are compiled for current CPU architecture
# This prevents "Illegal instruction" errors when binaries were compiled on a different machine
echo "Cleaning and rebuilding GloVe for current CPU architecture..."
make clean 2>/dev/null || true
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
# Allow MAX_ITER to be overridden via environment variable for testing
MAX_ITER=${MAX_ITER:-15}
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
if ! $BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE; then
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 132 ] || [ $EXIT_CODE -eq 139 ]; then
        # Exit code 132 = SIGILL (Illegal instruction), 139 = SIGSEGV (Segmentation fault)
        echo "Error: Illegal instruction or segfault detected (exit code $EXIT_CODE)."
        echo "This usually means GloVe was compiled on a different CPU."
        echo "Forcing clean rebuild..."
        make clean
        make
        echo "Retrying cooccur..."
        if ! $BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE; then
            echo "Error: cooccur still failing after rebuild. Check CPU compatibility."
            exit 1
        fi
    else
        echo "Error: cooccur failed with exit code $EXIT_CODE"
        exit 1
    fi
fi

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

# Try training with progressively fewer threads to avoid segfaults
# GloVe has known issues with very large datasets and high thread counts
TRAIN_SUCCESS=0
for THREAD_COUNT in $NUM_THREADS $((NUM_THREADS / 2)) $((NUM_THREADS / 4)) 4 2 1; do
    [ $THREAD_COUNT -lt 1 ] && THREAD_COUNT=1
    echo "Attempting training with $THREAD_COUNT thread(s)..."
    if $BUILDDIR/glove -save-file $SAVE_FILE -threads $THREAD_COUNT -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE; then
        TRAIN_SUCCESS=1
        break
    else
        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 139 ]; then
            echo "Segfault with $THREAD_COUNT thread(s). Trying fewer threads..."
            continue
        else
            echo "Training failed with exit code $EXIT_CODE"
            break
        fi
    fi
done

if [ $TRAIN_SUCCESS -eq 0 ]; then
    echo ""
    echo "Error: GloVe training failed with all thread counts."
    echo "This is a known issue with GloVe on very large datasets (400M+ lines)."
    echo ""
    echo "Possible solutions:"
    echo "  1. Use a machine with more RAM"
    echo "  2. Reduce dataset size (use a subset of the cooccurrence file)"
    echo "  3. Try reducing iterations: MAX_ITER=5 bash script/train_glove.sh ..."
    echo "  4. This is a known GloVe bug with very large datasets (400M+ lines)"
    echo ""
    echo "Note: The dataset has 422M+ lines which exceeds GloVe's tested limits."
    echo "You may need to use a subset of the data or patch GloVe's source code."
    echo ""
    echo "Check available memory: free -h"
    exit 1
fi
