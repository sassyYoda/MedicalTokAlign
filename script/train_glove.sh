#!/bin/bash
set -e
# args:
# corpus save_file

# Makes programs, downloads sample data, trains a GloVe model, and then evaluates it.
# One optional argument can specify the language used for eval script: matlab, octave or [default] python

# Clean and rebuild to ensure binaries are compiled for current CPU architecture
# This prevents "Illegal instruction" errors when binaries were compiled on a different machine
echo "Cleaning and rebuilding GloVe for current CPU architecture..."

# Check environment and increase stack size if needed (GloVe needs large stack for big datasets)
# This is CRITICAL - small stack size causes segfaults with large datasets
STACK_SIZE=$(ulimit -s 2>/dev/null || echo "unknown")
if [ "$STACK_SIZE" != "unlimited" ] && [ "$STACK_SIZE" != "unknown" ]; then
    if [ "$STACK_SIZE" -lt 32768 ]; then
        echo "⚠ Stack size is only ${STACK_SIZE}KB - too small for large datasets!"
        echo "Increasing stack size limit to unlimited..."
        ulimit -s unlimited 2>/dev/null || ulimit -s 65536 2>/dev/null || {
            echo "Warning: Could not increase stack size. This may cause segfaults."
            echo "Try running: ulimit -s unlimited"
        }
    else
        echo "Stack size: ${STACK_SIZE}KB (OK)"
    fi
else
    echo "Stack size: unlimited (OK)"
fi

# Fix Makefile compilation flags if needed (prevents segfaults with large datasets)
# Get current directory (should be GloVe directory when called from token_align.sh)
CURRENT_DIR=$(pwd)
if [ -f "${MAIN_DIR}/script/fix_glove_makefile.sh" ]; then
    echo "Checking and fixing GloVe Makefile compilation flags..."
    bash ${MAIN_DIR}/script/fix_glove_makefile.sh "$CURRENT_DIR" || {
        echo "Warning: Makefile fix script failed or had issues"
        echo "You may need to manually edit the Makefile to change -O3 to -O2 and remove -funroll-loops"
    }
elif [ -f "./script/fix_glove_makefile.sh" ]; then
    # Fallback if MAIN_DIR not set
    echo "Checking and fixing GloVe Makefile compilation flags..."
    bash ./script/fix_glove_makefile.sh "$CURRENT_DIR" || {
        echo "Warning: Makefile fix script failed or had issues"
    }
fi

# Patch shuffle.c to fix integer overflow bug
if [ -f "${MAIN_DIR}/script/patch_glove_shuffle.c.sh" ]; then
    echo "Patching shuffle.c to fix integer overflow bug..."
    bash ${MAIN_DIR}/script/patch_glove_shuffle.c.sh "$CURRENT_DIR" 2>/dev/null || {
        echo "Note: shuffle.c patch may have already been applied or failed"
    }
elif [ -f "./script/patch_glove_shuffle.c.sh" ]; then
    echo "Patching shuffle.c to fix integer overflow bug..."
    bash ./script/patch_glove_shuffle.c.sh "$CURRENT_DIR" 2>/dev/null || true
fi

# Verify and fix Makefile - do this manually to ensure it works
if [ -f "Makefile" ]; then
    # Check if Makefile needs fixing
    NEEDS_FIX=0
    if grep -qE "CFLAGS.*-O3" Makefile || grep -qE "CFLAGS.*-Ofast" Makefile; then
        NEEDS_FIX=1
    fi
    if grep -qE "CFLAGS.*-funroll-loops" Makefile; then
        NEEDS_FIX=1
    fi
    
    if [ "$NEEDS_FIX" = "1" ]; then
        echo "Fixing Makefile compilation flags..."
        
        # If GLOVE_USE_O0 is set, use -O0 (no optimization) for maximum stability
        if [ "${GLOVE_USE_O0:-0}" = "1" ]; then
            echo "Using -O0 (no optimization) for maximum stability..."
            sed -i 's/-O[0-9]/ -O0 /g' Makefile
            sed -i 's/-Ofast/ -O0 /g' Makefile
        else
            # Otherwise use -O2 (safer than -O3)
            echo "Changing -O3/-Ofast to -O2..."
            sed -i 's/-O3/ -O2 /g' Makefile
            sed -i 's/-Ofast/ -O2 /g' Makefile
        fi
        
        # Remove -funroll-loops (causes issues with large loops)
        echo "Removing -funroll-loops..."
        sed -i 's/-funroll-loops//g' Makefile
        
        # Clean up any double spaces
        sed -i 's/  */ /g' Makefile
        
        echo "✓ Makefile fixed"
        echo "Verifying:"
        grep "^CFLAGS" Makefile | head -1
    else
        echo "✓ Makefile already uses safe compilation flags"
    fi
fi

make clean 2>/dev/null || true
echo "Compiling GloVe..."
make 2>&1 | tail -5

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
    # Run with timeout to catch segfaults, and try to get more info
    if timeout 300 $BUILDDIR/glove -save-file $SAVE_FILE -threads $THREAD_COUNT -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE 2>&1; then
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
    echo ""
    echo "Since this worked on RunPod, the issue is likely environment-specific."
    echo ""
    echo "Try these solutions in order:"
    echo ""
    echo "1. Recompile with no optimization (most likely to work):"
    echo "   cd /path/to/GloVe"
    echo "   Edit Makefile: change -O2/-O3 to -O0"
    echo "   make clean && make"
    echo "   Then run token_align.sh again"
    echo ""
    echo "2. Or set environment variable to force -O0:"
    echo "   GLOVE_USE_O0=1 bash script/token_align.sh"
    echo ""
    echo "3. Check if GloVe version differs from RunPod:"
    echo "   cd /path/to/GloVe && git log --oneline -5"
    echo ""
    echo "4. Get a backtrace with gdb to see exact crash location:"
    echo "   cd /path/to/GloVe"
    echo "   gdb --batch --ex run --ex bt --args build/glove -save-file test -threads 1 -input-file cooccurrence.shuf.src.bin -x-max 10 -iter 1 -vector-size 300 -binary 2 -vocab-file vocab.src.txt -verbose 2"
    echo ""
    echo "5. Check system differences (libraries, compiler version):"
    echo "   bash script/check_glove_environment.sh"
    echo ""
    exit 1
fi
