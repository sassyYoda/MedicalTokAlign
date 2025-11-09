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
MEMORY=4096.0
VOCAB_MIN_COUNT=5
VECTOR_SIZE=300
MAX_ITER=15
WINDOW_SIZE=15
BINARY=2
NUM_THREADS=64
X_MAX=10

if hash python 2>/dev/null; then
    PYTHON=python
else
    PYTHON=python3
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
    $BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE || {
        echo "Error: Shuffle failed. Check if cooccurrence file exists and is valid."
        exit 1
    }
fi

echo "$ $BUILDDIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE"
$BUILDDIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE
