#!/bin/sh

# Auto-detect MAIN_DIR from script location
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export MAIN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# GloVe is a sibling directory to MedicalTokAlign
export GLOVE_DIR="$(cd "${MAIN_DIR}/.." && pwd)/GloVe"

export MODLE_PATH1="EleutherAI/pythia-1b"
export TOKENIZER_PATH1="EleutherAI/pythia-1b"
export GLOVE_TRAIN_PATH1="${MAIN_DIR}/data/pretrain-dataset/mix-pythia-glove"
export GLOVE_VECTOR_PATH1="${MAIN_DIR}/data/vec-mix-pythia.txt"

export MODLE_PATH2="microsoft/biogpt"
export TOKENIZER_PATH2="microsoft/biogpt"
export GLOVE_TRAIN_PATH2="${MAIN_DIR}/data/pretrain-dataset/mix-biogpt-glove"
export GLOVE_VECTOR_PATH2="${MAIN_DIR}/data/vec-mix-biogpt.txt"

export TGT_ID_2_SRC_ID_GOLD_PATH="${MAIN_DIR}/data/Vocab_count/biogpt2pythia.json"
# The output path of token alignment matrix
export TGT_ID_2_SRC_ID_RES_PATH="${MAIN_DIR}/data/pythia2biogpt/align_matrix.json"


# Stage-1: train glove vectors
cd ${GLOVE_DIR}
# Export MAIN_DIR so train_glove.sh can use it for Python shuffle script
export MAIN_DIR
GLOVE_VECTOR_NAME1=$(basename ${GLOVE_VECTOR_PATH1})
GLOVE_VECTOR_NAME1="${GLOVE_VECTOR_NAME1%.*}"
printf "\n### Train GloVe vector ${GLOVE_VECTOR_NAME1} with ${GLOVE_TRAIN_PATH1}  ###\n\n"
bash ${MAIN_DIR}/script/train_glove.sh ${GLOVE_TRAIN_PATH1} ${GLOVE_VECTOR_NAME1}

# GloVe creates .txt file (even with binary=2, it creates .txt for the text format)
# Check if file exists before moving
if [ -f "${GLOVE_VECTOR_NAME1}.txt" ]; then
    mv ${GLOVE_VECTOR_NAME1}.txt ${GLOVE_VECTOR_PATH1}
elif [ -f "${GLOVE_VECTOR_NAME1}.bin" ]; then
    echo "Warning: Found .bin file instead of .txt. GloVe may have saved in binary format."
    echo "Converting binary to text format..."
    # If binary format, we'd need to convert it, but for now just report the issue
    echo "Error: Binary format not supported. Please check GloVe training completed successfully."
    exit 1
else
    echo "Error: GloVe output file not found: ${GLOVE_VECTOR_NAME1}.txt or .bin"
    echo "This usually means GloVe training failed (segfault or other error)."
    echo "Check the training output above for errors."
    exit 1
fi

GLOVE_VECTOR_NAME2=$(basename ${GLOVE_VECTOR_PATH2})
GLOVE_VECTOR_NAME2="${GLOVE_VECTOR_NAME2%.*}"
printf "\n### Train GloVe vector ${GLOVE_VECTOR_NAME2} with ${GLOVE_TRAIN_PATH2}  ###\n\n"
bash ${MAIN_DIR}/script/train_glove.sh ${GLOVE_TRAIN_PATH2} ${GLOVE_VECTOR_NAME2}

# GloVe creates .txt file (even with binary=2, it creates .txt for the text format)
# Check if file exists before moving
if [ -f "${GLOVE_VECTOR_NAME2}.txt" ]; then
    mv ${GLOVE_VECTOR_NAME2}.txt ${GLOVE_VECTOR_PATH2}
elif [ -f "${GLOVE_VECTOR_NAME2}.bin" ]; then
    echo "Warning: Found .bin file instead of .txt. GloVe may have saved in binary format."
    echo "Converting binary to text format..."
    # If binary format, we'd need to convert it, but for now just report the issue
    echo "Error: Binary format not supported. Please check GloVe training completed successfully."
    exit 1
else
    echo "Error: GloVe output file not found: ${GLOVE_VECTOR_NAME2}.txt or .bin"
    echo "This usually means GloVe training failed (segfault or other error)."
    echo "Check the training output above for errors."
    exit 1
fi


# Stage-2: token ID align
cd ${MAIN_DIR}

# Check if GloVe vector files exist
if [ ! -f "${GLOVE_VECTOR_PATH1}" ]; then
    echo "Error: GloVe vector file not found: ${GLOVE_VECTOR_PATH1}"
    echo "Please ensure Stage-1 completed successfully."
    exit 1
fi
if [ ! -f "${GLOVE_VECTOR_PATH2}" ]; then
    echo "Error: GloVe vector file not found: ${GLOVE_VECTOR_PATH2}"
    echo "Please ensure Stage-1 completed successfully."
    exit 1
fi

export VOCAB_SIZE1=$($PYTHON src/count_vocab.py -m ${MODLE_PATH1})
export VOCAB_SIZE2=$($PYTHON src/count_vocab.py -m ${MODLE_PATH2})

$PYTHON src/count_dict.py \
    -s ${TOKENIZER_PATH1} \
    -t ${TOKENIZER_PATH2} \
    -o ${TGT_ID_2_SRC_ID_GOLD_PATH}

$PYTHON src/cal_trans_matrix.py \
    -s ${GLOVE_VECTOR_PATH1} \
    -s1 ${VOCAB_SIZE1} \
    -t ${GLOVE_VECTOR_PATH2} \
    -s2 ${VOCAB_SIZE2} \
    -r -n 300 \
    -g ${TGT_ID_2_SRC_ID_GOLD_PATH} \
    -o ${TGT_ID_2_SRC_ID_RES_PATH}
