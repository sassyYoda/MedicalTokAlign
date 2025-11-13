#!/bin/sh

# Auto-detect MAIN_DIR from script location
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export MAIN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd ${MAIN_DIR}
export CACHE_DIR="${MAIN_DIR}/data/cache"

export MODLE_PATH="./data/pythia2biogpt/TokAlign-Init-1B"
export TOKENIZER_PATH="./data/pythia2biogpt/TokAlign-Init-1B"

# export TRAIN_FILE="./data/pretrain-corpus/pile00.json"
export TRAIN_FILE="./data/pretrain-corpus/pubmed-corpus.json"

# export DATASET_PATH="./data/pretrain-dataset/pile00-biogpt-tokenized"
export DATASET_PATH="./data/pretrain-dataset/pubmed-biogpt-tokenized"

export NUM_WORKERS=60
export BLOCK_SIZE=2048

# HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1  # Commented out to allow model downloads if needed

python -u src/process_dataset.py \
  --model_name_or_path ${MODLE_PATH} \
  --tokenizer_name ${TOKENIZER_PATH} \
  --train_file ${TRAIN_FILE} \
  --cache_dir ${CACHE_DIR} \
  --dataset_path_in_disk ${DATASET_PATH} \
  --preprocessing_num_workers ${NUM_WORKERS} \
  --block_size ${BLOCK_SIZE} \
  --output_dir ./log 2>&1 | tee ./log/process_dataset.log