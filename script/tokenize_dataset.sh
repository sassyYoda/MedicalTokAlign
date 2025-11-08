#!/bin/sh

export MAIN_DIR="/path/2/TokAlign/"
cd ${MAIN_DIR}
export CACHE_DIR="${MAIN_DIR}/data/cache"

export MODLE_PATH="./data/pythia2biogpt/TokAlign-Init-1B"
export TOKENIZER_PATH="./data/pythia2biogpt/TokAlign-Init-1B"

# Use HuggingFace dataset directly
export DATASET_NAME="uiyunkim-hub/pubmed-abstract"

# export DATASET_PATH="./data/pretrain-dataset/pile00-biogpt-tokenized"
export DATASET_PATH="./data/pretrain-dataset/pubmed-biogpt-tokenized"

export NUM_WORKERS=60
export BLOCK_SIZE=2048

python -u src/process_dataset.py \
  --model_name_or_path ${MODLE_PATH} \
  --tokenizer_name ${TOKENIZER_PATH} \
  --dataset_name ${DATASET_NAME} \
  --cache_dir ${CACHE_DIR} \
  --dataset_path_in_disk ${DATASET_PATH} \
  --preprocessing_num_workers ${NUM_WORKERS} \
  --block_size ${BLOCK_SIZE} \
  --output_dir ./log 2>&1 | tee ./log/process_dataset.log