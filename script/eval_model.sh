#!/bin/sh

# Auto-detect MAIN_DIR from script location
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export MAIN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd ${MAIN_DIR}

# Path to trained model (after Stage 2)
export MODEL_PATH="${MAIN_DIR}/log/1b/0_biogpt_S2/checkpoint-2500"

# Path to evaluation dataset (optional - for perplexity)
export EVAL_DATASET_PATH="${MAIN_DIR}/data/pretrain-dataset/pubmed-biogpt-tokenized"

# Number of samples for perplexity evaluation
export NUM_SAMPLES=100

# Create prompts file for medical domain evaluation
cat > /tmp/medical_prompts.json << 'EOF'
[
  "The patient presented with",
  "Treatment options for",
  "The mechanism of action involves",
  "Clinical trials have shown that",
  "The diagnosis was confirmed by",
  "The symptoms include",
  "Risk factors for",
  "The prognosis is",
  "Side effects may include",
  "The recommended dosage is"
]
EOF

# Run evaluation
python src/eval_model.py \
    --model_path ${MODEL_PATH} \
    --dataset_path ${EVAL_DATASET_PATH} \
    --num_samples ${NUM_SAMPLES} \
    --prompts_file /tmp/medical_prompts.json \
    --device cuda

