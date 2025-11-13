#!/bin/sh

# Auto-detect MAIN_DIR from script location
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export MAIN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd ${MAIN_DIR}

# Auto-detect hardware (lean detection)
GPUNUM=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo 1)
CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPUNUM-1)) 2>/dev/null || echo 0)
export CUDA_VISIBLE_DEVICES
GPU_MEMORY_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 24000)
CPU_CORES=$(nproc 2>/dev/null || echo 4)

# Set batch size based on GPU memory
if [ $GPU_MEMORY_MB -ge 80000 ]; then
    TRAIN_BS=16
elif [ $GPU_MEMORY_MB -ge 40000 ]; then
    TRAIN_BS=8
elif [ $GPU_MEMORY_MB -ge 24000 ]; then
    TRAIN_BS=4
else
    TRAIN_BS=2
fi

# Calculate gradient accumulation (target effective batch = 128)
TARGET_EFFECTIVE_BATCH=128
GRADIENT_ACC=$((TARGET_EFFECTIVE_BATCH / (GPUNUM * TRAIN_BS)))
[ $GRADIENT_ACC -lt 1 ] && GRADIENT_ACC=1

# Set CPU workers (75% of cores, max 64)
NUM_WORKERS=$((CPU_CORES * 3 / 4))
[ $NUM_WORKERS -gt 64 ] && NUM_WORKERS=64
[ $NUM_WORKERS -lt 1 ] && NUM_WORKERS=1

export GPUNUM
export TRAIN_BS
export GRADIENT_ACC
export NUM_WORKERS
export MASTER_PORT=16899

export MODEL="1b"

export TGT="biogpt"

# Use the initialized model from init_model.sh
MODEL_NAME="./data/pythia2${TGT}/TokAlign-Init-1B"

# export DATASET_PATH="./data/pretrain-dataset/pile00-${TGT}-tokenized"
export DATASET_PATH="./data/pretrain-dataset/pubmed-${TGT}-tokenized"

# Generate DeepSpeed config dynamically based on GPU count
CONFIG_FILE="${MAIN_DIR}/data/Deepspeed-Configs/zero3.yaml"
mkdir -p "$(dirname ${CONFIG_FILE})"
cat > ${CONFIG_FILE} << EOF
compute_environment: LOCAL_MACHINE
deepspeed_config:
  deepspeed_multinode_launcher: standard
  offload_optimizer_device: none
  offload_param_device: none
  zero3_init_flag: true
  zero3_save_16bit_model: true
  zero_hpz_partition_size: ${GPUNUM}
  zero_stage: 3
distributed_type: DEEPSPEED
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: ${GPUNUM}
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF
export CONFIG_FILE

export EVAL_BS=1

export BLOCK_SIZE=2048

export SEED=0

export LR=6.4e-4
export NUM_STEPS=2500
export NUM_SAVE_STEPS=2500
export EVAL_STEP=10000
# NUM_WORKERS set by hardware detection above
export LOGGING_STEPS=1

export RESUME=False

export TRAIN_START_IDX=0

export ADD_PARAMETERS=""

PREFIX="${MODEL}/${SEED}_${TGT}_S1"

if [ "${RESUME}" != "False" ];
then
PREFIX="${PREFIX}_resume"
ADD_PARAMETERS="${ADD_PARAMETERS} --resume_from_checkpoint ${RESUME}"
fi

MODEL_DIR="${MAIN_DIR}/log/$PREFIX"
LOG_FILE="${MAIN_DIR}/log/${PREFIX}.log"

mkdir -p $MODEL_DIR


accelerate launch \
    --config_file ${CONFIG_FILE} \
    --main_process_port ${MASTER_PORT} \
    --num_processes ${GPUNUM} \
    --num_machines 1 src/clm_train.py \
    --model_name ${MODEL_NAME} \
    --tokenizer_path ${MODEL_NAME} \
    --dataset_name ${DATASET_PATH} \
    --max_seq_length ${BLOCK_SIZE} \
    --max_steps ${NUM_STEPS} \
    --logging_steps ${LOGGING_STEPS} \
    --save_steps ${NUM_SAVE_STEPS} \
    --num_workers ${NUM_WORKERS} \
    --bf16 True \
    --packing True \
    --output_dir ${MODEL_DIR} \
    --per_device_train_batch_size ${TRAIN_BS} \
    --gradient_accumulation_steps ${GRADIENT_ACC} \
    --use_gradient_checkpointing \
    --learning_rate ${LR}  \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --ignore_data_skip True \
    --train_start_idx ${TRAIN_START_IDX} \
    ${ADD_PARAMETERS} \
    --warmup_ratio 0.03 \
    --finetune_embed_only True \
    --use_flash_attn True 2>&1 >$LOG_FILE

# STAGE-2
MODEL_NAME="./$MODEL_DIR/checkpoint-$NUM_STEPS"
LR=5e-5
export TRAIN_START_IDX=2560000

export ADD_PARAMETERS=""

PREFIX="${MODEL}/${SEED}_${TGT}_S2"

MODEL_DIR="${MAIN_DIR}/log/$PREFIX"
LOG_FILE="${MAIN_DIR}/log/${PREFIX}.log"

mkdir -p $MODEL_DIR

accelerate launch \
    --config_file ${CONFIG_FILE} \
    --main_process_port ${MASTER_PORT} \
    --num_processes ${GPUNUM} \
    --num_machines 1 src/clm_train.py \
    --model_name ${MODEL_NAME} \
    --tokenizer_path ${MODEL_NAME} \
    --dataset_name ${DATASET_PATH} \
    --max_seq_length ${BLOCK_SIZE} \
    --max_steps ${NUM_STEPS} \
    --logging_steps ${LOGGING_STEPS} \
    --save_steps ${NUM_SAVE_STEPS} \
    --num_workers ${NUM_WORKERS} \
    --bf16 True \
    --packing True \
    --output_dir ${MODEL_DIR} \
    --per_device_train_batch_size ${TRAIN_BS} \
    --gradient_accumulation_steps ${GRADIENT_ACC} \
    --use_gradient_checkpointing \
    --learning_rate ${LR}  \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --ignore_data_skip True \
    --train_start_idx ${TRAIN_START_IDX} \
    ${ADD_PARAMETERS} \
    --warmup_ratio 0.03 \
    --use_flash_attn True 2>&1 >$LOG_FILE
  
