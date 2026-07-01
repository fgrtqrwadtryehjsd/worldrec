#!/bin/bash
# 单卡 LoRA 训练 (单张 3080 10GB)
# 用法：bash scripts/train_lora.sh [MODEL_DIR] [DATA_DIR] [OUTPUT_DIR]
set -e

MODEL_DIR="${1:-$HOME/models/OneReason-0.8B-pretrain-competition}"
DATA_DIR="${2:-dataset}"
OUTPUT_DIR="${3:-outputs/lora_v1}"

echo "============================================"
echo "  单卡 LoRA 训练 (RTX 3080 10GB)"
echo "  Model : ${MODEL_DIR}"
echo "  Data  : ${DATA_DIR}"
echo "  Output: ${OUTPUT_DIR}"
echo "============================================"

conda run -n worldrec --no-capture-output \
    python src/train_sft.py \
        --model_path   "${MODEL_DIR}" \
        --data_dir     "${DATA_DIR}" \
        --output_dir   "${OUTPUT_DIR}" \
        --lora_r       64 \
        --lora_alpha   128 \
        --lora_dropout 0.05 \
        --learning_rate 2e-4 \
        --num_epochs   3 \
        --batch_size   1 \
        --gradient_accumulation_steps 16 \
        --max_seq_length 2048 \
        --save_steps   200 \
        --logging_steps 10 \
        --bf16

echo ""
echo "训练完成，输出目录: ${OUTPUT_DIR}/final"
