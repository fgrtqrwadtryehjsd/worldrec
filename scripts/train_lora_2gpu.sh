#!/bin/bash
# 双卡 LoRA 训练 (两张 RTX 3080 10GB, 共 20GB)
# 使用 accelerate + DeepSpeed ZeRO Stage 2
# 用法：bash scripts/train_lora_2gpu.sh [MODEL_DIR] [DATA_DIR] [OUTPUT_DIR]
set -e

MODEL_DIR="${1:-$HOME/models/OneReason-0.8B-pretrain-competition}"
DATA_DIR="${2:-dataset}"
OUTPUT_DIR="${3:-outputs/lora_2gpu_v1}"

echo "============================================"
echo "  双卡 LoRA 训练 (2x RTX 3080 10GB)"
echo "  Model : ${MODEL_DIR}"
echo "  Data  : ${DATA_DIR}"
echo "  Output: ${OUTPUT_DIR}"
echo "============================================"

# 双卡用 torchrun 启动，ZeRO2 分担显存
# batch_size=1, grad_accum=8 => effective batch = 2*1*8 = 16
conda run -n worldrec --no-capture-output \
    torchrun \
        --nproc_per_node=2 \
        --master_port=29500 \
        src/train_sft.py \
            --model_path   "${MODEL_DIR}" \
            --data_dir     "${DATA_DIR}" \
            --output_dir   "${OUTPUT_DIR}" \
            --lora_r       64 \
            --lora_alpha   128 \
            --lora_dropout 0.05 \
            --learning_rate 2e-4 \
            --num_epochs   3 \
            --batch_size   1 \
            --gradient_accumulation_steps 8 \
            --max_seq_length 2048 \
            --save_steps   200 \
            --logging_steps 10 \
            --bf16

echo ""
echo "训练完成，输出目录: ${OUTPUT_DIR}/final"
