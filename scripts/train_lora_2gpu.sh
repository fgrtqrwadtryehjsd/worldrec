#!/bin/bash
# 双卡 LoRA 训练 (两张 RTX 3080 10GB, 共 20GB)
# 使用 torchrun 双卡并行
#
# 数据集分布（粗估 token 数）：
#   懂推荐 1-4 : 19204 条, avg~4800, p95~8000, max~10865
#   懂物料 1-7 : 10384 条, avg~150,  p95~200,  max~245
#   懂用户     :  2892 条, avg~7155, p95~11002, max~15441
#
# 显存估算 (0.8B bf16 + LoRA r64 + grad_ckpt, seq=8192, bs=1/卡):
#   ~7-9 GB/卡 → 双卡 3080 10GB 均可用
# effective batch = 2卡 × bs1 × grad_accum8 = 16
#
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
echo "  Effective batch = 2 x 1 x 8 = 16"
echo "============================================"

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
            --max_seq_length 8192 \
            --save_steps   200 \
            --logging_steps 10 \
            --bf16

echo ""
echo "训练完成，输出目录: ${OUTPUT_DIR}/final"
echo "上传万擎平台需要: adapter_model.safetensors + adapter_config.json"
