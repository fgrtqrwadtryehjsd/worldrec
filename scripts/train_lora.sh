#!/bin/bash
# 单卡 LoRA 训练 (单张 RTX 3080 10GB)
#
# 数据集分布（粗估 token 数）：
#   懂推荐 1-4 : 19204 条, avg~4800, p95~8000, max~10865
#   懂物料 1-7 : 10384 条, avg~150,  p95~200,  max~245
#   懂用户     :  2892 条, avg~7155, p95~11002, max~15441
#
# 显存估算 (0.8B bf16 + LoRA r64 + grad_ckpt, seq=8192, bs=1):
#   ~7-9 GB → RTX 3080 10GB 可用
#
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

# 懂用户 avg~7155 tokens，懂物料极短 avg~150
# 上调懂用户权重避免被短样本淹没；懂推荐本身占比最多权重保持 1.0
SAMPLE_WEIGHTS='{"懂推荐": 1.0, "懂物料": 1.5, "懂用户": 3.0}'

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
        --max_seq_length 8192 \
        --save_steps   200 \
        --logging_steps 10 \
        --bf16

echo ""
echo "训练完成，输出目录: ${OUTPUT_DIR}/final"
echo "上传万擎平台需要: adapter_model.safetensors + adapter_config.json"
