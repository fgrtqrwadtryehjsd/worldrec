#!/bin/bash
# Exp03: 双卡 LoRA 训练 v3 (2x RTX 3090 24GB)
# 变更 vs Exp02:
#   - 数据: 包含全部 _ext 扩展数据 (dataset_v3/)
#   - max_seq_length: 4096 → 8192 (恢复懂用户长序列)
#   - 取消 --filter_long_samples (允许超长数据截断，而不是整条抛弃)
#   - 保持良好的防过拟合超参数 (r=32, lr=1e-4, dropout=0.1, epochs=2)
#
set -e

MODEL_DIR="${1:-$HOME/models/OneReason-0.8B-pretrain-competition}"
OUTPUT_DIR="${2:-outputs/exp03_lora_ext}"

echo "============================================"
echo "  Exp03: 双卡 LoRA + 全量扩展数据 + 长序列"
echo "  Model : ${MODEL_DIR}"
echo "  Train : dataset_v3/train.jsonl"
echo "  Output: ${OUTPUT_DIR}"
echo "  max_seq_length = 8192 (无硬过滤)"
echo "============================================"

# 继续保持这个采样配重比例，为了挽救懂世界指标，将懂世界的权重从 1.0 提升到 2.0
SAMPLE_WEIGHTS='{"懂推荐": 2.0, "懂物料": 1.0, "懂用户": 5.0, "懂世界": 2.0}'

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

conda run -n worldrec --no-capture-output \
    torchrun \
        --nproc_per_node=2 \
        --master_port=29500 \
        src/train_sft.py \
            --model_path   "${MODEL_DIR}" \
            --train_file   "dataset_v3/train.jsonl" \
            --eval_file    "dataset_v3/eval.jsonl" \
            --eval_steps   500 \
            --output_dir   "${OUTPUT_DIR}" \
            --lora_r       32 \
            --lora_alpha   64 \
            --lora_dropout 0.1 \
            --learning_rate 1e-4 \
            --num_epochs   2 \
            --batch_size   1 \
            --gradient_accumulation_steps 8 \
            --max_seq_length 8192 \
            --sample_weights "${SAMPLE_WEIGHTS}" \
            --save_steps   500 \
            --logging_steps 10 \
            --bf16

echo "Exp03 训练启动完毕"
