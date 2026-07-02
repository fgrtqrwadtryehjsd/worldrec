#!/bin/bash
# Exp02: 双卡 LoRA 训练 v2 (2x RTX 3090 24GB)
# 变更 vs Exp01:
#   - 数据: 原始 dataset/ + 懂世界抽样30000条 (dataset_v2/train.jsonl)
#   - 新增 eval: dataset_v2/eval.jsonl, 每500步评估
#   - max_seq_length 8192 → 4096 + filter_long_samples (提速2x, 过滤超长)
#   - lora_r 64 → 32, lora_dropout 0.05 → 0.1 (防过拟合)
#   - learning_rate 2e-4 → 1e-4 (防遗忘)
#   - num_epochs 3 → 2 (Exp01 loss=0.32 过拟合)
#   - weight_decay 0.01 → 0.05 (L2正则防遗忘)
#   - sample_weights: 加入懂世界, 按baseline建议 懂用户5×/懂推荐2×/懂物料1×/懂世界1×
#   - flash-attn 已安装, auto检测启用
#
# 数据集 (dataset_v2/):
#   train: 53551 条 (懂推荐14240 + 懂物料9865 + 懂用户946 + 懂世界28500)
#   eval:  2817 条 (5% held-out)
#
# 预计: ~2.8s/it, 2 epochs × 3347 steps = 6694 steps, ~5.2h
#
# 用法：bash scripts/train_lora_2gpu_3090_v2.sh [MODEL_DIR] [OUTPUT_DIR]
set -e

MODEL_DIR="${1:-$HOME/models/OneReason-0.8B-pretrain-competition}"
OUTPUT_DIR="${2:-outputs/exp02_lora_world}"

echo "============================================"
echo "  Exp02: 双卡 LoRA + 懂世界 (2x RTX 3090)"
echo "  Model : ${MODEL_DIR}"
echo "  Train : dataset_v2/train.jsonl (53551 条)"
echo "  Eval  : dataset_v2/eval.jsonl (2817 条)"
echo "  Output: ${OUTPUT_DIR}"
echo "  Effective batch = 2 x 1 x 8 = 16"
echo "  max_seq_length   = 4096 + filter_long"
echo "  LoRA r=32, lr=1e-4, epochs=2"
echo "============================================"

SAMPLE_WEIGHTS='{"懂推荐": 2.0, "懂物料": 1.0, "懂用户": 5.0, "懂世界": 1.0}'

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

conda run -n worldrec --no-capture-output \
    torchrun \
        --nproc_per_node=2 \
        --master_port=29500 \
        src/train_sft.py \
            --model_path   "${MODEL_DIR}" \
            --train_file   "dataset_v2/train.jsonl" \
            --eval_file    "dataset_v2/eval.jsonl" \
            --eval_steps   500 \
            --output_dir   "${OUTPUT_DIR}" \
            --lora_r       32 \
            --lora_alpha   64 \
            --lora_dropout 0.1 \
            --learning_rate 1e-4 \
            --num_epochs   2 \
            --batch_size   1 \
            --gradient_accumulation_steps 8 \
            --max_seq_length 4096 \
            --filter_long_samples \
            --sample_weights "${SAMPLE_WEIGHTS}" \
            --save_steps   500 \
            --logging_steps 10 \
            --bf16

echo ""
echo "训练完成，输出目录: ${OUTPUT_DIR}/final"
echo "上传万擎平台需要: adapter_model.safetensors + adapter_config.json"
