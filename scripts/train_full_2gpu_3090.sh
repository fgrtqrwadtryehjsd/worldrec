#!/bin/bash
# 双卡全参微调 (两张 RTX 3090 24GB, 共 48GB)
# 相比 LoRA：更新全部 841M 参数，效果通常更好，但显存压力大
#
# 关键优化：
#   - 8-bit AdamW：优化器状态从 fp32(6.4GB) 压到 int8(1.6GB)，省 ~5GB
#   - DeepSpeed ZeRO Stage 2：梯度分片到各卡，减半梯度显存
#   - max_seq_length 8192：覆盖懂推荐 p95，懂用户会截断 ~5%（可接受）
#   - batch_size 2：双卡并行，吞吐量约翻倍
#
# 数据集分布（实测 token 估算）：
#   懂推荐 1-4 : 19204 条, avg~3860, p95~6400,  max~8700   → seq=8192 覆盖 p95
#   懂物料 1-7 : 10384 条, avg~125,  p95~165,   max~245    → 完全覆盖
#   懂用户     :  2892 条, avg~5725, p95~8800,  max~12300  → 截断 ~5%
#
# 显存估算 (0.8B bf16 + 全参 + grad_ckpt + ZeRO-2, seq=8192, bs=2/卡):
#   模型 1.6GB + 梯度 0.8GB(ZeRO分片) + 优化器 1.6GB(8bit) + 激活 5GB + logits 5GB ≈ 14-18 GB/卡
#
# 用法：bash scripts/train_full_2gpu_3090.sh [MODEL_DIR] [DATA_DIR] [OUTPUT_DIR]
set -e

MODEL_DIR="${1:-$HOME/models/OneReason-0.8B-pretrain-competition}"
DATA_DIR="${2:-dataset}"
OUTPUT_DIR="${3:-outputs/full_2gpu_3090_v1}"

echo "============================================"
echo "  双卡全参微调 (2x RTX 3090 24GB)"
echo "  Model : ${MODEL_DIR}"
echo "  Data  : ${DATA_DIR}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Effective batch = 2 x 2 x 4 = 16"
echo "  max_seq_length   = 8192 (懂用户截断~5%)"
echo "  Optimizer        = adamw_8bit"
echo "  DeepSpeed        = ZeRO Stage 2"
echo "============================================"

# 懂用户仅 2892 条（占 8.9%），但平均最长；上调权重避免被短样本淹没
SAMPLE_WEIGHTS='{"懂推荐": 1.0, "懂物料": 1.5, "懂用户": 3.0}'

# 减少显存碎片，避免 OOM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

conda run -n worldrec --no-capture-output \
    torchrun \
        --nproc_per_node=2 \
        --master_port=29500 \
        src/train_full.py \
            --model_path   "${MODEL_DIR}" \
            --data_dir     "${DATA_DIR}" \
            --output_dir   "${OUTPUT_DIR}" \
            --learning_rate 1e-5 \
            --num_epochs   2 \
            --batch_size   2 \
            --gradient_accumulation_steps 4 \
            --max_seq_length 8192 \
            --sample_weights "${SAMPLE_WEIGHTS}" \
            --optim       adamw_8bit \
            --deepspeed   configs/ds_stage2.json \
            --save_steps  500 \
            --logging_steps 10 \
            --bf16

echo ""
echo "训练完成，输出目录: ${OUTPUT_DIR}/final"
echo "上传万擎平台需要: model.safetensors + config.json + generation_config.json"
