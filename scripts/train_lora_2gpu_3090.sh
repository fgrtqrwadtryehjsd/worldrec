#!/bin/bash
# 双卡 LoRA 训练 (两张 RTX 3090 24GB, 共 48GB)
# 相比 3080 方案：
#   - max_seq_length 8192  覆盖懂推荐 p95，懂用户会截断 ~5%（max~12300）
#     注：曾试 12288 完整覆盖懂用户，但 step 1077 遇到 12300 长样本 OOM
#   - batch_size 1 (vocab=176k 巨大，logits 矩阵 1×8192×176k 已~5.7GB，无法用 bs=2)
#   - grad_accum 8               effective batch = 2卡 × bs1 × accum8 = 16
#   - 加入 sample_weights           上调懂用户权重，避免被短样本淹没
#   - PYTORCH_CUDA_ALLOC_CONF       减少显存碎片
#
# 数据集分布（实测 token 估算）：
#   懂推荐 1-4 : 19204 条, avg~3860, p95~6400,  max~8700   → seq=8192 覆盖绝大多数
#   懂物料 1-7 : 10384 条, avg~125,  p95~165,   max~245    → 完全覆盖
#   懂用户     :  2892 条, avg~5725, p95~8800,  max~12300  → 截断 ~5%
#
# 显存估算 (0.8B bf16 + LoRA r64 + grad_ckpt, seq=8192, bs=1/卡):
#   模型~1.6GB + LoRA~0.5GB + 激活~3.5GB + logits~5.7GB + 梯度~1GB ≈ 12-16 GB/卡
#
# 注意：全参微调在 24GB 上不可行（vocab=176k 导致 logits 矩阵 bs×seq×176k 太大，
#   加上优化器状态会 OOM）。LoRA r=64 已足够覆盖 0.8B 模型的微调信号。
#
# 用法：bash scripts/train_lora_2gpu_3090.sh [MODEL_DIR] [DATA_DIR] [OUTPUT_DIR]
set -e

MODEL_DIR="${1:-$HOME/models/OneReason-0.8B-pretrain-competition}"
DATA_DIR="${2:-dataset}"
OUTPUT_DIR="${3:-outputs/lora_2gpu_3090_v1}"

echo "============================================"
echo "  双卡 LoRA 训练 (2x RTX 3090 24GB)"
echo "  Model : ${MODEL_DIR}"
echo "  Data  : ${DATA_DIR}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Effective batch = 2 x 1 x 8 = 16"
echo "  max_seq_length   = 8192 (懂用户截断~5%, 避免长样本 OOM)"
echo "============================================"

# 懂用户仅 2892 条（占 8.9%），但平均最长；懂物料 10384 条但极短
# 上调懂用户权重到 3.0，让每个 epoch 中懂用户 step 占比足够
SAMPLE_WEIGHTS='{"懂推荐": 1.0, "懂物料": 1.5, "懂用户": 3.0}'

# 减少显存碎片，避免 OOM（logits 矩阵 batch × seq × vocab=176k 极大）
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
            --sample_weights "${SAMPLE_WEIGHTS}" \
            --save_steps   200 \
            --logging_steps 10 \
            --bf16

echo ""
echo "训练完成，输出目录: ${OUTPUT_DIR}/final"
echo "上传万擎平台需要: adapter_model.safetensors + adapter_config.json"
