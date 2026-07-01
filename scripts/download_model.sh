#!/bin/bash
# 模型下载脚本
# 用法：bash scripts/download_model.sh [模型保存目录]
# 示例：bash scripts/download_model.sh ~/models/OneReason-0.8B
set -e

MODEL_DIR="${1:-$HOME/models/OneReason-0.8B-pretrain-competition}"
MODEL_ID="OpenOneRec/OneReason-0.8B-pretrain-competition"

echo "============================================"
echo "  下载 OneReason-0.8B-pretrain-competition"
echo "  保存至: ${MODEL_DIR}"
echo "============================================"

mkdir -p "${MODEL_DIR}"

# 优先用 huggingface_hub（支持断点续传）
conda run -n worldrec --no-capture-output python - <<PYEOF
from huggingface_hub import snapshot_download
import os

model_dir = "${MODEL_DIR}"
print(f"Downloading to {model_dir} ...")
snapshot_download(
    repo_id="${MODEL_ID}",
    local_dir=model_dir,
    ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],  # 只下载 PyTorch 权重
)
print("Download complete!")
PYEOF

echo ""
echo "模型已保存至: ${MODEL_DIR}"
echo "下一步: bash scripts/train_lora.sh 或 bash scripts/train_lora_2gpu.sh"
