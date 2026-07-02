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

# Hugging Face 的大文件传输在部分网络环境下不稳定，关闭 xet/hf_transfer 并重试。
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0

# 优先用 huggingface_hub（支持断点续传）
conda run -n worldrec --no-capture-output python - <<PYEOF
from huggingface_hub import snapshot_download
import os
import time

model_dir = "${MODEL_DIR}"
print(f"Downloading to {model_dir} ...")
last_error = None
for attempt in range(1, 4):
    try:
        snapshot_download(
            repo_id="${MODEL_ID}",
            local_dir=model_dir,
            ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],  # 只下载 PyTorch 权重
        )
        last_error = None
        break
    except Exception as exc:
        last_error = exc
        print(f"Attempt {attempt} failed: {exc}")
        if attempt < 3:
            time.sleep(10 * attempt)

if last_error is not None:
    raise last_error
print("Download complete!")
PYEOF

echo ""
echo "模型已保存至: ${MODEL_DIR}"
echo "下一步: bash scripts/train_lora.sh 或 bash scripts/train_lora_2gpu.sh"
