#!/bin/bash
# 环境配置脚本 - 两张 RTX 3080 (Linux)
# 用法：bash scripts/setup.sh
set -e

echo "============================================"
echo "  OneReason LLM-Rec 2026 环境配置"
echo "============================================"

# -------- 1. conda 环境 --------
ENV_NAME="worldrec"

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "[1/6] conda 环境 '${ENV_NAME}' 已存在，跳过创建"
else
    echo "[1/6] 创建 conda 环境 python=3.11 ..."
    conda create -n "${ENV_NAME}" python=3.11 -y
fi

# 激活（在脚本内通过 conda run 调用，避免 source activate 的兼容问题）
CONDA_RUN="conda run -n ${ENV_NAME} --no-capture-output"

# -------- 2. PyTorch (CUDA 12.1) --------
echo "[2/6] 安装 PyTorch 2.5.1 + CUDA 12.1 ..."
$CONDA_RUN pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu121 -q

# -------- 3. 项目依赖 --------
echo "[3/6] 安装 requirements.txt ..."
$CONDA_RUN pip install -r requirements.txt -q

# -------- 4. DeepSpeed（多卡必需） --------
echo "[4/6] 安装 DeepSpeed ..."
$CONDA_RUN pip install deepspeed -q

# -------- 5. flash-attn（3080 是 Ampere，支持 FA2，可选，加速显著） --------
echo "[5/6] 安装 flash-attn (可选，编译较慢约 10min) ..."
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"
$CONDA_RUN pip install flash-attn --no-build-isolation -q || \
    echo "  [WARN] flash-attn 安装失败，将 fallback 到 sdpa，不影响训练"

# -------- 6. 验证 --------
echo "[6/6] 验证安装 ..."
$CONDA_RUN python - <<'EOF'
import torch
print(f"  torch      : {torch.__version__}")
print(f"  CUDA       : {torch.version.cuda}")
print(f"  GPU count  : {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}      : {props.name} ({props.total_memory / 1024**3:.1f} GB)")

import transformers, peft, accelerate, deepspeed
print(f"  transformers: {transformers.__version__}")
print(f"  peft        : {peft.__version__}")
print(f"  accelerate  : {accelerate.__version__}")
print(f"  deepspeed   : {deepspeed.__version__}")
EOF

echo ""
echo "============================================"
echo "  环境配置完成！"
echo "  激活命令: conda activate ${ENV_NAME}"
echo "  下一步:   bash scripts/download_model.sh"
echo "============================================"
