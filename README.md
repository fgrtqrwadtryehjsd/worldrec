# 快手探索者 LLM-Rec 挑战赛 2026

基于 [OneReason-0.8B-pretrain-competition](https://huggingface.co/OpenOneRec/OneReason-0.8B-pretrain-competition) 模型的 SFT 微调方案，支持 LoRA 微调和全参微调两种模式。

## 项目结构

```
worldrec/
├── README.md
├── requirements.txt
├── .gitignore
├── info.md                    # 万擎平台使用指南
├── dataset.tar.gz             # 压缩数据集（需自行放置）
├── dataset/                   # 解压后的数据集（gitignore）
│   ├── 懂推荐1~4.jsonl        # 推荐能力训练数据 (~19k 条)
│   ├── 懂物料part1~7.jsonl    # 物料理解训练数据 (~10k 条)
│   └── 懂用户.jsonl           # 用户理解训练数据 (~3k 条)
├── configs/
│   └── ds_stage2.json         # DeepSpeed ZeRO Stage 2 配置
├── scripts/                   # 一键脚本
│   ├── setup.sh               # 配置 conda 环境
│   ├── download_model.sh      # 下载基础模型
│   ├── prepare_data.sh        # 解压数据集
│   ├── train_lora.sh          # 单卡 LoRA 训练
│   └── train_lora_2gpu.sh     # 双卡 LoRA 训练
├── src/
│   ├── data_utils.py          # 数据加载与预处理
│   ├── train_sft.py           # SFT LoRA 训练脚本 (低显存)
│   ├── train_full.py          # 全参微调训练脚本 (高显存)
│   └── analyze_data.py        # 数据集分析脚本
└── outputs/                   # 训练输出目录 (运行时生成)
```

## 快速开始（两张 RTX 3080）

```bash
# 1. 克隆代码
git clone <your-repo-url>
cd worldrec

# 2. 配置环境（约 5-15 min，含 flash-attn 编译）
bash scripts/setup.sh

# 3. 激活环境
conda activate worldrec

# 4. 下载基础模型（~1.7GB）
bash scripts/download_model.sh ~/models/OneReason-0.8B-pretrain-competition

# 5. 解压数据集（需将 dataset.tar.gz 放到项目根目录）
bash scripts/prepare_data.sh

# 6a. 双卡 LoRA 训练（推荐，两卡并行速度约 2x）
bash scripts/train_lora_2gpu.sh ~/models/OneReason-0.8B-pretrain-competition

# 6b. 单卡 LoRA 训练（备用）
bash scripts/train_lora.sh ~/models/OneReason-0.8B-pretrain-competition
```

## 环境配置

```bash
# 一键配置（推荐）
bash scripts/setup.sh

# 或手动配置
conda create -n worldrec python=3.11 -y
conda activate worldrec
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install deepspeed
pip install flash-attn --no-build-isolation  # 可选，3080 支持
```

## 模型下载

```bash
# 一键下载（推荐）
bash scripts/download_model.sh ~/models/OneReason-0.8B-pretrain-competition

# 或手动下载
python -c "
from huggingface_hub import snapshot_download
snapshot_download('OpenOneRec/OneReason-0.8B-pretrain-competition',
                  local_dir='~/models/OneReason-0.8B-pretrain-competition')
"
```

## 数据集说明

数据集包含三个维度，共约 32,000 条样本：

| 维度 | 文件 | 样本数 | Avg Tokens | 说明 |
|------|------|--------|------------|------|
| 懂推荐 | 懂推荐1~4.jsonl | ~19,200 | ~1,855 | 多域推荐（直播/电商/视频/广告） |
| 懂物料 | 懂物料part1~7.jsonl | ~10,400 | ~161 | 物料理解（desc↔token 双向） |
| 懂用户 | 懂用户.jsonl | ~2,900 | ~4,770 | 用户行为序列建模 |

> 懂物料包含 7 个子任务：商品/主播/广告/短视频的 token 生成（desc→token）和描述生成（token→desc）。

每条数据格式：
```json
[
  {
    "system": "系统提示词",
    "prompt": "用户输入",
    "response": "<think>推理过程</think>\n<itemic_token_sequence>"
  }
]
```

## 训练

### 两张 RTX 3080（10GB × 2）推荐方案

**双卡 LoRA（推荐，effective batch = 16，速度最快）**

```bash
bash scripts/train_lora_2gpu.sh \
    ~/models/OneReason-0.8B-pretrain-competition \
    dataset \
    outputs/lora_2gpu_v1
```

**单卡 LoRA（备用，effective batch = 16）**

```bash
bash scripts/train_lora.sh \
    ~/models/OneReason-0.8B-pretrain-competition \
    dataset \
    outputs/lora_v1
```

### 方式一：LoRA 微调（低显存，8GB+）

```bash
python src/train_sft.py \
  --model_path "D:/models/OneReason-0.8B-pretrain-competition" \
  --data_dir "dataset" \
  --output_dir "outputs/lora_v1" \
  --lora_r 64 \
  --lora_alpha 128 \
  --learning_rate 2e-4 \
  --num_epochs 3 \
  --batch_size 1 \
  --gradient_accumulation_steps 16 \
  --max_seq_length 2048
```

### 方式二：全参微调（高显存，24GB+）

```bash
# 基础全参微调
python src/train_full.py \
  --model_path "D:/models/OneReason-0.8B-pretrain-competition" \
  --data_dir "dataset" \
  --output_dir "outputs/full_v1" \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --batch_size 2 \
  --gradient_accumulation_steps 8 \
  --max_seq_length 4096 \
  --optim adamw_torch

# 使用 8-bit AdamW 节省显存
python src/train_full.py \
  --model_path "D:/models/OneReason-0.8B-pretrain-competition" \
  --data_dir "dataset" \
  --output_dir "outputs/full_v1" \
  --optim adamw_8bit \
  --batch_size 1 \
  --gradient_accumulation_steps 16

# 使用 DeepSpeed ZeRO Stage 2 多卡训练
python src/train_full.py \
  --model_path "D:/models/OneReason-0.8B-pretrain-competition" \
  --data_dir "dataset" \
  --output_dir "outputs/full_v1" \
  --deepspeed configs/ds_stage2.json \
  --batch_size 4 \
  --gradient_accumulation_steps 4
```

### 数据配比控制

通过 `--sample_weights` 参数控制不同维度的采样权重：

```bash
# 方案 A：均衡型（推荐初期）
python src/train_full.py \
  --sample_weights '{"懂推荐": 1.0, "懂物料": 1.0, "懂用户": 3.0}' \
  ...

# 方案 B：推荐优先型
python src/train_full.py \
  --sample_weights '{"懂推荐": 2.0, "懂物料": 1.0, "懂用户": 4.0}' \
  ...
```

### 数据分析

```bash
python src/analyze_data.py
```

### 关键参数

| 参数 | LoRA 默认 | 全参默认 | 说明 |
|------|-----------|---------|------|
| `--model_path` | - | - | 模型路径 |
| `--data_dir` | `dataset` | `dataset` | 数据目录 |
| `--learning_rate` | 2e-4 | 1e-5 | 学习率 |
| `--num_epochs` | 3 | 2 | 训练轮次 |
| `--batch_size` | 1 | 2 | 批大小 |
| `--gradient_accumulation_steps` | 16 | 8 | 梯度累积 |
| `--max_seq_length` | 2048 | 4096 | 最大序列长度 |
| `--lora_r` | 64 | - | LoRA 秩 |
| `--optim` | - | adamw_torch | 优化器（全参支持 adamw_8bit） |
| `--deepspeed` | - | - | DeepSpeed 配置文件路径 |
| `--sample_weights` | - | - | 数据配比权重（JSON） |

## 硬件要求

| 配置 | LoRA 单卡 | LoRA 双卡 (3080×2) | 全参微调 |
|------|---------|-------------------|---------|
| GPU 显存 | 8GB+ | 10GB × 2 ✅ | 24GB+ |
| 内存 | 16GB | 16GB | 32GB |
| 磁盘 | 5GB | 5GB | 10GB |

> **RTX 3080 (10GB) × 2** 最推荐使用双卡 LoRA 方案（`train_lora_2gpu.sh`），显存充裕、速度快。

## 模型上传

### LoRA 微调上传文件
- `adapter_model.safetensors`
- `adapter_config.json`

### 全参微调上传文件
- `model.safetensors`（如分片还需 `model.safetensors.index.json`）
- `config.json`
- `generation_config.json`

## 比赛信息

- 官网：https://ks-llmrec.streamlake.com/
- 平台：https://www.streamlake.com/product/wanqing
- Baseline & 数据集：https://huggingface.co/OpenOneRec/Explorer_LLM_Rec_Competition
