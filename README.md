# 快手探索者 LLM-Rec 挑战赛 2026

基于 [OneReason-0.8B-pretrain-competition](https://huggingface.co/OpenOneRec/OneReason-0.8B-pretrain-competition) 模型的 SFT LoRA 微调方案。

## 项目结构

```
worldrec/
├── README.md
├── requirements.txt
├── .gitignore
├── info.md                    # 万擎平台使用指南
├── dataset.tar.gz             # 压缩数据集
├── dataset/                   # 解压后的数据集
│   ├── 懂推荐1~4.jsonl        # 推荐能力训练数据 (~19k 条)
│   ├── 懂物料part1~7.jsonl    # 物料理解训练数据 (~10k 条)
│   └── 懂用户.jsonl           # 用户理解训练数据 (~3k 条)
├── src/
│   ├── data_utils.py          # 数据加载与预处理
│   └── train_sft.py           # SFT LoRA 训练脚本
└── outputs/                   # 训练输出目录 (运行时生成)
```

## 环境配置

```bash
# 创建 conda 环境
conda create -n worldrec python=3.11 -y
conda activate worldrec

# 安装 PyTorch (CUDA 12.1)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# 安装依赖
pip install -r requirements.txt
```

## 模型下载

```bash
# 方式一：huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('OpenOneRec/OneReason-0.8B-pretrain-competition', local_dir='D:/models/OneReason-0.8B-pretrain-competition')"

# 方式二：git lfs
git lfs install
git clone https://huggingface.co/OpenOneRec/OneReason-0.8B-pretrain-competition D:/models/OneReason-0.8B-pretrain-competition
```

## 数据集说明

数据集包含三个维度，共约 32,000 条样本：

| 维度 | 文件 | 样本数 | 说明 |
|------|------|--------|------|
| 懂推荐 | 懂推荐1~4.jsonl | ~19,200 | 多域推荐能力 |
| 懂物料 | 懂物料part1~7.jsonl | ~10,400 | 物料理解与 token 生成 |
| 懂用户 | 懂用户.jsonl | ~2,900 | 用户行为建模 |

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

```bash
# LoRA 微调（推荐，适合 8GB 显存）
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

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_path` | - | 模型路径 |
| `--data_dir` | `dataset` | 数据目录 |
| `--lora_r` | 64 | LoRA 秩 |
| `--lora_alpha` | 128 | LoRA alpha |
| `--learning_rate` | 2e-4 | 学习率 |
| `--num_epochs` | 3 | 训练轮次 |
| `--batch_size` | 1 | 批大小 |
| `--gradient_accumulation_steps` | 16 | 梯度累积步数 |
| `--max_seq_length` | 2048 | 最大序列长度 |
| `--use_gradient_checkpointing` | True | 梯度检查点（节省显存） |

## 硬件要求

| 配置 | 最低要求 | 推荐配置 |
|------|---------|---------|
| GPU 显存 | 6GB (LoRA) | 8GB+ |
| 内存 | 16GB | 32GB |
| 磁盘 | 5GB | 10GB |

## 模型上传

训练完成后，将 LoRA 权重上传至万擎平台进行评测：
- `adapter_model.safetensors`
- `adapter_config.json`

## 比赛信息

- 官网：https://ks-llmrec.streamlake.com/
- 平台：https://www.streamlake.com/product/wanqing
- Baseline & 数据集：https://huggingface.co/OpenOneRec/Explorer_LLM_Rec_Competition
