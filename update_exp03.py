import sys
from pathlib import Path

content = Path("experiments.md").read_text(encoding="utf-8")

exp03_text = """
---

## 🚀 Exp03 — 引入全量扩展数据 + 恢复长序列 + 提升世界常识权重 (RTX 3090 ×2)

| 项目 | 值 |
|------|-----|
| **状态** | 训练中 |
| **开始时间** | 2026-07-03 |
| **脚本** | `scripts/train_lora_2gpu_3090_v3.sh` |
| **模型** | `~/models/OneReason-0.8B-pretrain-competition` |
| **数据** | `dataset_v3/train.jsonl` (144857 条) + `dataset_v3/eval.jsonl` (7623 条) |
| **输出目录** | `outputs/exp03_lora_ext/` |

### 数据准备 (v3)

为了挽救懂世界指标，并彻底解决“懂用户”因截断导致的分数倒退，本实验进行了大规模数据扩充：

| 类别 | train | eval | 来源/处理 |
|------|-------|------|------|
| 懂推荐 | 37244 | 1960 | 基础 + `懂推荐_ext` |
| 懂物料 | 28865 | 1519 | 基础 + `懂物料_ext` |
| 懂用户 | 21748 | 1144 | 基础 + `懂用户_ext` |
| 懂世界 | 57000 | 3000 | `懂世界.jsonl` (转换格式, 抽样由3w提升至**6w**条) |
| **合计** | **144857**| **7623** | |

### 训练配置（vs Exp02 调整）

| 参数 | Exp02 | Exp03 | 调整理由 |
|------|-------|-------|---------|
| max_seq_length | 4096 | **8192** | 恢复长序列，拯救“懂用户” |
| filter_long_samples | True | **False** | 取消硬过滤，采用默认尾部自然截断 |
| sample_weights | 用户5/推荐2/物料1/世界1 | **用户5/推荐2/物料1/世界2** | 提升懂世界权重，对抗灾难性遗忘 |
| lora_r, dropout | 32, 0.1 | 32, 0.1 | 保持防过拟合设定 |
| learning_rate | 1e-4 | 1e-4 | 保持不变 |

### 评测结果 (待更新)

| 指标 | Baseline | Exp02 | **Exp03** |
|------|----------|-------|-------|
| **总分** | 0.6731 | 0.7910 | |
| 懂物料 | 0.1533 | 0.1840 | |
| 懂用户-1 | 0.0000 | 0.0514 | |
| 懂用户-2 | 0.0055 | 0.0394 | |
| 懂推荐-1 | 0.0960 | 0.0384 | |
| 懂推荐-2 | 0.0544 | 0.1088 | |
| 懂推荐-3 | 0.1330 | 0.1260 | |
| 懂推荐-4 | 0.0900 | 0.1044 | |
| 懂世界 | 0.1409 | 0.1337 | |

"""

# Replace the old planned Exp03 with the real one
if "## 🚀 Exp03" in content:
    start_idx = content.find("## 🚀 Exp03")
    content = content[:start_idx] + exp03_text.strip() + "\n"
else:
    content += "\n" + exp03_text.strip() + "\n"

Path("experiments.md").write_text(content, encoding="utf-8")
print("experiments.md updated successfully.")
