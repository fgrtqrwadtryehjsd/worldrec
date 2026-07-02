# Baseline 数据复原与 dataset 对比分析

> 分析时间：2026-07-02

## 结论

**`baseline/data/*.parquet` 可以复原出 `dataset/*.jsonl` 的绝大部分内容，但两者不是完全等价。**

- `dataset/` 是 baseline 原始数据经过**采样、组装、格式转换**后的成品训练集。
- `baseline/` 包含更多原始素材，且 `dataset/` 缺少一个完整维度：**懂世界/通识**。

---

## 1. Baseline 数据结构

`baseline/data/` 下包含 5 类 parquet：

| 目录 | 文件数 | 大小 | 内容 |
|------|--------|------|------|
| `OneReason_UserProfile` | 10 | ~8GB | 用户多域行为序列（电商/短视频/直播/广告） |
| `OneReason_Pid2Sid` | 198 | ~500MB | PID → 三段语义 ID (`<s_a_X><s_b_Y><s_c_Z>`) |
| `OneReason_Pid2Caption` | 136 | ~12GB | PID → 文本描述 |
| `OneReason_Pid2Tag` | 31 | ~70MB | PID → 三级类目标签 |
| `OneReason_General` | 158 | ~2GB | 通用知识/通识问答数据 |

---

## 2. 各类数据能否直接转成 JSONL

### 2.1 `OneReason_General` → 可以直接转

`OneReason_General` 的 parquet 中已经有 `messages` 字段，可直接用 `baseline/demo/convertv2.py` 转成 Alpaca JSONL。

```bash
python baseline/demo/convertv2.py \
  --input baseline/data/OneReason_General/ \
  --output baseline_reconstructed/懂世界.jsonl \
  --report --summary baseline_reconstructed/懂世界_summary.json
```

**实测**：`part-00000.parquet`（12.8MB）→ 1000 条标准对话样本，包含数学、科学、写作、常识等问答。

**这是 dataset 中完全没有的维度**。

### 2.2 `OneReason_UserProfile` → 不能直接转

`OneReason_UserProfile` 的 parquet 没有 `messages` 字段，而是用户行为序列。需要结合 `Pid2Sid` + `Pid2Caption` 自己组装：

```
UserProfile[行为序列]  --┐
Pid2Sid[PID→token]     ─┼→ 组装成 system/prompt/response 的 JSONL
Pid2Caption[PID→描述] ─┘
```

- 组装后可得到 `懂推荐` 和 `懂用户` 两类训练样本。
- `dataset/懂推荐1~4.jsonl` 和 `懂用户.jsonl` 就是这种组装后的成品。

### 2.3 `OneReason_Pid2Sid` + `OneReason_Pid2Caption` → 不能直接转

需要双向构造：

- **desc → token**（part1-4）：caption → 找 sid → 输出 token
- **token → desc**（part5-7）：sid → 找 caption → 输出描述

- 组装后得到 `dataset/懂物料part1~7.jsonl`。

### 2.4 `OneReason_Pid2Tag` → 未在 dataset 中明显出现

三级类目标签数据在 `dataset/` 中未直接对应，可能用于扩展任务或辅助训练。

---

## 3. Baseline 复原的 dataset 结构

```
baseline/data/
├── OneReason_General/         ──────→  懂世界.jsonl  (dataset 中没有！)
├── OneReason_UserProfile/           ─┐
├── OneReason_Pid2Sid/               ├→ 懂推荐1~4.jsonl + 懂用户.jsonl
├── OneReason_Pid2Caption/           │
└── OneReason_Pid2Tag/               └→ 懂物料part1~7.jsonl
```

---

## 4. 与 dataset 的差异

| 维度 | dataset 中 | baseline 中 | 差异说明 |
|------|-----------|-------------|---------|
| 懂推荐 | 懂推荐1~4.jsonl（19,204 条） | UserProfile + Pid2Sid/Caption 组装 | **同源，结构一致** |
| 懂物料 | 懂物料part1~7.jsonl（10,384 条） | Pid2Sid + Pid2Caption 组装 | **同源，结构一致** |
| 懂用户 | 懂用户.jsonl（2,892 条） | UserProfile 组装 | **同源，结构一致** |
| 懂世界 | **无** | OneReason_General | **dataset 缺失** |

> 注：baseline 原始数据是 16GB，dataset 是 436MB，主要差距在于：
> 1. baseline 保存全量用户/商品，dataset 只保留采样后的训练样本；
> 2. dataset 没有包含 `OneReason_General` 懂世界数据；
> 3. dataset 的 `懂用户` 和 `懂推荐` 可能是从 50 万用户中采样得到，远少于 baseline 全量。

---

## 5. 对训练的意义

### 5.1 当前 dataset 的不足

- **缺少懂世界数据**：baseline 评测显示，基线模型在懂世界上有 0.1409 分，但 `dataset/` 中没有对应训练数据，这 0.1409 来自预训练阶段的知识。若要提高，必须补充 `OneReason_General`。
- **懂用户样本太少**：`懂用户.jsonl` 只有 2,892 条，且基线评测中懂用户几乎为 0，需要提高采样权重或从 UserProfile 中构造更多样本。

### 5.2 建议的改进训练数据

基于 baseline 原始数据，可以扩展出更强的训练集：

```text
懂用户  : 5×   (从 UserProfile 多采样)
懂推荐  : 2×   (dataset 已有，但可补充不同配比)
懂物料  : 1×   (dataset 已有，基本够用)
懂世界  : 1×   (从 OneReason_General 转制，dataset 缺失！)
```

### 5.3 具体可执行步骤

1. 用 `convertv2.py` 把 `OneReason_General` 全部转成 `懂世界.jsonl`，混入训练。
2. 从 `OneReason_UserProfile` 中按更高采样率构造更多 `懂用户` 和 `懂推荐` 样本。
3. 合并 `dataset/*.jsonl` + `懂世界.jsonl` + 扩展的 `懂用户/懂推荐` 一起训练。

---

## 6. 验证脚本

### 6.1 转换懂世界

```bash
python baseline/demo/convertv2.py \
  --input baseline/data/OneReason_General/ \
  --output dataset/懂世界.jsonl \
  --report --summary dataset/懂世界_summary.json
```

### 6.2 从 UserProfile 构造懂用户/懂推荐（Python 示例）

见 `reconstruct_baseline.py`（采样版）。完整版需要遍历 16GB 数据，耗时较长，适合在训练服务器上批量处理。

---

## 7. 总结

> **Baseline 是"原材料仓库"，dataset 是"半成品便当"。**
>
> - dataset 已经能吃，但缺少懂世界这一道菜；
> - 从 baseline 可以复原/扩展出更多、更均衡的训练数据；
> - 如果想在评测中提升，**强烈建议把 `OneReason_General` 转成懂世界.jsonl 加入训练**。
